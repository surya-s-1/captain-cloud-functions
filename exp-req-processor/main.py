import os
import json
import time
import logging
import datetime
import functools
import concurrent.futures as futures
from urllib.parse import urlparse
from typing import Any, Dict, TypeVar, List, Tuple

import functions_framework

# from dotenv import load_dotenv
# load_dotenv()

from google import genai
from google.genai.types import HttpOptions, Part, Content
from google.cloud import storage, firestore
from google.cloud.firestore_v1.transforms import Sentinel

# =====================
# Environment variables
# =====================
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
DUPE_SIM_THRESHOLD = float(os.getenv('DUPE_SIM_THRESHOLD'))
FIRESTORE_COMMIT_CHUNK = int(os.getenv('FIRESTORE_COMMIT_CHUNK'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE'))
MAX_PARALLEL_EMBEDDING_BATCHES = int(os.getenv('MAX_PARALLEL_EMBEDDING_BATCHES'))

# =====================
# Constants
# =====================

# Tunables (safe defaults for speed and cost-efficiency)
MAX_WORKERS = 16
MAX_DOC_SIZE_BYTES = 1048576 * 0.95

GENAI_API_VERSION = 'v1'
GENAI_TIMEOUT_SECONDS = 90  # Each LLM call safety timeout

# =====================
# Clients
# =====================
storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)

# Configure GenAI client (used for embeddings)
genai_client = genai.Client(http_options=HttpOptions(api_version=GENAI_API_VERSION))
logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('google.genai').setLevel(logging.WARNING)

# =====================
# Small utilities
# =====================


def _retry(max_attempts: int = 3, base_delay: float = 0.5):
    '''
    A decorator with exponential backoff and jitter for retrying transient errors.
    This helps to handle API rate limits and network issues gracefully.
    '''

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logging.warning(
                        f'Attempt {attempt}/{max_attempts} failed for {fn.__name__} with error: {e}'
                    )
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            return None  # Should be unreachable

        return wrapper

    return deco


def _update_firestore_status(project_id: str, version: str, status: str) -> None:
    '''Updates the status of a project version in Firestore.'''
    doc_ref = firestore_client.document('projects', project_id, 'versions', version)
    doc_ref.set({'status': status}, merge=True)
    print(f'Status => {status}')


def normalize_req_dict(req: Any) -> Any:
    '''
    Recursively normalizes keys in a dictionary (e.g., fixes inconsistent keys like ''requirement'' -> 'requirement').
    If the input is not a dict or list of dicts, it's returned as-is.
    '''
    if not isinstance(req, (dict, list)):
        return req

    if isinstance(req, list):
        return [normalize_req_dict(item) for item in req]

    fixed = {}
    for k, v in req.items():
        # Clean key of surrounding whitespace and quotes
        if isinstance(k, str):
            clean_key = k.strip().strip("'''").strip("'''").strip()
        else:
            clean_key = str(k).strip()

        # Recursively normalize nested dicts and lists
        fixed[clean_key] = normalize_req_dict(v)
    return fixed


T = TypeVar('T')


def _chunk_list(data: List[T], size: int) -> List[List[T]]:
    '''Yield successive n-sized chunks from a list.'''
    if not data:
        return []

    chunked_data = []
    for i in range(0, len(data), size):
        chunked_data.append(data[i : i + size])

    return chunked_data


def _firestore_json_converter(obj: Any) -> str:
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    if isinstance(obj, bytes):
        return obj.decode('utf-8')

    if isinstance(obj, Sentinel):
        return '<FIRESTORE_SENTINEL_PLACEHOLDER>'

    if isinstance(obj, (list, tuple)):
        return [_firestore_json_converter(item) for item in obj]

    if isinstance(obj, dict):
        return {k: _firestore_json_converter(v) for k, v in obj.items()}

    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _get_document_size_approx(data: Dict[str, Any]) -> int:
    try:
        data_json = json.dumps(data, default=_firestore_json_converter)
        return len(data_json.encode('utf-8'))
    except Exception as e:
        logging.warning(f'Failed to serialize document data for size check. Error: {e}')
        return -1


@_retry(max_attempts=3)
def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> None:
    batch = firestore_client.batch()
    batch_count = 0
    skipped_count = 0

    logging.info(f'Starting commit for {len(doc_tuples)} total documents...')

    for doc_ref, data in doc_tuples:
        data_size_bytes = _get_document_size_approx(data)

        if data_size_bytes == -1:
            logging.error(f'SKIPPING: {doc_ref.path} due to error during size check.')
            skipped_count += 1
            continue

        if data_size_bytes > MAX_DOC_SIZE_BYTES:
            logging.warning(
                f'SKIPPING: {doc_ref.path}. '
                f'Approximate size ({data_size_bytes / 1024:.2f} KiB) exceeds the '
                f'conservative limit ({MAX_DOC_SIZE_BYTES / 1024:.2f} KiB).'
            )
            skipped_count += 1
            continue

        batch.set(doc_ref, data)
        batch_count += 1

        if batch_count >= FIRESTORE_COMMIT_CHUNK:
            logging.info(f'Firestore => committing {batch_count} documents...')
            batch.commit()
            batch = firestore_client.batch()
            batch_count = 0

    if batch_count:
        logging.info(f'Firestore => committing final {batch_count} documents...')
        batch.commit()

    logging.warning(
        f'Commit finished. Total documents skipped due to size/error: {skipped_count}'
    )


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    '''Calculates the cosine similarity between two vectors using pure Python math.'''
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = sum(a * a for a in v1) ** 0.5
    magnitude_v2 = sum(b * b for b in v2) ** 0.5
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)


# =====================
# Core Processing Functions
# =====================


@_retry(max_attempts=3)
def _generate_embedding_batch(texts: List[str]) -> List[List[float]]:
    '''Generates vector embeddings for a list of texts using a single batch API call.'''
    if not texts:
        return []

    original_length = len(texts)

    try:
        contents = [Content(parts=[Part(text=t)]) for t in texts]

        with futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                lambda: genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents,
                )
            )
            response = future.result(timeout=GENAI_TIMEOUT_SECONDS)

        batch_embeddings = [e.values for e in response.embeddings]

        return batch_embeddings

    except Exception as e:
        logging.exception(
            f'Batch embedding generation failed for {len(texts)} texts. Error: {e}'
        )
        return [[]] * original_length


def _load_and_normalize_exp_req(version: str, obj_url: str) -> List[Dict[str, Any]]:
    print('Starting explicit requirement processing...')

    parsed = urlparse(obj_url)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip('/'))
    explicit_requirements_raw = json.loads(blob.download_as_text())

    if not explicit_requirements_raw:
        raise ValueError('Input data from GCS is empty.')

    normalized_list = normalize_req_dict(explicit_requirements_raw)

    final_list = []
    for i, r in enumerate(normalized_list, start=1):
        if not r.get('requirement') or not r.get('sources'):
            continue

        final_list.append(
            {
                'requirement_id': f'v{version}-REQ-E-{i:03d}',
                'requirement': r.get('requirement', ''),
                'requirement_category': r.get('requirement_type', 'Uncategorized'),
                'source_type': 'explicit',
                'sources': r.get('sources', []),
                'deleted': False,
                'duplicate': False,  # Initialize to False
                'near_duplicate_id': '',
                'embedding': [],
                'change_analysis_status': 'NEW',
                'change_analysis_status_reason': 'Newly created',
                'change_analysis_near_duplicate_id': '',
                'regulations': [],
                'parent_exp_req_ids': [],
                'testcase_status': 'NOT_STARTED',
                'updated_at': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP,
            }
        )

    print(f'Loaded and normalized {len(final_list)} explicit requirements.')

    return final_list


def _write_reqs_to_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    print(f'Generating embeddings for {len(requirements)} requirements...')

    texts_to_embed = [req.get('requirement', '') for req in requirements]
    embedding_vectors = []

    text_batches = _chunk_list(texts_to_embed, EMBEDDING_BATCH_SIZE)

    with futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EMBEDDING_BATCHES) as ex:
        batch_results = list(ex.map(_generate_embedding_batch, text_batches))
        for batch in batch_results:
            embedding_vectors.extend(batch)

    requirements_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    doc_tuples = []
    written_reqs = []

    for req, embedding_vector in zip(requirements, embedding_vectors):
        req_id = req.get('requirement_id')

        doc_data = {
            **req,
            'embedding': embedding_vector,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP,
        }

        doc_tuples.append(
            (
                requirements_collection_ref.document(req_id),
                doc_data,
            )
        )

        written_reqs.append(
            {
                'requirement_id': doc_data['requirement_id'],
                'embedding': doc_data['embedding'],
                'duplicate': doc_data['duplicate'],
                'source_type': doc_data['source_type'],
            }
        )

    _firestore_commit_many(doc_tuples)

    print(f'Explicit writes => {len(written_reqs)}')

    return written_reqs


def _mark_duplicates(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''
    Compares requirement embeddings and marks duplicates in Firestore using vector similarity.
    Returns (duplicates, originals)
    '''

    # Stores (duplicate_id, near_duplicate_id, duplicate_source_ids_to_merge)
    duplicates_to_update: List[Tuple[str, str, List[str]]] = []

    # Iterate through each requirement (i)
    for i in range(len(requirements)):
        req_i = requirements[i]

        # Compare req_i against all requirements (j) that came before it (j < i)
        for j in range(i):
            req_j = requirements[j]

            # Skip comparing against a requirement that has already been marked as a duplicate.
            if req_j.get('duplicate'):
                continue

            similarity = cosine_similarity(req_i['embedding'], req_j['embedding'])

            if similarity >= DUPE_SIM_THRESHOLD:
                # req_i is a duplicate of req_j (the earlier one is the original one)
                # Store the IDs and the source IDs from the duplicate req_i to be merged
                duplicates_to_update.append(
                    (
                        req_i['requirement_id'],
                        req_j['requirement_id'],
                        req_i.get('parent_exp_req_ids', []),
                    )
                )

                # Mark locally to prevent req_i from being a original source for future items
                # Once a match is found, we mark it and move to the next req_i
                req_i['duplicate'] = True
                break

    if not duplicates_to_update:
        print('No duplicates found using vector embeddings in this batch.')
        return [], requirements

    print(f'Found {len(duplicates_to_update)} duplicates to mark.')

    batch = firestore_client.batch()

    for req_id, near_duplicate_id, parent_exp_req_ids in duplicates_to_update:
        doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', req_id
        )
        batch.update(
            doc_ref, {'duplicate': True, 'near_duplicate_id': near_duplicate_id}
        )

        if parent_exp_req_ids:
            original_ref = firestore_client.document(
                'projects',
                project_id,
                'versions',
                version,
                'requirements',
                near_duplicate_id,
            )

            batch.update(
                original_ref,
                {'parent_exp_req_ids': firestore.ArrayUnion(parent_exp_req_ids)},
            )

    batch.commit()

    all_duplicates: List[Dict[str, Any]] = []
    all_originals: List[Dict[str, Any]] = []

    for req in requirements:
        if req.get('duplicate', False):
            all_duplicates.append(req)
        else:
            all_originals.append(req)

    print(f'Dedupe => Marked {len(all_duplicates)} explicit duplicates.')

    return all_duplicates, all_originals


# =====================
# Main HTTP Function
# =====================
@functions_framework.http
def process_explicit_requirements(request):
    '''
    Phase 2a: Loads, normalizes, persists, and de-duplicates
    EXPLICIT requirements from a GCS JSON file.
    '''
    project_id = None
    version = None

    try:
        payload = request.get_json(silent=True) or {}
        # Using mock payload for local testing
        # payload = {
        #     'project_id': 'abc',
        #     'version': 'v1',
        #     'requirements_p1_url': 'gs://genai-sage/projects/abc/v_v1/extractions/requirements-phase-1.json',
        # }

        project_id = payload.get('project_id')
        version = payload.get('version')
        reqs_url = payload.get('requirements_p1_url')

        if not all([project_id, version, reqs_url]):
            return (
                json.dumps(
                    {
                        'status': 'error',
                        'message': 'Required details (project_id, version, requirements_p1_url) are missing.',
                    }
                ),
                400,
            )

        _update_firestore_status(project_id, version, 'START_EXP_REQ_EXTRACT')

        normalised_reqs = _load_and_normalize_exp_req(version, reqs_url)

        _update_firestore_status(project_id, version, 'START_STORE_EXPLICIT')

        explicit_reqs = _write_reqs_to_firestore(project_id, version, normalised_reqs)

        _update_firestore_status(project_id, version, 'START_DEDUPE_EXPLICIT')

        _mark_duplicates(project_id, version, explicit_reqs)

        _update_firestore_status(project_id, version, 'CONFIRM_EXP_REQ_EXTRACT')

        return (
            json.dumps({'status': 'success'}),
            200,
        )

    except Exception as e:
        logging.exception('Error during requirements extraction phase 2a (EXPLICIT):')

        if project_id and version:
            _update_firestore_status(project_id, version, 'ERR_EXP_REQ_EXTRACT')

        return (
            json.dumps(
                {
                    'status': 'error',
                    'message': f'An unexpected error occurred during processing: {str(e)}',
                }
            ),
            500,
        )


# process_explicit_requirements(None)
