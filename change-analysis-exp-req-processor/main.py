import os
import json
import time
import logging
import datetime
import functools
import concurrent.futures as futures
from typing import Any, Dict, List, Tuple, TypeVar
from urllib.parse import urlparse
from google import genai
from google.genai.types import HttpOptions, Part, Content, EmbedContentConfig
from google.cloud import storage, firestore
from google.cloud.firestore_v1.transforms import Sentinel
import numpy as np
import faiss

import functions_framework

# from dotenv import load_dotenv
# load_dotenv()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# ===================== # Environment variables # =====================
# These are required dependencies for the functions in this file
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
FIRESTORE_COMMIT_CHUNK = os.getenv('FIRESTORE_COMMIT_CHUNK')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE'))
EMBEDDING_OUTPUT_DIMENSION = int(os.getenv('EMBEDDING_OUTPUT_DIMENSION'))
MAX_PARALLEL_EMBEDDING_BATCHES = int(os.getenv('MAX_PARALLEL_EMBEDDING_BATCHES'))
DUPE_SIM_THRESHOLD = float(os.getenv('DUPE_SIM_THRESHOLD'))
REQ_UNCHANGED_SIM_THRESHOLD = float(os.getenv('REQ_UNCHANGED_SIM_THRESHOLD'))
REQ_MODIFIED_SIM_THRESHOLD = float(os.getenv('REQ_MODIFIED_SIM_THRESHOLD'))
REQ_DEPRECATED_SIM_THRESHOLD = REQ_MODIFIED_SIM_THRESHOLD

MAX_WORKERS = 16
MAX_DOC_SIZE_BYTES = 1048576 * 0.95

SOURCE_TYPE_EXPLICIT = 'explicit'

CHANGE_STATUS_NEW = 'NEW'
CHANGE_STATUS_MODIFIED = 'MODIFIED'
CHANGE_STATUS_UNCHANGED = 'UNCHANGED'
CHANGE_STATUS_IGNORED = 'IGNORED'
CHANGE_STATUS_DEPRECATED = 'DEPRECATED'
CHANGE_STATUS_IGNORED_REASON_DELETED = (
    'Deleted in previous version, ignored for new version analysis.'
)
CHANGE_STATUS_IGNORED_REASON_DUPLICATE = (
    'Marked as duplicate in previous version, ignored for new version analysis.'
)
CHANGE_STATUS_IGNORED_REASON_DEPRECATED = (
    'Deprecated in previous version, ignored for new version analysis.'
)
CHANGE_STATUS_IGNORED_REASON_IGNORED = (
    'Ignored in previous version, sp ignored for new version analysis as well.'
)

# ===================== # Clients # =====================
storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)

# Configure GenAI client (used for embeddings)
GENAI_API_VERSION = 'v1'
GENAI_TIMEOUT_SECONDS = 90

genai_client = genai.Client(http_options=HttpOptions(api_version=GENAI_API_VERSION))


# ===================== # Small utilities # =====================
def _retry(max_attempts: int = 3, base_delay: float = 0.5):
    '''Decorator for retrying functions with exponential backoff.'''

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f'Attempt {attempt}/{max_attempts} failed for {fn.__name__} with error: {e}'
                    )
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2
            return None

        return wrapper

    return deco


def _update_version_status(project_id: str, version: str, status: str) -> None:
    '''Updates the status of a project version in Firestore.'''
    doc_ref = firestore_client.document('projects', project_id, 'versions', version)
    doc_ref.set({'status': status}, merge=True)
    logger.info(f'Status => {status}')


def _normalize_requirements(req: Any) -> Any:
    '''Recursively normalizes keys in a dictionary.'''
    if not isinstance(req, (dict, list)):
        return req
    if isinstance(req, list):
        return [_normalize_requirements(item) for item in req]
    fixed = {}
    for k, v in req.items():
        if isinstance(k, str):
            clean_key = k.strip("'''").strip("'''").strip()
        else:
            clean_key = str(k).strip()
        fixed[clean_key] = _normalize_requirements(v)
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
        logger.warning(f'Failed to serialize document data for size check. Error: {e}')
        return -1


@_retry(max_attempts=3)
def _commit_single_batch(
    batch_doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> None:
    '''Commits a single batch of documents to Firestore.'''
    if not batch_doc_tuples:
        return

    batch = firestore_client.batch()
    batch_count = 0
    skipped_count = 0

    for doc_ref, data in batch_doc_tuples:
        data_size_bytes = _get_document_size_approx(data)

        if data_size_bytes == -1:
            logger.info(f'SKIPPING: {doc_ref.path} due to error during size check.')
            skipped_count += 1
            continue

        if data_size_bytes > MAX_DOC_SIZE_BYTES:
            logger.info(
                f'SKIPPING: {doc_ref.path}. '
                f'Approximate size ({data_size_bytes / 1024:.2f} KiB) exceeds the '
                f'conservative limit ({MAX_DOC_SIZE_BYTES / 1024:.2f} KiB).'
            )

            skipped_count += 1
            continue

        batch.set(doc_ref, data)
        batch_count += 1

    if batch_count > 0:
        batch.commit()
        logger.info(f'Firestore => committed {batch_count} documents in a batch.')

    if skipped_count > 0:
        logger.info(f'Firestore => skipped {skipped_count} documents in a batch.')


@_retry(max_attempts=3)
def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> None:
    '''Commits documents to Firestore in batches, using threading for parallelism.'''

    logger.info(f'Starting commit for {len(doc_tuples)} total documents...')

    chunked_doc_tuples = _chunk_list(doc_tuples, FIRESTORE_COMMIT_CHUNK)

    logger.info(f'Total batches to commit => {len(chunked_doc_tuples)}')

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        ex.map(_commit_single_batch, chunked_doc_tuples)

    logger.info('Commit finished.')


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    '''Calculates the cosine similarity between two vectors.'''
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = sum(a * a for a in v1) ** 0.5
    magnitude_v2 = sum(b * b for b in v2) ** 0.5
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)


@_retry(max_attempts=3)
def _generate_embedding_batch(texts: List[str]) -> List[List[float]]:
    '''Generates vector embeddings for a list of texts using a single batch API call.'''
    if not texts:
        return []

    original_length = len(texts)

    logger.info(f'Generating embeddings for {len(texts)} texts...')

    try:
        contents = [Content(parts=[Part(text=t)]) for t in texts]

        perf_start = time.time()

        with futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                lambda: genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents,
                    config=EmbedContentConfig(
                        auto_truncate=True,
                        output_dimensionality=EMBEDDING_OUTPUT_DIMENSION,
                        task_type='SEMANTIC_SIMILARITY',
                    ),
                )
            )
            response = future.result(timeout=GENAI_TIMEOUT_SECONDS)

        batch_embeddings = [e.values for e in response.embeddings]

        perf_end = time.time()

        logger.info(
            f'Time taken for this embedding batch (in seconds): {perf_end - perf_start}'
        )

        return batch_embeddings

    except Exception as e:
        logger.exception(
            f'Batch embedding generation failed for {len(texts)} texts. Error: {e}'
        )
        return [[]] * original_length


def _load_newly_uploaded_exp_requirements(
    version: str, requirements_p1_url: str
) -> List[Dict[str, Any]]:
    '''Loads, normalizes, and embeds new explicit requirements from GCS.'''
    logger.info('Starting new explicit requirement processing from GCS...')
    parsed = urlparse(requirements_p1_url)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip('/'))
    explicit_requirements_raw = json.loads(blob.download_as_text())

    if not explicit_requirements_raw:
        raise ValueError('Input data from GCS is empty.')

    normalized_list: List[Dict[str, Any]] = _normalize_requirements(
        explicit_requirements_raw
    )

    embedding_vectors = []

    texts_to_embed = [r.get('requirement', '') for r in normalized_list]
    text_batches = _chunk_list(texts_to_embed, EMBEDDING_BATCH_SIZE)

    with futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EMBEDDING_BATCHES) as ex:
        batch_results = list(ex.map(_generate_embedding_batch, text_batches))
        for batch in batch_results:
            embedding_vectors.extend(batch)

    final_list = []
    for i, (r, embedding_vector) in enumerate(
        zip(normalized_list, embedding_vectors), start=1
    ):
        final_list.append(
            {
                'requirement_id': f'v{version}-REQ-E-{i:03d}',
                'requirement': r.get('requirement', ''),
                'requirement_category': r.get('requirement_type', 'Uncategorized'),
                'source_type': SOURCE_TYPE_EXPLICIT,
                'sources': r.get('sources', []),
                'deleted': False,
                'duplicate': False,  # Initialize to False
                'near_duplicate_id': '',
                'embedding': embedding_vector,
                'change_analysis_status': CHANGE_STATUS_NEW,
                'change_analysis_status_reason': 'Newly created',
                'change_analysis_near_duplicate_id': '',
                'regulations': [],
                'parent_exp_req_ids': [],
                'testcase_status': 'NOT_STARTED',
                'toolCreated': 'NOT_STARTED',
                'toolIssueKey': '',
                'toolIssueLink': '',
                'updated_at': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP,
            }
        )
    logger.info(
        f'Loaded, normalized, and embedded {len(final_list)} explicit requirements.'
    )
    return final_list


def _load_existing_exp_requirements(
    project_id: str, version: str
) -> List[Dict[str, Any]]:
    logger.info('Loading existing requirements from Firestore for change detection...')

    collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    ).where('source_type', '==', SOURCE_TYPE_EXPLICIT)

    batch = firestore_client.batch()

    deleted_query = collection_ref.where('deleted', '==', True)
    for doc in deleted_query.stream():
        batch.update(
            doc.reference,
            {
                'change_analysis_status': CHANGE_STATUS_IGNORED,
                'change_analysis_status_reason': CHANGE_STATUS_IGNORED_REASON_DELETED,
            },
        )

    duplicate_query = collection_ref.where('duplicate', '==', True)
    for doc in duplicate_query.stream():
        batch.update(
            doc.reference,
            {
                'change_analysis_status': CHANGE_STATUS_IGNORED,
                'change_analysis_status_reason': CHANGE_STATUS_IGNORED_REASON_DUPLICATE,
            },
        )

    deprecated_query = collection_ref.where(
        'change_analysis_status', '==', CHANGE_STATUS_DEPRECATED
    )
    for doc in deprecated_query.stream():
        batch.update(
            doc.reference,
            {
                'change_analysis_status': CHANGE_STATUS_IGNORED,
                'change_analysis_status_reason': CHANGE_STATUS_IGNORED_REASON_DEPRECATED,
            },
        )

    ignored_query = collection_ref.where(
        'change_analysis_status', '==', CHANGE_STATUS_IGNORED
    )
    for doc in ignored_query.stream():
        batch.update(
            doc.reference,
            {
                'change_analysis_status_reason': CHANGE_STATUS_IGNORED_REASON_IGNORED,
            },
        )

    logger.info('Committing all batch updates...')
    batch.commit()
    logger.info('Batch committed successfully.')

    query = (
        firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )
        .where('deleted', '==', False)
        .where('duplicate', '==', False)
        .where('source_type', '==', SOURCE_TYPE_EXPLICIT)
        .where('change_analysis_status', '!=', CHANGE_STATUS_IGNORED)
        # We only need to exclude IGNORED here because the batch operation above
        # moved all DEPRECATED, Deleted, and Duplicate docs into the IGNORED status.
    )

    existing_reqs = []
    for doc in query.stream():
        data = doc.to_dict()
        if 'requirement' in data and 'embedding' in data:
            existing_reqs.append(data)

    logger.info(f'Loaded {len(existing_reqs)} existing requirements.')

    return existing_reqs


def _mark_new_reqs_change_status(
    new_exp_reqs: List[Dict[str, Any]], existing_exp_reqs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''Compares new explicit requirements against existing ones to mark status (UNCHANGED/MODIFIED/NEW).'''

    logger.info(f'Starting change detection for {len(new_exp_reqs)} new texts...')

    old_ids_checked = set()

    for new_req in new_exp_reqs:
        best_match = None
        max_sim_score = -1.0

        # Find the best match among existing explicit requirements
        for old_req in existing_exp_reqs:
            if old_req['source_type'] != SOURCE_TYPE_EXPLICIT:
                continue

            sim_score = _cosine_similarity(new_req['embedding'], old_req['embedding'])

            if sim_score > max_sim_score:
                max_sim_score = sim_score
                best_match = old_req

        if max_sim_score >= REQ_UNCHANGED_SIM_THRESHOLD:
            new_req['change_analysis_status'] = CHANGE_STATUS_UNCHANGED
            new_req['change_analysis_status_reason'] = (
                'Did not detect any major changes in updated requirements'
            )
            new_req['change_analysis_near_duplicate_id'] = best_match['requirement_id']
            new_req['testcase_status'] = best_match.get('testcase_status', '')
            old_ids_checked.add(best_match['requirement_id'])

        elif max_sim_score >= REQ_MODIFIED_SIM_THRESHOLD:
            new_req['change_analysis_status'] = CHANGE_STATUS_MODIFIED
            new_req['change_analysis_status_reason'] = (
                'Detected considerable modifications in updated requirements'
            )
            new_req['change_analysis_near_duplicate_id'] = best_match['requirement_id']
            new_req['testcase_status'] = 'NOT_STARTED'
            old_ids_checked.add(best_match['requirement_id'])

        else:
            new_req['change_analysis_status'] = CHANGE_STATUS_NEW
            new_req['change_analysis_status_reason'] = (
                'Detected only in new version requirements'
            )
            new_req['change_analysis_near_duplicate_id'] = None
            new_req['testcase_status'] = 'NOT_STARTED'

    logger.info(
        f'Statuses: UNCHANGED={len([r for r in new_exp_reqs if r['change_analysis_status'] == CHANGE_STATUS_UNCHANGED])},'
        f' MODIFIED={len([r for r in new_exp_reqs if r['change_analysis_status'] == CHANGE_STATUS_MODIFIED])},'
        f' NEW={len([r for r in new_exp_reqs if r['change_analysis_status'] == CHANGE_STATUS_NEW])}'
    )

    # Filter existing explicit requirements to only include those that were NOT matched
    old_exp_to_check = [
        r
        for r in existing_exp_reqs
        if r['requirement_id'] not in old_ids_checked
        and r['source_type'] == SOURCE_TYPE_EXPLICIT
    ]

    return new_exp_reqs, old_exp_to_check


def _mark_old_reqs_deprecated(old_reqs_to_check: List[Dict[str, Any]]) -> List[str]:
    '''Step 2: Marks unmatched old explicit requirements as DEPRECATED.'''
    logger.info(
        f'Starting deprecation check for {len(old_reqs_to_check)} old explicit texts...'
    )
    deprecated_ids = [
        r.get('requirement_id') for r in old_reqs_to_check if r.get('requirement_id')
    ]
    logger.info(
        f'Marking {len(deprecated_ids)} old explicit requirements as DEPRECATED.'
    )
    return deprecated_ids


def _mark_deprecated_in_firestore(
    project_id: str, version: str, deprecated_exp_ids: List[str]
) -> None:
    logger.info('Committing status updates to Firestore (Explicit Deprecation)...')

    batch = firestore_client.batch()
    req_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    for req_id in deprecated_exp_ids:
        doc_ref = req_collection_ref.document(req_id)
        batch.update(
            doc_ref,
            {
                'change_analysis_status': CHANGE_STATUS_DEPRECATED,
                'change_analysis_status_reason': 'Not detected in updated requirements',
                'updated_at': firestore.SERVER_TIMESTAMP,
            },
        )

    batch.commit()


def _mark_unchanged_modified_new_in_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    '''Persists new, modified, and unchanged explicit requirements to Firestore.'''

    logger.info(f'Persisting {len(requirements)} explicit requirements to firestore...')

    requirements_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    doc_insertions_tuples_list = []
    doc_updates_tuples_list = []
    written_reqs = []

    for req in requirements:
        req_source_type = req.get('source_type', '')
        req_change_status = req.get('change_analysis_status', '')
        change_analysis_near_duplicate_id = req.get(
            'change_analysis_near_duplicate_id', ''
        )

        if req_source_type == SOURCE_TYPE_EXPLICIT:
            # Case 1: EXPLICIT NEW (New insertions)
            if req_change_status == CHANGE_STATUS_NEW:
                req_id = req.get('requirement_id')

                doc_data = {
                    **req,
                    'requirement_id': req_id,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                    'created_at': firestore.SERVER_TIMESTAMP,
                }

                doc_ref = requirements_collection_ref.document(req_id)
                doc_insertions_tuples_list.append((doc_ref, doc_data))

                req['requirement_id'] = req_id

            # Case 2: EXPLICIT MODIFIED/UNCHANGED (Update existing document)
            elif (
                req_change_status in (CHANGE_STATUS_MODIFIED, CHANGE_STATUS_UNCHANGED)
                and change_analysis_near_duplicate_id
            ):
                req_id = change_analysis_near_duplicate_id

                doc_data = {
                    **req,
                    'requirement_id': req_id,
                    'updated_at': firestore.SERVER_TIMESTAMP,
                }

                doc_ref = requirements_collection_ref.document(req_id)
                doc_updates_tuples_list.append((doc_ref, doc_data))

                req['requirement_id'] = req_id
            else:
                logger.info(
                    f'WARNING: Explicit requirement skipped due to unknown status \'{req_change_status}\' or missing \'change_analysis_near_duplicate_id\'.'
                )
                continue

        written_reqs.append(
            {
                'requirement_id': req.get('requirement_id'),
                'source_type': req_source_type,
                'requirement': req.get('requirement', ''),
                'embedding': req.get('embedding', []),
                'duplicate': req.get('duplicate', False),
                'parent_exp_req_ids': req.get('parent_exp_req_ids', []),
                'change_analysis_near_duplicate_id': req.get(
                    'change_analysis_near_duplicate_id', ''
                ),
                'change_analysis_status': req_change_status,
                'sources': req.get('sources', []),
            }
        )

    if doc_insertions_tuples_list:
        logger.info(
            f'Committing {len(doc_insertions_tuples_list)} INSERT/SET operations...'
        )
        _firestore_commit_many(doc_insertions_tuples_list)

    if doc_updates_tuples_list:
        logger.info(f'Committing {len(doc_updates_tuples_list)} UPDATE operations...')
        # Updates must use batch.update() which requires the document to exist.
        batch = firestore_client.batch()
        for doc_ref, data in doc_updates_tuples_list:
            batch.update(doc_ref, data)
        batch.commit()

    return written_reqs


def _mark_duplicates(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''
    Compares requirement embeddings and marks duplicates in Firestore using vector similarity.
    Uses Faiss for O(N log N) or better performance.
    Returns (duplicates, originals)
    '''

    # Stores (duplicate_id, near_duplicate_id, duplicate_source_ids_to_merge)
    duplicates_to_update: List[Tuple[str, str, List[str], List[Dict[str, Any]]]] = []

    perf_start = time.time()

    # 1. Prepare Data and Build Index
    if not requirements:
        return [], []

    # Filter out requirements that are already duplicates and prepare data
    original_requirements = [
        req for req in requirements if not req.get('duplicate', False)
    ]

    if len(original_requirements) < 2:
        # Cannot have duplicates if there's 0 or 1 item
        return [], requirements

    # Extract embeddings and map index back to original requirement_id
    embeddings = np.array(
        [req['embedding'] for req in original_requirements], dtype='float32'
    )

    req_ids = [req['requirement_id'] for req in original_requirements]

    dimension = embeddings.shape[1]

    # L2-normalize the embeddings (necessary for using L2 distance as cosine similarity)
    faiss.normalize_L2(embeddings)

    # Create the index (FlatIndex uses exact search, a good starting point)
    # For massive scale, you'd use IndexHNSWFlat or IndexIVFFlat for true ANN speedup
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 2. Perform Nearest Neighbors Search (All-vs-All in one go)
    # k=2 because the nearest neighbor to a vector is always itself (distance 0).
    # We want the *second* nearest neighbor (if it exists).
    k = 2

    # D is Distances (Squared L2), I is Indices
    D, I = index.search(embeddings, k)

    # Set to keep track of already marked duplicates by their index in the 'original_requirements' list
    marked_indices = set()

    # 3. Process Results
    # Iterate over the results of the nearest neighbor search
    for i in range(len(original_requirements)):
        # Skip if this requirement has already been marked as a duplicate
        if i in marked_indices:
            continue

        # The first index I[i, 0] is always 'i' itself (nearest neighbor)
        # The second index I[i, 1] is the near_duplicate_index
        near_duplicate_index = I[i, 1]

        # The squared L2 distance D[i, 1] is between req_i and req_j (its nearest neighbor)
        sq_dist = D[i, 1]

        # Squared L2 distance (d^2) and Cosine Similarity (cos_sim) are related for
        # L2-normalized vectors (where ||v||=1):
        # d^2 = 2 * (1 - cos_sim) => cos_sim = 1 - (d^2 / 2)
        # A squared L2 distance of 0.0 corresponds to a cos_sim of 1.0.
        # Threshold: 1 - (DUPE_SIM_THRESHOLD) * 2
        # Example: If DUPE_SIM_THRESHOLD is 0.95, max_sq_dist is 2 * (1 - 0.95) = 0.1
        MAX_SQ_DIST = 2 * (1 - DUPE_SIM_THRESHOLD)

        # Check if the near neighbor is close enough and is not the vector itself
        # Note: I[i, 1] < len(original_requirements) check is for safety,
        # Faiss returns max index if k > N.
        if near_duplicate_index < len(original_requirements) and sq_dist <= MAX_SQ_DIST:

            # Found a match! The original one should be the one that appeared earlier (lower index i)
            # The *duplicate* is the one being marked.

            req_i = original_requirements[i]
            req_j = original_requirements[near_duplicate_index]

            # Determine which is the 'original' and which is the 'duplicate' based on
            # their position in the *original list* of requirements passed to the function.
            # We assume the one with the earlier index in the *original* list is the 'original'.

            # Find the original index for sorting logic
            original_i_index = next(
                (
                    idx
                    for idx, req in enumerate(requirements)
                    if req['requirement_id'] == req_i['requirement_id']
                )
            )
            original_j_index = next(
                (
                    idx
                    for idx, req in enumerate(requirements)
                    if req['requirement_id'] == req_j['requirement_id']
                )
            )

            if original_i_index < original_j_index:
                # req_j is the duplicate of req_i (the earlier one)
                duplicate_req = req_j
                original_req = req_i
                duplicate_idx = near_duplicate_index
            else:
                # req_i is the duplicate of req_j (the earlier one)
                duplicate_req = req_i
                original_req = req_j
                duplicate_idx = i

            # Store the IDs and the source IDs from the duplicate to be merged
            duplicates_to_update.append(
                (
                    original_req['requirement_id'],
                    duplicate_req['requirement_id'],
                    duplicate_req.get('parent_exp_req_ids', []),
                    duplicate_req.get('sources', []),
                )
            )

            # Mark the duplicate's index to prevent it from being processed as an original later
            marked_indices.add(duplicate_idx)

            # Mark locally to filter results later
            duplicate_req['duplicate'] = True

    perf_end_search = time.time()

    logger.info(
        f'Time to find duplicates with Faiss (in seconds): {perf_end_search - perf_start}'
    )

    if not duplicates_to_update:
        logger.info('No duplicates found using vector embeddings in this batch.')
        return [], requirements

    logger.info(f'Found {len(duplicates_to_update)} duplicates to mark.')

    batch = firestore_client.batch()
    batch_count = 0

    for original_id, dupe_req_id, dupe_parent_ids, dupe_sources in duplicates_to_update:
        if batch_count >= FIRESTORE_COMMIT_CHUNK:
            batch.commit()
            batch_count = 0

        dupe_doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', dupe_req_id
        )

        batch.update(
            dupe_doc_ref, {'duplicate': True, 'near_duplicate_id': original_id}
        )
        batch_count += 1

        if dupe_parent_ids or dupe_sources:
            original_ref = firestore_client.document(
                'projects',
                project_id,
                'versions',
                version,
                'requirements',
                original_id,
            )

            updates = {
                'updated_at': firestore.SERVER_TIMESTAMP,
            }

            if dupe_parent_ids:
                updates['parent_exp_req_ids'] = firestore.ArrayUnion(dupe_parent_ids)

            if dupe_sources:
                updates['sources'] = firestore.ArrayUnion(dupe_sources)

            batch.update(original_ref, updates)
            batch_count += 1

    if batch_count > 0:
        batch.commit()

    perf_end = time.time()

    logger.info(
        f'Time taken to find and update duplicates (in seconds): {perf_end - perf_start}'
    )

    all_duplicates: List[Dict[str, Any]] = []
    all_originals: List[Dict[str, Any]] = []

    for req in requirements:
        if req.get('duplicate', False):
            all_duplicates.append(req)
        else:
            all_originals.append(req)

    logger.info(f'Dedupe => Marked {len(all_duplicates)} explicit duplicates.')

    return all_duplicates, all_originals


# ===================== # Main HTTP Function # =====================


@functions_framework.http
def explicit_req_processor_change_analysis(request):
    '''
    Main Cloud Function entry point for Explicit Requirement processing (Phase 2).
    This function handles change detection, deprecation, explicit persistence,
    and explicit deduplication. It stops before processing implicit requirements.
    '''
    project_id = None
    version = None
    try:
        payload = request.get_json(silent=True) or {}
        # Mock data for local testing
        # payload = {
        #     'project_id': 'abc',
        #     'version': '2',
        #     'requirements_p1_url': 'gs://genai-sage/projects/abc/v_2/extractions/requirements-phase-1.json',
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

        _update_version_status(project_id, version, 'START_EXP_REQ_EXTRACT')

        existing_exp_reqs = _load_existing_exp_requirements(project_id, version)

        newly_uploaded_exp_reqs = _load_newly_uploaded_exp_requirements(
            version, reqs_url
        )

        _update_version_status(project_id, version, 'START_CHANGE_DETECTION')

        # 1. Compare new list against old list (Find UNCHANGED, MODIFIED, NEW)
        new_exp_reqs, old_exp_to_check = _mark_new_reqs_change_status(
            newly_uploaded_exp_reqs, existing_exp_reqs
        )

        # 2. Mark old unmatched explicit requirements as DEPRECATED
        deprecated_exp_ids = _mark_old_reqs_deprecated(old_exp_to_check)

        _update_version_status(project_id, version, 'START_DEPRECATION_EXPLICIT')

        # 3. Commit explicit deprecation status updates
        _mark_deprecated_in_firestore(project_id, version, deprecated_exp_ids)

        _update_version_status(project_id, version, 'START_STORE_EXPLICIT')

        # 4. Store/Update new/modified/unchanged explicit requirements
        written_exp_reqs = _mark_unchanged_modified_new_in_firestore(
            project_id, version, new_exp_reqs
        )

        logger.info(
            f'New/Modified/Unchanged Explicit writes => {len(written_exp_reqs)}'
        )

        _update_version_status(project_id, version, 'START_DEDUPE_EXPLICIT')

        # 5. Deduplicate the newly written explicit requirements
        _mark_duplicates(project_id, version, written_exp_reqs)

        _update_version_status(project_id, version, 'CONFIRM_CHANGE_ANALYSIS_EXPLICIT')

        return (
            json.dumps(
                {
                    'status': 'success',
                    'message': 'Explicit requirement processing complete. Next step: Implicit Processor.',
                    'total_explicit_processed': len(written_exp_reqs),
                    'deprecated_explicit_count': len(deprecated_exp_ids),
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception('Error during explicit requirements extraction phase 2:')

        if project_id and version:
            _update_version_status(project_id, version, 'ERR_CHANGE_ANALYSIS_EXPLICIT')

        return (
            json.dumps(
                {
                    'status': 'error',
                    'message': f'An unexpected error occurred during explicit processing: {str(e)}',
                }
            ),
            500,
        )


# explicit_req_processor_change_analysis(None)
