import os
import json
import time
import logging
import datetime
import functools
import threading
import concurrent.futures as futures
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple, TypeVar
import numpy as np
import faiss

import functions_framework

# from dotenv import load_dotenv
# load_dotenv()

from google import genai
from google.genai.types import (
    HttpOptions,
    Part,
    Content,
    GenerateContentConfig,
    EmbedContentConfig,
)
from google.cloud import firestore, discoveryengine_v1
from google.cloud.firestore_v1.transforms import Sentinel

# =====================
# Environment variables
# =====================
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
DUPE_SIM_THRESHOLD = float(os.getenv('DUPE_SIM_THRESHOLD'))
DISCOVERY_RELEVANCE_THRESHOLD = float(os.getenv('DISCOVERY_RELEVANCE_THRESHOLD'))
FIRESTORE_COMMIT_CHUNK = int(os.getenv('FIRESTORE_COMMIT_CHUNK'))
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE'))
EMBEDDING_OUTPUT_DIMENSION = int(os.getenv('EMBEDDING_OUTPUT_DIMENSION'))
EMBEDDING_TIMEOUT_SECONDS = int(os.getenv('EMBEDDING_TIMEOUT_SECONDS'))
MAX_PARALLEL_EMBEDDING_BATCHES = int(os.getenv('MAX_PARALLEL_EMBEDDING_BATCHES'))
GENAI_MODEL = os.getenv('GENAI_MODEL')
GENAI_API_VERSION = os.getenv('GENAI_API_VERSION')
GENAI_TIMEOUT_SECONDS = int(os.getenv('GENAI_TIMEOUT_SECONDS'))
REFINEMENT_PROMPT_ENV = os.getenv('REFINEMENT_PROMPT')

# =====================
# Constants
# =====================

REGULATIONS = ['FDA', 'IEC 62304', 'ISO 9001', 'ISO 13485', 'ISO 27001', 'SaMD']
MAX_WORKERS = 16
MAX_DOC_SIZE_BYTES = 1048576 * 0.95

# System prompt for Gemini requirement refinement
REFINEMENT_PROMPT = (
    f'{REFINEMENT_PROMPT_ENV}' 'Here is the text you need to refine:\n\n{payload}'
)

# =====================
# Logging
# =====================

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# =====================
# Clients
# =====================
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
discovery_client = discoveryengine_v1.SearchServiceClient()

serving_config = (
    f'projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/'
    f'dataStores/{DATA_STORE_ID}/servingConfigs/default_serving_config'
)

# Configure GenAI client once (used for embeddings and generation)
genai_client = genai.Client(http_options=HttpOptions(api_version=GENAI_API_VERSION))

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
                    logger.exception(
                        f'Attempt {attempt}/{max_attempts} failed for {fn.__name__} with error: {e}'
                    )
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            return None  # Should be unreachable

        return wrapper

    return deco


T = TypeVar('T')


def _chunk_list(data: List[T], size: int) -> List[List[T]]:
    '''Yield successive n-sized chunks from a list.'''
    if not data:
        return []

    chunked_data = []
    for i in range(0, len(data), size):
        chunked_data.append(data[i : i + size])
    return chunked_data


def _update_firestore_status(project_id: str, version: str, status: str) -> None:
    '''Updates the status of a project version in Firestore.'''
    doc_ref = firestore_client.document('projects', project_id, 'versions', version)
    doc_ref.set({'status': status}, merge=True)
    logger.info(f'Status => {status}')


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


# =====================
# Core Processing Functions
# =====================


@_retry(max_attempts=3)
def _genai_json_call(model: str, prompt: str, schema: dict) -> list:
    '''
    Calls Gemini with a schema and returns the parsed JSON.
    Includes a timeout and retry logic for resilience.
    '''
    with futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            lambda: genai_client.models.generate_content(
                model=model,
                contents=[Content(parts=[Part(text=prompt)], role='user')],
                config=GenerateContentConfig(
                    response_mime_type='application/json',
                    response_json_schema=schema,
                ),
            )
        )
        resp = future.result(timeout=GENAI_TIMEOUT_SECONDS)
    return json.loads(resp.text)


@_retry(max_attempts=3)
def _refine_requirement_with_gemini(text: str) -> List[str]:
    '''Refines a raw text snippet into an objective requirement using Gemini.'''
    if not text:
        return ''

    try:
        perf_start = time.time()

        response = _genai_json_call(
            model=GENAI_MODEL,
            prompt=REFINEMENT_PROMPT.format(payload=text),
            schema={
                'type': 'ARRAY',
                'items': {
                    'type': 'OBJECT',
                    'properties': {
                        'text': {'type': 'string'},
                    },
                    'required': ['text'],
                },
            },
        )

        perf_end = time.time()

        logger.info(
            f'Time taken for this refinement (in seconds): {perf_end - perf_start}'
        )

        return [resp.get('text', '') for resp in response]

    except Exception as e:
        logger.exception(
            f'Gemini refinement failed for text: \'{text[:50]}...\'. Error: {e}'
        )

        return [text]


def _refine_disc_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    logger.info(
        f'Starting parallel refinement of {len(results)} candidates with Gemini...'
    )

    texts_to_refine = [res.get('snippet', '') for res in results]

    perf_start = time.time()

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # Note: _refine_requirement_with_gemini returns a list
        refined_text_responses = list(
            ex.map(_refine_requirement_with_gemini, texts_to_refine)
        )

    perf_end = time.time()

    logger.info(
        f'Total time taken for refinement (in seconds): {perf_end - perf_start}'
    )

    refined_candidates = []

    for candidate, refined_texts in zip(results, refined_text_responses):
        for refined_text in refined_texts:
            new_candidate = candidate.copy()
            new_candidate['refined_text'] = refined_text
            refined_candidates.append(new_candidate)

    logger.info(f'Refinement complete. Final candidates => {len(refined_candidates)}')

    return refined_candidates


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

        with futures.ThreadPoolExecutor() as ex:
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

            response = future.result(timeout=EMBEDDING_TIMEOUT_SECONDS)

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


@_retry(max_attempts=3)
def _query_discovery_engine_single(query_text: str) -> List[Dict[str, Any]]:
    '''
    Queries Discovery Engine for a single text string and returns top results.
    '''
    processed = []

    page_token = ''

    while True:
        request = discoveryengine_v1.SearchRequest(
            serving_config=serving_config,
            query=query_text,
            content_search_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True
                ),
                search_result_mode=discoveryengine_v1.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,
            ),
            page_token=page_token,
        )

        response = discovery_client.search(request=request)

        for page in response.pages:
            for result in page.results:
                relevance_score = 0.0
                if result.chunk.relevance_score:
                    relevance_score = float(result.chunk.relevance_score)

                link = result.chunk.document_metadata.uri
                filename = urlparse(link).path.split('/')[-1]
                regulation = next(
                    (prefix for prefix in REGULATIONS if filename.startswith(prefix)),
                    '',
                )

                processed.append(
                    {
                        'regulation': regulation,
                        'filename': filename,
                        'snippet': result.chunk.content,
                        'relevance_score': relevance_score,
                    }
                )

        if not response.next_page_token:
            break

        page_token = response.next_page_token

    processed.sort(key=lambda x: x['relevance_score'], reverse=True)

    logger.info(f'Discovery processed => {len(processed)}')

    return [
        el for el in processed if el['relevance_score'] > DISCOVERY_RELEVANCE_THRESHOLD
    ]


def _query_discovery_engine_wrapper(req_tuple: Tuple[str, str]) -> List[Dict[str, Any]]:
    '''
    A wrapper to execute _query_discovery_engine_single and attach the source ID.
    '''
    req_text, req_id = req_tuple

    discovery_results = _query_discovery_engine_single(
        f'Find the regulations, standards and procedures that apply to the following requirement: {req_text}'
    )

    logger.info(f'{req_id} => Discovery results => {len(discovery_results)}')

    for res in discovery_results:
        res['explicit_requirement_id'] = req_id

    return discovery_results


def _query_discovery_engine_parallel(
    requirements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    req_inputs = [(r.get('requirement'), r.get('requirement_id')) for r in requirements]

    all_results = []

    with futures.ThreadPoolExecutor(
        max_workers=min(MAX_WORKERS, len(req_inputs) or 1)
    ) as ex:
        for res_list in ex.map(_query_discovery_engine_wrapper, req_inputs):
            all_results.extend(res_list)

    logger.info(f'Discovery implicit candidates => {len(all_results)}')

    return all_results


def _format_disc_results(
    version: str, results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    '''
    Transforms Discovery Engine search results (which have been refined by Gemini)
    into a format suitable for direct persistence, including the explicit source ID
    and setting 'source_type' to 'implicit'.
    '''
    formatted_list = []
    for idx, res in enumerate(results, start=1):
        parent_req = res.pop('explicit_requirement_id', None)

        formatted_list.append(
            {
                'requirement_id': f'v{version}-REQ-I-{idx:03d}',
                'requirement': res.get('refined_text', ''),
                'requirement_category': 'regulation',
                'source_type': 'implicit',
                'sources': [],
                'deleted': False,
                'duplicate': False,  # Initialize to False
                'near_duplicate_id': '',
                'embedding': [],
                'change_analysis_status': 'NEW',
                'change_analysis_status_reason': 'Newly created',
                'change_analysis_near_duplicate_id': '',
                'regulations': [
                    {
                        'regulation': res.get('regulation', None),
                        'source': {
                            'filename': res.get('filename', None),
                            'snippet': res.get('snippet', None),
                            'relevance_score': res.get('relevance_score', None),
                        },
                    }
                ],
                'parent_exp_req_ids': [parent_req] if parent_req else [],
                'testcase_status': 'NOT_STARTED',
                'toolCreated': 'NOT_STARTED',
                'toolIssueKey': '',
                'toolIssueLink': '',
                'updated_at': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP,
            }
        )

    return formatted_list


def _write_reqs_to_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    logger.info(
        f'Generating embeddings for {len(requirements)} requirements using batching...'
    )

    texts_to_embed = [
        req.get('requirement') for req in requirements if req.get('requirement', '')
    ]

    embedding_vectors = []

    text_batches = _chunk_list(texts_to_embed, EMBEDDING_BATCH_SIZE)

    perf_start = time.time()

    with futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EMBEDDING_BATCHES) as ex:
        batch_results = list(ex.map(_generate_embedding_batch, text_batches))
        for batch in batch_results:
            embedding_vectors.extend(batch)

    perf_end = time.time()

    logger.info(f'Total time taken for embedding (in seconds): {perf_end - perf_start}')

    requirements_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    doc_tuples = []
    written_reqs = []

    for req, embedding_vector in zip(requirements, embedding_vectors):
        doc_data = {
            **req,
            'embedding': embedding_vector,
            'updated_at': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP,
        }

        doc_tuples.append(
            (
                requirements_collection_ref.document(doc_data['requirement_id']),
                doc_data,
            )
        )

        written_reqs.append(
            {
                'requirement_id': doc_data['requirement_id'],
                'embedding': doc_data['embedding'],
                'duplicate': doc_data['duplicate'],
                'parent_exp_req_ids': doc_data['parent_exp_req_ids'],
                'source_type': doc_data['source_type'],
            }
        )

    _firestore_commit_many(doc_tuples)

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
    duplicates_to_update: List[Tuple[str, str, List[str]]] = []

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
                    duplicate_req['requirement_id'],
                    original_req['requirement_id'],
                    duplicate_req.get('parent_exp_req_ids', []),
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

    for req_id, near_duplicate_id, parent_exp_req_ids in duplicates_to_update:
        if batch_count >= FIRESTORE_COMMIT_CHUNK:
            batch.commit()
            batch_count = 0

        doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', req_id
        )

        batch.update(
            doc_ref, {'duplicate': True, 'near_duplicate_id': near_duplicate_id}
        )
        batch_count += 1

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
            batch_count += 1

    if batch_count > 0:
        batch.commit()

    perf_end = time.time()

    logger.info(
        f'Time to find and update duplicates (in seconds): {perf_end - perf_start}'
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


def _fetch_explicit_reqs(project_id: str, version: str) -> List[Dict[str, Any]]:
    '''
    Fetches all non-duplicate, non-deleted, explicit requirements
    from Firestore to be used as a source for implicit requirement generation.
    '''
    logger.info(
        f'Fetching original explicit requirements for {project_id}/{version}...'
    )

    reqs_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    query = (
        reqs_ref.where(filter=firestore.FieldFilter('source_type', '==', 'explicit'))
        .where(filter=firestore.FieldFilter('duplicate', '==', False))
        .where(filter=firestore.FieldFilter('deleted', '==', False))
        .select(['requirement', 'requirement_id'])
    )

    documents = query.stream()

    req_list = [doc.to_dict() for doc in documents]

    logger.info(f'Found {len(req_list)} original explicit requirements to process.')

    return req_list


# =====================
# Main HTTP Function
# =====================
@functions_framework.http
def process_implicit_requirements(request):
    '''
    Starts the implicit requirements extraction process asynchronously.
    Returns 202 immediately while processing continues in the background.
    '''
    try:
        payload = request.get_json(silent=True) or {}
        # Mock data for local testing
        # payload = {'project_id': 'EUz0pMnqmNkBfh8FHMYZ', 'version': '1'}

        project_id = payload.get('project_id')
        version = payload.get('version')

        if not all([project_id, version]):
            return (
                json.dumps(
                    {
                        'status': 'error',
                        'message': 'Required details (project_id, version) are missing.',
                    }
                ),
                400,
            )

        _update_firestore_status(project_id, version, 'START_IMPLICIT_REQ_EXTRACT')

        exp_reqs = _fetch_explicit_reqs(project_id, version)

        if not exp_reqs:
            logger.info('No explicit requirements found. Skipping implicit generation.')

            _update_firestore_status(project_id, version, 'CONFIRM_IMP_REQ_EXTRACT')

            return

        _update_firestore_status(project_id, version, 'START_IMPLICIT_DISCOVERY')

        disc_results = _query_discovery_engine_parallel(exp_reqs)

        if not disc_results:
            logger.info('No implicit candidates found from Discovery Engine.')
            _update_firestore_status(project_id, version, 'CONFIRM_IMP_REQ_EXTRACT')
            return

        _update_firestore_status(project_id, version, 'START_IMPLICIT_REFINE')

        refined_results = _refine_disc_results(disc_results)

        formatted_results = _format_disc_results(version, refined_results)

        _update_firestore_status(project_id, version, 'START_STORE_IMPLICIT')

        implicit_reqs = _write_reqs_to_firestore(project_id, version, formatted_results)

        _update_firestore_status(project_id, version, 'START_DEDUPE_IMPLICIT')

        _mark_duplicates(project_id, version, implicit_reqs)

        _update_firestore_status(project_id, version, 'CONFIRM_IMP_REQ_EXTRACT')

        logger.info(
            f'Background task completed successfully for {project_id}/{version}.'
        )

        return (
            json.dumps(
                {
                    'status': 'success',
                    'message': f'Implicit requirement extracted for {project_id}/{version}.',
                }
            ),
            200,
        )

    except Exception as e:
        logger.exception('Error during implicit extraction:')

        _update_firestore_status(project_id, version, 'ERR_IMP_REQ_EXTRACT')

        return (
            json.dumps(
                {
                    'status': 'error',
                    'message': f'Failed to start process: {str(e)}',
                }
            ),
            500,
        )


# process_implicit_requirements(None)
