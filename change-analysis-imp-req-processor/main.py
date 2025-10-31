import os
import json
import time
import logging
import datetime
import functools
import threading
import concurrent.futures as futures
from typing import Any, Dict, List, Tuple, TypeVar
from urllib.parse import urlparse
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
# Required environment variables
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


# ===================== # Constants # =====================

REGULATIONS = ['FDA', 'IEC62304', 'ISO9001', 'ISO13485', 'ISO27001', 'SaMD']
MAX_WORKERS = 16
MAX_DOC_SIZE_BYTES = 1048576 * 0.95

SOURCE_TYPE_IMPLICIT = 'implicit'

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

REFINEMENT_PROMPT = (
    f'{REFINEMENT_PROMPT_ENV}' ' Here is the text you need to refine:\n\n{payload}'
)

# ===================== # Clients # =====================
# Initialize clients globally
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
discovery_client = discoveryengine_v1.SearchServiceClient()
serving_config = (
    f'projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/'
    f'dataStores/{DATA_STORE_ID}/servingConfigs/default_serving_config'
)
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
def _genai_json_call(model: str, prompt: str, schema: dict) -> dict:
    '''Makes a GenAI call configured for JSON output.'''
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
def _refine_requirement_with_gemini(text: str) -> str:
    '''Refines a raw text snippet into a formal requirement using Gemini.'''
    if not text:
        return ''
    try:
        response = _genai_json_call(
            model=GENAI_MODEL,
            prompt=REFINEMENT_PROMPT.format(payload=text),
            schema={
                'type': 'object',
                'properties': {
                    'text': {'type': 'string'},
                },
                'required': ['text'],
            },
        )
        return response.get('text', '')
    except Exception as e:
        logger.error(
            f'Gemini refinement failed for text:\'{text[:50]}...\'. Error: {e}'
        )
        return text


def _refine_disc_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''Runs Gemini refinement on a list of candidates in parallel.'''

    logger.info(
        f'Starting parallel refinement of {len(results)} candidates with Gemini...'
    )

    snippets_to_refine = [res.get('snippet', '') for res in results]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        refined_texts = list(
            ex.map(_refine_requirement_with_gemini, snippets_to_refine)
        )

    for candidate, refined_text in zip(results, refined_texts):
        candidate['refined_text'] = refined_text

    logger.info('Refinement complete.')

    return results


# ===================== # Implicit Core Functions # =====================


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
    req_text, req_id = req_tuple

    discovery_results = _query_discovery_engine_single(
        f'Find the regulations and standards and procedures that apply to the following requirement: {req_text}'
    )

    logger.info(f'{req_id} => Discovery results => {len(discovery_results)}')

    for res in discovery_results:
        res['explicit_requirement_id'] = req_id

    return discovery_results


def _query_discovery_engine_parallel(
    requirements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''Queries Discovery Engine for a list of requirements in parallel.'''
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
    version: str,
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''Transforms Discovery Engine results (refined by Gemini) into persistence format.'''

    formatted_list = []
    embedding_vectors = []

    texts_to_embed = [res.get('refined_text', '') for res in results]
    text_batches = _chunk_list(texts_to_embed, EMBEDDING_BATCH_SIZE)

    with futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EMBEDDING_BATCHES) as ex:
        batch_results = list(ex.map(_generate_embedding_batch, text_batches))
        for batch in batch_results:
            embedding_vectors.extend(batch)

    for idx, (res, embedding_vector) in enumerate(zip(results, embedding_vectors)):
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
                'embedding': embedding_vector,
                'change_analysis_status': CHANGE_STATUS_NEW,
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
                'updated_at': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP,
            }
        )

    return formatted_list


def _write_reqs_to_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    '''Persists new implicit requirements to Firestore.'''

    logger.info(f'Persisting {len(requirements)} implicit requirements to firestore...')

    requirements_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    doc_insertions_tuples_list = []

    written_reqs = []

    for req in requirements:
        if req.get('source_type') == SOURCE_TYPE_IMPLICIT:
            doc_data = {**req, 'created_at': firestore.SERVER_TIMESTAMP}
            req_id = req.get('requirement_id')

            doc_ref = requirements_collection_ref.document(req_id)
            doc_insertions_tuples_list.append((doc_ref, doc_data))

            written_reqs.append(
                {
                    'requirement_id': req_id,
                    'source_type': SOURCE_TYPE_IMPLICIT,
                    'requirement': req.get('requirement', ''),
                    'embedding': req.get('embedding', []),
                    'duplicate': req.get('duplicate', False),
                    'parent_exp_req_ids': req.get('parent_exp_req_ids', []),
                    'change_analysis_status': req.get(
                        'change_analysis_status', CHANGE_STATUS_NEW
                    ),
                    'regulations': req.get('regulations', []),
                }
            )

    if doc_insertions_tuples_list:

        logger.info(
            f'Committing {len(doc_insertions_tuples_list)} INSERT/SET operations...'
        )

        _firestore_commit_many(doc_insertions_tuples_list)

    logger.info(f'New Implicit writes (Discovery) => {len(written_reqs)}')

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
                    duplicate_req.get('regulations', []),
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

    for (
        original_id,
        dupe_req_id,
        dupe_parent_ids,
        dupe_regulations,
    ) in duplicates_to_update:
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

        if dupe_parent_ids or dupe_regulations:
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

            if dupe_regulations:
                updates['regulations'] = firestore.ArrayUnion(dupe_regulations)

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


def _get_current_explicit_statuses(project_id: str, version: str) -> Dict[str, str]:
    '''Retrieves the current change_analysis_status for all active explicit requirements.'''
    exp_status_map = {}
    exp_reqs_query = (
        firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )
        .where('source_type', '==', 'explicit')
        .where('deleted', '==', False)
        .where('duplicate', '==', False)
        .select(['change_analysis_status'])
    )

    for doc in exp_reqs_query.stream():
        data = doc.to_dict()
        exp_status_map[doc.id] = data.get(
            'change_analysis_status', CHANGE_STATUS_UNCHANGED
        )
    return exp_status_map


def update_existing_implicit_reqs(project_id: str, version: str) -> None:
    logger.info('Starting implicit requirement status link analysis...')

    collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    ).where('source_type', '==', SOURCE_TYPE_IMPLICIT)

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

    exp_status_map = _get_current_explicit_statuses(project_id, version)

    # Set of explicit IDs that are UNCHANGED
    unchanged_exp_ids = {
        k for k, v in exp_status_map.items() if v == CHANGE_STATUS_UNCHANGED
    }

    # Query only active, non-duplicate implicit requirements
    imp_reqs_query = (
        firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )
        .where('source_type', '==', SOURCE_TYPE_IMPLICIT)
        .where('deleted', '==', False)
        .where('duplicate', '==', False)
        .where('change_analysis_status', '!=', CHANGE_STATUS_IGNORED)
        .select(['requirement_id', 'parent_exp_req_ids'])
    )

    updates = []  # (doc_ref, data)

    for doc in imp_reqs_query.stream():
        imp_data = doc.to_dict()
        imp_req_id = imp_data['requirement_id']
        parent_exp_req_ids = set(imp_data.get('parent_exp_req_ids', []))

        # Explicit IDs that currently exist and are linked
        valid_linked_exp_ids = parent_exp_req_ids.intersection(exp_status_map.keys())

        # Explicit IDs that are linked and are UNCHANGED
        unchanged_links = valid_linked_exp_ids.intersection(unchanged_exp_ids)

        new_status = ''
        new_status_reason = ''
        updated_data = {}

        # --- If any of the parent_exp_req_ids are in UNCHANGED state, Then the implicit requirement is UNCHANGED. ---
        if unchanged_links:
            new_status = CHANGE_STATUS_UNCHANGED
            new_status_reason = (
                'Atleast one of its original parent explicit requirements is UNCHANGED.'
            )

            # Prune: keep only the UNCHANGED links
            parent_exp_req_ids_to_keep = list(unchanged_links)
            updated_data['parent_exp_req_ids'] = parent_exp_req_ids_to_keep

        # --- For Non-Unchanged Links ---
        elif parent_exp_req_ids:
            new_status = CHANGE_STATUS_DEPRECATED
            new_status_reason = 'All of its original parent explicit requirements are either MODIFIED / DEPRECATED / IGNORED.'

        if new_status:
            updated_data['change_analysis_status'] = new_status
            updated_data['change_analysis_status_reason'] = new_status_reason
            updated_data['updated_at'] = firestore.SERVER_TIMESTAMP

            doc_ref = firestore_client.document(
                'projects', project_id, 'versions', version, 'requirements', imp_req_id
            )
            updates.append((doc_ref, updated_data))

    _firestore_commit_many(updates)

    logger.info(f'Committed {len(updates)} implicit requirement status/link updates.')


# ===================== # Main HTTP Function for Implicit Processing # =====================


@functions_framework.http
def process_implicit_requirements(request):
    '''
    Cloud Function entry point to manage implicit requirements:
    1. Update existing implicit statuses based on explicit links (with pruning logic).
    2. Search and create new implicit requirements based on NEW/MODIFIED explicit requirements.
    '''
    project_id = None
    version = None
    try:
        payload = request.get_json(silent=True) or {}
        # Mock data
        # payload = {
        #     'project_id': 'abc',
        #     'version': 'v2',
        # }
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

        def background_task(project_id: str, version: str):
            try:
                _update_version_status(project_id, version, 'START_IMPLICIT_ANALYSIS')

                # Update status of existing implicit requirements based on explicit links
                update_existing_implicit_reqs(project_id, version)

                _update_version_status(project_id, version, 'START_IMPLICIT_DISCOVERY')

                logger.info(
                    'Starting implicit search for NEW/MODIFIED explicit requirements...'
                )

                # Query Firestore for explicit reqs with NEW or MODIFIED status
                search_reqs_query = (
                    firestore_client.collection(
                        'projects', project_id, 'versions', version, 'requirements'
                    )
                    .where('source_type', '==', 'explicit')
                    .where('deleted', '==', False)
                    .where(
                        'change_analysis_status',
                        'in',
                        [CHANGE_STATUS_NEW, CHANGE_STATUS_MODIFIED],
                    )
                    .select(['requirement_id', 'requirement'])
                )

                search_reqs = [doc.to_dict() for doc in search_reqs_query.stream()]

                disc_results = _query_discovery_engine_parallel(search_reqs)

                _update_version_status(project_id, version, 'START_IMPLICIT_REFINE')

                refined_results = _refine_disc_results(disc_results)

                implicit_reqs = _format_disc_results(version, refined_results)

                _update_version_status(project_id, version, 'START_STORE_IMPLICIT')

                written_imp_reqs = _write_reqs_to_firestore(
                    project_id, version, implicit_reqs
                )

                _update_version_status(project_id, version, 'START_DEDUPE_IMPLICIT')

                _mark_duplicates(project_id, version, written_imp_reqs)

                _update_version_status(
                    project_id, version, 'CONFIRM_CHANGE_ANALYSIS_IMPLICIT'
                )

            except Exception as e:
                logger.exception('Error during background implicit extraction:')

                _update_version_status(
                    project_id, version, 'ERR_CHANGE_ANALYSIS_IMPLICIT'
                )

        # Launch the process in background
        thread = threading.Thread(
            target=background_task, args=(project_id, version), daemon=True
        )

        thread.start()

        # Return immediately
        return (
            json.dumps(
                {
                    'status': 'accepted',
                    'message': f'Implicit requirement extraction with change analysis started for {project_id}/{version}.',
                }
            ),
            202,
        )

    except Exception as e:
        logger.exception('Error initiating process:')

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
