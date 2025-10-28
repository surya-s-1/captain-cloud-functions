import os
import json
import time
import logging
import datetime
import functools
import concurrent.futures as futures
from typing import Any, Dict, List, Tuple, Iterable, TypeVar
from urllib.parse import urlparse
from google import genai
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig
from google.cloud import firestore, discoveryengine_v1
from google.cloud.firestore_v1.transforms import Sentinel

import functions_framework

# from dotenv import load_dotenv
# load_dotenv()

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
GENAI_MODEL = os.getenv('GENAI_MODEL')
GENAI_API_VERSION = os.getenv('GENAI_API_VERSION')
GENAI_TIMEOUT_SECONDS = int(os.getenv('GENAI_TIMEOUT_SECONDS'))

# Constants
REGULATIONS = ['FDA', 'IEC62304', 'ISO9001', 'ISO13485', 'ISO27001', 'SaMD']
MAX_WORKERS = 16
MAX_DOC_SIZE_BYTES = 1048576 * 0.95
EMBEDDING_BATCH_SIZE = 150

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
    'You are a Medical Quality Assurance Document Specialist. Your task is to take raw text,'
    ' which is often a snippet from a regulatory document or an informal comment, and'
    ' rewrite it into a single, objective, formal software or system requirement.'
    ' Break into multiple requirements (maximum 2) if needed.'
    ' Make sure the rewritten requirement is size is less than 300 characters.'
    ' The rewritten requirement must be clear, concise, verifiable, and written in'
    ' the third person (e.g., \'The system shall...\' or \'The device must...\').'
    ' Remove all conversational language, first/second/third-person comments,'
    ' introductions, conclusions, or narrative elements. Focus only on the core action or constraint.'
    ' Here is the text you need to refine:\n\n{payload}'
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

logging.getLogger('google.cloud').setLevel(logging.WARNING)
logging.getLogger('google.genai').setLevel(logging.WARNING)

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
                    logging.warning(
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
    print(f'Status => {status}')


T = TypeVar('T')


def _chunk_list(data: List[T], size: int) -> Iterable[List[T]]:
    '''Yield successive n-sized chunks from a list.'''
    for i in range(0, len(data), size):
        yield data[i : i + size]


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


@_retry(max_attempts=3)
def _generate_embedding(text: str) -> List[float]:
    '''Generates a vector embedding for a given text using the GenAI API.'''
    try:
        if not text:
            return []
        with futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                lambda: genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=[Content(parts=[Part(text=text)])],
                )
            )
            response = future.result(timeout=GENAI_TIMEOUT_SECONDS)
        embeddings = response.embeddings
        if embeddings:
            return embeddings[0].values
        return []
    except Exception as e:
        logging.exception(f'Embedding generation failed. Error: {e}')
        return []


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
        logging.error(
            f'Gemini refinement failed for text:\'{text[:50]}...\'. Error: {e}'
        )
        return text


def _refine_disc_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''Runs Gemini refinement on a list of candidates in parallel.'''

    print(f'Starting parallel refinement of {len(results)} candidates with Gemini...')

    snippets_to_refine = [res.get('snippet', '') for res in results]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        refined_texts = list(
            ex.map(_refine_requirement_with_gemini, snippets_to_refine)
        )

    for candidate, refined_text in zip(results, refined_texts):
        candidate['refined_text'] = refined_text

    print('Refinement complete.')

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

    print(f'Discovery processed => {len(processed)}')

    return [
        el for el in processed if el['relevance_score'] > DISCOVERY_RELEVANCE_THRESHOLD
    ]


def _query_discovery_engine_wrapper(req_tuple: Tuple[str, str]) -> List[Dict[str, Any]]:
    req_text, req_id = req_tuple

    discovery_results = _query_discovery_engine_single(
        f'Find the regulations and standards and procedures that apply to the following requirement: {req_text}'
    )

    print(f'{req_id} => Discovery results => {len(discovery_results)}')

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

    print(f'Discovery implicit candidates => {len(all_results)}')

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

    for i, batch in enumerate(text_batches):
        logging.info(
            f'Embedding batch {i+1} of {len(texts_to_embed)//EMBEDDING_BATCH_SIZE + 1} with {len(batch)} items.'
        )

        batch_results = _generate_embedding_batch(batch)

        embedding_vectors.extend(batch_results)

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

    print(f'Persisting {len(requirements)} implicit requirements to firestore...')

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
                }
            )

    if doc_insertions_tuples_list:

        print(f'Committing {len(doc_insertions_tuples_list)} INSERT/SET operations...')

        _firestore_commit_many(doc_insertions_tuples_list)

    print(f'New Implicit writes (Discovery) => {len(written_reqs)}')

    return written_reqs


def _mark_duplicates(
    project_id: str, version: str, newly_written_reqs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''Compares newly written requirement embeddings and marks duplicates in Firestore.'''

    duplicates_to_update: List[Tuple[str, str, List[str]]] = []

    existing_query = (
        firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )
        .where('source_type', '==', SOURCE_TYPE_IMPLICIT)
        .where('duplicate', '==', False)
        .where('deleted', '==', False)
        .where('change_analysis_status', '!=', CHANGE_STATUS_IGNORED)
    )

    existing_originals = [doc.to_dict() for doc in existing_query.stream()]

    all_reqs_for_dedupe = existing_originals + newly_written_reqs

    for i in range(len(newly_written_reqs)):
        req_i = newly_written_reqs[i]  # New implicit req
        if req_i.get('duplicate', False):
            continue

        best_match_id = None
        max_sim_score = -1.0

        # Compare against existing originals and other new items
        for req_j in all_reqs_for_dedupe:
            if req_i['requirement_id'] == req_j['requirement_id']:
                continue
            if req_j.get('duplicate', False):
                continue

            similarity = _cosine_similarity(req_i['embedding'], req_j['embedding'])

            if similarity >= DUPE_SIM_THRESHOLD and similarity > max_sim_score:
                max_sim_score = similarity
                best_match_id = req_j['requirement_id']

        if best_match_id:
            # req_i is a duplicate of best_match_id (which could be an old original or a new original)
            duplicates_to_update.append(
                (
                    req_i['requirement_id'],
                    best_match_id,
                    req_i.get('parent_exp_req_ids', []),
                )
            )
            req_i['duplicate'] = True

    if not duplicates_to_update:
        print('No implicit duplicates found using vector embeddings in this batch.')
        return [], newly_written_reqs

    print(f'Found {len(duplicates_to_update)} implicit duplicates to mark.')
    batch = firestore_client.batch()

    for req_id, near_duplicate_id, parent_exp_req_ids in duplicates_to_update:
        # Mark the new/duplicate requirement as duplicate
        doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', req_id
        )
        batch.update(
            doc_ref, {'duplicate': True, 'near_duplicate_id': near_duplicate_id}
        )

        # Merge the source explicit IDs into the original implicit document
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

    for req in newly_written_reqs:
        if req.get('duplicate', False):
            all_duplicates.append(req)
        else:
            all_originals.append(req)

    print(f'Vector Dedupe (Implicit Only) => Marked {len(all_duplicates)} duplicates.')

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
    print('Starting implicit requirement status link analysis...')

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

    print('Committing all batch updates...')
    batch.commit()
    print('Batch committed successfully.')

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

    print(f'Committed {len(updates)} implicit requirement status/link updates.')


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

        _update_version_status(project_id, version, 'START_IMPLICIT_ANALYSIS')

        # Update status of existing implicit requirements based on explicit links
        update_existing_implicit_reqs(project_id, version)

        _update_version_status(project_id, version, 'START_IMPLICIT_DISCOVERY')

        # Search and create new implicit requirements for NEW/MODIFIED explicit sources
        print('Starting implicit search for NEW/MODIFIED explicit requirements...')

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

        if not search_reqs:
            print(
                'No new or modified explicit requirements found to trigger implicit search.'
            )
            return

        disc_results = _query_discovery_engine_parallel(search_reqs)

        _update_version_status(project_id, version, 'START_IMPLICIT_REFINE')

        refined_results = _refine_disc_results(disc_results)

        implicit_reqs = _format_disc_results(version, refined_results)

        _update_version_status(project_id, version, 'START_STORE_IMPLICIT')

        written_imp_reqs = _write_reqs_to_firestore(project_id, version, implicit_reqs)

        _update_version_status(project_id, version, 'START_DEDUPE_IMPLICIT')

        _mark_duplicates(project_id, version, written_imp_reqs)

        _update_version_status(project_id, version, 'CONFIRM_CHANGE_ANALYSIS_IMPLICIT')

        return ('OK', 200)

    except Exception as e:
        logging.exception('Error during implicit requirements processing:')

        if project_id and version:
            _update_version_status(project_id, version, 'ERR_CHANGE_ANALYSIS_IMPLICIT')

        return (
            json.dumps(
                {
                    'status': 'error',
                    'message': f'An unexpected error occurred during implicit processing: {str(e)}',
                }
            ),
            500,
        )


# process_implicit_requirements(None)
