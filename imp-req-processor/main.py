import os
import json
import time
import logging
import functools
import concurrent.futures as futures
from urllib.parse import urlparse
from typing import Any, Dict, List, Tuple

import functions_framework

# from dotenv import load_dotenv
# load_dotenv()

from google import genai
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig
from google.cloud import firestore, discoveryengine_v1

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
GENAI_MODEL = os.getenv('GENAI_MODEL')
GENAI_API_VERSION = os.getenv('GENAI_API_VERSION')
GENAI_TIMEOUT_SECONDS = int(os.getenv('GENAI_TIMEOUT_SECONDS'))

# Tunables (safe defaults for speed and cost-efficiency)
REGULATIONS = ['FDA', 'IEC 62304', 'ISO 9001', 'ISO 13485', 'ISO 27001', 'SaMD']
MAX_WORKERS = 16  # Thread pool concurrency for parallel API calls

# System prompt for Gemini requirement refinement
REFINEMENT_PROMPT = (
    'You are a Medical Quality Assurance Document Specialist. Your task is to take raw text, '
    'which is often a snippet from a regulatory document or an informal comment, and '
    'rewrite it into a single, objective, formal software or system requirement. '
    'The rewritten requirement must be clear, concise, verifiable, and written in '
    'the third person (e.g., \'The system shall...\' or \'The device must...\'). '
    'Remove all conversational language, first/second/third-person comments, '
    'introductions, conclusions, or narrative elements. Focus only on the core action or constraint.'
    'Here is the text you need to refine:\n\n{payload}'
)


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


def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> None:
    '''Commits a list of documents to Firestore in batches.'''
    batch = firestore_client.batch()
    count = 0
    for doc_ref, data in doc_tuples:
        batch.set(doc_ref, data)
        count += 1
        if count >= FIRESTORE_COMMIT_CHUNK:
            print(f'Firestore => committing {count} documents')
            batch.commit()
            batch = firestore_client.batch()
            count = 0
    if count:
        batch.commit()


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
def _refine_requirement_with_gemini(text: str) -> str:
    '''Refines a raw text snippet into an objective requirement using Gemini.'''
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
            f'Gemini refinement failed for text: \'{text[:50]}...\'. Error: {e}'
        )
        # Fallback to the raw text if refinement fails
        return text


def _refine_disc_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    print(f'Starting parallel refinement of {len(results)} candidates with Gemini...')

    texts_to_refine = [res.get('snippet', '') for res in results]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        refined_texts = list(ex.map(_refine_requirement_with_gemini, texts_to_refine))

    for candidate, refined_text in zip(results, refined_texts):
        candidate['refined_text'] = refined_text

    print('Refinement complete.')

    return results


@_retry(max_attempts=3)
def _generate_embedding(text: str) -> List[float]:
    '''Generates a vector embedding for a given text using the GenAI API.'''
    try:
        if not text:
            return []

        # Use the same timeout mechanism as generation calls for safety
        with futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                lambda: genai_client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=[Content(parts=[Part(text=text)])],
                )
            )
            response = future.result(timeout=GENAI_TIMEOUT_SECONDS)

        # The response is a list of ContentEmbedding objects (one per text input).
        embeddings = response.embeddings
        if embeddings:
            return embeddings[0].values
        return []
    except Exception as e:
        logging.exception(
            f'Embedding generation failed for text: \'{text}...\'. Error: {e}'
        )
        return []


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
    '''
    A wrapper to execute _query_discovery_engine_single and attach the source ID.
    '''
    req_text, req_id = req_tuple

    discovery_results = _query_discovery_engine_single(
        f'Find the regulations, standards and procedures that apply to the following requirement: {req_text}'
    )

    print(f'{req_id} => Discovery results => {len(discovery_results)}')

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

    print(f'Discovery implicit candidates => {len(all_results)}')

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
                'updated_at': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP,
            }
        )

    return formatted_list


def _write_reqs_to_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    print(f'Generating embeddings for {len(requirements)} requirements...')

    texts_to_embed = [req.get('requirement', '') for req in requirements]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        embedding_vectors = list(ex.map(_generate_embedding, texts_to_embed))

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


def _fetch_explicit_reqs(project_id: str, version: str) -> List[Dict[str, Any]]:
    '''
    Fetches all non-duplicate, non-deleted, explicit requirements
    from Firestore to be used as a source for implicit requirement generation.
    '''
    print(f'Fetching original explicit requirements for {project_id}/{version}...')

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

    print(f'Found {len(req_list)} original explicit requirements to process.')

    return req_list


# =====================
# Main HTTP Function
# =====================
@functions_framework.http
def process_implicit_requirements(request):
    '''
    Fetches original explicit reqs from Firestore, finds,
    refines, persists, and de-duplicates IMPLICIT requirements.
    '''
    project_id = None
    version = None

    try:
        payload = request.get_json(silent=True) or {}
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
            print('No explicit requirements found. Skipping implicit generation.')

            _update_firestore_status(project_id, version, 'CONFIRM_IMP_REQ_EXTRACT')

            return (
                json.dumps(
                    {
                        'status': 'success',
                        'message': 'No explicit requirements to process.',
                    }
                ),
                200,
            )

        _update_firestore_status(project_id, version, 'START_IMPLICIT_DISCOVERY')

        disc_results = _query_discovery_engine_parallel(exp_reqs)

        if not disc_results:
            print('No implicit candidates found from Discovery Engine.')

            _update_firestore_status(project_id, version, 'CONFIRM_IMP_REQ_EXTRACT')

            return (
                json.dumps(
                    {
                        'status': 'success',
                        'message': 'No implicit candidates found.',
                    }
                ),
                200,
            )

        _update_firestore_status(project_id, version, 'START_IMPLICIT_REFINE')

        refined_results = _refine_disc_results(disc_results)

        formatted_results = _format_disc_results(version, refined_results)

        _update_firestore_status(project_id, version, 'START_STORE_IMPLICIT')

        implicit_reqs = _write_reqs_to_firestore(project_id, version, formatted_results)

        _update_firestore_status(project_id, version, 'START_DEDUPE_IMPLICIT')

        _mark_duplicates(project_id, version, implicit_reqs)

        _update_firestore_status(project_id, version, 'CONFIRM_IMP_REQ_EXTRACT')

        return (
            json.dumps(
                {
                    'status': 'success',
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception('Error during requirements extraction (IMPLICIT):')

        if project_id and version:
            _update_firestore_status(project_id, version, 'ERR_IMP_REQ_EXTRACT')

        return (
            json.dumps(
                {
                    'status': 'error',
                    'message': f'An unexpected error occurred during processing: {str(e)}',
                }
            ),
            500,
        )


# process_implicit_requirements(None)
