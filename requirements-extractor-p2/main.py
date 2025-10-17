import os
import json
import time
import logging
import functools
import concurrent.futures as futures
from urllib.parse import urlparse
from typing import Any, Dict, Iterator, List, Tuple

import functions_framework

from google import genai
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig
from google.cloud import storage, firestore, discoveryengine_v1

# =====================
# Environment variables
# =====================
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')

# Tunables (safe defaults for speed and cost-efficiency)
REGULATIONS = ['FDA', 'IEC 62304', 'ISO 9001', 'ISO 13485', 'ISO 27001', 'SaMD']
MAX_WORKERS = 16  # Thread pool concurrency for parallel API calls
FIRESTORE_COMMIT_CHUNK = 450  # <= 500 per batch write limit
EMBEDDING_MODEL = 'text-embedding-004'
DUPLICATE_SIM_THRESHOLD = 0.95
GENAI_MODEL = 'gemini-2.5-flash'
GENAI_API_VERSION = 'v1'
GENAI_TIMEOUT_SECONDS = 90  # Each LLM call safety timeout

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
storage_client = storage.Client()
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


def _chunk(lst: list, n: int) -> Iterator[list]:
    '''Yields successive n-sized chunks from a list.'''
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


def _refine_candidates_parallel(
    implicit_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print(
        f'Starting parallel refinement of {len(implicit_candidates)} candidates with Gemini...'
    )

    # Extract texts for parallel processing. Use 'content' as it holds the main text.
    texts_to_refine = [res.get('content', '') for res in implicit_candidates]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # Map the refinement function to all texts
        refined_texts = list(ex.map(_refine_requirement_with_gemini, texts_to_refine))

    # Re-integrate the refined text back into the candidates list
    for candidate, refined_text in zip(implicit_candidates, refined_texts):
        # Preserve the *original* content/snippet for provenance
        candidate['raw_snippet'] = candidate.get(
            'snippet', candidate.get('content', '')
        )
        # Overwrite 'content' and 'snippet' with the refined text for later use
        candidate['content'] = refined_text
        candidate['snippet'] = refined_text

    print('Refinement complete.')

    return implicit_candidates


@_retry(max_attempts=3)
def _generate_embedding(text: str) -> List[float]:
    '''Generates a vector embedding for a given text using the GenAI API.'''
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


@_retry(max_attempts=3)
def _query_discovery_engine_single(query_text: str) -> List[Dict[str, Any]]:
    '''
    Queries Discovery Engine for a single text string and returns top results.
    '''
    request = discoveryengine_v1.SearchRequest(
        serving_config=serving_config,
        query=query_text,
        page_size=10,
        content_search_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
            search_result_mode=discoveryengine_v1.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,
        ),
    )
    response = discovery_client.search(request=request)

    processed = []
    for page in response.pages:
        for result in page.results:
            relevance = 0.0
            if result.model_scores and hasattr(
                result.model_scores.get('relevance_score'), 'values'
            ):
                relevance = float(result.model_scores.get('relevance_score').values[0])

            link = result.chunk.document_metadata.uri
            filename = urlparse(link).path.split('/')[-1]
            regulation = next(
                (prefix for prefix in REGULATIONS if filename.startswith(prefix)), ''
            )
            processed.append(
                {
                    'relevance': relevance,
                    'content': result.chunk.content,
                    'regulation': regulation,
                    'filename': filename,
                    'page_start': result.chunk.page_span.page_start,
                    'page_end': result.chunk.page_span.page_end,
                    'snippet': result.chunk.content,
                }
            )

    processed.sort(key=lambda x: x['relevance'], reverse=True)

    return [el for el in processed if el['relevance'] > 0.15]


def _query_discovery_engine_wrapper(req_tuple: Tuple[str, str]) -> List[Dict[str, Any]]:
    '''
    A wrapper to execute _query_discovery_engine_single and attach the source ID.
    '''
    req_text, req_id = req_tuple
    discovery_results = _query_discovery_engine_single(
        f'Regulations related to: {req_text}'
    )
    # Attach the source ID to each result
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


def _load_and_normalize_exp_req(obj_url: str) -> List[Dict[str, Any]]:
    print('Starting explicit requirement processing...')

    parsed = urlparse(obj_url)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip('/'))
    explicit_requirements_raw = json.loads(blob.download_as_text())

    if not explicit_requirements_raw:
        raise ValueError('Input data from GCS is empty.')

    normalized_list = normalize_req_dict(explicit_requirements_raw)

    final_list = []
    for r in normalized_list:
        req_text = r.get('requirement')

        final_list.append(
            {
                'requirement': req_text,
                'requirement_type': r.get('requirement_type', 'functional'),
                'exp_req_ids': [],
                'sources': r.get('sources', []),
                'source_type': 'explicit',  # Mark as explicit
            }
        )

    print(f'Loaded and normalized {len(final_list)} explicit requirements.')

    return final_list


def _format_discovery_results(
    discovery_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    '''
    Transforms Discovery Engine search results (which have been refined by Gemini)
    into a format suitable for direct persistence, including the explicit source ID
    and setting 'source_type' to 'implicit'.
    '''
    formatted_list = []
    for res in discovery_results:
        # Use the refined text (overwritten into 'snippet' and 'content') as the primary requirement text
        req_text = res.get('snippet', res.get('content'))

        # Extract and format the explicit source ID
        explicit_requirement_id = res.pop('explicit_requirement_id', None)

        # Structure the result into the expected persistence format
        formatted_list.append(
            {
                'requirement': req_text,  # REFINED TEXT
                'requirement_type': 'regulation',
                'exp_req_ids': (
                    [explicit_requirement_id] if explicit_requirement_id else []
                ),  # Store the explicit req ID(s) that led to its creation
                'source_type': 'implicit',  # Mark as implicit
                'regulations': [
                    {
                        'regulation': res.get('regulation', 'N/A'),
                        'source': {
                            'filename': res.get('filename', 'N/A'),
                            'page_start': res.get('page_start', 'N/A'),
                            'page_end': res.get('page_end', 'N/A'),
                            # Store the refined text as the snippet
                            'snippet': req_text,
                            # Store the raw text for traceability
                            'raw_snippet': res.get('raw_snippet', req_text),
                        },
                    }
                ],
            }
        )
    return formatted_list


def _persist_requirements_to_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]], start_id: int = 1
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

    for i, (req, embedding_vector) in enumerate(
        zip(requirements, embedding_vectors), start=start_id
    ):
        req_id = f'{version}-REQ-{i:03d}'

        doc_data = {
            'requirement_id': req_id,
            'embedding': embedding_vector,
            'source_type': req.get('source_type', ''),
            'requirement': req.get('requirement', ''),
            'embedding': req.get('embedding', []),
            'requirement_type': req.get('requirement_type', ''),
            'deleted': req.get('deleted', False),
            'duplicate': req.get('duplicate', False),  # Initialize to False
            'change_analysis_status': req.get('change_analysis_status', ''),
            'change_analysis_near_duplicate_id': req.get('change_analysis_near_duplicate_id', ''),
            'deprecation_reason': req.get('deprecation_reason', ''),
            'sources': req.get('sources', []),
            'regulations': req.get('regulations', []),
            'exp_req_ids': req.get('exp_req_ids', []),
            'testcase_status': req.get('testcase_status', ''),
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
                'exp_req_ids': doc_data['exp_req_ids'],
                'source_type': doc_data['source_type'],
            }
        )

    _firestore_commit_many(doc_tuples)

    return written_reqs


def _mark_duplicates(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> int:
    '''
    Compares requirement embeddings and marks duplicates in Firestore using vector similarity.
    Upon finding a duplicate (req_i), it marks req_i as a duplicate and unions
    its 'exp_req_ids' into the original requirement (req_j).

    Note: The order of requirements in the input list determines original status:
    the requirement with the lower index (i.e., appearing earlier) is the original one.
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

            if similarity >= DUPLICATE_SIM_THRESHOLD:
                # req_i is a duplicate of req_j (the earlier one is the original one)
                # Store the IDs and the source IDs from the duplicate req_i to be merged
                duplicates_to_update.append(
                    (
                        req_i['requirement_id'],
                        req_j['requirement_id'],
                        req_i.get('exp_req_ids', []),
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

    for req_id, near_duplicate_id, exp_req_ids in duplicates_to_update:
        doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', req_id
        )

        batch.update(doc_ref, {'duplicate': True, 'near_duplicate_id': near_duplicate_id})

        if exp_req_ids:
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
                {'exp_req_ids': firestore.ArrayUnion(exp_req_ids)},
            )

    batch.commit()

    all_duplicates: List[Dict[str, Any]] = []
    all_originals: List[Dict[str, Any]] = []

    for req in requirements:
        if req.get('duplicate', False):
            all_duplicates.append(req)
        else:
            all_originals.append(req)

    return all_duplicates, all_originals


# =====================
# Main HTTP Function
# =====================
@functions_framework.http
def process_requirements_phase_2(request):
    '''
    Main Cloud Function entry point for requirements processing Phase 2. It orchestrates the
    loading, explicit storage, vector deduplication, Discovery Engine search, and persistence
    of implicit requirements, with an added Gemini refinement step.
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

        _update_firestore_status(project_id, version, 'START_REQ_EXTRACT_P2')

        normalised_reqs = _load_and_normalize_exp_req(reqs_url)

        _update_firestore_status(project_id, version, 'START_STORE_EXPLICIT')

        explicit_reqs = _persist_requirements_to_firestore(
            project_id, version, normalised_reqs
        )

        print(f'Explicit writes => {len(explicit_reqs)}')

        _update_firestore_status(project_id, version, 'START_DEDUPE_EXPLICIT')

        dupe_exps, orig_exps = _mark_duplicates(project_id, version, explicit_reqs)

        print(f'Dedupe => Marked {len(dupe_exps)} explicit duplicates.')

        _update_firestore_status(project_id, version, 'START_IMPLICIT_DISCOVERY')

        implicit_candidates = _query_discovery_engine_parallel(orig_exps)

        _update_firestore_status(project_id, version, 'START_REFINE_IMPLICIT')

        implicit_candidates = _refine_candidates_parallel(implicit_candidates)

        implicit_reqs = _format_discovery_results(implicit_candidates)

        _update_firestore_status(project_id, version, 'START_STORE_IMPLICIT')

        implicit_reqs = _persist_requirements_to_firestore(
            project_id,
            version,
            implicit_reqs,
            len(explicit_reqs) + 1,
        )

        _update_firestore_status(project_id, version, 'START_DEDUPE_IMPLICIT')

        dupe_imps, orig_imps = _mark_duplicates(project_id, version, implicit_reqs)

        _update_firestore_status(project_id, version, 'CONFIRM_REQ_EXTRACT')

        print(f'Vector Dedupe (Implicit Only) => Marked {len(dupe_imps)} duplicates.')

        return ('OK', 200)

    except Exception as e:
        logging.exception(
            'Error during requirements extraction phase 2 (VECTOR DEDUPE):'
        )

        if project_id and version:
            _update_firestore_status(project_id, version, 'ERR_REQ_EXTRACT_P2')

        return (
            json.dumps(
                {
                    'status': 'error',
                    'message': f'An unexpected error occurred during processing: {str(e)}',
                }
            ),
            500,
        )
