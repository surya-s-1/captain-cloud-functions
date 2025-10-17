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
DUPE_SIM_THRESHOLD = 0.95
REQ_UNCHANGED_SIM_THRESHOLD = 0.95  # Dedup/Exact match: >= 0.95
REQ_DEPRECATED_SIM_THRESHOLD = (
    0.65  # Semantic difference: < 0.65 (old text is deprecated)
)
REQ_MODIFIED_SIM_THRESHOLD = 0.65  # Change detected: 0.65 <= score < 0.95

GENAI_MODEL = 'gemini-2.5-flash'
GENAI_API_VERSION = 'v1'
GENAI_TIMEOUT_SECONDS = 90  # Each LLM call safety timeout

CHANGE_STATUS_NEW = 'NEW'
CHANGE_STATUS_MODIFIED = 'MODIFIED'
CHANGE_STATUS_UNCHANGED = 'UNCHANGED'
CHANGE_STATUS_DEPRECATED = 'DEPRECATED'

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


def _load_and_normalize_exp_req(version: str, obj_url: str) -> List[Dict[str, Any]]:
    print('Starting new explicit requirement processing from GCS...')

    parsed = urlparse(obj_url)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip('/'))
    explicit_requirements_raw = json.loads(blob.download_as_text())

    if not explicit_requirements_raw:
        raise ValueError('Input data from GCS is empty.')

    normalized_list = normalize_req_dict(explicit_requirements_raw)

    texts_to_embed = [r.get('requirement', '') for r in normalized_list]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        embedding_vectors = list(ex.map(_generate_embedding, texts_to_embed))

    final_list = []
    for i, r in enumerate(normalized_list):
        requirement_id = f'{version}-REQ-{i:03d}'
        final_list.append(
            {
                'requirement': r.get('requirement', ''),
                'requirement_type': r.get('requirement_type', 'functional'),
                'exp_req_ids': [],
                'sources': r.get('sources', []),
                'source_type': 'explicit',
                'embedding': embedding_vectors[i],
                'requirement_id': requirement_id,
                'change_analysis_status': 'NEW',  # Initial status
            }
        )

    print(f'Loaded, normalized, and embedded {len(final_list)} explicit requirements.')

    return final_list


def _load_existing_requirements(project_id: str, version: str) -> List[Dict[str, Any]]:
    '''
    Loads all *existing*, non-deleted, non-duplicate explicit and implicit requirements
    from the current Firestore version.
    '''
    print('Loading existing requirements from Firestore for change detection...')

    query = (
        firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )
        .where('deleted', '==', False)
        .where('change_analysis_status', '!=', CHANGE_STATUS_DEPRECATED)
        .where('is_duplicate', '==', False)
    )

    existing_reqs = []
    for doc in query.stream():
        data = doc.to_dict()
        if 'requirement' in data and 'embedding' in data:
            existing_reqs.append(
                {
                    'requirement_id': data.get('requirement_id'),
                    'requirement': data['requirement'],
                    'embedding': data['embedding'],
                    'source_type': data.get('source_type'),
                    'exp_req_ids': data.get('exp_req_ids', []),
                }
            )

    print(f'Loaded {len(existing_reqs)} existing requirements.')
    return existing_reqs


def _mark_new_reqs_change_status(
    new_reqs: List[Dict[str, Any]], existing_exp_reqs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    print(f'Starting change detection for {len(new_reqs)} new texts...')

    old_ids_checked = set()

    for new_req in new_reqs:
        best_match = None
        max_sim_score = -1.0

        # Find the best match among existing explicit requirements
        for old_req in existing_exp_reqs:
            # Only check explicit against explicit
            if old_req['source_type'] != 'explicit':
                continue

            sim_score = cosine_similarity(new_req['embedding'], old_req['embedding'])

            if sim_score > max_sim_score:
                max_sim_score = sim_score
                best_match = old_req

        if max_sim_score >= REQ_UNCHANGED_SIM_THRESHOLD:
            new_req['change_analysis_status'] = CHANGE_STATUS_UNCHANGED
            new_req['change_analysis_original_id'] = best_match['requirement_id']
            old_ids_checked.add(best_match['requirement_id'])

        elif max_sim_score >= REQ_MODIFIED_SIM_THRESHOLD:
            new_req['change_analysis_status'] = CHANGE_STATUS_MODIFIED
            new_req['change_analysis_original_id'] = best_match['requirement_id']
            old_ids_checked.add(best_match['requirement_id'])

        else:
            new_req['change_analysis_status'] = CHANGE_STATUS_NEW
            new_req['change_analysis_original_id'] = None

    print(
        f"Statuses: UNCHANGED={len([r for r in new_reqs if r['change_analysis_status'] == 'UNCHANGED'])}, "
        f"MODIFIED={len([r for r in new_reqs if r['change_analysis_status'] == 'MODIFIED'])}, "
        f"NEW={len([r for r in new_reqs if r['change_analysis_status'] == 'NEW'])}"
    )

    # Filter existing requirements to only include those that were NOT matched
    old_exp_to_check = [
        r
        for r in existing_exp_reqs
        if r['requirement_id'] not in old_ids_checked and r['source_type'] == 'explicit'
    ]

    return new_reqs, old_exp_to_check


def _mark_old_reqs_deprecated(
    old_reqs_to_check: List[Dict[str, Any]], new_reqs: List[Dict[str, Any]]
) -> List[str]:
    '''
    Step 2: Compares unmatched old explicit requirements against all new requirements
    to determine if they are DEPRECATED.
    '''
    deprecated_ids = []
    new_exp_reqs = [
        r for r in new_reqs if r['change_analysis_status'] != CHANGE_STATUS_UNCHANGED
    ]

    if not new_exp_reqs:
        return []

    print(
        f'Starting deprecation check for {len(old_reqs_to_check)} old explicit texts...'
    )

    for old_req in old_reqs_to_check:
        max_sim_score = -1.0

        # Find the best match among the new requirements
        for new_req in new_exp_reqs:
            sim_score = cosine_similarity(old_req['embedding'], new_req['embedding'])
            if sim_score > max_sim_score:
                max_sim_score = sim_score

        # If the best match is below the deprecation threshold, the old requirement is obsolete.
        if max_sim_score < REQ_DEPRECATED_SIM_THRESHOLD:
            deprecated_ids.append(old_req['requirement_id'])

    print(f'Marking {len(deprecated_ids)} old explicit requirements as DEPRECATED.')
    return deprecated_ids


def _mark_firestore_updates(
    project_id: str,
    version: str,
    deprecated_exp_ids: List[str],
    existing_all_reqs: List[Dict[str, Any]],
) -> None:
    print('Committing status updates to Firestore (Deprecated/Implicit propagation)...')

    batch = firestore_client.batch()
    req_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    # 1. Mark deprecated explicit requirements
    for req_id in deprecated_exp_ids:
        doc_ref = req_collection_ref.document(req_id)
        batch.update(
            doc_ref,
            {
                'change_analysis_status': CHANGE_STATUS_DEPRECATED,
                'deprecation_reason': 'New requirements doesn\'t have this requirement anymore',
            },
        )

    # 1. Mark deprecated implicit requirements
    for req in existing_all_reqs:
        if req.get('source_type', None) == 'implicit':
            # Get all currently non-deprecated explicit IDs linked to this implicit one
            linked_exp_ids = set(req.get('exp_req_ids', []))
            non_deprecated_linked_ids = linked_exp_ids.difference(
                set(deprecated_exp_ids)
            )

            # Check if all sources are now deprecated
            if not non_deprecated_linked_ids:
                # Only mark implicit as deleted if it had sources, and ALL sources are now deprecated
                if linked_exp_ids:
                    doc_ref = req_collection_ref.document(req.get('requirement_id'))
                    batch.update(
                        doc_ref,
                        {
                            'change_analysis_status': CHANGE_STATUS_DEPRECATED,
                            'deprecation_reason': 'All source explicit requirements deprecated',
                        },
                    )

    batch.commit()


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
                'deleted': False,  # New requirements are not deleted
                'is_duplicate': False,  # To be determined later
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
    print(f'Persisting {len(requirements)} requirements starting from ID {start_id}...')

    requirements_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    doc_insertions_tuples_list = []  # Stores (DocumentReference, Data) for batch operations
    doc_updates_tuples_list = []  # Stores (DocumentReference, Data) for update operations
    written_reqs = []

    current_index = start_id

    for req in requirements:
        req_source_type = req.get('source_type', '')
        req_change_status = req.get('change_analysis_status', 'NEW')

        doc_data = {
            **req,
            'created_at': firestore.SERVER_TIMESTAMP
        }
        doc_data.pop('temp_id', None)

        # Case 1: IMPLICIT (Always new insertions)
        if req_source_type == 'implicit':
            req_id = (
                f'{version}-REQ-{current_index:03d}'
                if not req.get('requirement_id')
                else req.get('requirement_id')
            )
            doc_data['requirement_id'] = req_id
            doc_data['testcase_status'] = ''

            doc_ref = requirements_collection_ref.document(req_id)
            doc_insertions_tuples_list.append((doc_ref, doc_data))
            current_index += 1

            req['requirement_id'] = req_id

        elif req_source_type == 'explicit':
            change_analysis_original_id = req.get('change_analysis_original_id', '')

            # Case 2: EXPLICIT NEW (New insertions)
            if req_change_status == CHANGE_STATUS_NEW:
                req_id = (
                    f'{version}-REQ-{current_index:03d}'
                    if not req.get('requirement_id')
                    else req.get('requirement_id')
                )
                doc_data['requirement_id'] = req_id
                doc_data['testcase_status'] = ''

                doc_ref = requirements_collection_ref.document(req_id)
                doc_insertions_tuples_list.append((doc_ref, doc_data))

                current_index += 1

                req['requirement_id'] = req_id

            # Case 3: EXPLICIT MODIFIED/UNCHANGED (Update existing document)
            elif (
                req_change_status in (CHANGE_STATUS_MODIFIED, CHANGE_STATUS_UNCHANGED)
                and change_analysis_original_id
            ):
                req_id = change_analysis_original_id
                doc_data['requirement_id'] = req_id
                doc_data['testcase_status'] = '' if req_change_status == CHANGE_STATUS_MODIFIED else req.get('testcase_status', '')

                doc_ref = requirements_collection_ref.document(req_id)
                doc_updates_tuples_list.append((doc_ref, doc_data))

                req['requirement_id'] = req_id

            else:
                print(
                    f"WARNING: Explicit requirement skipped due to unknown status '{req_change_status}' or missing 'change_analysis_original_id'."
                )
                continue

        written_reqs.append(
            {
                'requirement_id': req['requirement_id'],
                'embedding': req['embedding'],
                'is_duplicate': req.get('is_duplicate', False),
                'exp_req_ids': req.get('exp_req_ids', []),
                'source_type': req_source_type,
                'change_analysis_original_id': req.get('change_analysis_original_id', ''),
                'change_analysis_status': req_change_status,
            }
        )

    if doc_insertions_tuples_list:
        print(f"Committing {len(doc_insertions_tuples_list)} INSERT/SET operations...")
        _firestore_commit_many(doc_insertions_tuples_list)

    if doc_updates_tuples_list:
        print(f"Committing {len(doc_updates_tuples_list)} UPDATE operations...")
        # NOTE: Updates cannot use _firestore_commit_many (which uses batch.set).
        # They must be committed with batch.update().
        batch = firestore_client.batch()
        for doc_ref, data in doc_updates_tuples_list:
            update_data = {
                'requirement_id': data.get('requirement_id', ''),
                'source_type': data.get('source_type', ''),
                'requirement': data.get('requirement', ''),
                'embedding': data.get('embedding', []),
                'requirement_type': data.get('requirement_type', ''),
                'deleted': data.get('deleted', False),
                'is_duplicate': data.get('is_duplicate', False),
                'change_analysis_status': data.get('change_analysis_status', ''),
                'change_analysis_original_id': data.get(
                    'change_analysis_original_id', ''
                ),
                'deprecation_reason': data.get('deprecation_reason', ''),
                'sources': data.get('sources', []),
                'regulations': data.get('regulations', []),
                'exp_req_ids': data.get('exp_req_ids', []),
                'testcase_status': data.get('testcase_status', ''),
                'updated_at': firestore.SERVER_TIMESTAMP,
                'created_at': data.get('created_at', firestore.SERVER_TIMESTAMP),
            }

            batch.update(doc_ref, update_data)
        batch.commit()

    return written_reqs


def _mark_duplicates(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''
    Step 4 & 6: Compares requirement embeddings and marks duplicates in Firestore.
    This function is used for both explicit and implicit deduplication.
    '''

    # Stores (duplicate_id, original_id, duplicate_source_ids_to_merge)
    duplicates_to_update: List[Tuple[str, str, List[str]]] = []

    # Iterate through each requirement (i)
    for i in range(len(requirements)):
        req_i = requirements[i]

        # Compare req_i against all requirements (j) that came before it (j < i)
        for j in range(i):
            req_j = requirements[j]

            # Skip comparing against a requirement that has already been marked as a duplicate.
            if req_j.get('is_duplicate', False):
                continue

            similarity = cosine_similarity(req_i['embedding'], req_j['embedding'])

            if similarity >= DUPE_SIM_THRESHOLD:
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
                req_i['is_duplicate'] = True
                break

    if not duplicates_to_update:
        print('No duplicates found using vector embeddings in this batch.')
        return [], requirements

    print(f'Found {len(duplicates_to_update)} duplicates to mark.')

    batch = firestore_client.batch()

    for req_id, original_id, exp_req_ids in duplicates_to_update:
        doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', req_id
        )

        batch.update(doc_ref, {'is_duplicate': True, 'original_id': original_id})

        # Only merge source IDs if the duplicate is an explicit requirement
        if exp_req_ids and req_i['source_type'] == 'explicit':
            original_ref = firestore_client.document(
                'projects',
                project_id,
                'versions',
                version,
                'requirements',
                original_id,
            )

            batch.update(
                original_ref,
                {'exp_req_ids': firestore.ArrayUnion(exp_req_ids)},
            )

    batch.commit()

    all_duplicates: List[Dict[str, Any]] = []
    all_originals: List[Dict[str, Any]] = []

    for req in requirements:
        if req.get('is_duplicate', False):
            all_duplicates.append(req)
        else:
            all_originals.append(req)

    return all_duplicates, all_originals


# =====================
# Main HTTP Function
# =====================
@functions_framework.http
def process_requirements_phase_2_change_analysis(request):
    '''
    Main Cloud Function entry point for requirements processing Phase 2. It orchestrates the
    change detection, explicit storage, vector deduplication, Discovery Engine search, and persistence
    of implicit requirements, with an added Gemini refinement step.
    '''
    project_id = None
    version = None

    try:
        payload = request.get_json(silent=True) or {}
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

        existing_all_reqs = _load_existing_requirements(project_id, version)
        existing_exp_reqs = [
            r for r in existing_all_reqs if r['source_type'] == 'explicit'
        ]

        print(f'Loaded {len(existing_exp_reqs)} existing explicit requirements.')

        new_exp_reqs = _load_and_normalize_exp_req(reqs_url)

        _update_firestore_status(project_id, version, 'START_CHANGE_DETECTION')

        # old_exp_to_check is the list of existing explicit reqs not covered by the new list
        new_exp_reqs, old_exp_to_check = _mark_new_reqs_change_status(
            new_exp_reqs, existing_exp_reqs
        )

        # Compare the unmatched old texts against all new texts to see if they're obsolete
        deprecated_exp_ids = _mark_old_reqs_deprecated(old_exp_to_check, new_exp_reqs)

        _update_firestore_status(project_id, version, 'START_DEPRECATION_COMMIT')

        _mark_firestore_updates(
            project_id, version, deprecated_exp_ids, existing_all_reqs
        )

        _update_firestore_status(project_id, version, 'START_STORE_EXPLICIT')

        persisted_new_exp_reqs = _persist_requirements_to_firestore(
            project_id, version, new_exp_reqs
        )

        print(f'New/Modified/Unchanged Explicit writes => {len(persisted_new_exp_reqs)}')

        _update_firestore_status(project_id, version, 'START_DEDUPE_EXPLICIT')

        dupe_exps, orig_exps = _mark_duplicates(
            project_id, version, persisted_new_exp_reqs
        )

        print(f'Dedupe => Marked {len(dupe_exps)} new explicit duplicates.')

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
            start_id=len(persisted_new_exp_reqs),
        )

        _update_firestore_status(project_id, version, 'START_DEDUPE_IMPLICIT')

        dupe_imps, orig_imps = _mark_duplicates(project_id, version, implicit_reqs)

        print(f'Vector Dedupe (Implicit Only) => Marked {len(dupe_imps)} duplicates.')

        _update_firestore_status(project_id, version, 'CONFIRM_REQ_EXTRACT')

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
