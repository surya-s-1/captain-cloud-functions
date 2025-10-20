import os
import json
import time
import logging
import functools
import concurrent.futures as futures
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from google import genai
from google.genai.types import HttpOptions, Part, Content
from google.cloud import storage, firestore
import functions_framework

from dotenv import load_dotenv

load_dotenv()

# ===================== # Environment variables # =====================
# These are required dependencies for the functions in this file
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
PROJECT_ID = os.getenv('PROJECT_ID')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
DUPE_SIM_THRESHOLD = float(os.getenv('DUPE_SIM_THRESHOLD'))
REQ_UNCHANGED_SIM_THRESHOLD = float(os.getenv('DUPE_SIM_THRESHOLD'))
REQ_MODIFIED_SIM_THRESHOLD = float(os.getenv('DUPE_SIM_THRESHOLD'))
REQ_DEPRECATED_SIM_THRESHOLD = REQ_MODIFIED_SIM_THRESHOLD
MAX_WORKERS = 16
FIRESTORE_COMMIT_CHUNK = 450

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


def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> None:
    '''Commits a list of documents to Firestore in batches using batch.set().'''
    batch = firestore_client.batch()
    count = 0
    for doc_ref, data in doc_tuples:
        batch.set(doc_ref, data, merge=True)  # Use merge=True for updates/sets
        count += 1
        if count >= FIRESTORE_COMMIT_CHUNK:
            batch.commit()
            batch = firestore_client.batch()
            count = 0
    if count:
        batch.commit()


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    '''Calculates the cosine similarity between two vectors.'''
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = sum(a * a for a in v1) ** 0.5
    magnitude_v2 = sum(b * b for b in v2) ** 0.5
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)


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


def _load_updated_exp_requirements(
    version: str, requirements_p1_url: str
) -> List[Dict[str, Any]]:
    '''Loads, normalizes, and embeds new explicit requirements from GCS.'''
    print('Starting new explicit requirement processing from GCS...')
    parsed = urlparse(requirements_p1_url)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip('/'))
    explicit_requirements_raw = json.loads(blob.download_as_text())

    if not explicit_requirements_raw:
        raise ValueError('Input data from GCS is empty.')

    normalized_list: List[Dict[str, Any]] = _normalize_requirements(
        explicit_requirements_raw
    )
    texts_to_embed = [r.get('requirement', '') for r in normalized_list]

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        embedding_vectors = list(ex.map(_generate_embedding, texts_to_embed))

    final_list = []
    for i, (r, embedding_vector) in enumerate(
        zip(normalized_list, embedding_vectors), start=1
    ):
        requirement_id = f'{version}-REQ-E-{i:03d}'
        final_list.append(
            {
                'requirement_id': requirement_id,
                'requirement': r.get('requirement', ''),
                'requirement_type': r.get('requirement_type', 'functional'),
                'exp_req_ids': [],
                'sources': r.get('sources', []),
                'source_type': SOURCE_TYPE_EXPLICIT,
                'embedding': embedding_vector,
                'change_analysis_status': CHANGE_STATUS_NEW,  # Initial status
                'deleted': False,
                'duplicate': False,
            }
        )
    print(f'Loaded, normalized, and embedded {len(final_list)} explicit requirements.')
    return final_list


def _load_existing_exp_requirements(
    project_id: str, version: str
) -> List[Dict[str, Any]]:
    print('Loading existing requirements from Firestore for change detection...')

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

    deprecated_query = collection_ref.where(
        'change_analysis_status', '==', CHANGE_STATUS_IGNORED
    )
    for doc in deprecated_query.stream():
        batch.update(
            doc.reference,
            {
                'change_analysis_status_reason': CHANGE_STATUS_IGNORED_REASON_IGNORED,
            },
        )

    print('Committing all batch updates...')
    batch.commit()
    print('Batch committed successfully.')

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
            existing_reqs.append(
                {
                    'requirement_id': data.get('requirement_id'),
                    'requirement': data.get('requirement'),
                    'embedding': data.get('embedding'),
                    'source_type': data.get('source_type'),
                    'exp_req_ids': data.get('exp_req_ids', []),
                }
            )
    print(f'Loaded {len(existing_reqs)} existing requirements.')
    return existing_reqs


def _mark_new_reqs_change_status(
    new_exp_reqs: List[Dict[str, Any]], existing_exp_reqs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''Compares new explicit requirements against existing ones to mark status (UNCHANGED/MODIFIED/NEW).'''
    print(f'Starting change detection for {len(new_exp_reqs)} new texts...')
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
            new_req['change_analysis_near_duplicate_id'] = best_match['requirement_id']
            old_ids_checked.add(best_match['requirement_id'])
        elif max_sim_score >= REQ_MODIFIED_SIM_THRESHOLD:
            new_req['change_analysis_status'] = CHANGE_STATUS_MODIFIED
            new_req['change_analysis_near_duplicate_id'] = best_match['requirement_id']
            old_ids_checked.add(best_match['requirement_id'])
        else:
            new_req['change_analysis_status'] = CHANGE_STATUS_NEW
            new_req['change_analysis_near_duplicate_id'] = None

    print(
        f'Statuses: UNCHANGED={len([r for r in new_exp_reqs if r['change_analysis_status'] == CHANGE_STATUS_UNCHANGED])},'
        f' MODIFIED={len([r for r in new_exp_reqs if r['change_analysis_status'] == CHANGE_STATUS_MODIFIED])},'
        f' NEW={len([r for r in new_exp_reqs if r['change_analysis_status'] == CHANGE_STATUS_NEW])}'
    )

    # Filter existing explicit requirements to only include those that were NOT matched
    old_exp_to_check = [
        r
        for r in existing_exp_reqs
        if r['requirement_id'] not in old_ids_checked and r['source_type'] == SOURCE_TYPE_EXPLICIT
    ]
    return new_exp_reqs, old_exp_to_check


def _mark_old_reqs_deprecated(old_reqs_to_check: List[Dict[str, Any]]) -> List[str]:
    '''Step 2: Marks unmatched old explicit requirements as DEPRECATED.'''
    # NOTE: The provided logic marks all unmatched explicit requirements as DEPRECATED.
    # We should compare against the *new* list to confirm obsolescence.
    print(
        f'Starting deprecation check for {len(old_reqs_to_check)} old explicit texts...'
    )
    deprecated_ids = [
        r.get('requirement_id') for r in old_reqs_to_check if r.get('requirement_id')
    ]
    print(f'Marking {len(deprecated_ids)} old explicit requirements as DEPRECATED.')
    return deprecated_ids


def _mark_deprecated_in_firestore(
    project_id: str, version: str, deprecated_exp_ids: List[str]
) -> None:
    print('Committing status updates to Firestore (Explicit Deprecation)...')

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
    project_id: str, version: str, requirements: List[Dict[str, Any]], start_id: int = 1
) -> List[Dict[str, Any]]:
    '''Persists new, modified, and unchanged explicit requirements to Firestore.'''
    print(f'Persisting {len(requirements)} explicit requirements to firestore...')
    requirements_collection_ref = firestore_client.collection(
        'projects', project_id, 'versions', version, 'requirements'
    )

    doc_insertions_tuples_list = []
    doc_updates_tuples_list = []
    written_reqs = []
    current_index = start_id

    for req in requirements:
        req_source_type = req.get('source_type', '')
        req_change_status = req.get('change_analysis_status', 'NEW')
        change_analysis_near_duplicate_id = req.get(
            'change_analysis_near_duplicate_id', ''
        )

        if req_source_type == SOURCE_TYPE_EXPLICIT:
            # Case 1: EXPLICIT NEW (New insertions)
            if req_change_status == CHANGE_STATUS_NEW:
                req_id = f'{version}-REQ-E-{current_index:03d}'

                doc_data = {
                    **req,
                    'requirement_id': req_id,
                    'testcase_status': '',
                    'change_analysis_status_reason': 'Detected only in updated requirements',
                    'created_at': firestore.SERVER_TIMESTAMP,
                }

                doc_ref = requirements_collection_ref.document(req_id)
                doc_insertions_tuples_list.append((doc_ref, doc_data))
                current_index += 1
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
                    'change_analysis_status_reason': (
                        'Did not detect any major changes in updated requirements'
                        if req_change_status == CHANGE_STATUS_UNCHANGED
                        else 'Detected considerable modifications in updated requirements'
                    ),
                    'testcase_status': (
                        ''
                        if req_change_status == CHANGE_STATUS_MODIFIED
                        else req.get('testcase_status', '')
                    ),
                }

                doc_ref = requirements_collection_ref.document(req_id)

                # Prepare update payload (use batch.update() structure for clarity)
                update_data = {
                    'requirement': doc_data.get('requirement', ''),
                    'embedding': doc_data.get('embedding', []),
                    'requirement_type': doc_data.get('requirement_type', ''),
                    'deleted': doc_data.get('deleted', False),
                    'duplicate': doc_data.get('duplicate', False),
                    'change_analysis_status': doc_data.get(
                        'change_analysis_status', ''
                    ),
                    'change_analysis_near_duplicate_id': doc_data.get(
                        'change_analysis_near_duplicate_id', ''
                    ),
                    'change_analysis_status_reason': doc_data.get(
                        'change_analysis_status_reason', ''
                    ),
                    'sources': doc_data.get('sources', []),
                    'testcase_status': doc_data.get('testcase_status', ''),
                    'updated_at': firestore.SERVER_TIMESTAMP,
                }

                doc_updates_tuples_list.append((doc_ref, update_data))
                req['requirement_id'] = req_id
            else:
                print(
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
                'exp_req_ids': req.get('exp_req_ids', []),
                'change_analysis_near_duplicate_id': req.get(
                    'change_analysis_near_duplicate_id', ''
                ),
                'change_analysis_status': req_change_status,
            }
        )

    if doc_insertions_tuples_list:
        print(f'Committing {len(doc_insertions_tuples_list)} INSERT/SET operations...')
        _firestore_commit_many(doc_insertions_tuples_list)

    if doc_updates_tuples_list:
        print(f'Committing {len(doc_updates_tuples_list)} UPDATE operations...')
        # Updates must use batch.update() which requires the document to exist.
        batch = firestore_client.batch()
        for doc_ref, data in doc_updates_tuples_list:
            batch.update(doc_ref, data)
        batch.commit()

    return written_reqs


def _mark_duplicates(
    project_id: str, version: str, requirements: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    '''Compares requirement embeddings and marks duplicates in Firestore.'''
    duplicates_to_update: List[Tuple[str, str, List[str]]] = []

    for i in range(len(requirements)):
        req_i = requirements[i]
        for j in range(i):
            req_j = requirements[j]
            if req_j.get('duplicate', False):
                continue

            similarity = _cosine_similarity(req_i['embedding'], req_j['embedding'])

            if similarity >= DUPE_SIM_THRESHOLD:
                duplicates_to_update.append(
                    (
                        req_i['requirement_id'],
                        req_j['requirement_id'],
                        req_i.get('exp_req_ids', []),
                    )
                )
                req_i['duplicate'] = True
                break

    if not duplicates_to_update:
        print('No explicit duplicates found using vector embeddings in this batch.')
        return [], requirements

    print(f'Found {len(duplicates_to_update)} explicit duplicates to mark.')
    batch = firestore_client.batch()

    for req_id, near_duplicate_id, exp_req_ids in duplicates_to_update:
        doc_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', req_id
        )
        batch.update(
            doc_ref, {'duplicate': True, 'near_duplicate_id': near_duplicate_id}
        )

        # Merge source IDs into the original document
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
                original_ref, {'exp_req_ids': firestore.ArrayUnion(exp_req_ids)}
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
        #     'version': 'v2',
        #     'requirements_p1_url': 'gs://genai-sage/projects/abc/v_v2/extractions/requirements-phase-1.json',
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

        _update_version_status(project_id, version, 'START_REQ_EXTRACT_P2')

        existing_exp_reqs = _load_existing_exp_requirements(project_id, version)
        print(f'Loaded {len(existing_exp_reqs)} existing explicit requirements.')

        new_exp_reqs = _load_updated_exp_requirements(version, reqs_url)

        _update_version_status(project_id, version, 'START_CHANGE_DETECTION')

        # 1. Compare new list against old list (Find UNCHANGED, MODIFIED, NEW)
        new_exp_reqs, old_exp_to_check = _mark_new_reqs_change_status(
            new_exp_reqs, existing_exp_reqs
        )

        # 2. Mark old unmatched explicit requirements as DEPRECATED
        deprecated_exp_ids = _mark_old_reqs_deprecated(old_exp_to_check)

        _update_version_status(project_id, version, 'START_DEPRECATION_COMMIT_EXPLICIT')

        # 3. Commit explicit deprecation status updates
        _mark_deprecated_in_firestore(project_id, version, deprecated_exp_ids)

        _update_version_status(project_id, version, 'START_STORE_EXPLICIT')

        # 4. Store/Update new/modified/unchanged explicit requirements
        written_exp_reqs = _mark_unchanged_modified_new_in_firestore(
            project_id, version, new_exp_reqs
        )
        print(f'New/Modified/Unchanged Explicit writes => {len(written_exp_reqs)}')

        _update_version_status(project_id, version, 'START_DEDUPE_EXPLICIT')

        # 5. Deduplicate the newly written explicit requirements
        dupe_exps, orig_exps = _mark_duplicates(project_id, version, written_exp_reqs)
        print(
            f'Vector Dedupe (Explicit)=> Marked {len(dupe_exps)} new explicit duplicates.'
        )

        _update_version_status(project_id, version, 'CONFIRM_EXPLICIT_PROCESSED')

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
        logging.exception('Error during explicit requirements extraction phase 2:')
        if project_id and version:
            _update_version_status(project_id, version, 'ERR_REQ_EXTRACT_P2_EXPLICIT')
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
