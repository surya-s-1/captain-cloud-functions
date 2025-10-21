import os
import json
import logging
import functools
import time
from typing import Any, Dict, List, Tuple

import functions_framework
from google import genai
from google.cloud import firestore
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

# =====================
# Environment variables
# =====================
FIRESTORE_DATABASE = os.environ.get('FIRESTORE_DATABASE')
GENAI_MODEL = os.environ.get('GENAI_MODEL')
FIRESTORE_COMMIT_CHUNK = int(os.environ.get('FIRESTORE_COMMIT_CHUNK', '450'))

# =====================
# Clients
# =====================
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))
logging.basicConfig(level=logging.INFO)


# =====================
# Utilities
# =====================
def _update_requirement_status(
    project_id: str, version: str, requirement_id: str, status: str
):
    doc_ref = firestore_client.document(
        'projects', project_id, 'versions', version, 'requirements', requirement_id
    )
    doc_ref.set({'testcase_status': status}, merge=True)
    logging.info(f'Requirement {requirement_id} status => {status}')


def _retry(max_attempts=3, base_delay=0.5, exc_types=(Exception,)):
    '''A decorator with exponential backoff and jitter for retrying transient errors.'''

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exc_types as e:
                    if attempt == max_attempts:
                        raise
                    logging.warning(f'Retry {attempt}/{max_attempts} after error: {e}')
                    time.sleep(delay)
                    delay *= 2
            return None

        return wrapper

    return deco


@_retry(max_attempts=3)
def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> int:
    batch = firestore_client.batch()
    count = 0
    total = 0

    for doc_ref, data in doc_tuples:
        batch.set(doc_ref, data)
        count += 1
        total += 1

        if count >= FIRESTORE_COMMIT_CHUNK:
            batch.commit()
            batch = firestore_client.batch()
            count = 0

    if count:
        batch.commit()

    return total


# =====================
# Testcase generation
# =====================
@_retry(max_attempts=3)
def _generate_test_cases(requirement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    '''Generate test cases for one requirement using Gemini.'''
    requirement_text = requirement_data.get('requirement', '')
    prompt = f'''
    You are a medical industry QA engineer. Based on the following requirement, generate JSON test cases.
    Each test case must include: title, description (markdown bullets), acceptance_criteria (markdown bullets), priority (High/Medium/Low).
    
    Requirement: {requirement_text}
    '''

    response_schema = {
        'type': 'ARRAY',
        'items': {
            'type': 'OBJECT',
            'properties': {
                'title': {'type': 'STRING'},
                'description': {'type': 'STRING'},
                'acceptance_criteria': {'type': 'STRING'},
                'priority': {'type': 'STRING', 'enum': ['High', 'Medium', 'Low']},
            },
            'required': ['title', 'description', 'acceptance_criteria', 'priority'],
        },
    }

    resp = genai_client.models.generate_content(
        model=GENAI_MODEL,
        contents=[Content(parts=[Part(text=prompt)], role='user')],
        config=GenerateContentConfig(
            response_mime_type='application/json', response_json_schema=response_schema
        ),
    )

    return json.loads(resp.text)


# =======================================================
# Cloud Function: Cloud Tasks Worker
# Processes a single requirement to generate test cases
# =======================================================
@functions_framework.http
def generate_test_cases(request):
    '''
    Cloud Task worker that generates test cases for a single requirement.
    '''
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return {'error': 'JSON body not provided.'}, 400

        project_id = request_json.get('project_id')
        version = request_json.get('version')
        requirement_id = request_json.get('requirement_id')

        if not project_id or not version or not requirement_id:
            return {'error': 'Missing project_id, version, or requirement_id'}, 400

        logging.info(f'Processing requirement {requirement_id}...')

        # Get the specific requirement to check its status
        req_ref = firestore_client.document(
            'projects', project_id, 'versions', version, 'requirements', requirement_id
        )
        req_doc = req_ref.get()

        if not req_doc.exists:
            logging.warning(f'Requirement {requirement_id} not found.')
            return {
                'status': 'skipped',
                'message': 'Requirement not found, skipping.',
            }, 200

        req_data = req_doc.to_dict()
        if req_data.get('testcase_status') == 'TESTCASES_CREATION_COMPLETE':
            logging.info(f'Requirement {requirement_id} already processed. Skipping.')
            return {'status': 'skipped', 'message': 'Already processed, skipping.'}, 200

        _update_requirement_status(
            project_id, version, requirement_id, 'TESTCASES_CREATION_STARTED'
        )

        # Generate test cases using Gemini
        try:
            testcases = _generate_test_cases(req_data)
        except Exception as e:
            logging.error(f'Gemini call failed for {requirement_id}: {e}')
            _update_requirement_status(
                project_id, version, requirement_id, 'ERR_GEMINI_CALL'
            )
            return {'error': 'Gemini call failed.'}, 500

        # Prepare test case documents for batch write
        testcase_docs = []
        for i, tc in enumerate(testcases, start=1):
            tc_id = f'{requirement_id}-TC-{i}'
            testcase_docs.append(
                (
                    firestore_client.collection('projects')
                    .document(project_id)
                    .collection('versions')
                    .document(version)
                    .collection('testcases')
                    .document(tc_id),
                    {
                        **tc,
                        'testcase_id': tc_id,
                        'requirement_id': requirement_id,
                        'change_analysis_status': 'NEW',
                        'toolCreated': False,
                        'toolIssueLink': '',
                        'deleted': False,
                        'datasets': [],
                        'dataset_status': 'NOT_STARTED',
                    },
                )
            )

        # Bulk Firestore write
        try:
            total_written = _firestore_commit_many(testcase_docs)
        except Exception as e:
            logging.error(f'Firestore write failed for {requirement_id}: {e}')
            return {'error': 'Saving to DB failed.'}, 500

        logging.info(
            f'Stored {total_written} testcases for requirement {requirement_id}.'
        )

        _update_requirement_status(
            project_id, version, requirement_id, 'TESTCASES_CREATION_COMPLETE'
        )

        collection_ref = firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )
        requirements = [
            doc.to_dict() for doc in collection_ref.get() if doc.get('deleted') != True
        ]

        if len(requirements) == len(
            [
                req
                for req in requirements
                if req.get('testcase_status', '') == 'TESTCASES_CREATION_COMPLETE'
            ]
        ):
            version_ref = firestore_client.document(
                'projects', project_id, 'versions', version
            )
            version_ref.update({'status': 'CONFIRM_TESTCASES'})

        return (
            json.dumps(
                {'status': 'success', 'message': 'Test cases generated and stored.'}
            ),
            200,
        )

    except Exception as e:
        logging.exception(f'Error processing task for requirement {requirement_id}')

        _update_requirement_status(
            project_id, version, requirement_id, 'ERR_TESTCASE_CREATION'
        )

        return {'error': str(e)}, 500
