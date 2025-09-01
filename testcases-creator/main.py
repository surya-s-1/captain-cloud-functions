import os
import json
import threading
import functions_framework

from google import genai
from google.cloud import firestore
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

FIRESTORE_DATABASE = os.environ.get('FIRESTORE_DATABASE')

firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))


def _update_firestore_status(project_id, version, status):
    '''Updates the status of a specific project version in Firestore.'''
    doc_ref = (
        firestore_client.collection('projects')
        .document(project_id)
        .collection(f'versions')
        .document(version)
    )
    update_data = {'status': status}
    doc_ref.set(update_data, merge=True)
    print(f'Updated status for project {project_id} version {version} to {status}.')


def _generate_test_cases(requirement_data):
    '''
    Generates test cases for a given requirement using the Gemini model.
    '''
    requirement_text = requirement_data['requirement']

    prompt = f'''
    You are a medical insdustry QA engineer for medical software development. Based on the following software requirement, generate one or more test cases. 
    The test cases should be in a JSON format as a list of objects. Each test case object must have the following fields:
    - 'testcase_id': A string in the format 'TC-001', 'TC-002', etc.
    - 'title': A concise title for the test case.
    - 'description': A detailed description of the test case, formatted in markdown with bullet points.
    - 'acceptance_criteria': The criteria that must be met for the test to pass, formatted in markdown with bullet points.
    - 'priority': A string value of 'High', 'Medium', or 'Low'.

    Requirement: {requirement_text}

    Example JSON output format:
    [
        {{
            'testcase_id': 'TC-001',
            'title': 'Verify user registration with valid data',
            'description': '- Enter valid email and password.\n- Click on 'Register' button.',
            'acceptance_criteria': '- The user should be successfully registered.\n- A confirmation email should be sent.',
            'priority': 'High'
        }}
    ]
    '''

    response_schema = {
        'type': 'ARRAY',
        'items': {
            'type': 'OBJECT',
            'properties': {
                'testcase_id': {'type': 'STRING'},
                'title': {'type': 'STRING'},
                'description': {'type': 'STRING'},
                'acceptance_criteria': {'type': 'STRING'},
                'priority': {'type': 'STRING', 'enum': ['High', 'Medium', 'Low']},
            },
            'required': [
                'testcase_id',
                'title',
                'description',
                'acceptance_criteria',
                'priority',
            ],
        },
    }

    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[Content(parts=[Part(text=prompt)], role='user')],
            config=GenerateContentConfig(
                response_mime_type='application/json',
                response_json_schema=response_schema,
            ),
        )

        test_cases_json = json.loads(response.text)

        print(f'Generated test cases: {len(test_cases_json)}')

        return test_cases_json
    except Exception as e:
        print(f'Error generating test cases: {e}')
        return []


def _create_testcases(project_id, version):
    try:
        _update_firestore_status(project_id, version, 'START_TESTCASE_CREATION')

        requirements_ref = (
            firestore_client.collection('projects')
            .document(project_id)
            .collection('versions')
            .document(version)
            .collection('requirements')
        )

        requirements_docs = requirements_ref.get()

        for req_doc in requirements_docs:
            requirement_data = req_doc.to_dict()
            requirement_id = req_doc.id

            test_cases = _generate_test_cases(requirement_data)

            for i, test_case in enumerate(test_cases):
                test_case_id = test_case.get('testcase_id', f'TC-{i+1}')
                test_case_ref = (
                    firestore_client.collection('projects')
                    .document(project_id)
                    .collection('versions')
                    .document(version)
                    .collection('requirements')
                    .document(requirement_id)
                    .collection('test_cases')
                    .document(test_case_id)
                )

                test_case_ref.set(test_case)

                print(
                    f'Stored test case {test_case_id} for requirement {requirement_id}'
                )

        _update_firestore_status(project_id, version, 'COMPLETE_TESTCASE_CREATION')

    except Exception as e:
        print(f'An error occurred: {e}')

        _update_firestore_status(project_id, version, 'ERR_TESTCASE_CREATION')


@functions_framework.http
def process_for_testcases(request):
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return {'error': 'JSON body not provided.'}, 400

        project_id = request_json.get('project_id')
        version = request_json.get('version')

        if not project_id or not version:
            return {'error': 'Missing project_id or version in the request body.'}, 400

        print(f'Extracted project_id: {project_id}, version: {version}')

        worker_thread = threading.Thread(
            target=_create_testcases,
            args=(project_id, version),
        )
        worker_thread.start()

        return (
            json.dumps(
                {
                    'status': 'success',
                    'message': 'Requirement analysis process started asynchronously.',
                }
            ),
            202,
        )  # 202 Accepted status indicates the request has been accepted for processing.
    except Exception as e:
        return {'error': f'An unexpected error occurred: {e}'}, 500
