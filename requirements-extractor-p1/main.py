import os
import json
import requests
import threading
import functions_framework

from google import genai
from google.cloud import firestore, storage
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig
import google.auth.transport.requests as auth_requests
import google.oauth2.id_token as oauth2_id_token

GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
REQ_EXTRACT_P2_URL = os.getenv('REQ_EXTRACT_P2_URL')

storage_client = storage.Client()
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


# --- Asynchronous Worker Function ---
def _process_req_p1_async(project_id, version, extracted_text_url):
    '''Performs the core logic of text analysis and makes the final POST request asynchronously.'''
    try:
        _update_firestore_status(project_id, version, 'START_REQ_EXTRACT_P1')

        print(
            f'Starting asynchronous processing for project {project_id}, version {version}.'
        )

        bucket_name, file_path = extracted_text_url[5:].split('/', 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        extracted_content = blob.download_as_bytes()
        extracted_data_list = json.loads(extracted_content.decode('utf-8'))

        print(
            f'Successfully downloaded and parsed extracted text from {extracted_text_url}.'
        )

        requirement_types = [
            'functional',
            'non-functional',
            'performance',
            'security',
            'usability',
        ]

        prompt = f'''
        
        You are a highly skilled software requirements analyst for medical devices. Your task is to analyze raw text
        extracted from various documents. You must identify distinct, individual software requirements, deduplicate
        them, and categorize each one. The output must be a single JSON array of requirement objects.
        
        Categorize each requirement into one of the following types: {', '.join(requirement_types)}.
        If a requirement does not fit into these categories, use 'non-functional'.
        
        Here is the extracted text data, formatted as a JSON array of objects, each containing text and location metadata.
        
        {json.dumps(extracted_data_list, indent=2)}
        
        Your response MUST be a valid JSON array, and nothing else.
        
        '''

        response_schema = {
            'type': 'ARRAY',
            'items': {
                'type': 'OBJECT',
                'properties': {
                    'requirement': {'type': 'STRING'},
                    'requirement_type': {
                        'type': 'STRING',
                        'enum': requirement_types,
                    },
                    'sources': {
                        'type': 'ARRAY',
                        'items': {
                            'type': 'OBJECT',
                            'properties': {
                                'file_name': {'type': 'STRING'},
                                'text_used': {'type': 'STRING'},
                                'location': {'type': 'STRING'},
                            },
                            'required': ['file_name', 'text_used', 'location'],
                        },
                    },
                },
                'required': ['requirement', 'requirement_type', 'sources'],
            },
        }

        print(f'Generated prompt and response schema for project {project_id}.')

        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[Content(parts=[Part(text=prompt)], role='user')],
            config=GenerateContentConfig(
                response_mime_type='application/json',
                response_json_schema=response_schema,
            ),
        )

        print(f'Received response from GenAI for project {project_id}.')

        requirements_json = json.loads(response.text)

        output_path = f'requirements/{project_id}/v_{version}/requirements-phase-1.json'
        output_blob = storage_client.bucket(OUTPUT_BUCKET).blob(output_path)
        output_blob.upload_from_string(
            json.dumps(requirements_json, indent=2), content_type='application/json'
        )
        requirements_p1_url = f'gs://{OUTPUT_BUCKET}/{output_path}'

        print(f'Uploaded generated requirements to GCS: {requirements_p1_url}')

        print(f'Successfully wrote requirements to {requirements_p1_url}')

        _update_firestore_status(project_id, version, 'COMPLETE_REQ_EXTRACT_P1')

    except Exception as e:
        print(f'An error occurred during asynchronous processing: {e}')

        _update_firestore_status(project_id, version, 'ERR_REQ_EXTRACT_P1')

    final_message_data = {
        'project_id': project_id,
        'version': version,
        'requirements_p1_url': requirements_p1_url,
    }

    request = auth_requests.Request()
    id_token = oauth2_id_token.fetch_id_token(request, REQ_EXTRACT_P2_URL)

    response = requests.post(
        REQ_EXTRACT_P2_URL,
        headers={'Authorization': f'Bearer {id_token}'},
        json=final_message_data,
        timeout=30,
    )

    print(
        f'POST request sent to {REQ_EXTRACT_P2_URL} for project {project_id} version {version}.'
    )

    response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

    print(
        f'Successfully sent POST request to {REQ_EXTRACT_P2_URL}. Response status: {response.status_code}'
    )


# --- Main Cloud Function (HTTP Trigger) ---
@functions_framework.http
def process_requirements_phase_1(request):
    '''
    Cloud Function triggered by an HTTP POST request.
    It returns immediately and processes the request asynchronously.
    '''
    try:
        message_payload = request.get_json(silent=True)
        if not message_payload:
            return (
                json.dumps(
                    {
                        'status': 'error',
                        'message': 'No JSON payload found in request body.',
                    }
                ),
                400,
            )

        project_id = message_payload.get('project_id', None)
        version = message_payload.get('version', None)
        extracted_text_url = message_payload.get('extracted_text_url', None)

        if not project_id or not version or not extracted_text_url:
            return (
                json.dumps(
                    {
                        'status': 'error',
                        'message': 'Required details (project_id, version, or extracted_text_url) are missing.',
                    }
                ),
                400,
            )

        print(
            f'Validated project_id: {project_id}, version: {version}, extracted_text_url: {extracted_text_url}.'
        )

        # Start the heavy lifting in a new thread and return immediately
        worker_thread = threading.Thread(
            target=_process_req_p1_async, args=(project_id, version, extracted_text_url)
        )
        worker_thread.start()

        print(f'Asynchronous worker thread started for project {project_id}.')

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
        print(f'An error occurred in the main function: {e}')
        return json.dumps({'status': 'error', 'message': str(e)}), 500
