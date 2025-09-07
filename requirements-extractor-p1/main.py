import os
import json
import functions_framework

from google import genai
from google.cloud import firestore, storage
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

# --- Environment variables ---

GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')

storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))

# --- Helper functions ---


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


# --- Main Cloud Function (HTTP Trigger) ---


@functions_framework.http
def process_requirements_phase_1(request):
    '''
    Cloud Function triggered by an HTTP POST request.
    It processes the request synchronously and returns the results upon completion.
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

        _update_firestore_status(project_id, version, 'START_REQ_EXTRACT_P1')

        print(
            f'Starting synchronous processing for project {project_id}, version {version}.'
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

        _update_firestore_status(project_id, version, 'COMPLETE_REQ_EXTRACT_P1')

        return (
            json.dumps(
                {
                    'status': 'success',
                    'project_id': project_id,
                    'version': version,
                    'requirements_p1_url': requirements_p1_url,
                }
            ),
            200,
        )

    except Exception as e:
        print(f'An error occurred: {e}')

        _update_firestore_status(project_id, version, 'ERR_REQ_EXTRACT_P1')

        return json.dumps({'status': 'error', 'message': str(e)}), 500
