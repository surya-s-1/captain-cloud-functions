import os
import ast
import json
import base64
import functions_framework

from google import genai
from google.cloud import firestore, pubsub_v1, storage
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC')

pubsub_client = pubsub_v1.PublisherClient()
storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))

def update_firestore_status(project_id, status):
    '''Updates the status of a project in Firestore.'''
    doc_ref = firestore_client.collection('projects').document(project_id)
    update_data = {'status': status}
    doc_ref.set(update_data, merge=True)
    print(f'Updated status for project {project_id} to {status}.')


@functions_framework.cloud_event
def process_req_p1(event):
    '''
    Cloud Function triggered by a Pub/Sub message on 'requirement-extraction-p1'.
    It updates the project status and publishes a message to 'requirement-extraction-p2'
    after completing the entire extraction process.
    '''
    if (
        not event
        or not event.data
        or not event.data.get('message', None)
        or not event.data.get('message', {}).get('data', None)
    ):
        raise ValueError('Pub/Sub message \'data\' field is missing.')

    try:
        message_payload_str = base64.b64decode(event.data.get('message').get('data')).decode('utf-8')
        message_payload = ast.literal_eval(message_payload_str)

        project_id = message_payload.get('project_id', None)
        version = message_payload.get('version', None)
        extracted_text_url = message_payload.get('extracted_text_url', None)
    except Exception as e:
        print(f'Error decoding message data: {e}. Exiting.')
        return

    if not project_id or not version or not extracted_text_url:
        print('REQUIRED DETAILS NOT PROVIDED:', project_id, version, extracted_text_url)
        return

    update_firestore_status(project_id, 'START_REQ_EXTRACT_P1')

    try:
        bucket_name, file_path = extracted_text_url[5:].split('/', 1)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        extracted_content = blob.download_as_bytes()
        extracted_data_list = json.loads(extracted_content.decode('utf-8'))
    except Exception as e:
        print(f'Error fetching or parsing extracted text file: {e}. Exiting.')
        update_firestore_status(project_id, 'ERROR_REQ_EXTRACT')
        return

    requirement_types = [
        'functional',
        'non-functional',
        'performance',
        'security',
        'usability'
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

    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[ Content(parts=[Part(text=prompt)], role='user') ],
            config=GenerateContentConfig(
                response_mime_type='application/json',
                response_json_schema=response_schema,
            ),
        )
        requirements_json = json.loads(response.text)
    except Exception as e:
        print(f'Error calling Gemini API or parsing response: {e}. Exiting.')
        update_firestore_status(project_id, 'ERROR_REQ_EXTRACT')
        return

    output_path = f'requirements/{project_id}/v{version}/requirements-p1.json'
    output_blob = storage_client.bucket(OUTPUT_BUCKET).blob(output_path)
    output_blob.upload_from_string(
        json.dumps(requirements_json, indent=2),
        content_type='application/json'
    )
    requirements_p1_url = f'gs://{OUTPUT_BUCKET}/{output_path}'
    print(f'Successfully wrote requirements to {requirements_p1_url}')

    update_firestore_status(project_id,'COMPLETE_REQ_EXTRACT_P2')

    final_topic_name = f'projects/{GOOGLE_CLOUD_PROJECT}/topics/{OUTPUT_TOPIC}'
    final_message_data = {
        'project_id': project_id,
        'version': version,
        'requirements_p1_url': requirements_p1_url,
    }

    future = pubsub_client.publish(
        final_topic_name, json.dumps(final_message_data).encode('utf-8')
    )

    print(f'Published final message to {final_topic_name}: {future.result()}')
