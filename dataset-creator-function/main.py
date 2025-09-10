import os
import json
import logging
import functions_framework
from google.cloud import firestore, storage
from google import genai
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

logging.basicConfig(level=logging.INFO)

# Environment
OUTPUT_BUCKET = os.environ.get('OUTPUT_BUCKET')
FIRESTORE_DATABASE = os.environ.get('FIRESTORE_DATABASE')
SCHEMA_MODEL = os.environ.get('SCHEMA_MODEL', 'gemini-2.5-pro')
DATA_MODEL = os.environ.get('DATA_MODEL', 'gemini-2.5-flash')

firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
storage_client = storage.Client()
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))

EXCLUDED_STATES = {
    'DATASET_GENERATION_STARTED',
    'DATASET_GENERATION_COMPLETED',
}


def generate_data_schema_and_rules(title, description, acceptance_criteria):
    test_case_text = f'Title: {title}\nDescription: {description}\nAcceptance Criteria: {acceptance_criteria}'

    prompt = f'''
    You are an expert software QA engineer specializing in test data generation. Your task is to analyze test cases and, based on the requirements, generate a precise and detailed JSON schema and set of rules for automated data creation. Be meticulous and thorough.

    Given the following software test case, analyze it and extract:
    1. A flat JSON object schema with snake_case field names and simple types: string, integer, number, boolean, or array.
    2. A set of generation rules or constraints for each field whether they are inferred or explicit, write it extremely clearly to guide the data generation to make sense. Also mention what data to give if there is a need for edge cases.
    3. An appropriate number of records to generate that provides full test coverage and realistic variation. Suggest at least 10 records, more if needed for edge cases.

    Test Case:
    {test_case_text}

    Example JSON Response:
    {{
        'schema': {{ 
            'first_name': 'string'
            'last_name': 'string'
            'age': 'integer'
        }},
        'rules': {{ 
            'first_name': 'should be a valid American first name'
            'last_name': 'should be a valid American last name'
            'age': 'must be more than 18' 
        }},
        'num_of_records': 10
    }}
    '''
    resp = genai_client.models.generate_content(
        model=SCHEMA_MODEL,
        contents=[Content(parts=[Part(text=prompt)], role='user')],
        config=GenerateContentConfig(
            response_mime_type='application/json',
            response_schema={
                'type': 'object',
                'properties': {
                    'schema': {
                        'type': 'object',
                        'additionalProperties': {'type': 'string'},
                    },
                    'rules': {
                        'type': 'object',
                        'additionalProperties': {'type': 'string'},
                    },
                    'num_of_records': {'type': 'integer'},
                },
                'required': ['schema', 'rules'],
            },
        ),
    )
    return json.loads(resp.text)


def generate_synthetic_data(schema_and_rules_and_num):
    num_of_records = schema_and_rules_and_num.get('num_of_records', 10)
    payload = dict(schema_and_rules_and_num)
    payload.pop('num_of_records', None)
    schema_json_string = json.dumps(payload, indent=2)

    prompt = f'''
    You are a master test data generator. Your sole purpose is to create synthetic data records that precisely match the provided schema and rules. The data must be realistic and valid according to the rules.

    Using the following schema and generation rules, create exactly {num_of_records} valid synthetic records. Each record must:
    - Match the field types in the schema
    - Follow the constraints in the rules
    - Be realistic and representative for test case validation

    Schema and Rules:
    {schema_json_string}

    The output must be a JSON array of objects.
    '''

    resp = genai_client.models.generate_content(
        model=DATA_MODEL,
        contents=[Content(parts=[Part(text=prompt)], role='user')],
        config=GenerateContentConfig(response_mime_type='application/json'),
    )
    return json.loads(resp.text)


def upload_json_to_gcs(bucket_name, path, data):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
    return f'gs://{bucket_name}/{path}'


@functions_framework.http
def worker_function(request):
    try:
        body = request.get_json(silent=True)
        if not body:
            return ('Bad request: JSON body required', 400)

        project_id = body.get('project_id')
        version = body.get('version')
        testcase_id = body.get('testcase_id')
        if not project_id or not version or not testcase_id:
            return ('Bad request: project_id, version, testcase_id required', 400)

        doc_ref = firestore_client.document(
            f'projects/{project_id}/versions/{version}/testcases/{testcase_id}'
        )
        doc = doc_ref.get()
        if not doc.exists:
            logging.warning('Testcase %s not found', testcase_id)
            return ('Not Found', 404)

        testcase = doc.to_dict()
        if testcase.get('deleted', False):
            logging.info('Skipping deleted testcase %s', testcase_id)
            return ('Skipped', 200)

        if testcase.get('dataset_status') in EXCLUDED_STATES:
            logging.info('Skipping comepleted testcase %s', testcase_id)
            return ('Skipped', 200)

        doc_ref.set(
            {
                'dataset_status': 'DATASET_GENERATION_STARTED',
            },
            merge=True,
        )

        title = testcase.get('title', '')
        description = testcase.get('description', '')
        acceptance_criteria = testcase.get('acceptance_criteria', '')

        schema_and_rules = generate_data_schema_and_rules(
            title, description, acceptance_criteria
        )
        schema_path = f'projects/{project_id}/v_{version}/testcases/{testcase_id}/datasets/schema.json'
        gs_schema = upload_json_to_gcs(OUTPUT_BUCKET, schema_path, schema_and_rules)

        synthetic = generate_synthetic_data(schema_and_rules)
        dataset_path = f'projects/{project_id}/v_{version}/testcases/{testcase_id}/datasets/dataset.json'
        gs_dataset = upload_json_to_gcs(OUTPUT_BUCKET, dataset_path, synthetic)

        doc_ref.set(
            {
                'schema': gs_schema,
                'datasets': firestore.ArrayUnion([gs_dataset]),
                'dataset_status': 'DATASET_GENERATION_COMPLETED',
            },
            merge=True,
        )

        logging.info('Completed testcase %s', testcase_id)
        return (json.dumps({'message': 'success', 'testcase_id': testcase_id}), 200)

    except Exception as e:
        logging.exception('Worker failed: %s', e)
        if 'doc_ref' in locals():
            try:
                doc_ref.set(
                    {
                        'status': 'FAILED',
                    },
                    merge=True,
                )
            except Exception:
                logging.exception('Failed to write failure status')
        return (f'Internal server error: {e}', 500)
