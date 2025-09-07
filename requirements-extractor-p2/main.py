import os
import json
import functions_framework
from urllib.parse import urlparse
from google import genai
from google.cloud import storage, firestore, discoveryengine_v1
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

# Environment variables
GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')

REGULATIONS = ['FDA', 'IEC 62304', 'ISO 9001', 'ISO 13485', 'ISO 27001']
BATCH_SIZE = 6

storage_client = storage.Client()
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))
discovery_client = discoveryengine_v1.SearchServiceClient()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)

serving_config = f'projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_serving_config'


def _update_firestore_status(project_id, version, status):

    doc_ref = firestore_client.document('projects', project_id, 'versions', version)
    update_data = {'status': status}
    doc_ref.set(update_data, merge=True)

    print(f'Updated status for project {project_id} version {version} to {status}.')

def query_discovery_engine(query_text):
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
            valid_prefixes = REGULATIONS
            score_list = result.model_scores.get('relevance_score')
            relevance = (
                float(score_list.values[0])
                if score_list and len(score_list.values) > 0
                else 0.0
            )

            content = result.chunk.content

            regulation = ''
            link = result.chunk.document_metadata.uri
            filename = link.split('/')[-1]

            for prefix in valid_prefixes:
                if filename.startswith(prefix):
                    regulation = prefix
                    break

            processed.append(
                {
                    'relevance': relevance,
                    'content': content,
                    'regulation': regulation,
                    'filename': filename,
                    'page_start': result.chunk.page_span.page_start,
                    'page_end': result.chunk.page_span.page_end,
                    'snippet': content,
                }
            )

    processed = sorted(processed, key=lambda x: x['relevance'], reverse=True)[:2]

    return [
        {
            'content': p['content'],
            'regulation': p['regulation'],
            'filename': p['filename'],
            'page_start': p['page_start'],
            'page_end': p['page_end'],
            'snippet': p['snippet'],
        }
        for p in processed
    ]


##################################################################################
##################################################################################
##################################################################################
##################################################################################

# --- Main Cloud Function (HTTP Trigger) ---
@functions_framework.http
def process_requirements_phase_2(request):
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
        requirements_p1_url = message_payload.get('requirements_p1_url', None)

        print(
            f'Extracted project_id: {project_id}, version: {version}, requirements_p1_url: {requirements_p1_url}'
        )

        if not all([project_id, version, requirements_p1_url]):
            return (
                json.dumps(
                    {
                        'status': 'error',
                        'message': 'Required details (project_id, version, or requirements_p1_url) are missing.',
                    }
                ),
                400,
            )

        _update_firestore_status(project_id, version, 'START_REQ_EXTRACT_P2')

        parsed_url = urlparse(requirements_p1_url)
        bucket_name = parsed_url.netloc
        blob_path = parsed_url.path.lstrip('/')

        print(
            f'Downloading requirements from GCS bucket: {bucket_name}, path: {blob_path}'
        )

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        explicit_requirements_raw = json.loads(blob.download_as_text())

        if not explicit_requirements_raw:
            return (
                json.dumps(
                    {
                        'status': 'error',
                        'message': 'Input data missing from payload.',
                    }
                ),
                400,
            )

        print(f'Successfully received explicit requirements.')

        #################################################################################
        #################################################################################
        #################################################################################
        #################################################################################

        # --- Step 1: Deduplicate explicit requirements with Gemini ---
        print('Deduplicating explicit requirements with Gemini...')

        explicit_requirements_str = json.dumps(explicit_requirements_raw, indent=2)

        gemini_explicit_dedupe_prompt = f'''
            You are an expert in medical device regulations. Your task is to review a list of functional, security, and regulatory requirements. 
            Some of these requirements may be duplicates or redundant.

            You must perform the following actions:
            1.  Combine requirements that are semantically identical or express the same core idea.
            2.  If a requirement is big, split it into multiple smaller requirements.
            3.  For any merged requirements, combine the `sources` values into a single, comprehensive list without duplicates.
            4.  Maintain the original `requirement_type` if the requirement is kept.
            5.  Each object in `sources` should be of format {{ filename: <filename>, location:  <location>, snippet: <text_used> }}.
            6.  Summarize the requirement text in each requirement if they are in 1st person or more than 100 words. And remove any filler words or things you think are unnecessay to understand the requirement.
            7.  Make each requirement into markdown format, if possible, and remove any HTML tags if present.
            
            Return the final, deduplicated list in a single JSON array, following this exact schema. Do not provide a requirement_id.
            [
                {{
                    'requirement': string,
                    'requirement_type': string,
                    'sources': list of objects of type {{
                        'filename': string,
                        'location': string,
                        'snippet': string
                    }}
                }}
            ]
            
            Here is the list of requirements to process:
            
            ```json
            {explicit_requirements_str}
            ```
        '''

        explicit_schema = {
            'type': 'ARRAY',
            'items': {
                'type': 'OBJECT',
                'properties': {
                    'requirement': {'type': 'STRING'},
                    'requirement_type': {
                        'type': 'STRING',
                        'enum': [
                            'functional',
                            'non-functional',
                            'performance',
                            'security',
                            'usability',
                            'regulation',
                        ],
                    },
                    'sources': {
                        'type': 'ARRAY',
                        'items': {'type': 'OBJECT'},
                        'properties': {
                            'filename': {'type': 'STRING'},
                            'location': {'type': 'STRING'},
                            'snippet': {'type': 'STRING'},
                        },
                    },
                },
                'required': ['requirement', 'requirement_type', 'sources'],
            },
        }

        explicit_deduped_response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                Content(parts=[Part(text=gemini_explicit_dedupe_prompt)], role='user')
            ],
            config=GenerateContentConfig(
                response_mime_type='application/json',
                response_json_schema=explicit_schema,
            ),
        )

        deduplicated_explicit_requirements = json.loads(explicit_deduped_response.text)

        print(
            f'Found {len(deduplicated_explicit_requirements)} unique requirements.'
        )

        requirements_collection_ref = firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )

        req_id_counter = 1

        batch = firestore_client.batch()

        for req in deduplicated_explicit_requirements:
            req_id = f'REQ-{req_id_counter:03d}'
            doc_ref = requirements_collection_ref.document(req_id)
            req_data = {**req, 'requirement_id': req_id, 'deleted': False}
            batch.set(doc_ref, req_data)
            req_id_counter += 1

        batch.commit()

        _update_firestore_status(project_id, version, 'COMPLETE_EXP_REQ')

        #################################################################################
        #################################################################################
        #################################################################################
        #################################################################################

        # --- Step 2: Get implicit requirements based on the new list ---

        print('Searching for implicit requirements using Vertex AI Discovery Engine...')

        all_implicit_requirements = []

        for req in deduplicated_explicit_requirements:
            query = f'Regulations related to: {req.get('requirement')}'

            discovery_results = query_discovery_engine(query)

            all_implicit_requirements.extend(discovery_results)

        print(f'Found {len(all_implicit_requirements)} implicit requirements.')

        _update_firestore_status(project_id, version, 'COMPLETE_IMP_REQ')

        ##################################################################################
        #################################################################################
        #################################################################################
        #################################################################################

        # --- Step 3: Deduplicate implicit requirements in batches using Gemini ---
        print('Processing implicit requirements in batches with Gemini...')

        implicit_schema = {
            'type': 'ARRAY',
            'items': {
                'type': 'OBJECT',
                'properties': {
                    'requirement': {'type': 'STRING'},
                    'requirement_type': {'type': 'STRING', 'enum': ['regulation']},
                    'regulations': {
                        'type': 'ARRAY',
                        'items': {'type': 'OBJECT'},
                        'properties': {
                            'regulation': {'type': 'STRING', 'enum': REGULATIONS},
                            'source': {
                                'type': 'OBJECT',
                                'properties': {
                                    'filename': {'type': 'STRING'},
                                    'page_start': {'type': 'STRING'},
                                    'page_end': {'type': 'STRING'},
                                    'snippet': {'type': 'STRING'},
                                },
                            },
                        },
                    },
                },
                'required': ['requirement', 'requirement_type', 'regulations'],
            },
        }

        _update_firestore_status(project_id, version, 'START_PROCESS_IMP_REQ')

        current_batch = 1
        
        num_of_batches = int(len(all_implicit_requirements) / BATCH_SIZE) + 1

        for i in range(0, len(all_implicit_requirements), BATCH_SIZE):

            batch = all_implicit_requirements[i : i + BATCH_SIZE]

            batch_str = json.dumps(batch, indent=2)

            gemini_implicit_dedupe_prompt = f'''
                You are an expert in medical device regulations. Your task is to review a list of regulatory requirements retrieved from a discovery engine.
                Some of these requirements may be duplicates or redundant.

                You must perform the following actions:
                1.  Combine requirements that are semantically identical or express the same core idea.
                2.  If a requirement is big, split it into multiple (maximum 3) smaller requirements.
                3.  Change any requirement text from 1st person to a regular, objective voice.
                4.  For any merged requirements, combine the `regulations` values into a single, comprehensive list, without duplicates.
                5.  Summarize the requirement text to be concise and easy to understand, removing any unnecessary filler words or content.
                6.  Maintain the `requirement_type` as 'regulation'.
                7.  Make each requirement into markdown format, if possible, and remove any HTML tags if present.
                
                Return the final, deduplicated list in a single JSON array, following this exact schema. Do not provide a requirement_id.
                [
                    {{
                        'requirement': string,
                        'requirement_type': string,
                        'regulations': list of objects of type {{
                            'regulation': string,
                            'source': {{
                                'filename': string,
                                'page_start': string,
                                'page_end': string,
                                'snippet': string
                            }}
                        }}
                    }}
                ]
                
                Here is the list of regulatory requirements to process:
                
                ```json
                {batch_str}
                ```
            '''

            implicit_deduped_response = genai_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    Content(
                        parts=[Part(text=gemini_implicit_dedupe_prompt)], role='user'
                    )
                ],
                config=GenerateContentConfig(
                    response_mime_type='application/json',
                    response_json_schema=implicit_schema,
                ),
            )

            implicit_requirements = json.loads(implicit_deduped_response.text)

            print(f'implicit requirements for batch: {len(implicit_requirements)}')

            batch = firestore_client.batch()

            for req in implicit_requirements:
                req_id = f'REQ-{req_id_counter:03d}'
                doc_ref = requirements_collection_ref.document(req_id)
                req_data = {**req, 'requirement_id': req_id, 'deleted': False}
                batch.set(doc_ref, req_data)
                req_id_counter += 1

            batch.commit()

            _update_firestore_status(
                project_id,
                version,
                f'PROCESS_IMP_REQ_{current_batch}/{num_of_batches}',
            )

            current_batch += 1

        _update_firestore_status(project_id, version, 'CONFIRM_REQ_EXTRACT')

        return 'OK', 200

    except Exception as e:
        print(f'An error occurred: {e}')
        # If an error occurs, delete all documents in the 'requirements' collection
        # for the given project_id and version.

        docs = requirements_collection_ref.get()

        for doc in docs:
            doc.reference.delete()

        print(
            'Deleted all documents in the requirements collection due to an error.'
        )

        _update_firestore_status(project_id, version, 'ERR_REQ_EXTRACT_P2')

        return str(e), 500
