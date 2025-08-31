import os
import ast
import json
import base64
import functions_framework
from urllib.parse import urlparse
from google import genai
from google.cloud import storage, pubsub_v1
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig
from google.cloud import discoveryengine_v1

GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')
OUTPUT_TOPIC = os.environ.get('OUTPUT_TOPIC')

REGULATIONS = ['FDA', 'IEC 62304', 'ISO 9001', 'ISO 13485', 'ISO 27001']

storage_client = storage.Client()
pubsub_publisher = pubsub_v1.PublisherClient()
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))
discovery_client = discoveryengine_v1.SearchServiceClient()

serving_config = f'projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_serving_config'

@functions_framework.cloud_event
def process_requirements(event, context = None):
    try:
        print('Starting function execution.')

        if (
            not event
            or not event.data
            or not event.data.get('message', None)
            or not event.data.get('message', {}).get('data', None)
        ):
            raise ValueError('Pub/Sub message \'data\' field is missing.')

        message_payload_str = base64.b64decode(event.data.get('message').get('data')).decode('utf-8')
        message_payload = ast.literal_eval(message_payload_str)

        project_id = message_payload.get('project_id')
        version = message_payload.get('version')
        requirements_p1_url = message_payload.get('requirements_p1_url')

        if not all([project_id, version, requirements_p1_url]):
            raise ValueError('Missing required fields in Pub/Sub message.')

        print(f'Processing requirements for project: {project_id}, version: {version}')

        parsed_url = urlparse(requirements_p1_url)
        bucket_name = parsed_url.netloc
        blob_path = parsed_url.path.lstrip('/')

        print(f'Downloading requirements from GCS bucket: {bucket_name}, path: {blob_path}')

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        input_data = json.loads(blob.download_as_text())

        explicit_requirements = []
        implicit_requirements = []

        print('Searching for implicit requirements using Vertex AI Discovery Engine...')

        for req in input_data:
            explicit_requirements.append(
                {
                    'requirement': req['requirement'],
                    'requirement_type': req['requirement_type'],
                    'sources': req.get('sources', []),
                    'regulations': [],
                }
            )

            query = f'Regulations related to: "{req['requirement']}"'

            discovery_results = query_discovery_engine(query)

            for doc in discovery_results:
                content = doc.get('content')
                regulation = doc.get('regulation')

                implicit_requirements.append(
                    {
                        'requirement': content,
                        'requirement_type': 'regulation',
                        'sources': [],
                        'regulations': [regulation],
                    }
                )

        print('Deduplicating and merging requirements with Gemini...')

        all_requirements = explicit_requirements + implicit_requirements

        requirements_str = json.dumps(all_requirements, indent=2)

        gemini_prompt = f'''
            You are an expert in medical device regulations. Your task is to review a list of functional, security, and regulatory requirements. 
            Some of these requirements may be duplicates or redundant.
            
            You must perform the following actions:
            1.  Combine requirements that are semantically identical or express the same core idea, regardless of their source or type.
            2.  For any merged requirements, combine the `sources` values among each other and `regulations` values among each other to form arrays into a single, comprehensive list without duplicates.
            3.  The final `requirement_id` should be a unique sequential number starting from 1.
            4.  Maintain the original `requirement_type` if the requirement is kept. If it's a new, combined requirement, use the type that best describes the merged content (e.g., if a `functional` and `regulation` requirement are merged, the new type should be `regulation`). If both are the same, keep that type.
            5. Absolutely do not ignore the requiremnts of any type if they are not being combined with another requirement.
            
            Return the final, deduplicated list in a single JSON array, following this exact schema:
            [
                {{
                    'requirement_id': integer,
                    'requirement': string,
                    'requirement_type': string,
                    'sources': list of strings,
                    'regulations': list of strings
                }}
            ]
            
            Here is the list of requirements to process:
            
            ```json
            {requirements_str}
            ```
        '''

        requirement_types = [
            'functional',
            'non-functional',
            'performance',
            'security',
            'usability',
            'regulation'
        ]

        response_schema = {
            'type': 'ARRAY',
            'items': {
                'type': 'OBJECT',
                'properties': {
                    'requirement_id': {'type': 'INTEGER'},
                    'requirement': {'type': 'STRING'},
                    'requirement_type': {
                        'type': 'STRING',
                        'enum': requirement_types,
                    },
                    'sources': {
                        'type': 'ARRAY',
                        'items': {'type': 'STRING'},
                    },
                    'regulations': {
                        'type': 'ARRAY',
                        'items': {'type': 'STRING', 'enum': REGULATIONS},
                    },
                },
                'required': [
                    'requirement_id',
                    'requirement',
                    'requirement_type',
                    'sources',
                    'regulations',
                ],
            },
        }

        response = genai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[Content(parts=[Part(text=gemini_prompt)], role='user')],
            config=GenerateContentConfig(
                response_mime_type='application/json',
                response_json_schema=response_schema,
            ),
        )

        final_requirements = json.loads(response.text)

        output_blob_path = f'requirements/{project_id}/v{version}/requirements-p2.json'
        output_url = f'gs://{bucket_name}/{output_blob_path}'

        output_blob = bucket.blob(output_blob_path)
        output_blob.upload_from_string(
            json.dumps(final_requirements, indent=4), content_type='application/json'
        )

        print(f'Saved final requirements to GCS at: {output_url}')

        pubsub_message = {
            'project_id': project_id,
            'version': version,
            'requirements_p2_url': output_url,
        }

        pubsub_topic_path = pubsub_publisher.topic_path(GOOGLE_CLOUD_PROJECT, OUTPUT_TOPIC)

        future = pubsub_publisher.publish(
            pubsub_topic_path, json.dumps(pubsub_message).encode('utf-8')
        )

        print(f'Published message with URL: {future.result()}')

    except Exception as e:
        print(f'An error occurred: {e}')


def query_discovery_engine(query_text):
    def process_results(response):
        processed = []

        valid_prefixes = REGULATIONS

        for result in response.results:

            score_list = result.model_scores.get('relevance_score')

            relevance = float(score_list.values[0]) if score_list and len(score_list.values) > 0 else 0.0

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
                }
            )

        processed = sorted(processed, key=lambda x: x['relevance'], reverse=True)[:2]

        return [
            {
                'content': p['content'],
                'regulation': p['regulation'],
            }
            for p in processed
        ]

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

    processed_response = process_results(response)

    return processed_response
