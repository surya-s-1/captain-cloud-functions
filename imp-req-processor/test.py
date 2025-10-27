from dotenv import load_dotenv

load_dotenv()

import os
from google.cloud.discoveryengine_v1 import SearchServiceClient, SearchRequest

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
DATA_STORE_ID = os.getenv('DATA_STORE_ID')

discovery_client = SearchServiceClient()
SERVING_CONFIG = (
    f'projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/'
    f'dataStores/{DATA_STORE_ID}/servingConfigs/default_serving_config'
)

QUERY_TEXT = 'Find the regulations, standards and procedures that apply to the following requirement: Authorized users should be able to create, view and edit user accounts'


def perform_extractive_search_and_paginate(serving_config: str, query_text: str):
    '''
    Performs a search query, paginates results, and prints extractive answers
    with source and page number.
    '''
    page_token = ''

    # Loop continues as long as a non-empty page_token is available
    while True:
        # 1. Configure the Search Request
        request = SearchRequest(
            serving_config=serving_config,
            query=query_text,
            # page_size=10,
            content_search_spec=SearchRequest.ContentSearchSpec(
                extractive_content_spec=SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_segment_count=3,
                    return_extractive_segment_score=True
                )
            ),
            # content_search_spec=SearchRequest.ContentSearchSpec(
            #     snippet_spec=SearchRequest.ContentSearchSpec.SnippetSpec(
            #         return_snippet=True
            #     ),
            #     search_result_mode=SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,
            # ),
            page_token=page_token,
        )

        print(f'Fetching page with token: \'{page_token}\'...')
        response = discovery_client.search(request=request)

        print('\n--- Results for Current Page ---')
        print(response)

        page_token = response.next_page_token

        if not page_token:
            print('\n*** Reached the end of search results. ***')
            break


# --- Main Execution ---
if __name__ == '__main__':
    if not all([PROJECT_ID, LOCATION, DATA_STORE_ID]):
        print(
            'ERROR: Please set PROJECT_ID, LOCATION, and DATA_STORE_ID in your .env file.'
        )
    else:
        perform_extractive_search_and_paginate(SERVING_CONFIG, QUERY_TEXT)
