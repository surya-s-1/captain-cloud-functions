import os
import json
import time
import logging
import functools
import concurrent.futures as futures
from typing import Any, Dict, List

import functions_framework
from google import genai
from google.cloud import firestore, storage
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

# --- Environment variables ---
GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET')
FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
MAX_INPUT_CHARS_FOR_CONTEXT = int(os.getenv('MAX_INPUT_CHARS_FOR_CONTEXT', '600000'))
CONTEXT_PROMPT = os.getenv('CONTEXT_PROMPT')
EXTRACTION_PROMPT = os.getenv('EXTRACTION_PROMPT')


# --- Clients ---
storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version='v1'))
logging.basicConfig(level=logging.INFO)


# --- Config ---
MAX_WORKERS = 8
GENAI_MODEL = 'gemini-2.5-flash'
GENAI_TIMEOUT = 90
REQUIREMENT_TYPES = [
    'functional',
    'non-functional',
    'performance',
    'security',
    'usability',
]


# --- Schemas ---
REQUIREMENT_SCHEMA = {
    'type': 'ARRAY',
    'items': {
        'type': 'OBJECT',
        'properties': {
            'requirement': {'type': 'STRING'},
            'requirement_type': {
                'type': 'STRING',
                'enum': REQUIREMENT_TYPES,
            },
        },
        'required': ['requirement', 'requirement_type'],
    },
}


# --- Helpers ---
def _update_firestore_status(project_id: str, version: str, status: str):
    '''Updates the status of a project version in Firestore and logs it.'''
    doc_ref = firestore_client.document('projects', project_id, 'versions', version)
    doc_ref.set({'status': status}, merge=True)
    logging.info(f'Status => {status}')


def _retry(max_attempts: int = 3, base_delay: int = 2):
    '''Retry decorator with exponential backoff and jitter.'''

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logging.warning(f'Retry {attempt}/{max_attempts} after error: {e}')
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2
            return None

        return wrapper

    return deco


@_retry(max_attempts=3)
def _call_genai_for_context(extracted_text_data: List[Dict[str, Any]]) -> str:
    '''
    Call Gemini to synthesize the System Context Header from all extracted text.
    '''
    all_text = []

    # Extract only the 'text' field from all files for context generation
    for file_data in extracted_text_data:
        for snippet_data in file_data.get('extracted_text', []):
            text = snippet_data.get('text', '').strip()
            # Only include text that is likely a full sentence/snippet, filtering out IDs/headers
            if text and len(text.split()) > 3:
                all_text.append(text)

    full_text_for_context = '\n'.join(all_text)

    # --- Truncation Logic Added Here ---
    truncated_text = full_text_for_context
    is_truncated = False

    if len(full_text_for_context) > MAX_INPUT_CHARS_FOR_CONTEXT:
        truncated_text = full_text_for_context[:MAX_INPUT_CHARS_FOR_CONTEXT]
        # Append a note to the model that the text has been cut off
        truncated_text += '\n\n[...TEXT TRUNCATED HERE. Please synthesize the context header only from the available text.]'
        is_truncated = True

    if is_truncated:
        logging.warning(
            f'Context input text was truncated from {len(full_text_for_context)} '
            f'to {len(truncated_text)} characters.'
        )

    # Use a concise set of instructions for the prompt
    context_prompt = f'''
    {CONTEXT_PROMPT}

    Full Text Snippets for Analysis:
    ---
    {truncated_text}
    ---
    '''

    logging.info('Generating System Context Header...')
    resp = genai_client.models.generate_content(
        model=GENAI_MODEL,
        contents=[Content(parts=[Part(text=context_prompt)], role='user')],
        # Use a simple text response, not JSON schema, for the context
    )

    # Clean up the response for use as a header
    system_context = resp.text.strip().replace('\n', ' ').replace('  ', ' ')
    logging.info(f'Generated Context: {system_context}')

    return system_context


@_retry(max_attempts=3)
def _call_genai_for_snippet(
    snippet: Dict[str, str], system_context: str
) -> List[Dict[str, Any]]:
    '''
    Call Gemini for a single text snippet to split it into requirements, using the system_context for better inference.
    The snippet contains 'file_name', 'text_used', and 'location'.
    '''
    # Isolate the text for the prompt
    text_to_analyze = snippet.get('text_used', '')

    if not text_to_analyze or len(text_to_analyze.split()) <= 4:
        logging.info(f'Skipping snippet, too short/non-content: \'{text_to_analyze}\'')
        return []

    prompt = f'''
    SYSTEM CONTEXT: {system_context}

    {EXTRACTION_PROMPT}

    Categorize each requirement into: {', '.join(REQUIREMENT_TYPES)}.
    If none fit, default to 'functional'.

    Return ONLY a valid JSON array of requirements, each with the keys 'requirement' and 'requirement_type'.

    Input Text Snippet:
    ```
    {text_to_analyze}
    ```
    '''

    with futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            lambda: genai_client.models.generate_content(
                model=GENAI_MODEL,
                contents=[Content(parts=[Part(text=prompt)], role='user')],
                config=GenerateContentConfig(
                    response_mime_type='application/json',
                    response_json_schema=REQUIREMENT_SCHEMA,
                ),
            )
        )

        resp = future.result(timeout=GENAI_TIMEOUT)

        extracted_requirements = json.loads(resp.text)

        source_info = {
            'file_name': snippet.get('file_name', 'unknown'),
            'text_used': snippet.get('text_used', ''),
            'location': snippet.get('location', 'unknown'),
        }

        final_requirements = []
        for req in extracted_requirements:
            final_requirements.append(
                {
                    'requirement': req['requirement'],
                    'requirement_type': req['requirement_type'],
                    'sources': [source_info],
                }
            )

        return final_requirements


# --- Main Cloud Function ---
@functions_framework.http
def process_requirements_phase_1(request):
    '''
    Main HTTP entrypoint for requirements extraction Phase 1.
    It orchestrates the loading, processing, and saving of requirements.
    '''
    project_id, version, system_context = None, None, ''

    try:
        payload = request.get_json(silent=True) or {}
        project_id = payload.get('project_id')
        version = payload.get('version')
        extracted_text_url = payload.get('extracted_text_url')

        if not all([project_id, version, extracted_text_url]):
            return (json.dumps({'status': 'error', 'message': 'Missing params'}), 400)

        _update_firestore_status(project_id, version, 'START_REQ_EXTRACT_P1')

        # Load extracted text from GCS
        try:
            bucket_name, file_path = extracted_text_url[5:].split('/', 1)
            blob = storage_client.bucket(bucket_name).blob(file_path)

            extracted_file_data: List[Dict[str, Any]] = json.loads(
                blob.download_as_text()
            )

        except Exception as e:
            logging.error(f'Failed to load data from GCS: {e}')
            raise RuntimeError('Failed to load data from GCS')

        logging.info(f'Loaded {len(extracted_file_data)} files data.')

        _update_firestore_status(project_id, version, 'START_CONTEXT_GENERATION')

        system_context = _call_genai_for_context(extracted_file_data)

        logging.info(f'System Context for all snippets: {system_context}')

        _update_firestore_status(project_id, version, 'START_REQ_EXTRACTION')

        all_snippets_to_process: List[Dict[str, str]] = []

        for file_data in extracted_file_data:
            file_name = file_data.get('file_name', 'unknown')
            for snippet_data in file_data.get('extracted_text', []):
                all_snippets_to_process.append(
                    {
                        'file_name': file_name,
                        'text_used': snippet_data.get('text', '').strip(),
                        'location': snippet_data.get('location', 'Unknown'),
                    }
                )

        logging.info(
            f'Generated {len(all_snippets_to_process)} total snippets for processing.'
        )

        # Process each snippet individually in parallel
        all_requirements: List[Dict[str, Any]] = []

        # Create a partial function with the fixed system_context argument
        partial_genai_call = functools.partial(
            _call_genai_for_snippet, system_context=system_context
        )

        # Use ThreadPoolExecutor to run the partial function on every snippet
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results_iterator = ex.map(
                partial_genai_call, all_snippets_to_process, timeout=GENAI_TIMEOUT
            )

            for result in results_iterator:
                all_requirements.extend(result)

        logging.info(
            f'Extracted {len(all_requirements)} requirements after splitting/categorizing snippets.'
        )

        _update_firestore_status(project_id, version, 'START_REQ_PERSIST')

        # Save to GCS
        output_path = (
            f'projects/{project_id}/v_{version}/extractions/requirements-phase-1.json'
        )
        output_blob = storage_client.bucket(OUTPUT_BUCKET).blob(output_path)
        output_blob.upload_from_string(
            json.dumps(all_requirements, indent=2), content_type='application/json'
        )
        requirements_p1_url = f'gs://{OUTPUT_BUCKET}/{output_path}'

        _update_firestore_status(project_id, version, 'COMPLETE_REQ_EXTRACT_P1')

        return requirements_p1_url, 200

    except Exception as e:
        logging.exception('Phase 1 failed')

        if project_id and version:
            _update_firestore_status(project_id, version, 'ERR_REQ_EXTRACT_P1')

        return (json.dumps({'status': 'error', 'message': str(e)}), 500)
