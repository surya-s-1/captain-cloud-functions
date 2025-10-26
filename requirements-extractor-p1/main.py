import os
import json
import time
import logging
import functools
import concurrent.futures as futures
from typing import Any, Dict, List, Iterator

import functions_framework
from google import genai
from google.cloud import firestore, storage
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

# --- Environment variables ---
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE")

# --- Clients ---
storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
logging.basicConfig(level=logging.INFO)

# --- Config ---
MAX_WORKERS = 8
PROCESS_BATCH_SIZE = 50
GENAI_MODEL = "gemini-2.5-flash"
GENAI_TIMEOUT = 60
REQUIREMENT_TYPES = [
    "functional",
    "non-functional",
    "performance",
    "security",
    "usability",
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
    """Updates the status of a project version in Firestore and logs it."""
    doc_ref = firestore_client.document("projects", project_id, "versions", version)
    doc_ref.set({"status": status}, merge=True)
    logging.info(f"Status => {status}")


def _retry(max_attempts: int = 3, base_delay: int = 2):
    """Retry decorator with exponential backoff and jitter."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"Retry {attempt}/{max_attempts} after error: {e}")
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2
            return None

        return wrapper

    return deco


@_retry(max_attempts=3)
def _call_genai_for_snippet(snippet: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Call Gemini for a single text snippet to split it into requirements.
    The snippet contains 'file_name', 'text_used', and 'location'.
    """
    # Isolate the text for the prompt
    text_to_analyze = snippet.get('text_used', '')
    if not text_to_analyze:
        return []

    prompt = f"""
    You are a software requirements analyst for medical devices.
    Analyze the single, provided text snippet. Do not invent any new requirements or derive implicit requirements.
    Your sole task is to take the text, split it into its constituent requirements if multiple exist, and categorize each one. If not, return the original provided text.

    Categorize into: {', '.join(REQUIREMENT_TYPES)}.
    If none fit, default to 'non-functional'.

    Return ONLY a valid JSON array of requirements, each with the keys "requirement" and "requirement_type".

    Input Text Snippet:
    ```
    {text_to_analyze}
    ```
    """

    resp = genai_client.models.generate_content(
        model=GENAI_MODEL,
        contents=[Content(parts=[Part(text=prompt)], role='user')],
        config=GenerateContentConfig(
            response_mime_type='application/json',
            response_json_schema=REQUIREMENT_SCHEMA,
            timeout=GENAI_TIMEOUT,
        ),
    )

    extracted_requirements = json.loads(resp.text)

    source_info = {
        "file_name": snippet.get("file_name", "unknown"),
        "text_used": snippet.get("text_used", ""),
        "location": snippet.get("location", "unknown"),
    }

    final_requirements = []
    for req in extracted_requirements:
        final_requirements.append(
            {
                "requirement": req["requirement"],
                "requirement_type": req["requirement_type"],
                "sources": [source_info],
            }
        )

    return final_requirements


# --- Main Cloud Function ---
@functions_framework.http
def process_requirements_phase_1(request):
    """
    Main HTTP entrypoint for requirements extraction Phase 1.
    It orchestrates the loading, processing, and saving of requirements.
    """
    try:
        payload = request.get_json(silent=True) or {}
        project_id = payload.get("project_id")
        version = payload.get("version")
        extracted_text_url = payload.get("extracted_text_url")

        if not all([project_id, version, extracted_text_url]):
            return (json.dumps({"status": "error", "message": "Missing params"}), 400)

        _update_firestore_status(project_id, version, "START_REQ_EXTRACT_P1")

        # Load extracted text from GCS
        try:
            bucket_name, file_path = extracted_text_url[5:].split("/", 1)
            blob = storage_client.bucket(bucket_name).blob(file_path)
            extracted_data_list: List[Dict[str, str]] = json.loads(
                blob.download_as_text()
            )
        except Exception as e:
            logging.error(f"Failed to load data from GCS: {e}")
            raise RuntimeError("Failed to load data from GCS")

        logging.info(f"Loaded {len(extracted_data_list)} extracted text chunks")

        # Process each snippet individually in parallel
        all_requirements: List[Dict[str, Any]] = []

        # Use ThreadPoolExecutor to run _call_genai_for_snippet on every item
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            results_iterator = ex.map(_call_genai_for_snippet, extracted_data_list)

            for result in results_iterator:
                all_requirements.extend(result)

        logging.info(
            f"Extracted {len(all_requirements)} requirements after splitting/categorizing snippets."
        )

        # Save to GCS
        output_path = f"projects/{project_id}/v_{version}/extractions/requirements-phase-1.json"
        output_blob = storage_client.bucket(OUTPUT_BUCKET).blob(output_path)
        output_blob.upload_from_string(
            json.dumps(all_requirements, indent=2), content_type="application/json"
        )
        requirements_p1_url = f"gs://{OUTPUT_BUCKET}/{output_path}"

        _update_firestore_status(project_id, version, "COMPLETE_REQ_EXTRACT_P1")

        return requirements_p1_url, 200

    except Exception as e:
        logging.exception("Phase 1 failed")

        if project_id and version:
            _update_firestore_status(project_id, version, "ERR_REQ_EXTRACT_P1_SPLIT")

        return (json.dumps({"status": "error", "message": str(e)}), 500)
