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
CHUNK_SIZE = 20  # number of text snippets per Gemini request
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


def _chunk(lst: list, n: int) -> Iterator[list]:
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@_retry(max_attempts=3)
def _call_genai(batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Call Gemini for one chunk of extracted text and return parsed JSON."""
    prompt = f"""
    You are a software requirements analyst for medical devices.
    Analyze the following extracted text snippets, deduplicate, and extract requirements.

    Categorize into: {', '.join(REQUIREMENT_TYPES)}.
    If none fit, default to 'non-functional'.

    Return ONLY valid JSON array with each object of form:
    {{
        "requirement": str, 
        "requirement_type": str, 
        "sources":[
            {{
                "file_name":str,
                "text_used":str,
                "location":str
            }}
        ]
    }}

    Input JSON:
    ```json
    {json.dumps(batch, indent=2)}
    ```
    """

    resp = genai_client.models.generate_content(
        model=GENAI_MODEL,
        contents=[Content(parts=[Part(text=prompt)], role='user')],
        config=GenerateContentConfig(
            response_mime_type='application/json',
            response_json_schema=REQUIREMENT_SCHEMA,
        ),
    )

    return json.loads(resp.text)


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
            extracted_data_list = json.loads(blob.download_as_text())
        except Exception as e:
            logging.error(f"Failed to load data from GCS: {e}")
            raise RuntimeError("Failed to load data from GCS")

        logging.info(f"Loaded {len(extracted_data_list)} extracted text chunks")

        # Split into manageable chunks
        batches = list(_chunk(extracted_data_list, CHUNK_SIZE))
        logging.info(f"Processing {len(batches)} batches in parallel...")

        all_requirements = []
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for result in ex.map(_call_genai, batches):
                all_requirements.extend(result)

        logging.info(f"Extracted {len(all_requirements)} requirements")

        # Save to GCS
        output_path = f"requirements/{project_id}/v_{version}/requirements-phase-1.json"
        output_blob = storage_client.bucket(OUTPUT_BUCKET).blob(output_path)
        output_blob.upload_from_string(
            json.dumps(all_requirements, indent=2), content_type="application/json"
        )
        requirements_p1_url = f"gs://{OUTPUT_BUCKET}/{output_path}"

        _update_firestore_status(project_id, version, "COMPLETE_REQ_EXTRACT_P1")

        return requirements_p1_url, 200

    except Exception as e:
        logging.exception("Phase 1 failed")

        _update_firestore_status(project_id, version, "ERR_REQ_EXTRACT_P1")

        return (json.dumps({"status": "error", "message": str(e)}), 500)