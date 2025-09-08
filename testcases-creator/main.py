import os
import json
import logging
import functools
import time
import concurrent.futures as futures
from typing import Any, Dict, List, Tuple

import functions_framework
from google import genai
from google.cloud import firestore
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig

# =====================
# Environment variables
# =====================
FIRESTORE_DATABASE = os.environ.get("FIRESTORE_DATABASE")
MAX_WORKERS = 16  # concurrency for Gemini calls
FIRESTORE_COMMIT_CHUNK = 450  # Firestore limit is 500 per batch
GENAI_TIMEOUT_SECONDS = 90
GENAI_MODEL = "gemini-2.5-flash"

# =====================
# Clients
# =====================
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client(http_options=HttpOptions(api_version="v1"))
logging.basicConfig(level=logging.INFO)


# =====================
# Utilities
# =====================
def _update_firestore_status(project_id: str, version: str, status: str):
    """Updates the status of a project version in Firestore."""
    doc_ref = firestore_client.document("projects", project_id, "versions", version)
    doc_ref.set({"status": status}, merge=True)
    logging.info(f"Status => {status}")


def _retry(max_attempts=3, base_delay=0.5, exc_types=(Exception,)):
    """A decorator with exponential backoff and jitter for retrying transient errors."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exc_types as e:
                    if attempt == max_attempts:
                        raise
                    logging.warning(f"Retry {attempt}/{max_attempts} after error: {e}")
                    time.sleep(delay)
                    delay *= 2
            return None

        return wrapper

    return deco


def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> int:
    """Commits documents in chunks to Firestore."""
    batch = firestore_client.batch()
    count = 0
    total = 0
    for doc_ref, data in doc_tuples:
        batch.set(doc_ref, data)
        count += 1
        total += 1
        if count >= FIRESTORE_COMMIT_CHUNK:
            batch.commit()
            batch = firestore_client.batch()
            count = 0
    if count:
        batch.commit()
    return total


# =====================
# Testcase generation
# =====================
@_retry(max_attempts=3)
def _generate_test_cases(requirement_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate test cases for one requirement using Gemini."""
    requirement_text = requirement_data.get("requirement", "")
    prompt = f"""
    You are a medical industry QA engineer. Based on the following requirement, generate JSON test cases.
    Each test case must include: title, description (markdown bullets), acceptance_criteria (markdown bullets), priority (High/Medium/Low).
    
    Requirement: {requirement_text}
    """

    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "description": {"type": "STRING"},
                "acceptance_criteria": {"type": "STRING"},
                "priority": {"type": "STRING", "enum": ["High", "Medium", "Low"]},
            },
            "required": ["title", "description", "acceptance_criteria", "priority"],
        },
    }

    resp = genai_client.models.generate_content(
        model=GENAI_MODEL,
        contents=[Content(parts=[Part(text=prompt)], role="user")],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_schema,
        ),
    )

    return json.loads(resp.text)


def _create_testcases(project_id: str, version: str) -> None:
    """Main function to orchestrate testcase creation."""
    try:
        _update_firestore_status(project_id, version, "START_TESTCASE_CREATION")

        requirements_ref = firestore_client.collection(
            "projects", project_id, "versions", version, "requirements"
        )
        requirements_docs = list(requirements_ref.stream())

        logging.info(f"Loaded {len(requirements_docs)} requirements")

        # Parallel Gemini calls
        all_testcases = []
        with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_to_req = {
                ex.submit(_generate_test_cases, req_doc.to_dict()): req_doc
                for req_doc in requirements_docs
            }
            for future in futures.as_completed(future_to_req):
                req_doc = future_to_req[future]
                req_id = req_doc.id
                try:
                    testcases = future.result()
                    for i, tc in enumerate(testcases, start=1):
                        tc_id = f"{req_id}-TC-{i}"
                        all_testcases.append(
                            (
                                firestore_client.collection("projects")
                                .document(project_id)
                                .collection("versions")
                                .document(version)
                                .collection("testcases")
                                .document(tc_id),
                                {
                                    **tc,
                                    "testcase_id": tc_id,
                                    "requirement_id": req_id,
                                    "deleted": False,
                                },
                            )
                        )
                    logging.info(f"✓ {req_id} => {len(testcases)} testcases")
                except Exception as e:
                    logging.warning(f"Error for {req_id}: {e}")

        # Bulk Firestore write
        total_written = _firestore_commit_many(all_testcases)
        logging.info(f"Stored {total_written} testcases in Firestore")

        _update_firestore_status(project_id, version, "COMPLETE_TESTCASE_CREATION")

    except Exception as e:
        logging.exception("Error during testcase creation")

        _update_firestore_status(project_id, version, "ERR_TESTCASE_CREATION")


# =====================
# HTTP entrypoint
# =====================
@functions_framework.http
def process_for_testcases(request):
    """Main HTTP entrypoint for testcase generation."""
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return {"error": "JSON body not provided."}, 400

        project_id = request_json.get("project_id")
        version = request_json.get("version")

        if not project_id or not version:
            return {"error": "Missing project_id or version"}, 400

        logging.info(
            f"Start testcase generation for project={project_id}, version={version}"
        )

        # Run async in background thread so HTTP returns immediately
        import threading

        worker = threading.Thread(target=_create_testcases, args=(project_id, version))
        worker.start()

        return (
            json.dumps(
                {
                    "status": "success",
                    "message": "Testcase generation started asynchronously",
                }
            ),
            202,
        )

    except Exception as e:
        return {"error": str(e)}, 500