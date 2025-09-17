import os
import json
import math
import time
import logging
import functools
import concurrent.futures as futures
from urllib.parse import urlparse
from typing import Any, Dict, Iterator, List, Optional, Tuple

import functions_framework
from rapidfuzz import fuzz

from google import genai
from google.genai.types import HttpOptions, Part, Content, GenerateContentConfig
from google.cloud import storage, firestore, discoveryengine_v1

# =====================
# Environment variables
# =====================
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
DATA_STORE_ID = os.getenv("DATA_STORE_ID")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE")

# Tunables (safe defaults for speed and cost-efficiency)
REGULATIONS = ["FDA", "IEC 62304", "ISO 9001", "ISO 13485", "ISO 27001", "SaMD"]
EXPL_BATCH_SIZE = 10  # Gemini explicit-dedupe batch size
IMPL_BATCH_SIZE = 10  # Gemini implicit-dedupe batch size
MAX_WORKERS = 16  # Thread pool concurrency for parallel API calls
FUZZY_SIM_THRESHOLD = 90  # Pre-dedupe local similarity threshold (0-100)
FIRESTORE_COMMIT_CHUNK = 450  # <= 500 per batch write limit
GENAI_MODEL = "gemini-2.5-flash"  # The model used for all LLM-based processing
GENAI_API_VERSION = "v1"
GENAI_TIMEOUT_SECONDS = 90  # Each LLM call safety timeout

# =====================
# Clients
# =====================
storage_client = storage.Client()
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
discovery_client = discoveryengine_v1.SearchServiceClient()

serving_config = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/"
    f"dataStores/{DATA_STORE_ID}/servingConfigs/default_serving_config"
)

# Configure GenAI client once
genai_client = genai.Client(http_options=HttpOptions(api_version=GENAI_API_VERSION))
logging.getLogger("google.cloud").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

# =====================
# Small utilities
# =====================


def _retry(max_attempts: int = 3, base_delay: float = 0.5):
    """
    A decorator with exponential backoff and jitter for retrying transient errors.
    This helps to handle API rate limits and network issues gracefully.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logging.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {fn.__name__} with error: {e}"
                    )
                    if attempt == max_attempts:
                        raise
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
            return None  # Should be unreachable

        return wrapper

    return deco


def _update_firestore_status(project_id: str, version: str, status: str) -> None:
    """Updates the status of a project version in Firestore."""
    doc_ref = firestore_client.document("projects", project_id, "versions", version)
    doc_ref.set({"status": status}, merge=True)
    print(f"Status => {status}")


def _normalize_text(s: str) -> str:
    """Normalizes a string for fuzzy matching by removing noise."""
    if not s:
        return ""
    s = s.strip().lower()
    # Remove markdown formatting noise that hurts fuzzy scores
    for ch in ["*", "#", "`", "+", "-", ">", "|", "_", "\n"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())  # Collapse multiple spaces
    return s


def normalize_req_dict(req: Any) -> Any:
    """
    Recursively normalizes keys in a dictionary (e.g., fixes inconsistent keys like "'requirement'" -> "requirement").
    If the input is not a dict or list of dicts, it's returned as-is.
    """
    if not isinstance(req, (dict, list)):
        return req
    if isinstance(req, list):
        return [normalize_req_dict(item) for item in req]

    fixed = {}
    for k, v in req.items():
        # Clean key of surrounding whitespace and quotes
        if isinstance(k, str):
            clean_key = k.strip().strip("'").strip('"').strip()
        else:
            clean_key = str(k).strip()

        # Recursively normalize nested dicts and lists
        fixed[clean_key] = normalize_req_dict(v)
    return fixed


def _pre_dedupe_local(requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Performs a cheap local deduplication using fuzzy string matching to reduce
    the payload size before sending to Gemini.
    """
    seen_texts: List[str] = []
    unique_reqs: List[Dict[str, Any]] = []

    for r in requirements:
        req_text = (
            r.get("requirement") or r.get("text") or r.get("title") or json.dumps(r)
        )
        norm_text = _normalize_text(req_text)
        is_duplicate = any(
            fuzz.token_set_ratio(norm_text, prev) >= FUZZY_SIM_THRESHOLD
            for prev in seen_texts
        )

        if not is_duplicate:
            seen_texts.append(norm_text)
            # Normalize structure for downstream processes
            unique_reqs.append(
                {
                    "requirement": req_text,
                    "requirement_type": r.get("requirement_type", "functional"),
                    "sources": r.get("sources", []),
                }
            )

    print(
        f"Pre-dedupe => {len(unique_reqs)} items from {len(requirements)} initial items."
    )
    return unique_reqs


def _chunk(lst: list, n: int) -> Iterator[list]:
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _firestore_commit_many(
    doc_tuples: List[Tuple[firestore.DocumentReference, Dict[str, Any]]],
) -> int:
    """Commits a list of documents to Firestore in batches."""
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
# Prompts and Schemas
# =====================

# A short, targeted prompt for explicit requirement deduplication.
EXPLICIT_PROMPT_TPL = (
    "You are an expert in medical device regulations. Deduplicate, split, and clean the "
    "following requirements. Keep the original 'requirement_type' and merge any "
    "associated 'sources'. The final output should be a single, concise markdown text "
    "string per requirement. Do not use HTML or first-person language. "
    "Return ONLY a JSON array with the following schema.\n\n"
    "Input JSON:\n```json\n{payload}\n```"
)

# Schema for the explicit requirements deduplication.
EXPLICIT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "requirement": {"type": "STRING"},
            "requirement_type": {"type": "STRING"},
            "requirement_title": {"type": "STRING"},
            "priority": {"type": "STRING", "enum": ["High", "Medium", "Low"]},
            "sources": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "filename": {"type": "STRING"},
                        "location": {"type": "STRING"},
                        "snippet": {"type": "STRING"},
                    },
                },
            },
        },
        "required": [
            "requirement",
            "requirement_type",
            "requirement_title",
            "priority",
            "sources",
        ],
    },
}


# A short, targeted prompt for implicit (regulatory) requirement deduplication.
IMPLICIT_PROMPT_TPL = "You are an expert in medical device regulations. Deduplicate and split the regulatory items below. Keep the 'requirement_type' as 'regulation', and use a concise, objective tone in markdown. Merge regulations that are the same. Return ONLY a JSON array with the following schema. Requirement title should be within 5-10 words representative of the main purpose of the requirement.\n\nInput JSON:\n```json\n{payload}\n```"

# Schema for the implicit requirements deduplication.
IMPLICIT_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "requirement": {"type": "STRING"},
            "requirement_type": {"type": "STRING"},
            "requirement_title": {"type": "STRING"},
            "priority": {"type": "STRING", "enum": ["High", "Medium", "Low"]},
            "regulations": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "regulation": {"type": "STRING"},
                        "source": {
                            "type": "OBJECT",
                            "properties": {
                                "filename": {"type": "STRING"},
                                "page_start": {"type": "STRING"},
                                "page_end": {"type": "STRING"},
                                "snippet": {"type": "STRING"},
                            },
                        },
                    },
                },
            },
        },
        "required": [
            "requirement",
            "requirement_type",
            "requirement_title",
            "priority",
            "regulations",
        ],
    },
}


# =====================
# Core Processing Functions
# =====================


@_retry(max_attempts=3)
def _genai_json_call(model: str, prompt: str, schema: dict) -> list:
    """
    Calls Gemini with a schema and returns the parsed JSON.
    Includes a timeout and retry logic for resilience.
    """
    with futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(
            lambda: genai_client.models.generate_content(
                model=model,
                contents=[Content(parts=[Part(text=prompt)], role="user")],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
        )
        resp = future.result(timeout=GENAI_TIMEOUT_SECONDS)
    return json.loads(resp.text)


@_retry(max_attempts=3)
def _query_discovery_engine_single(query_text: str) -> List[Dict[str, Any]]:
    """
    Queries Discovery Engine for a single text string and returns top results.
    """
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
            relevance = 0.0
            if result.model_scores and hasattr(
                result.model_scores.get("relevance_score"), 'values'
            ):
                relevance = float(result.model_scores.get("relevance_score").values[0])

            link = result.chunk.document_metadata.uri
            filename = link.split("/")[-1]
            regulation = next(
                (prefix for prefix in REGULATIONS if filename.startswith(prefix)), ""
            )
            processed.append(
                {
                    "relevance": relevance,
                    "content": result.chunk.content,
                    "regulation": regulation,
                    "filename": filename,
                    "page_start": result.chunk.page_span.page_start,
                    "page_end": result.chunk.page_span.page_end,
                    "snippet": result.chunk.content,
                }
            )

    processed.sort(key=lambda x: x["relevance"], reverse=True)
    return processed[:2]  # Return top 2 results


def _query_discovery_engine_parallel(
    requirements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Queries Discovery Engine for multiple requirements in parallel using a thread pool.
    """
    queries = [f"Regulations related to: {r.get('requirement')}" for r in requirements]
    all_results = []
    with futures.ThreadPoolExecutor(
        max_workers=min(MAX_WORKERS, len(queries) or 1)
    ) as ex:
        for res in ex.map(_query_discovery_engine_single, queries):
            all_results.extend(res)
    return all_results


def _load_and_normalize_explicit_requirements(
    requirements_p1_url: str,
) -> List[Dict[str, Any]]:
    """Loads and normalizes explicit requirements from a GCS bucket."""
    parsed = urlparse(requirements_p1_url)
    bucket = storage_client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))
    explicit_requirements_raw = json.loads(blob.download_as_text())

    if not explicit_requirements_raw:
        raise ValueError("Input data from GCS is empty.")

    # Normalize all incoming raw items immediately
    normalized_list = normalize_req_dict(explicit_requirements_raw)

    # Perform local fuzzy deduplication as a fast pre-processing step
    print("Pre-deduping locally (fuzzy)...")
    deduped_local = _pre_dedupe_local(normalized_list)
    return deduped_local


def _dedupe_requirements_with_gemini(
    requirements: List[Dict[str, Any]],
    prompt_template: str,
    schema: Dict[str, Any],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Deduplicates a list of requirements using Gemini in parallel batches."""
    batches = list(_chunk(requirements, batch_size))
    deduped_results: List[Dict[str, Any]] = []

    def process_batch(batch):
        # Normalize keys in batch defensively
        prompt = prompt_template.format(payload=json.dumps(batch, indent=2))
        return _genai_json_call(GENAI_MODEL, prompt, schema)

    with futures.ThreadPoolExecutor(
        max_workers=min(MAX_WORKERS, len(batches) or 1)
    ) as ex:
        for out in ex.map(process_batch, batches):
            # Ensure each returned item is normalized and appended
            deduped_results.extend(normalize_req_dict(out))

    return deduped_results


def _persist_requirements_to_firestore(
    project_id: str, version: str, requirements: List[Dict[str, Any]], start_id: int
) -> int:
    """Writes a list of requirements to Firestore in batches."""
    requirements_collection_ref = firestore_client.collection(
        "projects", project_id, "versions", version, "requirements"
    )
    doc_tuples = [
        (
            requirements_collection_ref.document(f"REQ-{i:03d}"),
            {
                **req,
                "requirement_id": f"REQ-{i:03d}",
                "deleted": False,
                "created_at": firestore.SERVER_TIMESTAMP,
            },
        )
        for i, req in enumerate(requirements, start=start_id)
    ]
    total_written = _firestore_commit_many(doc_tuples)
    return total_written


# =====================
# Main HTTP Function
# =====================
@functions_framework.http
def process_requirements_phase_2(request):
    """
    Main Cloud Function entry point for requirements processing Phase 2. It orchestrates the deduplication and ingestion of explicit and implicit
    requirements.
    """
    try:
        payload = request.get_json(silent=True) or {}
        project_id = payload.get("project_id")
        version = payload.get("version")
        requirements_p1_url = payload.get("requirements_p1_url")

        if not all([project_id, version, requirements_p1_url]):
            return (
                json.dumps(
                    {
                        "status": "error",
                        "message": "Required details (project_id, version, requirements_p1_url) are missing.",
                    }
                ),
                400,
            )

        _update_firestore_status(project_id, version, "START_REQ_EXTRACT_P2")

        # ------- Step 1: Load and deduplicate explicit requirements locally
        print("Starting explicit requirement processing...")
        explicit_reqs_shrunk = _load_and_normalize_explicit_requirements(
            requirements_p1_url
        )
        print(f"Loaded and locally deduped {len(explicit_reqs_shrunk)} items.")

        # ------- Step 2: Deduplicate explicit requirements with Gemini
        _update_firestore_status(project_id, version, "DEDUP_EXPLICIT_WITH_GEMINI")
        deduped_explicit = _dedupe_requirements_with_gemini(
            explicit_reqs_shrunk, EXPLICIT_PROMPT_TPL, EXPLICIT_SCHEMA, EXPL_BATCH_SIZE
        )
        print(f"Gemini explicit => {len(deduped_explicit)} unique requirements.")

        _update_firestore_status(project_id, version, "WRITE_EXPLICIT_TO_FIRESTORE")
        total_written = _persist_requirements_to_firestore(
            project_id, version, deduped_explicit, start_id=1
        )
        print(f"Explicit writes => {total_written}")
        _update_firestore_status(project_id, version, "COMPLETE_EXP_REQ")

        # ------- Step 3: Search for implicit regulations using Discovery Engine
        _update_firestore_status(project_id, version, "SEARCH_IMPLICIT_DISCOVERY")
        implicit_candidates = _query_discovery_engine_parallel(deduped_explicit)
        print(f"Discovery implicit candidates => {len(implicit_candidates)}")
        _update_firestore_status(project_id, version, "COMPLETE_IMP_REQ_FETCH")

        # ------- Step 4: Deduplicate implicit requirements with Gemini
        _update_firestore_status(project_id, version, "PROCESS_IMPLICIT_WITH_GEMINI")
        deduped_implicit = _dedupe_requirements_with_gemini(
            implicit_candidates, IMPLICIT_PROMPT_TPL, IMPLICIT_SCHEMA, IMPL_BATCH_SIZE
        )
        print(f"Gemini implicit => {len(deduped_implicit)} requirements.")

        _update_firestore_status(project_id, version, "WRITE_IMPLICIT_TO_FIRESTORE")
        total_written += _persist_requirements_to_firestore(
            project_id, version, deduped_implicit, start_id=len(deduped_explicit) + 1
        )
        print(f"Total writes (explicit+implicit) => {total_written}")

        _update_firestore_status(project_id, version, "CONFIRM_REQ_EXTRACT")
        return ("OK", 200)

    except Exception as e:
        logging.exception("Error during requirements extraction phase 2:")
        _update_firestore_status(project_id, version, "ERR_REQ_EXTRACT_P2")
        # Return a more descriptive error message
        return (
            json.dumps(
                {
                    "status": "error",
                    "message": f"An unexpected error occurred during processing: {str(e)}",
                }
            ),
            500,
        )
