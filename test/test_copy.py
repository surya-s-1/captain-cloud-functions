from google.cloud import firestore
import datetime
import copy
import sys

try:
    db = firestore.Client(database='sage')
    print("Firestore client initialized successfully.")
except Exception as e:
    print(f"Error initializing Firestore client: {e}")
    sys.exit(1)

# --- Example Usage ---
# NOTE: Replace these with your actual document paths and subcollection ID

PREV_VERSION = "1"
SOURCE_DOC_PATH = "projects/EUz0pMnqmNkBfh8FHMYZ/versions/1"
SUBCOLLECTION_ID = "requirements"
TARGET_DOC_PATH = "projects/EUz0pMnqmNkBfh8FHMYZ/versions/2"


def process_document_data(version: str, doc_data: dict) -> dict:
    existing_history = []
    prev_history = doc_data.pop('history', [])

    if isinstance(prev_history, list):
        existing_history.extend(prev_history)

    current_state_entry = {
        "version": version,
        "fields": copy.deepcopy(doc_data),
        "copied_at": datetime.datetime.now(datetime.timezone.utc),
    }

    new_history = [current_state_entry] + existing_history

    doc_data['history'] = new_history

    return doc_data


def copy_subcollection_with_history(
    source_doc_path: str, subcollection_id: str, target_doc_path: str, prev_version: str
):
    print(f"\n--- Starting Copy Operation ---")
    print(f"Source: {source_doc_path}/{subcollection_id}")
    print(f"Target: {target_doc_path}/{subcollection_id}")

    try:
        source_subcollection_ref = (
            db.collection(f'{source_doc_path}/{subcollection_id}')
            # .where('source_type', '==', 'implicit')
        )

        target_subcollection_ref = db.collection(
            f'{target_doc_path}/{subcollection_id}'
        )

        docs_to_copy = source_subcollection_ref.stream()

        batch = db.batch()
        copy_count = 0
        batch_size = 0

        for doc in docs_to_copy:
            doc_id = doc.id
            doc_data = doc.to_dict()

            # --- History Processing ---
            # Process the data to include the new history entry and manage old history
            processed_data = process_document_data(prev_version, doc_data)

            target_doc_ref = target_subcollection_ref.document(doc_id)
            batch.set(target_doc_ref, processed_data)

            copy_count += 1
            batch_size += 1

            if batch_size >= 50:
                print(f"Committing batch of {batch_size} documents...")
                batch.commit()
                batch = db.batch()  # Start a new batch
                batch_size = 0

        if batch_size > 0:
            print(f"Committing final batch of {batch_size} documents...")
            batch.commit()

        print(f"\nSuccessfully copied {copy_count} documents.")
        print(
            "All documents now contain an updated 'history' array reflecting this migration."
        )

    except Exception as e:
        print(f"\n--- A critical error occurred ---")
        print(f"Error: {e}")
        print(
            "Note: If the script fails due to permission errors, ensure your service account or ADC has 'Cloud Datastore User' or 'Cloud Firestore User' roles."
        )


copy_subcollection_with_history(
    SOURCE_DOC_PATH, SUBCOLLECTION_ID, TARGET_DOC_PATH, PREV_VERSION
)


### How the History Logic Works:

# 1.  **`doc_data.pop('history', [])`**: This is the key line. It attempts to extract the value of the `history` field. If it exists, it is stored in `existing_history` and simultaneously **deleted** from the `doc_data` dictionary. If it doesn't exist, `existing_history` is an empty list `[]`.
# 2.  **`copy.deepcopy(doc_data)`**: A copy of the history-less `doc_data` is made. This ensures the new history entry contains only the document's actual data fields and not the old `history` array, fulfilling your requirement.
# 3.  **`new_history.extend(...)`**: The old history entries (if any) are copied first.
# 4.  **`new_history.append(current_state_entry)`**: The new entry for the migration is appended last.
# 5.  **`doc_data['history'] = new_history`**: The complete, new `history` array is added back to `doc_data` before it is written to the target location.

# This approach ensures the `history` field is always a clean record of previous states, and the state *before* the move is captured as the latest entry.
