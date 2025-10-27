from google.cloud import firestore

COLLECTION_NAME = "projects/7J4TD03eXpwnuoPxkB5M/versions/1/requirements"
BATCH_SIZE = 25

try:
    db = firestore.Client(database='sage')
except Exception as e:
    print(f"ERROR: Failed to initialize Firestore client: {e}")
    print(
        "Please ensure your Google Cloud Project ID is correct and you are authenticated."
    )
    exit()

collection_ref = db.collection(COLLECTION_NAME)

# ----------------------------------------------------------------------


def delete_collection_in_batches(coll_ref, batch_size):
    """
    Deletes all documents in a collection in batches using the Firestore Client SDK.

    Args:
        coll_ref: The Firestore collection reference.
        batch_size: The number of documents to delete in each batch.
    """
    print(f"Starting deletion from collection: '{coll_ref.id}'...")

    total_deleted = 0

    while True:
        # Get the next batch of documents (limited by batch_size)
        # Using .limit() fetches the documents needed for the current batch.
        docs = coll_ref.limit(batch_size).stream()

        deleted_count = 0
        batch = db.batch()

        # Add each document's reference to the transaction batch for deletion
        for doc in docs:
            batch.delete(doc.reference)
            deleted_count += 1

        # If no documents were fetched, we are done
        if deleted_count == 0:
            print(f"Finished. Total documents deleted: {total_deleted}.")
            break

        # Commit the deletion batch (this sends the 25 delete operations to Firestore)
        batch.commit()
        total_deleted += deleted_count
        print(
            f"Successfully deleted batch of {deleted_count} documents. Total deleted: {total_deleted}"
        )


# ----------------------------------------------------------------------

# --- Execute Script ---
if __name__ == "__main__":
    delete_collection_in_batches(collection_ref, BATCH_SIZE)
