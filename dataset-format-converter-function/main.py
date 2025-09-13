import os
import re
import json
import logging
import pandas as pd
from io import BytesIO, StringIO
import xml.etree.ElementTree as ET
from google.cloud import storage, firestore
import functions_framework

storage_client = storage.Client()
db = firestore.Client(database=os.getenv('FIRESTORE_DATABASE'))

def upload_and_update_firestore(
    bucket, output_prefix, filename, content, content_type, 
    project_id, version, testcase_id
):
    """
    Upload file to GCS and update Firestore datasets array field.
    """
    gcs_path = f"gs://{bucket}/{output_prefix}/{filename}"

    # Upload file
    bucket_obj = storage_client.bucket(bucket)
    blob = bucket_obj.blob(f"{output_prefix}/{filename}")
    if isinstance(content, bytes):
        blob.upload_from_string(content, content_type=content_type)
    else:
        blob.upload_from_string(str(content), content_type=content_type)

    logging.info(f"Uploaded {gcs_path}")

    # Update Firestore
    doc_ref = (
        db.collection('projects')
        .document(project_id)
        .collection('versions')
        .document(version)
        .collection('testcases')
        .document(testcase_id)
    )

    doc_ref.set(
        {"datasets": firestore.ArrayUnion([gcs_path])},
        merge=True
    )
    logging.info(f"Updated Firestore with {gcs_path}")


def convert_json_to_formats(data, bucket_name, output_prefix, project_id, version, testcase_id):
    """
    Converts JSON to XLSX, CSV, XML, uploads them, and updates Firestore.
    """
    try:
        df = pd.read_json(StringIO(data))
    except ValueError as e:
        logging.error(f"Error reading JSON data: {e}")
        return

    # Convert to CSV
    csv_content = df.to_csv(index=False)
    upload_and_update_firestore(
        bucket_name, output_prefix, "dataset.csv", csv_content, "text/csv",
        project_id, version, testcase_id
    )

    # Convert to XLSX
    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    xlsx_buffer.seek(0)
    upload_and_update_firestore(
        bucket_name, output_prefix, "dataset.xlsx", xlsx_buffer.read(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        project_id, version, testcase_id
    )

    # Convert to XML
    root = ET.Element("data")
    for row in df.itertuples(index=False):
        item = ET.SubElement(root, "item")
        for col, value in zip(df.columns, row):
            child = ET.SubElement(item, col)
            child.text = str(value)

    xml_string = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    upload_and_update_firestore(
        bucket_name, output_prefix, "dataset.xml", xml_string, "application/xml",
        project_id, version, testcase_id
    )


# Cloud Function entrypoint
@functions_framework.cloud_event
def hello_gcs(cloud_event):
    """
    Cloud Function triggered by a GCS file upload (CloudEvent format).
    """
    data = cloud_event.data

    bucket = data["bucket"]
    name = data["name"]  # path inside bucket
    file_path = f"gs://{bucket}/{name}"

    logging.info(f"Received file: {file_path}")

    # Regex pattern to match required path
    pattern = re.compile(
        r"^projects/([^/]+)/v_([^/]+)/testcases/([^/]+)/datasets/dataset\.json$"
    )

    match = pattern.match(name)
    if not match:
        logging.info(f"Skipping file: {name}. Path does not match required pattern.")
        return

    project_id, version, testcase_id = match.groups()
    output_prefix = os.path.dirname(name)

    logging.info(f"Valid dataset.json found for project={project_id}, version={version}, testcase={testcase_id}")
    logging.info(f"Output directory: {output_prefix}")

    try:
        bucket_obj = storage_client.bucket(bucket)
        blob = bucket_obj.blob(name)
        json_data = json.loads(blob.download_as_text())

        json_data_str = json.dumps(json_data)
        convert_json_to_formats(json_data_str, bucket, output_prefix, project_id, version, testcase_id)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)