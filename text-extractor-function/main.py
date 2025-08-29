import os
import io
import ast
import json
import docx
import base64
import mimetypes
import pandas as pd
import functions_framework
from urllib.parse import urlparse
from google.cloud import storage, documentai_v1 as documentai

storage_client = storage.Client()
documentai_client = documentai.DocumentProcessorServiceClient()

GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
PROCESSOR_ID = os.getenv('DOC_AI_PROCESSOR_ID')
LOCATION = os.getenv('DOC_AI_LOCATION')
PROCESSOR_NAME = documentai_client.processor_path(
    GOOGLE_CLOUD_PROJECT, LOCATION, PROCESSOR_ID
)
OUTPUT_BUCKET = os.getenv('OUTPUT_BUCKET')

# --- Helper Functions ---


def _download_blob_to_memory(bucket_name, source_blob_name):
    '''Downloads a blob from a GCS bucket to an in-memory byte stream.'''
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    file_stream = io.BytesIO()
    blob.download_to_file(file_stream)
    file_stream.seek(0)
    return file_stream


def _extract_from_structured(file_url):
    """Extracts text from a CSV or Excel file using pandas, with location data."""
    parsed_url = urlparse(file_url)
    bucket_name = parsed_url.netloc
    blob_name = parsed_url.path.lstrip('/')

    file_stream = _download_blob_to_memory(bucket_name, blob_name)

    if blob_name.lower().endswith('.csv'):
        df = pd.read_csv(file_stream)
    elif blob_name.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_stream, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported structured file type: {blob_name}")

    extracted_data = []
    # Iterate through rows and create a structured output with location
    for row_index, row in df.iterrows():
        extracted_data.append(
            {
                'text': row.to_string(),
                'location': {'row': int(row_index), 'column_headers': list(df.columns)},
            }
        )

    return {
        'file_type': 'structured',
        'extracted_by': 'pandas',
        'extracted_text': extracted_data,
    }


def _extract_from_word(file_url):
    """Extracts text from a Word document using python-docx, with location data."""
    parsed_url = urlparse(file_url)
    bucket_name = parsed_url.netloc
    blob_name = parsed_url.path.lstrip('/')

    file_stream = _download_blob_to_memory(bucket_name, blob_name)

    doc = docx.Document(file_stream)
    extracted_data = []

    for para_index, para in enumerate(doc.paragraphs):
        if para.text.strip():  # Avoid empty paragraphs
            extracted_data.append(
                {'text': para.text, 'location': {'paragraph_number': para_index}}
            )

    return {
        'file_type': 'unstructured',
        'extracted_by': 'python-docx',
        'extracted_text': extracted_data,
    }


def _extract_from_document_ai(file_url):
    """Extracts text from a PDF or image using Google Document AI OCR, with location data."""
    parsed_url = urlparse(file_url)
    bucket_name = parsed_url.netloc
    blob_name = parsed_url.path.lstrip('/')

    mime_type, _ = mimetypes.guess_type(file_url)
    if not mime_type:
        raise ValueError(f"Could not determine MIME type for file: {file_url}")

    file_stream = _download_blob_to_memory(bucket_name, blob_name)
    content = file_stream.read()

    raw_document = documentai.RawDocument(content=content, mime_type=mime_type)

    request = documentai.ProcessRequest(
        name=PROCESSOR_NAME,
        raw_document=raw_document,
    )

    response = documentai_client.process_document(request=request)

    extracted_data = []
    # Document AI provides detailed page and layout information
    for page_index, page in enumerate(response.document.pages):
        for paragraph in page.paragraphs:
            paragraph_text = ''
            for segment in paragraph.layout.text_anchor.text_segments:
                start_index = int(segment.start_index)
                end_index = int(segment.end_index)
                paragraph_text += response.document.text[start_index:end_index]

            if paragraph_text.strip():
                extracted_data.append(
                    {'text': paragraph_text, 'location': {'page': page_index + 1}}
                )

    return {
        'file_type': 'semistructured',
        'extracted_by': 'document_ai',
        'extracted_text': extracted_data,
    }


# --- Main Cloud Function ---


@functions_framework.cloud_event
def extract_text_from_files(event, context = None):
    '''
    Cloud Function to extract text from files specified in a Pub/Sub message.
    '''
    try:
        if (
            not event
            or not event.data
            or not event.data.get('message', None)
            or not event.data.get('message', {}).get('data', None)
        ):
            raise ValueError('Pub/Sub message \'data\' field is missing.')

        # Decode and parse the Pub/Sub message
        message_payload_str = base64.b64decode(event.data.get('message', {}).get('data', None)).decode('utf-8')
        message_payload = ast.literal_eval(message_payload_str)

        project_id = message_payload.get('project_id', None)
        file_urls = message_payload.get('files', [])

        extracted_results = []

        for file_url in file_urls:
            try:
                file_name = os.path.basename(urlparse(file_url).path)
                file_extension = file_name.split('.')[-1].lower()

                result_object = {'file_name': file_name, 'file_url': file_url}

                if file_extension in ['csv', 'xlsx', 'xls']:
                    result_object.update(_extract_from_structured(file_url))
                elif file_extension in ['docx']:
                    result_object.update(_extract_from_word(file_url))
                elif file_extension in ['pdf', 'jpg', 'jpeg', 'png']:
                    result_object.update(_extract_from_document_ai(file_url))
                else:
                    print(f'Skipping unsupported file type: {file_url}')
                    continue

                extracted_results.append(result_object)
            except Exception as e:
                print(f'Error processing file {file_url}: {e}')

        # Write the results to a JSON file in GCS
        output_blob_path = f'extracted-text/{project_id}.json'
        output_bucket = storage_client.bucket(OUTPUT_BUCKET)
        output_blob = output_bucket.blob(output_blob_path)

        output_blob.upload_from_string(
            data=json.dumps(extracted_results, indent=2),
            content_type='application/json',
        )
        print(f'Successfully wrote results to gs://{OUTPUT_BUCKET}/{output_blob_path}')

    except Exception as e:
        print(f'An error occurred: {e}')
        # Re-raise the exception to indicate failure to the Cloud Function runner
        raise
