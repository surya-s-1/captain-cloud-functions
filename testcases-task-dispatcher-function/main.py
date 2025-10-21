import os
import json
import logging

import functions_framework
from google.cloud import firestore, tasks_v2

# from dotenv import load_dotenv
# load_dotenv()

# =====================
# Environment variables
# =====================
FIRESTORE_DATABASE = os.environ.get('FIRESTORE_DATABASE')
QUEUE_NAME = os.environ.get('QUEUE_NAME')
WORKER_URL = os.environ.get('WORKER_URL')  # URL of the worker Cloud Function
CLOUD_TASKS_SERVICE_ACCOUNT = os.environ.get('CLOUD_TASKS_SERVICE_ACCOUNT')

# =====================
# Clients
# =====================
firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
tasks_client = tasks_v2.CloudTasksClient()
logging.basicConfig(level=logging.INFO)

EXCLUDED_CHANGE_STATUSES = ['IGNORED', 'DEPRECATED', 'UNCHANGED']
EXCLUDED_TESTCASE_STATUSES = [
    'TESTCASES_CREATION_QUEUED',
    'TESTCASES_CREATION_STARTED',
    'TESTCASES_CREATION_COMPLETE',
]


# =======================================================
# Cloud Function: HTTP Orchestrator
# Enqueues tasks for each requirement
# =======================================================
@functions_framework.http
def process_for_testcases(request):
    '''
    Main HTTP entrypoint. Retrieves requirements and enqueues them as Cloud Tasks.
    '''
    try:
        request_json = request.get_json(silent=True)
        # Mock data
        # request_json = {
        #     'project_id': 'pJ6Q09OchNLLE86eUu1f',
        #     'version': 'v2'
        # }

        if not request_json:
            return {'error': 'JSON body not provided.'}, 400

        project_id = request_json.get('project_id')
        version = request_json.get('version')

        if not project_id or not version:
            return {'error': 'Missing project_id or version'}, 400

        logging.info(f'Start orchestration for project={project_id}, version={version}')

        firestore_client.document(f'projects/{project_id}/versions/{version}').update(
            {'testcase_status': 'START_TESTCASE_CREATION'}
        )

        requirements_ref = firestore_client.collection(
            'projects', project_id, 'versions', version, 'requirements'
        )

        query = (
            requirements_ref.where('deleted', '!=', True)
            .select(['requirement_id', 'deleted', 'duplicate', 'change_analysis_status', 'testcase_status'])
        )

        requirements_to_process = query.get()
        requirements_to_process = [r.to_dict() for r in requirements_to_process]

        requirements_to_process = [
            r
            for r in requirements_to_process
            if r.get('duplicate', '') != True
            and r.get('change_analysis_status', '') not in EXCLUDED_CHANGE_STATUSES
            and r.get('testcase_status', '') not in EXCLUDED_TESTCASE_STATUSES
        ]

        logging.info(f'Found {len(requirements_to_process)} requirements to enqueue.')

        docs_to_update = []

        for req_doc in requirements_to_process:
            payload = {
                'project_id': project_id,
                'version': version,
                'requirement_id': req_doc.id,
            }

            task = {
                'http_request': {
                    'http_method': 'POST',
                    'url': WORKER_URL,
                    'body': json.dumps(payload).encode('utf-8'),
                    'headers': {'Content-Type': 'application/json'},
                    'oidc_token': {
                        'service_account_email': CLOUD_TASKS_SERVICE_ACCOUNT,
                        'audience': WORKER_URL,
                    },
                }
            }

            try:
                tasks_client.create_task(parent=QUEUE_NAME, task=task)
                firestore_client.document(
                    f'projects/{project_id}/versions/{version}/requirements/{req_doc.id}'
                ).update({'testcase_status': 'TESTCASES_CREATION_QUEUED'})
                docs_to_update.append(req_doc.id)

            except Exception as e:
                logging.error(f'Failed to enqueue task for {req_doc.id}: {e}')
                # The loop continues to the next requirement

        return (
            json.dumps(
                {
                    'status': 'success',
                    'message': f'Successfully enqueued {len(docs_to_update)} requirements for processing',
                }
            ),
            200,
        )

    except Exception as e:
        firestore_client.document(f'projects/{project_id}/versions/{version}').update(
            {'status': 'CONFIRM_REQ_EXTRACT_RETRY'}
        )

        logging.exception(e)

        return {'error': str(e)}, 500
