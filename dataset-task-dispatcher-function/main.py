import os
import json
import logging
import functions_framework
from google.cloud import firestore, tasks_v2

logging.basicConfig(level=logging.INFO)

# Environment
FIRESTORE_DATABASE = os.environ.get('FIRESTORE_DATABASE')
QUEUE_ID = os.environ.get('QUEUE_ID')  # Cloud Tasks queue name
QUEUE_LOCATION = os.environ.get('QUEUE_LOCATION')  # Region for Cloud Tasks
WORKER_URL = os.environ.get('WORKER_URL')  # URL of Cloud Function 2
CLOUD_TASKS_SERVICE_ACCOUNT = os.environ.get('CLOUD_TASKS_SERVICE_ACCOUNT')

firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
tasks_client = tasks_v2.CloudTasksClient()

# Statuses where we should NOT dispatch again
EXCLUDED_STATUSES = {
    'DATASET_GENERATION_QUEUED',
    'DATASET_GENERATION_STARTED',
    'DATASET_GENERATION_COMPLETED',
}


@functions_framework.http
def dispatcher_function(request):
    try:
        body = request.get_json(silent=True)
        if not body:
            return ('Bad request: JSON body required', 400)

        project_id = body.get('project_id')
        version = body.get('version')
        if not project_id or not version:
            return ('Bad request: project_id and version required', 400)

        collection_ref = firestore_client.collection(
            f'projects/{project_id}/versions/{version}/testcases'
        )
        docs = list(collection_ref.stream())

        # Filter out deleted docs + excluded statuses
        valid_testcases = []
        for d in docs:
            data = d.to_dict() or {}
            
            if data.get('deleted', False):
                continue

            status = data.get('dataset_status')
            if status in EXCLUDED_STATUSES:
                logging.info('Skipping testcase %s due to dataset_status=%s', d.id, status)
                continue

            valid_testcases.append(d)

        if not valid_testcases:
            return (
                json.dumps(
                    {
                        'message': 'no eligible testcases found',
                        'project_id': project_id,
                        'version': version,
                    }
                ),
                200,
            )

        parent = tasks_client.queue_path(
            os.environ['GOOGLE_CLOUD_PROJECT'], QUEUE_LOCATION, QUEUE_ID
        )

        enqueued = []
        for d in valid_testcases:
            tcid = d.id
            task_payload = {
                'project_id': project_id,
                'version': version,
                'testcase_id': tcid,
            }

            task = {
                'http_request': {
                    'http_method': tasks_v2.HttpMethod.POST,
                    'url': WORKER_URL,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps(task_payload).encode(),
                    'oidc_token': {
                        'service_account_email': CLOUD_TASKS_SERVICE_ACCOUNT,
                        'audience': WORKER_URL
                    }
                }
            }

            try:
                tasks_client.create_task(request={'parent': parent, 'task': task})
                logging.info('Enqueued task for testcase %s', tcid)

                # ✅ Update Firestore status after successful enqueue
                d.reference.update({'dataset_status': 'DATASET_GENERATION_QUEUED'})
                enqueued.append(tcid)

            except Exception as enqueue_err:
                logging.error('Failed to enqueue testcase %s: %s', tcid, enqueue_err)

        return (
            json.dumps(
                {
                    'message': 'tasks enqueued (and dataset_status updated)',
                    'count': len(enqueued),
                    'project_id': project_id,
                    'version': version,
                    'testcases': enqueued,
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception('Dispatcher failed: %s', e)
        return (f'Internal server error: {e}', 500)