import os
import json
import logging
import functions_framework
from google.cloud import firestore, tasks_v2

# from dotenv import load_dotenv
# load_dotenv()

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

        collection_ref = (
            firestore_client.collection(
                f'projects/{project_id}/versions/{version}/testcases'
            )
            .where('deleted', '==', False)
            .where('change_analysis_status', 'in', ['NEW', 'UNCHANGED'])
            .select(['testcase_id', 'dataset_status'])
            .order_by('testcase_id', 'ASCENDING')
        )

        docs = collection_ref.get()

        valid_testcases = [d.to_dict() for d in docs if d.get('dataset_status') not in EXCLUDED_STATUSES]

        if not valid_testcases:
            logging.info('No eligible testcases found')
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

        for tc in valid_testcases:
            if not tc.get('testcase_id'):
                continue

            tc_id = tc.get('testcase_id')

            task_payload = {
                'project_id': project_id,
                'version': version,
                'testcase_id': tc_id,
            }

            task = {
                'http_request': {
                    'http_method': tasks_v2.HttpMethod.POST,
                    'url': WORKER_URL,
                    'headers': {'Content-Type': 'application/json'},
                    'body': json.dumps(task_payload).encode(),
                    'oidc_token': {
                        'service_account_email': CLOUD_TASKS_SERVICE_ACCOUNT,
                        'audience': WORKER_URL,
                    },
                }
            }

            try:
                tasks_client.create_task(request={'parent': parent, 'task': task})
                logging.info('Enqueued task for testcase %s', tc_id)

                firestore_client.document(
                    'projects', project_id, 'versions', version, 'testcases', tc_id
                ).update({'dataset_status': 'DATASET_GENERATION_QUEUED'})

                enqueued.append(tc_id)

            except Exception as enqueue_err:
                logging.exception('Failed to enqueue testcase %s: %s', tc_id, enqueue_err)

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

# dispatcher_function(None)
