import os
import json
import functions_framework
from google.cloud import firestore
from google import genai
from google.genai.types import Content, Part

FIRESTORE_DATABASE = os.getenv('FIRESTORE_DATABASE')
MODEL_NAME = os.getenv('MODEL_NAME')
ENHANCEMENT_PROMPT = os.getenv('ENHANCEMENT_PROMPT')

firestore_client = firestore.Client(database=FIRESTORE_DATABASE)
genai_client = genai.Client()

SCHEMA = {
    'type': 'object',
    'properties': {
        'title': {'type': 'string'},
        'description': {'type': 'string'},
        'acceptance_criteria': {'type': 'string'},
        'priority': {'type': 'string', 'enum': ['High', 'Medium', 'Low']},
    },
    'required': ['title', 'description', 'acceptance_criteria', 'priority'],
}


@functions_framework.http
def update_testcase(request):
    try:
        data = request.get_json(silent=True)
        if not data:
            return (
                json.dumps({'error': 'Invalid request, must be JSON.'}),
                400,
                {'Content-Type': 'application/json'},
            )

        project_id = data.get('project_id')
        version = data.get('version')
        testcase_id = data.get('testcase_id')
        prompt = data.get('prompt')
        uid = data.get('uid')

        if not all([project_id, version, testcase_id, prompt, uid]):
            return (
                json.dumps({'error': 'Missing required parameters.'}),
                400,
                {'Content-Type': 'application/json'},
            )

        # Firestore path
        doc_ref = (
            firestore_client.collection('projects')
            .document(project_id)
            .collection('versions')
            .document(version)
            .collection('testcases')
            .document(testcase_id)
        )

        # Prefix the prompt with role instruction
        role_prompt = f'{ENHANCEMENT_PROMPT}\n\n{prompt}'

        # Generate structured output from Gemini
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=[Content(parts=[Part(text=role_prompt)], role='user')],
            config=genai.types.GenerateContentConfig(
                response_mime_type='application/json', response_schema=SCHEMA
            )
        )

        output = json.loads(response.text)

        doc_ref.update(
            {
                'title': output.get('title'),
                'description': output.get('description'),
                'acceptance_criteria': output.get('acceptance_criteria'),
                'priority': output.get('priority'),
                'last_updated_by': uid
            }
        )

        return (
            json.dumps({'success': True, 'updated': output}),
            200,
            {'Content-Type': 'application/json'},
        )

    except Exception as e:
        return (
            json.dumps({'error': str(e)}),
            500,
            {'Content-Type': 'application/json'},
        )
