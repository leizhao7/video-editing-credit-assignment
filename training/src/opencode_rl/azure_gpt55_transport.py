"""Parent-only Azure transport; OpenCode owns all messages, tools and history.

Deployment/parameter names and the explicitly declared effort are mapped.
Native GPT options request a Responses-only automatic reasoning summary; this
Chat endpoint rejects that option. Omit only that unavailable return-metadata
request (not reasoning effort). Original and mapped requests are both archived.
No role rewriting, hidden history, image removal or tool-result adaptation.
"""
import json
import os
from urllib.parse import urlparse
from .contract import ParityError, canonical

AZURE_GPT55_MODEL = {
    'provider': 'azure-openai-compatible', 'model_id': 'gpt-5.5',
    'deployment': os.environ.get('AZURE_OPENAI_DEPLOYMENT', ''), 'reasoning_effort': 'xhigh',
    'model_revision': 'API-managed; actual response model recorded',
    'internal_tensors_available': False,
}


class AzureGPT55Transport:
    def __init__(self, key, *, endpoint=None):
        self.endpoint = endpoint or os.environ.get('AZURE_OPENAI_CHAT_COMPLETIONS_URL', '')
        self.deployment = AZURE_GPT55_MODEL['deployment']
        parsed = urlparse(self.endpoint)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.fragment or not self.deployment):
            raise ParityError('Configure an HTTPS Azure chat-completions URL and deployment')
        self.key = key
        self.streams = {}

    @property
    def auth_headers(self):
        return {'api-key': self.key}

    def prepare(self, body):
        obj = json.loads(body)
        if obj.get('model') != 'gpt-5.5' or obj.get('temperature') != 1 or obj.get('top_p', 1) != 1:
            raise ParityError('API model/sampling differs from contract')
        if obj.get('reasoning_effort', 'xhigh') != 'xhigh':
            raise ParityError('GPT-5.5 reasoning effort differs from contract')
        if 'max_tokens' in obj:
            value = obj.pop('max_tokens')
            if 'max_completion_tokens' in obj and obj['max_completion_tokens'] != value:
                raise ParityError('Conflicting completion token budgets')
            obj['max_completion_tokens'] = value
        if obj.get('max_completion_tokens') != 16384:
            raise ParityError('GPT-5.5 output budget differs from contract')
        if 'reasoningSummary' in obj:
            if obj.pop('reasoningSummary') != 'auto':
                raise ParityError('Unreviewed reasoning summary mapping')
        obj['model'] = self.deployment
        obj['reasoning_effort'] = AZURE_GPT55_MODEL['reasoning_effort']
        return canonical(obj)

    def feed(self, request_id, chunk):
        state = self.streams.setdefault(request_id, {'buffer': b'', 'done': False})
        state['buffer'] += chunk
        while b'\n' in state['buffer']:
            line, state['buffer'] = state['buffer'].split(b'\n', 1)
            if line.startswith(b'data:'):
                data = line[5:].strip()
                if data == b'[DONE]':
                    state['done'] = True
                else:
                    obj = json.loads(data)
                    if obj.get('error'):
                        raise ParityError('GPT-5.5 returned an SSE error')
        return chunk  # Byte-exact provider stream, including usage and reasoning.

    def finish(self, request_id):
        state = self.streams.pop(request_id, None)
        if state and (state['buffer'].strip() or not state['done']):
            raise ParityError('Incomplete API SSE stream')
