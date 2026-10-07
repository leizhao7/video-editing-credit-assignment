"""Inspect actual wire records, not self-reported successful process exit."""
import json
import tarfile
from pathlib import Path
from .store import TraceStore


def audit_preflight(path):
    root = Path(path)
    store = TraceStore(root / 'trace')
    attempt = json.loads((root / 'attempt.json').read_bytes())['attempt_id']
    requests = []
    observations = 0
    tool_names = set()
    for ref in store.attempt_requests(attempt):
        events, seal = store.read_request(ref['request_id'])
        for event in events:
            item = event['data']
            if event['kind'] == 'runtime_media_observation':
                observations += 1
            if event['kind'] == 'relay_request':
                body = json.loads(store.read_blob(item['body']))
                requests.append(body)
                for message in body.get('messages', []):
                    for call in message.get('tool_calls', []):
                        tool_names.add(call.get('function', {}).get('name', ''))
    has_images = any(part.get('type') == 'image_url' for request in requests
        for message in request.get('messages', []) if isinstance(message.get('content'), list)
        for part in message['content'] if isinstance(part, dict))
    marker = False
    with tarfile.open(root / 'complete_trajectory.tar.gz') as archive:
        try:
            info = archive.getmember('work/output/marker.txt')
            if info.isfile() and info.size < 100:
                marker = archive.extractfile(info).read() == b'OCRL_PREFLIGHT_OK'
        except KeyError:
            pass
    return {'passed': len(requests) >= 3 and has_images and observations > 0 and 'bash' in tool_names and marker
            and all(r.get('temperature') == 1 for r in requests),
        'requests': len(requests), 'has_image_input': has_images, 'media_observations': observations, 'shell_marker': marker,
        'tool_names': sorted(tool_names), 'temperatures': [r.get('temperature') for r in requests]}
