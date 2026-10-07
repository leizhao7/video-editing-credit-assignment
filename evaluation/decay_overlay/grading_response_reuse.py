"""Reuse audited, parseable answers only inside one private grading invocation.

This module does not schedule retries, accept partial grades, or change deadlines.
The service must create a new scope after reconstructing its private task and
must keep that task's evidence unchanged between scoring rounds. Existing judge
parsers and aggregators remain responsible for the actual grade.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

FORMAT = 'outcome-response-reuse-v1'
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024


class ReuseIntegrityError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _folder(task):
    task = Path(task)
    if not task.is_absolute() or task.is_symlink():
        raise ReuseIntegrityError('Expected absolute private task directory')
    folder = task / 'evidence' / 'retry_reuse'
    if (task / 'evidence').is_symlink() or folder.is_symlink():
        raise ReuseIntegrityError('Unexpected reuse-directory symlink')
    return folder


def _read(path, limit=MAX_METADATA_BYTES):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ReuseIntegrityError('Invalid reuse metadata')
    try:
        return json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ReuseIntegrityError('Malformed reuse metadata') from exc


def _publish_first(path, value):
    temporary = path.with_name('.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass  # First successful result wins; never select a better score.
    finally:
        temporary.unlink(missing_ok=True)


def create_scope(task, identity, deadline_unix, *, now=None):
    now = time.time() if now is None else now
    if (type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix)
            or not 0 < deadline_unix - now <= 600):
        raise ReuseIntegrityError('Expected the existing, unexpired grading deadline')
    required = {'input_contract', 'grader_sha256', 'judge', 'review_policy'}
    if set(identity) != required or identity['review_policy'] != 'judge_only':
        raise ReuseIntegrityError('Incomplete or unsupported grading identity')
    if any(not isinstance(identity[key], dict) or not identity[key]
           for key in ('input_contract', 'grader_sha256', 'judge')):
        raise ReuseIntegrityError('Empty grading identity')
    folder = _folder(task)
    folder.mkdir(parents=True, exist_ok=False)
    value = {'format': FORMAT, 'invocation': uuid.uuid4().hex,
             'identity_sha256': digest(identity), 'deadline_unix': deadline_unix}
    _publish_first(folder / 'scope.json', value)
    return value


def _scope(task, *, now=None):
    folder = _folder(task)
    if not (folder / 'scope.json').exists():
        return None
    value = _read(folder / 'scope.json')
    if (value.get('format') != FORMAT
            or not re.fullmatch('[0-9a-f]{32}', str(value.get('invocation', '')))
            or not re.fullmatch('[0-9a-f]{64}', str(value.get('identity_sha256', '')))
            or type(value.get('deadline_unix')) not in (int, float)
            or not math.isfinite(value['deadline_unix'])):
        raise ReuseIntegrityError('Invalid grading reuse scope')
    now = time.time() if now is None else now
    if value['deadline_unix'] <= now:
        return None
    return value


def _key(scope, rubric_id, request):
    return digest({'scope': scope, 'rubric': rubric_id, 'request': request})


def _audit_path(task, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'evidence':
        raise ReuseIntegrityError('Audit source is outside private evidence')
    current = Path(task)
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ReuseIntegrityError('Audit source contains a symlink')
    return current


def _parseable(text, error, parser):
    if not isinstance(text, str) or not text or error != '':
        return False
    band, _, _ = parser(text)
    return type(band) is int and 0 <= band <= 4


def _proven_response(task, prefix_relative, request, parser):
    prefix = _audit_path(task, prefix_relative)
    paths = {kind: Path(str(prefix) + '.' + kind)
             for kind in ('request.json', 'response.json', 'provider.json', 'http.jsonl')}
    request_saved = _read(paths['request.json'], MAX_REQUEST_BYTES)
    response = _read(paths['response.json'])
    if not _parseable(response.get('text'), response.get('error'), parser):
        return None
    provider = _read(paths['provider.json'])
    if digest(request_saved) != digest(request):
        raise ReuseIntegrityError('Source request differs from lookup request')
    http_path = paths['http.jsonl']
    if http_path.is_symlink() or not http_path.is_file() or http_path.stat().st_size > MAX_METADATA_BYTES:
        raise ReuseIntegrityError('Missing or invalid original HTTP audit')
    try:
        events = [json.loads(line) for line in http_path.read_bytes().splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise ReuseIntegrityError('Malformed source HTTP audit') from exc
    if any(not isinstance(event, dict) for event in events):
        raise ReuseIntegrityError('Invalid source HTTP event')
    finished = [event for event in events if event.get('event') == 'finished']
    successful = (finished and finished[-1].get('http_status') == 200
                  and finished[-1].get('status') == 'completed')
    blocked = bool((provider.get('promptFeedback') or {}).get('blockReason'))
    blocked |= any(reason in ('PROHIBITED_CONTENT', 'SAFETY', 'BLOCKLIST', 'RECITATION')
                   for reason in provider.get('finishReasons', []))
    if not successful or blocked or not _parseable(response.get('text'), response.get('error'), parser):
        return None
    hashes = {kind: hashlib.sha256(path.read_bytes()).hexdigest() for kind, path in paths.items()}
    return response, provider, hashes


def remember_response(task, rubric_id, request, prefix, parser, *, now=None):
    scope = _scope(task, now=now)
    if scope is None:
        return False
    relative = str(Path(prefix).relative_to(Path(task)))
    proven = _proven_response(task, relative, request, parser)
    if proven is None:
        return False
    _, _, hashes = proven
    value = {'format': FORMAT, 'scope': scope, 'rubric': rubric_id,
             'request_sha256': digest(request), 'source_prefix': relative,
             'source_sha256': hashes}
    _publish_first(_folder(task) / (_key(scope, rubric_id, request) + '.json'), value)
    return True


def reuse_response(task, rubric_id, request, prefix, parser, *, now=None):
    scope = _scope(task, now=now)
    if scope is None:
        return None
    path = _folder(task) / (_key(scope, rubric_id, request) + '.json')
    if not path.exists():
        return None
    value = _read(path)
    if (value.get('format') != FORMAT or value.get('scope') != scope
            or value.get('rubric') != rubric_id or value.get('request_sha256') != digest(request)):
        raise ReuseIntegrityError('Cached response binding mismatch')
    proven = _proven_response(task, value['source_prefix'], request, parser)
    if proven is None or proven[2] != value.get('source_sha256'):
        raise ReuseIntegrityError('Original provider evidence changed')
    response, provider, hashes = proven
    # Normal response artifacts are preserved; a reuse receipt explicitly records
    # zero network requests. Never fabricate a new successful HTTP attempt.
    prefix = _audit_path(task, str(Path(prefix).relative_to(Path(task))))
    Path(str(prefix) + '.response.json').write_text(json.dumps(response, ensure_ascii=False))
    Path(str(prefix) + '.provider.json').write_text(json.dumps(provider, ensure_ascii=False))
    Path(str(prefix) + '.reuse.json').write_text(json.dumps({
        'format': FORMAT, 'scope': scope, 'source_prefix': value['source_prefix'],
        'source_sha256': hashes, 'request_sha256': digest(request),
        'actual_http_requests': 0, 'reused_at': time.time() if now is None else now}))
    return response['text'], response['error']


def _file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def mark_evidence_ready(task, evidence_name):
    """Called only after the evidence subprocess has returned successfully."""
    scope = _scope(task)
    if scope is None:
        return
    task = Path(task)
    if Path(evidence_name).name != evidence_name or not (task/'evidence'/evidence_name).is_file():
        raise ReuseIntegrityError('Missing completed evidence summary')
    inventory = {}
    total_bytes = 0
    for directory in ('evidence', 'output'):
        for path in sorted((task/directory).rglob('*')):
            relative = path.relative_to(task)
            if directory == 'evidence' and len(relative.parts)>1 and relative.parts[1] in (
                    'judge_requests', 'retry_reuse', 'scoring_rounds'):
                continue
            if path.is_symlink():
                raise ReuseIntegrityError('Unexpected generated-evidence symlink')
            if not path.is_file():
                continue
            # Scoring summaries belong to rounds, not to the prepared evidence.
            if directory == 'evidence' and len(relative.parts)==2 and re.fullmatch(
                    r's(?:3|4e)_(?:score(?:_ref)?|final|consistency)\.json', path.name):
                continue
            total_bytes += path.stat().st_size
            if total_bytes > 1024**3 or len(inventory) >= 4096:
                # Evidence reuse is optional. A larger task keeps full preparation
                # on later rounds; it must not acquire a new submission limit.
                (_folder(task)/'evidence_ready.json').unlink(missing_ok=True)
                return
            inventory[str(relative)] = {'bytes':path.stat().st_size,'sha256':_file_sha(path)}
    value = {'format':FORMAT,'scope':scope,'evidence_name':evidence_name,'files':inventory}
    path = _folder(task)/'evidence_ready.json'
    path.unlink(missing_ok=True)
    _publish_first(path,value)


def evidence_reusable(task, evidence_name):
    scope = _scope(task)
    path = _folder(task)/'evidence_ready.json'
    if scope is None or not path.exists():
        return False
    value = _read(path)
    if value.get('format')!=FORMAT or value.get('scope')!=scope or value.get('evidence_name')!=evidence_name:
        return False
    files = value.get('files')
    if not isinstance(files,dict) or 'evidence/'+evidence_name not in files:
        return False
    for relative, metadata in files.items():
        child = Path(relative)
        if child.is_absolute() or '..' in child.parts or child.parts[0] not in ('output','evidence'):
            raise ReuseIntegrityError('Invalid prepared-evidence path')
        target = Path(task)
        for part in child.parts:
            target = target/part
            if target.is_symlink():
                raise ReuseIntegrityError('Prepared evidence became a symlink')
        if (not target.is_file() or target.stat().st_size!=metadata['bytes']
                or _file_sha(target)!=metadata['sha256']):
            return False
    return True
