"""Bounded async, observation-only PRM sidecar for the shared append loop.

No synchronous offline HTTP worker or archive reader is used. A failed/unknown
process decision has score=0, valid=False; it never overwrites outcome reward.
"""
import asyncio
import base64
import binascii
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import judge_v5, transport_v3
from .live_config_v5 import validate


MEDIA_FORMAT = 'prm-live-media-v1'
REQUEST_FORMAT = 'prm-live-request-refs-v1'
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_IMAGE_MIMES = {'image/jpeg', 'image/png', 'image/webp'}


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _checked_media(record, expected_sha):
    """Pure byte/hash/MIME validation, no path access or pixel re-encoding."""
    if (type(record) is not dict or set(record) != {'format', 'sha256', 'mime_type', 'bytes', 'base64'}
            or record['format'] != MEDIA_FORMAT or record['sha256'] != expected_sha
            or not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha)
            or record['mime_type'] not in _IMAGE_MIMES or type(record['bytes']) is not int
            or not 0 < record['bytes'] <= 8_000_000 or not isinstance(record['base64'], str)):
        raise ValueError('Invalid content-addressed media record')
    if len(record['base64']) > 4 * ((record['bytes'] + 2) // 3):
        raise ValueError('Media encoding exceeds declared length')
    try:
        raw = base64.b64decode(record['base64'], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('Invalid media encoding') from exc
    if (len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != expected_sha
            or base64.b64encode(raw).decode('ascii') != record['base64']):
        raise ValueError('Media hash/length/canonical encoding mismatch')
    return record


def _media_identity(record, sha):
    _checked_media(record, sha)
    return record['mime_type'], record['bytes'], _canonical_sha(record)


def freeze_payload(payload, model):
    """Pure lossless JSON-reference encoding; API payload remains untouched.

    Returns (saved_request, records_by_sha). Only exact inlineData.data values
    become null placeholders; labels, order, MIME, prompts and all parameters
    are retained. Canonical SHA uses UTF-8, sorted keys and compact separators.
    """
    if type(payload) is not dict or not isinstance(model, str) or not model:
        raise ValueError('Request payload/model missing')
    records, refs = {}, []
    def encode(value, path):
        if type(value) is list:
            return [encode(v, path + [i]) for i, v in enumerate(value)]
        if type(value) is not dict:
            return deepcopy(value)
        result = {}
        for key, child in value.items():
            if key != 'inlineData':
                result[key] = encode(child, path + [key])
                continue
            if (type(child) is not dict or not isinstance(child.get('data'), str)
                    or child.get('mimeType') not in _IMAGE_MIMES):
                raise ValueError('Unsupported inlineData in live image payload')
            encoded = child['data']
            if len(encoded) > 4 * ((8_000_000 + 2) // 3):
                raise ValueError('Oversized inline image')
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError('Invalid inline image encoding') from exc
            sha = hashlib.sha256(raw).hexdigest()
            record = {'format': MEDIA_FORMAT, 'sha256': sha, 'mime_type': child['mimeType'],
                      'bytes': len(raw), 'base64': encoded}
            _checked_media(record, sha)
            if sha in records and records[sha] != record:
                raise ValueError('Same image SHA has conflicting media metadata')
            records[sha] = record
            result[key] = deepcopy(child)
            result[key]['data'] = None
            refs.append({'path': path + [key, 'data'], 'sha256': sha,
                         'mime_type': record['mime_type'], 'bytes': record['bytes'],
                         'relative_path': f'media/{sha}.json'})
        return result
    template = encode(payload, [])
    saved = {'format': REQUEST_FORMAT, 'model': model, 'payload': template, 'media_refs': refs,
             'payload_sha256': _canonical_sha(payload),
             'request_sha256': _canonical_sha({'model': model, 'payload': payload})}
    return saved, records


def restore_payload(saved_request, media_records):
    """Rebuild exact network payload from already-loaded records; no file I/O.

    media_records maps SHA -> JSON media record, and may contain other images
    from the same trajectory. Missing, corrupt, wrong-MIME or modified request
    data fail closed. A hash verifies archival integrity, not judge correctness.
    """
    if (type(saved_request) is not dict or set(saved_request) != {
            'format', 'model', 'payload', 'media_refs', 'payload_sha256', 'request_sha256'}
            or saved_request['format'] != REQUEST_FORMAT or type(media_records) is not dict
            or type(saved_request['payload']) is not dict or type(saved_request['media_refs']) is not list):
        raise ValueError('Invalid saved request reference schema')
    payload = deepcopy(saved_request['payload'])
    seen, checked = set(), {}
    for ref in saved_request['media_refs']:
        if type(ref) is not dict or set(ref) != {'path', 'sha256', 'mime_type', 'bytes', 'relative_path'}:
            raise ValueError('Invalid media reference')
        path, sha = ref['path'], ref['sha256']
        if (type(path) is not list or len(path) < 2 or path[-2:] != ['inlineData', 'data']
                or any(type(p) not in (str, int) for p in path) or tuple(path) in seen
                or not isinstance(sha, str) or not _SHA256.fullmatch(sha)
                or ref['relative_path'] != f'media/{sha}.json'):
            raise ValueError('Invalid/repeated inline media reference path')
        seen.add(tuple(path))
        if sha not in media_records:
            raise ValueError('Required media record missing')
        if sha not in checked:
            checked[sha] = _checked_media(media_records[sha], sha)
        record = checked[sha]
        if ref['mime_type'] != record['mime_type'] or type(ref['bytes']) is not int or ref['bytes'] != record['bytes']:
            raise ValueError('Reference disagrees with verified media metadata')
        parent = payload
        for key in path[:-1]:
            if type(parent) is dict and type(key) is str and key in parent:
                parent = parent[key]
            elif type(parent) is list and type(key) is int and 0 <= key < len(parent):
                parent = parent[key]
            else:
                raise ValueError('Media reference path does not exist')
        if (type(parent) is not dict or parent.get('data', 'missing') is not None
                or parent.get('mimeType') != record['mime_type']):
            raise ValueError('Reference does not identify an empty matching inlineData slot')
        parent['data'] = record['base64']
    if (_canonical_sha(payload) != saved_request['payload_sha256']
            or _canonical_sha({'model': saved_request['model'], 'payload': payload}) != saved_request['request_sha256']):
        raise ValueError('Restored request canonical hash mismatch')
    return payload


class ProviderFailure(Exception):
    def __init__(self, kind, *, retryable=True, response=None):
        super().__init__(kind)
        self.kind, self.retryable, self.response = kind, retryable, response


async def _bounded_body(content, limit=2_000_000):
    # StreamReader.read(n) may return only the first available network chunk;
    # it is NOT readexactly(n) or read-to-EOF. Preserve the complete reply while
    # bounding accumulation and remaining under the enclosing aiohttp timeout.
    chunks, size = [], 0
    while True:
        chunk = await content.read(min(65536, limit + 1 - size))
        if not chunk:
            return b''.join(chunks)
        size += len(chunk)
        if size > limit:
            raise ProviderFailure('oversized_provider_response')
        chunks.append(chunk)


async def gemini_request(payload, config):
    """Header-only credential, hard aiohttp total deadline; cancellation closes IO."""
    import aiohttp
    key = os.environ.get('GOOGLE_AI_STUDIO_API_KEY')
    if not key:
        raise ProviderFailure('missing_provider_secret', retryable=False)
    url = 'https://generativelanguage.googleapis.com/v1beta/models/' + config['model'] + ':generateContent'
    timeout = aiohttp.ClientTimeout(total=config['request_timeout_s'], connect=15)
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async with client.post(url, headers={'x-goog-api-key': key}, json=payload) as response:
            # Keep complete bounded replies, not exception representations that
            # might include authentication headers or request objects.
            body = await _bounded_body(response.content)
            try:
                value = json.loads(body)
            except (ValueError, UnicodeError):
                value = {'unparsed_body': body.decode('utf8', errors='replace')}
            if response.status != 200:
                raise ProviderFailure('http_' + str(response.status),
                    retryable=response.status in (408, 429) or response.status >= 500, response=value)
            return value


def payload_for(item):
    payload = transport_v3.payload_for(item['packet'], item['context'], item['image_parts'], 'v3')
    payload['systemInstruction'] = {'parts': [{'text': judge_v5.SYSTEM_PROMPT}]}
    payload['generationConfig']['responseJsonSchema'] = judge_v5.response_schema(len(item['packet']['targets']))
    return payload


def parse_response(response, item):
    if not isinstance(response, dict):
        raise ProviderFailure('malformed_provider_envelope')
    feedback = response.get('promptFeedback') or {}
    if feedback.get('blockReason'):
        raise ProviderFailure('provider_refusal', retryable=False, response=response)
    candidates = response.get('candidates') or []
    if len(candidates) != 1:
        raise ProviderFailure('missing_or_multiple_candidates', response=response)
    candidate = candidates[0]
    reason = candidate.get('finishReason')
    if reason in ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'RECITATION', 'SPII'):
        raise ProviderFailure('provider_refusal', retryable=False, response=response)
    if reason != 'STOP':
        raise ProviderFailure('incomplete_provider_output', response=response)
    raw = ''.join(p.get('text', '') for p in candidate.get('content', {}).get('parts', [])
                  if not p.get('thought'))
    rich, errors, normalization = judge_v5.parse_wire_rows(raw, item['packet'], item['context'])
    # Do not manufacture neutral/abstention rows for invalid decisions. Retain
    # positional None slots so scoring/metrics distinguish protocol failure.
    canonical = {'assessments': [None if row is None else
        judge_v5.canonical_result({'assessments': [row]}, item['context'])['assessments'][0]
        for row in rich['assessments']]}
    return {'typed': rich, 'canonical': canonical, 'normalization': normalization,
            'row_errors': errors}


class OnlineJudge:
    """One shared admission/rate gate per generator, never one gate per sample."""
    def __init__(self, config, *, save, request=None):
        self.config = validate(config)
        self.save = save
        self.request = request or gemini_request
        self.gate = asyncio.Semaphore(config['concurrency'])
        self.pacer = asyncio.Lock()
        self.next_request = 0.
        self.pending = 0
        self.sessions = {}

    def open(self, *, attempt_id, task_text, task_id, stage, root):
        if attempt_id in self.sessions:
            raise ValueError('Duplicate process trajectory identity')
        session = TrajectoryJudge(self, attempt_id, task_text, task_id, stage, root)
        self.sessions[attempt_id] = session
        return session

    async def result(self, attempt_id):
        session = self.sessions[attempt_id]
        return await session.result()

    async def release(self, attempt_id):
        session = self.sessions.pop(attempt_id, None)
        if session is not None:
            await session.cancel()

    async def _pace(self):
        async with self.pacer:
            now = time.monotonic()
            delay = self.next_request - now
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_request = time.monotonic() + 1. / self.config['requests_per_second']

    async def _vote(self, item, payload, root, vote):
        queue_total_s = 0.
        for attempt in range(self.config['max_attempts']):
            request_started = None
            record = {'vote': vote, 'attempt': attempt, 'started_at': time.time(),
                      'status': 'pending', 'response': None}
            path = root / f'vote_{vote}_attempt_{attempt}.json'
            await self.save(path, record)
            queue_started = time.monotonic()  # Gate/pacer only; audit I/O is in window time.
            try:
                async with self.gate:
                    await self._pace()
                    record['request_started_at'] = time.time()
                    request_started = time.monotonic()
                    record['queue_s'] = request_started - queue_started
                    queue_total_s += record['queue_s']
                    response = await asyncio.wait_for(self.request(deepcopy(payload), deepcopy(self.config)),
                                                       self.config['request_timeout_s'])
                    record['request_latency_s'] = time.monotonic() - request_started
                record['response'] = response
                try:
                    parsed = parse_response(response, item)
                except judge_v5.judge.ProtocolError as exc:
                    # Parser diagnostics originate only from the already
                    # archived evidence/reply, never HTTP exception objects
                    # which may contain credential-bearing request details.
                    record['validation_error'] = str(exc)[:512]
                    raise
                parsed['_transport'] = {'attempts': attempt + 1,
                    'request_latency_s': record['request_latency_s'],
                    'queue_s': queue_total_s,
                    'usage': deepcopy(response.get('usageMetadata') or {})}
                # A row-local error is not an API failure. Keep this independent
                # vote's original valid rows; do not resample to cherry-pick them.
                record.update(status='partial' if parsed['row_errors'] else 'valid', parsed=parsed)
                return parsed
            except asyncio.CancelledError:
                record['status'] = 'cancelled_or_deadline'
                raise
            except ProviderFailure as exc:
                record.update(status='failed', error=exc.kind)
                if exc.response is not None:
                    record['response'] = exc.response
                if not exc.retryable:
                    return None
            except Exception as exc:
                # Validation, timeout and transport failure: one immediate
                # retry with the identical immutable evidence, not more context.
                record.update(status='failed', error=type(exc).__name__)
            finally:
                if request_started is not None and 'request_latency_s' not in record:
                    record['request_latency_s'] = time.monotonic() - request_started
                record['finished_at'] = time.time()
                await self.save(path, record)
        return None

    async def window(self, session, turns, start, end, *, execution_receipts=(), harness_notices=()):
        from .live_packet_v5 import prepare
        started, started_at = time.monotonic(), time.time()
        root = session.root / f'window_{start:03d}_{end:03d}'
        fallback = {'start_turn': start, 'end_turn': end, 'scores': [0.] * (end-start+1),
                    'valid': [False] * (end-start+1), 'status': 'unavailable', 'started_at': started_at}
        jobs = []
        try:
            async def work():
                item = await asyncio.to_thread(prepare, task_text=session.task_text,
                    attempt_id=session.attempt_id, task_id=session.task_id, stage=session.stage,
                    turns=turns, start_turn=start, end_turn=end,
                    execution_receipts=list(execution_receipts), harness_notices=harness_notices)
                payload = payload_for(item)
                # Original image bytes occur once per trajectory, not again in
                # both evidence/request or in every later history window. The
                # exact API payload remains unchanged and is hash-restorable.
                saved_request, media_records = await asyncio.to_thread(freeze_payload, payload, self.config['model'])
                await session.save_media(media_records)
                if payload['contents'][0]['parts'][1:] != item['image_parts']:
                    raise ValueError('Image reconstruction slice differs from original payload')
                saved_evidence = {k: v for k, v in item.items() if k != 'image_parts'}
                saved_evidence['media_refs'] = saved_request['media_refs']
                saved_evidence['image_parts_source'] = {
                    'request_file': 'request.json', 'restore': 'restore_payload',
                    'payload_path': ['contents', 0, 'parts'], 'slice_start': 1,
                    'payload_sha256': saved_request['payload_sha256']}
                await self.save(root / 'evidence.json', saved_evidence)
                await self.save(root / 'request.json', saved_request)
                jobs.extend(asyncio.create_task(self._vote(item, payload, root, i)) for i in range(3))
                votes = await asyncio.gather(*jobs)
                scores, valid = [], []
                for offset in range(end-start+1):
                    values = [v['canonical']['assessments'][offset]['score']
                              if v and v['canonical']['assessments'][offset] is not None else None
                              for v in votes]
                    usable = all(type(x) in (int, float) for x in values)
                    scores.append(sum(values)/3. if usable else 0.)
                    valid.append(usable)
                return {**fallback, 'status': 'complete', 'scores': scores, 'valid': valid,
                        'packet_id': item['packet']['packet_id'], 'votes': votes}
            value = await asyncio.wait_for(work(), self.config['window_timeout_s'])
        except asyncio.CancelledError:
            fallback['status'] = 'cancelled_or_deadline'
            fallback.update(finished_at=time.time(), elapsed_s=time.monotonic() - started)
            await self.save(root / 'result.json', fallback)
            raise
        except Exception as exc:
            value = {**fallback, 'error': type(exc).__name__}
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
        value.update(finished_at=time.time(), elapsed_s=time.monotonic() - started)
        await self.save(root / 'result.json', value)
        return value


class TrajectoryJudge:
    def __init__(self, owner, attempt_id, task_text, task_id, stage, root):
        self.owner, self.attempt_id, self.task_text = owner, attempt_id, task_text
        self.task_id, self.stage, self.root = task_id, stage, Path(root) / 'process_reward'
        self.turns, self.windows, self.jobs, self.request_ids = [], [], [], []
        self.finished, self.error, self.final_job = False, None, None
        self.scheduled = {}
        self.media_lock = asyncio.Lock()
        self.saved_media = {}
        self.execution_receipts = []
        self.harness_notices = []

    async def save_media(self, records):
        """One successful save per SHA per trajectory, serialized across windows.

        Mark only after the durable callback succeeds. A failed/uncertain save
        may be retried with the exact same content, never skipped as complete.
        The cache stores small hashes/metadata, not another copy of base64.
        """
        async with self.media_lock:
            for sha, record in records.items():
                identity = await asyncio.to_thread(_media_identity, record, sha)
                previous = self.saved_media.get(sha)
                if previous is not None:
                    if previous != identity:
                        raise ValueError('Content-addressed image identity changed between windows')
                    continue
                await self.owner.save(self.root / 'media' / (sha + '.json'), record)
                self.saved_media[sha] = identity

    def record_harness_notice(self, notice):
        from append_agent_rl.budget_notice import checked_notices
        if self.error:
            return
        if self.finished or type(notice) is not dict or notice.get("before_turn") != len(self.turns) + 1:
            self.error = "process_harness_notice_identity_mismatch"
            return
        try:
            checked = checked_notices([*self.harness_notices, notice], end_turn=len(self.turns) + 1)
            self.harness_notices = checked
        except Exception:
            self.error = "process_harness_notice_invalid"

    def bind_task(self, task_text):
        if not isinstance(task_text, str) or not task_text.strip() or self.turns:
            raise ValueError('Actual renderer task must bind before the first PRM turn')
        if self.task_text is not None and self.task_text != task_text:
            raise ValueError('PRM task differs from the actual model task')
        self.task_text = task_text

    async def observe(self, *, turn, request_id, action, observation):
        """Called after the same native tool result; never changes model context."""
        if self.finished or turn != len(self.turns)+1 or request_id in self.request_ids:
            self.error = 'process_turn_identity_mismatch'
            return
        from .live_packet_v5 import action_text, observation_messages, native_execution_receipts
        try:
            action_text(action)  # Validate, but preserve the original native view.
            # prepare() strips reasoning once from original assistant content.
            # Re-normalizing serialized tool XML would reinterpret a literal
            # '</think>' inside a command argument as an assistant delimiter.
            row = {'turn': turn, 'action': deepcopy(action),
                   'messages': observation_messages(observation) if observation is not None else []}
            receipts = native_execution_receipts(observation, turn=turn) if observation is not None else []
            self.turns.append(row)
            self.execution_receipts.extend(receipts)
            self.request_ids.append(request_id)
            if turn % self.owner.config['window'] == 0:
                self._submit(turn - self.owner.config['window'] + 1, turn)
        except Exception as exc:
            self.error = type(exc).__name__

    def _submit(self, start, end):
        if self.error or self.owner.pending >= self.owner.config['max_pending_windows']:
            self.windows.append({'start_turn': start, 'end_turn': end,
                'scores': [0.] * (end-start+1), 'valid': [False] * (end-start+1),
                'status': self.error or 'admission_overflow'})
            return
        self.owner.pending += 1
        # Only the immutable prefix through end enters this task; no future
        # turn is appended to its list. The callback never mutates old rows.
        kwargs = {"execution_receipts": deepcopy(self.execution_receipts)}
        if self.harness_notices:
            from append_agent_rl.budget_notice import checked_notices
            kwargs["harness_notices"] = checked_notices(self.harness_notices, end_turn=end)
        job = asyncio.create_task(self.owner.window(self, list(self.turns[:end]), start, end, **kwargs))
        self.jobs.append(job)
        self.scheduled[job] = (start, end)
        # A task can be cancelled before its coroutine first runs; release
        # admission from its done callback, not from coroutine finally alone.
        def released(_):
            self.owner.pending -= 1
        job.add_done_callback(released)

    def finish(self, request_ids):
        if self.finished:
            raise ValueError('Duplicate PRM finalization')
        self.finished = True
        expected = list(request_ids)
        if expected != self.request_ids:
            self.error = 'process_request_alignment_mismatch'
        self.expected_ids = expected
        n = len(self.turns)
        if n % self.owner.config['window']:
            self._submit(n - n % self.owner.config['window'] + 1, n)
        self.final_job = asyncio.create_task(self._finish())

    async def _finish(self):
        try:
            results = await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True),
                                             self.owner.config['finish_timeout_s'])
        except asyncio.TimeoutError:
            results = [job.result() if job.done() and not job.cancelled() and job.exception() is None
                       else None for job in self.jobs]
        n = len(self.expected_ids)
        scores, valid = [0.] * n, [False] * n
        windows = list(self.windows)
        for job, result in zip(self.jobs, results):
            if isinstance(result, dict):
                windows.append(result)
            else:
                start, end = self.scheduled[job]
                windows.append({'start_turn': start, 'end_turn': end,
                    'scores': [0.] * (end-start+1), 'valid': [False] * (end-start+1),
                    'status': 'cancelled_or_deadline'})
        for window in windows:
            # Includes overflow and a coroutine cancelled before first entry,
            # whose window() finally block could never persist a result.
            await self.owner.save(self.root / f"window_{window['start_turn']:03d}_{window['end_turn']:03d}" / 'result.json', window)
            lo, hi = window['start_turn']-1, window['end_turn']
            if not self.error and 0 <= lo < hi <= n:
                scores[lo:hi], valid[lo:hi] = window['scores'], window['valid']
        value = {'protocol': judge_v5.PROTOCOL_VERSION, 'config': self.owner.config,
            'attempt_id': self.attempt_id, 'request_ids': self.expected_ids,
            'scores': scores, 'valid': valid, 'windows': sorted(windows, key=lambda w: w['start_turn']),
            'error': self.error, 'valid_turns': sum(valid), 'total_turns': n,
            'fallback': 'zero_process_only_keep_outcome',
            'prompt_sha256': hashlib.sha256(judge_v5.SYSTEM_PROMPT.encode()).hexdigest()}
        if self.harness_notices:
            from append_agent_rl.budget_notice import checked_notices
            value["harness_budget_notices"] = checked_notices(self.harness_notices, end_turn=n)
        await self.owner.save(self.root / 'result.json', value)
        return value

    async def result(self):
        if self.final_job is None:
            raise ValueError('PRM result requested before rollout completion')
        return await self.final_job

    async def cancel(self):
        jobs = self.jobs + ([self.final_job] if self.final_job else [])
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        # An early-cancelled coroutine may have executed no save/finally block.
        # Retain a terminal record for every admitted window even then. Do not
        # overwrite successfully completed results during normal release.
        for job in self.jobs:
            if job.cancelled() or job.exception() is not None:
                start, end = self.scheduled[job]
                await self.owner.save(self.root / f'window_{start:03d}_{end:03d}' / 'result.json', {
                    'start_turn': start, 'end_turn': end, 'scores': [0.] * (end-start+1),
                    'valid': [False] * (end-start+1), 'status': 'cancelled_or_deadline'})
