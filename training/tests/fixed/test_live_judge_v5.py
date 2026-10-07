"""Deterministic asyncio integration tests: fake provider and in-memory saves.

No credentials/API/files/Modal/training. These test transport/protocol timing,
not semantic judge quality or learning improvement.
"""
import asyncio
import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prm_v2 import live_config_v5, live_judge_v5 as live


def inline_image():
    from PIL import Image
    output = BytesIO()
    Image.new('RGB', (2, 2), 'red').save(output, format='PNG')
    return {'mimeType': 'image/png', 'data': base64.b64encode(output.getvalue()).decode()}


class PayloadArchiveTests(unittest.TestCase):
    def payload(self):
        image = inline_image()
        return {'contents': [{'role': 'user', 'parts': [{'text': 'original text 中文'},
            {'inlineData': image}, {'text': 'same bytes, different evidence label'},
            {'inlineData': deepcopy(image)}]}],
            'systemInstruction': {'parts': [{'text': 'original system'}]},
            'generationConfig': {'temperature': 0, 'maxOutputTokens': 8192, 'responseMimeType': 'application/json'}}

    def test_exact_roundtrip_preserves_all_parameters_and_duplicate_image_positions(self):
        payload = self.payload()
        original = deepcopy(payload)
        saved, records = live.freeze_payload(payload, 'gemini-3.8-flash')
        before_saved, before_records = deepcopy(saved), deepcopy(records)
        self.assertEqual(payload, original)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(saved['media_refs']), 2)
        self.assertIsNone(saved['payload']['contents'][0]['parts'][1]['inlineData']['data'])
        self.assertEqual(live.restore_payload(saved, records), original)
        self.assertEqual(saved, before_saved)
        self.assertEqual(records, before_records)
        self.assertNotIn(inline_image()['data'], json.dumps(saved))

    def test_text_only_payload_roundtrips_without_media(self):
        payload = {'contents': [{'role': 'user', 'parts': [{'text': 'no images'}]}], 'generationConfig': {'temperature': 0}}
        saved, records = live.freeze_payload(payload, 'gemini-3.8-flash')
        self.assertEqual(records, {})
        self.assertEqual(saved['media_refs'], [])
        self.assertEqual(live.restore_payload(saved, records), payload)

    def test_missing_corrupt_mime_and_modified_request_fail_closed(self):
        saved, records = live.freeze_payload(self.payload(), 'gemini-3.8-flash')
        sha = next(iter(records))
        with self.assertRaises(ValueError):
            live.restore_payload(saved, {})
        for key, value in [('base64', base64.b64encode(b'corruption').decode()), ('bytes', 1),
                           ('mime_type', 'image/jpeg'), ('sha256', '0' * 64)]:
            changed = deepcopy(records)
            changed[sha][key] = value
            with self.assertRaises(ValueError):
                live.restore_payload(saved, changed)
        for change in ('temperature', 'text', 'model', 'repeated_ref', 'path', 'traversal'):
            modified = deepcopy(saved)
            if change == 'temperature':
                modified['payload']['generationConfig']['temperature'] = 1
            elif change == 'text':
                modified['payload']['contents'][0]['parts'][0]['text'] = 'changed'
            elif change == 'model':
                modified['model'] = 'another-model'
            elif change == 'repeated_ref':
                modified['media_refs'].append(deepcopy(modified['media_refs'][0]))
            elif change == 'path':
                modified['media_refs'][0]['path'][1] = -1
            else:
                modified['media_refs'][0]['relative_path'] = '../another.json'
            with self.assertRaises(ValueError):
                live.restore_payload(modified, records)


def packet_from_payload(payload):
    text = payload['contents'][0]['parts'][0]['text']
    prefix = 'UNTRUSTED EVIDENCE PACKET\n'
    return json.JSONDecoder().raw_decode(text[len(prefix):])[0]


def response_for(payload, *, scores=None):
    packet = packet_from_payload(payload)
    rows = []
    for row in packet['targets']:
        n = row['turn']
        score = scores.get(n, 1) if scores is not None else 1
        obs = row['evidence_ids']
        event, scope = ('new_information', 'metadata') if score == 1 else ('routine_io', 'io')
        if score is None:
            event, scope = 'insufficient_evidence', 'unknown'
        rows.append({'turn': n, 'event': event, 'scope': scope, 'score': score,
            'status': 'insufficient_evidence' if score is None else 'scored',
            'prior_evidence_ids': [], 'current_evidence_ids': [f'action.t{n:03d}'] + obs,
            'support_evidence_ids': obs if score == 1 else [], 'fact_refs': [],
            'limitations': ['semantics_unverified'] if score is None else []})
    return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [
        {'text': json.dumps({'assessments': rows})}]}}]}


class MemoryStore:
    def __init__(self):
        self.values, self.events = {}, []

    async def save(self, path, value):
        self.values[str(path)] = deepcopy(value)
        self.events.append((str(path), deepcopy(value)))


class LiveJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.owners = []
        self.store = MemoryStore()

    async def asyncTearDown(self):
        for owner in self.owners:
            for ident in list(owner.sessions):
                await owner.release(ident)

    def owner(self, request, **overrides):
        cfg = live_config_v5.make_config(request_timeout_s=1., window_timeout_s=3., finish_timeout_s=3.,
                                         requests_per_second=1_000_000., **overrides)
        owner = live.OnlineJudge(cfg, save=self.store.save, request=request)
        self.owners.append(owner)
        return owner

    def session(self, owner, ident='a'):
        return owner.open(attempt_id=ident, task_text='Measure the output.', task_id='task-'+ident,
                          stage='s_4e', root=Path('/in-memory-test') / ident)

    async def observe(self, session, start=1, end=10):
        for turn in range(start, end+1):
            await session.observe(turn=turn, request_id=f'request-{turn}',
                action={'text': 'inspect', 'calls': [{'name': 'bash', 'arguments': {'command': f'probe asset{turn}'}}]},
                observation={'results': [{'id': f'call-{turn}', 'name': 'bash',
                    'output': f'measured_property={turn}', 'attachments': []}]})

    async def settle(self, session):
        session.finish(list(session.request_ids))
        value = await session.result()
        await asyncio.sleep(0)
        return value

    def window_results(self):
        return {k: v for k, v in self.store.values.items() if '/window_' in k and k.endswith('/result.json')}

    async def test_tenth_turn_starts_three_independent_requests_before_rollout_done(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def request(payload, config):
            calls.append(deepcopy(payload))
            if len(calls) == 3:
                entered.set()
            await release.wait()
            return response_for(payload)
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=9)
        self.assertEqual(owner.pending, 0)
        self.assertEqual(calls, [])
        await self.observe(session, start=10, end=10)
        await asyncio.wait_for(entered.wait(), 2.)
        self.assertFalse(session.finished)
        self.assertEqual(owner.pending, 1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[1], calls[2])
        release.set()
        value = await self.settle(session)
        self.assertEqual(value['scores'], [1.] * 10)
        self.assertEqual(value['valid'], [True] * 10)
        self.assertEqual(owner.pending, 0)

    async def test_tail_window_and_last_no_tool_action_are_scored_once(self):
        calls = []
        async def request(payload, config):
            p = packet_from_payload(payload)
            calls.append([t['turn'] for t in p['targets']])
            return response_for(payload, scores={13: 0})
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=12)
        await session.observe(turn=13, request_id='request-13', action={'text': 'Done', 'calls': []}, observation=None)
        value = await self.settle(session)
        self.assertEqual(calls.count(list(range(1, 11))), 3)
        self.assertEqual(calls.count([11, 12, 13]), 3)
        self.assertEqual(value['scores'], [1.] * 12 + [0.])
        self.assertTrue(all(value['valid']))
        self.assertEqual(len(self.window_results()), 2)

    async def test_null_is_invalid_process_but_real_neutral_zero_is_valid(self):
        counter = 0
        async def request(payload, config):
            nonlocal counter
            counter += 1
            return response_for(payload, scores={1: 0, 2: None if counter == 2 else 1})
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=3)
        value = await self.settle(session)
        self.assertEqual(counter, 3)
        self.assertEqual(value['scores'], [0., 0., 1.])
        self.assertEqual(value['valid'], [True, False, True])
        self.assertEqual(value['fallback'], 'zero_process_only_keep_outcome')
        self.assertNotIn('outcome_reward', value)
        self.assertEqual(len(value['windows'][0]['votes']), 3)

    async def test_one_failed_vote_invalidates_window_without_dropping_outcome(self):
        counter = 0
        async def request(payload, config):
            nonlocal counter
            counter += 1
            if counter == 2:
                raise live.ProviderFailure('provider_refusal', retryable=False)
            return response_for(payload)
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=2)
        outcome = {'reward': .73}
        value = await self.settle(session)
        self.assertEqual(outcome, {'reward': .73})
        self.assertEqual(value['scores'], [0., 0.])
        self.assertEqual(value['valid'], [False, False])
        self.assertEqual(counter, 3)
        self.assertEqual(owner.pending, 0)

    async def test_all_failed_requests_exact_two_attempts_each_and_save_replies(self):
        counter = 0
        async def request(payload, config):
            nonlocal counter
            counter += 1
            raise live.ProviderFailure('http_503', response={'error': 'fixture'}, retryable=True)
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=1)
        value = await self.settle(session)
        self.assertEqual(counter, 6)
        self.assertEqual(value['valid'], [False])
        records = [v for k, v in self.store.values.items() if '/vote_' in k]
        self.assertEqual(len(records), 6)
        self.assertTrue(all(v['status'] == 'failed' and v['response'] == {'error': 'fixture'} for v in records))
        self.assertEqual(owner.gate._value, 64)

    async def test_first_schema_failure_retries_identical_payload(self):
        count = 0
        payloads = []
        async def request(payload, config):
            nonlocal count
            count += 1
            payloads.append(deepcopy(payload))
            if count == 1:
                return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{}'}]}}]}
            return response_for(payload)
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=1)
        value = await self.settle(session)
        self.assertEqual(count, 4)
        self.assertTrue(all(p == payloads[0] for p in payloads))
        self.assertEqual(value['valid'], [True])
        vote_metadata = [v['_transport'] for v in value['windows'][0]['votes']]
        self.assertEqual(sorted(v['attempts'] for v in vote_metadata), [1, 1, 2])
        self.assertTrue(all(v['request_latency_s'] >= 0 and v['queue_s'] >= 0 for v in vote_metadata))
        failed = [v for k, v in self.store.values.items() if '/vote_' in k and v['status'] == 'failed']
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['error'], 'ProtocolError')
        self.assertEqual(failed[0]['validation_error'], 'expected only assessments array')

    async def test_transport_exception_text_is_not_added_to_parser_diagnostics(self):
        async def request(payload, config):
            raise RuntimeError('credential-bearing-request-fixture')
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=1)
        value = await self.settle(session)
        self.assertEqual(value['valid'], [False])
        records = [v for k, v in self.store.values.items() if '/vote_' in k]
        self.assertEqual(len(records), 6)
        self.assertTrue(all(v['error'] == 'RuntimeError' and 'validation_error' not in v for v in records))
        self.assertNotIn('credential-bearing-request-fixture', json.dumps(records))

    async def test_payload_mutation_in_one_request_cannot_pollute_other_votes(self):
        seen = []
        async def request(payload, config):
            seen.append(deepcopy(payload))
            response = response_for(payload)
            payload['contents'][0]['parts'][0]['text'] = 'client mutation'
            config['model'] = 'client mutation'
            return response
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=1)
        value = await self.settle(session)
        self.assertEqual(value['valid'], [True])
        self.assertTrue(all(p == seen[0] for p in seen))
        self.assertEqual(owner.config['model'], 'gemini-3.8-flash')

    async def test_shared_gate_never_exceeds_64_across_23_sessions(self):
        active, peak, started = 0, 0, 0
        filled, release = asyncio.Event(), asyncio.Event()
        async def request(payload, config):
            nonlocal active, peak, started
            active += 1
            started += 1
            peak = max(peak, active)
            if active == 64:
                filled.set()
            try:
                await release.wait()
                return response_for(payload)
            finally:
                active -= 1
        owner = self.owner(request)
        sessions = [self.session(owner, str(i)) for i in range(23)]
        for session in sessions:
            await self.observe(session, end=1)
            session.finish(session.request_ids)
        await asyncio.wait_for(filled.wait(), 2.)
        self.assertEqual(started, 64)
        self.assertEqual(peak, 64)
        release.set()
        results = await asyncio.gather(*(s.result() for s in sessions))
        await asyncio.sleep(0)
        self.assertEqual(started, 69)
        self.assertEqual(active, 0)
        self.assertTrue(all(r['valid'] == [True] for r in results))
        self.assertEqual(owner.pending, 0)
        self.assertEqual(owner.gate._value, 64)

    async def test_overflow_is_zero_prm_without_blocking_and_has_window_file(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0
        async def request(payload, config):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return response_for(payload)
        owner = self.owner(request, max_pending_windows=1)
        first, second = self.session(owner, 'first'), self.session(owner, 'second')
        await self.observe(first)
        await asyncio.wait_for(entered.wait(), 2.)
        await self.observe(second)
        second.finish(second.request_ids)
        value = await asyncio.wait_for(second.result(), .5)
        self.assertEqual(value['valid'], [False] * 10)
        self.assertEqual(value['windows'][0]['status'], 'admission_overflow')
        self.assertTrue(any('/second/' in k for k in self.window_results()))
        self.assertEqual(owner.pending, 1)
        release.set()
        await self.settle(first)
        self.assertEqual(calls, 3)
        self.assertEqual(owner.pending, 0)

    async def test_cancel_before_coroutine_entry_releases_pending_and_saves_window(self):
        async def request(payload, config):
            self.fail('Cancelled-before-start window must not call provider')
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session)
        self.assertEqual(owner.pending, 1)
        await session.cancel()
        await asyncio.sleep(0)
        self.assertEqual(owner.pending, 0)
        self.assertTrue(all(job.done() for job in session.jobs))
        self.assertEqual(list(self.window_results().values())[0]['status'], 'cancelled_or_deadline')

    async def test_finish_deadline_cancels_provider_tasks_and_retains_fallback(self):
        entered = asyncio.Event()
        active, cancelled = 0, 0
        async def request(payload, config):
            nonlocal active, cancelled
            active += 1
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled += 1
                raise
            finally:
                active -= 1
        cfg = live_config_v5.make_config(request_timeout_s=1., window_timeout_s=2., finish_timeout_s=.03,
                                         requests_per_second=1_000_000.)
        owner = live.OnlineJudge(cfg, save=self.store.save, request=request)
        self.owners.append(owner)
        session = self.session(owner)
        await self.observe(session)
        await asyncio.wait_for(entered.wait(), 2.)
        value = await self.settle(session)
        self.assertEqual(value['valid'], [False] * 10)
        self.assertEqual(active, 0)
        self.assertEqual(cancelled, 3)
        self.assertEqual(owner.pending, 0)
        self.assertEqual(owner.gate._value, 64)
        self.assertTrue(all(job.done() for job in session.jobs))
        self.assertEqual(list(self.window_results().values())[0]['status'], 'cancelled_or_deadline')

    async def test_request_deadline_retries_then_zero_without_residual_io(self):
        active, calls = 0, 0
        async def request(payload, config):
            nonlocal active, calls
            calls += 1
            active += 1
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1
        cfg = live_config_v5.make_config(request_timeout_s=.02, window_timeout_s=1., finish_timeout_s=1.,
                                         requests_per_second=1_000_000.)
        owner = live.OnlineJudge(cfg, save=self.store.save, request=request)
        self.owners.append(owner)
        session = self.session(owner)
        await self.observe(session, end=1)
        value = await self.settle(session)
        self.assertEqual(value['valid'], [False])
        self.assertEqual(calls, 6)
        self.assertEqual(active, 0)
        self.assertEqual(owner.gate._value, 64)

    async def test_request_alignment_failure_zeroes_process_only(self):
        async def request(payload, config):
            return response_for(payload)
        owner = self.owner(request)
        session = self.session(owner)
        await self.observe(session, end=1)
        session.finish(['different-engine-request'])
        value = await session.result()
        self.assertEqual(value['error'], 'process_request_alignment_mismatch')
        self.assertEqual(value['valid'], [False])
        self.assertEqual(value['fallback'], 'zero_process_only_keep_outcome')

    async def test_original_native_action_not_double_stripped_when_argument_has_think_marker(self):
        command = "printf '</think>'"
        observed = []
        async def request(payload, config):
            observed.append(packet_from_payload(payload)['targets'][0]['action'])
            return response_for(payload, scores={1: 0})
        owner = self.owner(request)
        session = self.session(owner)
        native = {'content': 'private</think>public', 'tool_calls': [
            {'function': {'name': 'bash', 'arguments': json.dumps({'command': command})}}]}
        action = {'text': json.dumps(native), 'calls': []}
        await session.observe(turn=1, request_id='request-1', action=action,
            observation={'results': [{'id': 'call-1', 'name': 'bash', 'output': '</think>', 'attachments': []}]})
        action['text'] = 'caller mutation after observe'
        value = await self.settle(session)
        self.assertEqual(value['valid'], [True])
        self.assertTrue(all(command in text and '<function=bash>' in text for text in observed))
        self.assertTrue(all('private' not in text and 'caller mutation' not in text for text in observed))

    async def test_window_deadline_cancels_votes_and_saves_invalid_result(self):
        active = 0
        async def request(payload, config):
            nonlocal active
            active += 1
            try:
                await asyncio.Event().wait()
            finally:
                active -= 1
        cfg = live_config_v5.make_config(request_timeout_s=.02, window_timeout_s=.03, finish_timeout_s=1.,
                                         requests_per_second=1_000_000.)
        owner = live.OnlineJudge(cfg, save=self.store.save, request=request)
        self.owners.append(owner)
        session = self.session(owner)
        await self.observe(session, end=1)
        value = await self.settle(session)
        self.assertEqual(value['valid'], [False])
        self.assertEqual(active, 0)
        self.assertEqual(owner.pending, 0)
        self.assertEqual(owner.gate._value, 64)
        record = list(self.window_results().values())[0]
        self.assertEqual(record['status'], 'unavailable')
        self.assertEqual(record['error'], 'TimeoutError')

    async def test_bounded_body_reads_all_chunks_and_rejects_oversize(self):
        class Chunks:
            def __init__(self, chunks):
                self.chunks = list(chunks)
            async def read(self, n):
                return self.chunks.pop(0) if self.chunks else b''
        self.assertEqual(await live._bounded_body(Chunks([b'{', b'"x":', b'1}'])), b'{"x":1}')
        with self.assertRaises(live.ProviderFailure) as exc:
            await live._bounded_body(Chunks([b'ab', b'cd']), limit=3)
        self.assertEqual(exc.exception.kind, 'oversized_provider_response')

    async def test_two_concurrent_windows_same_image_save_once_and_requests_rebuild_exactly(self):
        captured, media_saves = [], []
        async def request(payload, config):
            captured.append(deepcopy(payload))
            response = response_for(payload)
            response['usageMetadata'] = {'promptTokenCount': 123, 'candidatesTokenCount': 45}
            return response
        owner = self.owner(request)
        async def save(path, value):
            if '/media/' in str(path):
                media_saves.append(str(path))
                # Yield long enough for the other window to contend on the
                # same SHA, exercising the session-wide lock rather than luck.
                await asyncio.sleep(.015)
            await self.store.save(path, value)
        owner.save = save
        session = self.session(owner)
        image = inline_image()
        for turn in range(1, 21):
            await session.observe(turn=turn, request_id=f'request-{turn}',
                action={'text': 'Inspect returned image', 'calls': []},
                observation={'results': [{'id': f'call-{turn}', 'name': 'read', 'output': f'clip property {turn}',
                    'attachments': [{'mime': image['mimeType'], 'url': 'data:' + image['mimeType'] + ';base64,' + image['data']}]}]})
        value = await self.settle(session)
        self.assertEqual(value['valid'], [True] * 20)
        self.assertEqual(len(captured), 6)
        for window in value['windows']:
            self.assertLessEqual(window['started_at'], window['finished_at'])
            self.assertGreaterEqual(window['elapsed_s'], 0)
            self.assertEqual(window['votes'][0]['_transport']['usage'], {'promptTokenCount': 123, 'candidatesTokenCount': 45})
        self.assertEqual(len(media_saves), 1)
        self.assertEqual(len(session.saved_media), 1)
        media = {v['sha256']: v for k, v in self.store.values.items() if '/media/' in k}
        requests = {k: v for k, v in self.store.values.items() if k.endswith('/request.json')}
        self.assertEqual(len(requests), 2)
        for network_payload in captured:
            targets = packet_from_payload(network_payload)['targets']
            suffix = f"window_{targets[0]['turn']:03d}_{targets[-1]['turn']:03d}/request.json"
            filename = next(k for k in requests if k.endswith(suffix))
            archived = requests[filename]
            restored = live.restore_payload(archived, media)
            self.assertEqual(restored, network_payload)
            evidence = self.store.values[filename.replace('/request.json', '/evidence.json')]
            self.assertNotIn('image_parts', evidence)
            self.assertTrue(evidence['image_receipts'])
            self.assertEqual(evidence['image_parts_source']['payload_sha256'], archived['payload_sha256'])
            self.assertEqual(restored['contents'][0]['parts'][1:], network_payload['contents'][0]['parts'][1:])
            self.assertNotIn(image['data'], json.dumps(archived))
            self.assertNotIn(image['data'], json.dumps(evidence))

    async def test_media_save_failure_is_not_marked_and_successful_retry_is_deduplicated(self):
        async def request(payload, config):
            self.fail('Direct media persistence test must not call provider')
        owner = self.owner(request)
        session = self.session(owner)
        _, records = live.freeze_payload({'contents': [{'parts': [{'inlineData': inline_image()}]}]}, owner.config['model'])
        calls = 0
        async def save(path, value):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError('fixture persistence failure')
            await self.store.save(path, value)
        owner.save = save
        with self.assertRaises(OSError):
            await session.save_media(records)
        self.assertEqual(session.saved_media, {})
        self.assertFalse(session.media_lock.locked())
        await session.save_media(records)
        await session.save_media(records)
        self.assertEqual(calls, 2)
        self.assertEqual(len(session.saved_media), 1)
        self.assertEqual(len(self.store.values), 1)

    async def test_cancelled_media_save_is_unmarked_and_lock_released(self):
        async def request(payload, config):
            self.fail('Direct media persistence test must not call provider')
        owner = self.owner(request)
        session = self.session(owner)
        _, records = live.freeze_payload({'contents': [{'parts': [{'inlineData': inline_image()}]}]}, owner.config['model'])
        entered = asyncio.Event()
        async def save(path, value):
            entered.set()
            await asyncio.Event().wait()
        owner.save = save
        job = asyncio.create_task(session.save_media(records))
        await entered.wait()
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        self.assertEqual(session.saved_media, {})
        self.assertFalse(session.media_lock.locked())
        owner.save = self.store.save
        await session.save_media(records)
        self.assertEqual(len(session.saved_media), 1)
        self.assertEqual(len(self.store.values), 1)


if __name__ == '__main__':
    unittest.main()
