"""Real OpenCode + real MCP + real stdio bridge, scripted policy (NO MODEL).

Only run in the network-blocked dedicated Modal Sandbox. This proves protocol
behavior, never numerical train/inference parity and never authorizes training.
"""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid
import time

from .contract import Contract, canonical, digest
from .relay_parent import forward_runtime
from .runtime import configuration, task_prompt
from .store import TraceStore, atomic_bytes


async def main(*, music=False, editing=False, context_overflow=False):
    from aiohttp import web
    from PIL import Image
    if not Path('/opt/OCRL_SANDBOX_IMAGE').is_file() or os.geteuid() != 0:
        raise RuntimeError('Sandbox only')
    started = time.monotonic()
    attempt = uuid.uuid4().hex
    root = Path('/root/smoke') / attempt
    root.mkdir(parents=True)
    Path('/work/frames').mkdir(parents=True, exist_ok=True)
    Path('/work/output').mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 32), 'red').save('/work/frames/test.jpg')
    editing_calls = []
    if editing:
        from .editing_tools import EditingTools
        Path('/work/initial_state/clips').mkdir(parents=True, exist_ok=True)
        EditingTools.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=10:duration=2',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-map', '0:v', '-map', '1:a',
            '-c:v', 'ffv1', '-c:a', 'pcm_s16le', '/work/initial_state/clips/c01.mkv'])
        Image.new('RGB', (896, 896), 'blue').save('/work/output/read_test.png')
        Image.new('RGB', (1792, 1792), 'green').save('/work/output/big.png')
        atomic_bytes(Path('/work/output/cuts.json'), canonical({'cuts': [{'clip': 'c01', 'in_s': 0, 'out_s': 2}]}))
        editing_calls = [
            ('s12_contact_sheet', {'items': [{'path': 'initial_state/clips/c01.mkv', 'timestamp_s': t}
                for t in (.1, .5, 1, 1.5)] + [{'path': 'frames/test.jpg'}]}),
            ('s12_inspect_frames', {'path': 'initial_state/clips/c01.mkv', 'frame_indices': [5, 6],
                                   'layout': 'individual', 'crop': [.25, .25, .5, .5]}),
            ('read', {'filePath': '/work/output/read_test.png'}),
            ('s12_analyze_audio', {'path': 'initial_state/clips/c01.mkv', 'duration_s': 2}),
            ('s12_validate_media', {'path': 'initial_state/clips/c01.mkv', 'plan_path': 'output/cuts.json',
                                    'expected': {'has_audio': True, 'duration_s': 2}}),
            *[('read', {'filePath': '/work/output/big.png'}) for _ in range(6)],
            ('s12_inspect_video', {'path': 'initial_state/clips/c01.mkv', 'timestamps_s': [99]}),
            ('s12_inspect_video', {'path': 'initial_state/clips/c01.mkv', 'timestamps_s': [1.99, 0]}),
            ('bash', {'command': 'printf OCRL_AFTER_BUDGET_OK', 'description': 'Check tool errors are recoverable'}),
            *[('s12_view_image', {'paths': ['frames/test.jpg'] * 32}) for _ in range(7)],
            *[('read', {'filePath': '/work/output/big.png'}) for _ in range(7)],
        ]
    os.chmod('/work', 0o777)
    os.chmod('/work/output', 0o777)
    config = configuration(context=32768, output=1024, temperature=1, top_p=.95, thinking=True,
                           max_turns=36 if editing else (8 if context_overflow else 5), music=music)
    music_catalog = None
    if music:
        from .music_observation import policy_sha256
        # Synthetic catalogue verifies transport/permissions only, not hearing quality.
        music_catalog = {'policy_sha256': policy_sha256(), 'assets': [{'music_id': 'music_0',
            'public_path': 'initial_state/music.wav', 'source_sha256': 'a' * 64, 'windows': [
                {'index': 0, 'start_s': 0, 'duration_s': 15, 'labels': {
                    'music_presence': 'present', 'style': 'orchestral', 'vocals': 'none', 'tempo': 'fast',
                    'pulse': 'steady', 'dynamics': 'building', 'instruments': ['brass', 'strings'],
                    'mood': ['tense'], 'energy': ['medium', 'high', 'high']}}]}]}
        atomic_bytes(root / 'music_catalog.json', canonical(music_catalog))
    contract = Contract(**{key: '1' * 64 for key in (
        'tokenizer_sha256', 'template_sha256', 'processor_sha256', 'tools_sha256', 'sandbox_sha256', 'integration_sha256')},
        binary_sha256=hashlib.sha256(Path('/usr/local/bin/opencode').read_bytes()).hexdigest(),
        config_sha256=digest(config), max_context=32768, max_output=1024, thinking=True,
        temperature=1, top_p=.95, top_k=-1, image_token_id=99, vision_patch_dim=6, spatial_merge_size=2)
    for name, data in (('contract.json', contract.to_dict()), ('config.json', config)):
        atomic_bytes(root / name, canonical(data))
    atomic_bytes(root / 'task.txt', task_prompt('Inspect frames/test.jpg, run pwd, then answer DONE.').encode())
    store = TraceStore(root / 'trace')
    journal = store.new_request({'attempt_id': attempt, 'kind': 'scripted_protocol_test'}, {})
    requests = []

    async def chat(request):
        body = await request.json()
        headers = {k.lower(): v for k, v in request.headers.items() if k.lower().startswith('x-ocrl-')}
        requests.append({'headers': headers, 'body': body})
        journal.event('scripted_policy_request', requests[-1])
        primary = headers.get('x-ocrl-primary') == '1'
        turn = int(headers.get('x-ocrl-turn', 0))
        if context_overflow and primary and turn == 1:
            # Reproduce the real vLLM HTTP rejection. The unchanged native
            # OpenCode binary must compact and continue this same session.
            return web.json_response({'error': {'message':
                "This model's maximum context length is 131072 tokens. However, you requested 16384 output tokens "
                "and your prompt contains at least 114689 input tokens, for a total of at least 131073 tokens. "
                "Please reduce the length of the input prompt or the number of requested output tokens. "
                "(parameter=input_tokens, value=114689)",
                'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}}, status=400)
        delta = {'role': 'assistant', 'reasoning_content': 'smoke reasoning sentinel'}
        if context_overflow and headers.get('x-ocrl-agent') == 'compaction':
            delta['content'] = 'OCRL_COMPACTION_SUMMARY: image inspected; next run pwd and then answer DONE.'
        elif editing and primary and turn < len(editing_calls):
            name, arguments = editing_calls[turn]
            delta['tool_calls'] = [{'index': 0, 'id': f'editing_{turn}', 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}]
        elif editing:
            delta['content'] = 'DONE'
        elif music and primary and turn == 0:
            delta['tool_calls'] = [{'index': 0, 'id': 'call_music', 'type': 'function', 'function': {
                'name': 's12_listen_music', 'arguments': json.dumps({'music_id': 'music_0', 'window_index': 0})}}]
        elif primary and turn == int(music):
            delta['tool_calls'] = [{'index': 0, 'id': 'call_image', 'type': 'function', 'function': {
                'name': 's12_view_image', 'arguments': json.dumps({'paths': ['frames/test.jpg']})}}]
        elif primary and turn == 1 + int(music) + int(context_overflow):
            delta['tool_calls'] = [{'index': 0, 'id': 'call_bash', 'type': 'function', 'function': {
                'name': 'bash', 'arguments': json.dumps({'command': (
                    'test ! -w "$OCRL_MUSIC_CATALOG" && test -z "$GOOGLE_AI_STUDIO_API_KEY" && pwd'
                    if music else 'pwd'), 'description': 'Check sandbox work directory and read-only music'})}}]
        else:
            delta['content'] = 'DONE'
        finish = 'tool_calls' if 'tool_calls' in delta else 'stop'
        stream = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await stream.prepare(request)
        for d, f in ((delta, None), ({}, finish)):
            obj = {'id': 'chatcmpl-smoke', 'object': 'chat.completion.chunk', 'created': 1,
                   'model': body['model'], 'choices': [{'index': 0, 'delta': d, 'finish_reason': f}],
                   'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}}
            await stream.write(b'data: ' + canonical(obj) + b'\n\n')
        await stream.write(b'data: [DONE]\n\n')
        await stream.write_eof()
        return stream

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post('/v1/chat/completions', chat)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, '127.0.0.1', 18081).start()
    process = await asyncio.create_subprocess_exec('/usr/local/bin/python', '-m', 'opencode_rl.stdio_relay',
        '--contract', str(root / 'contract.json'), '--config', str(root / 'config.json'),
        '--task', str(root / 'task.txt'), '--attempt', attempt, '--stage', 's_4e' if editing else 's_1',
        *(['--music-catalog', str(root / 'music_catalog.json')] if music else []),
        cwd='/work', env={**os.environ, 'PYTHONPATH': '/opt'},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        limit=100_000_000)
    adapter = SimpleNamespace(stdout=process.stdout,
        stdin=SimpleNamespace(write=process.stdin.write, drain=SimpleNamespace(aio=process.stdin.drain)))
    error = None
    try:
        await forward_runtime(adapter, 'http://127.0.0.1:18081', journal,
            attempt_id=attempt, contract_sha256=contract.sha256, timeout=240)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if process.returncode is None:
            process.kill()
    finally:
        await process.wait()
        journal.event('relay_stderr', {'text': (await process.stderr.read()).decode(errors='replace')})
        await runner.cleanup()
    primary = [r for r in requests if r['headers'].get('x-ocrl-primary') == '1']
    def has_image(messages):
        return any(m.get('role') == 'user' and isinstance(m.get('content'), list) and
                   any(p.get('type') == 'image_url' for p in m['content']) for m in messages)
    checks = {
        'real_opencode_expected_primary_turns': len(primary) == 3 + int(music),
        'mcp_image_in_next_context': len(primary) >= 2 + int(music) and has_image(primary[1 + int(music)]['body']['messages']),
        'reasoning_in_next_context': len(primary) >= 2 and 'smoke reasoning sentinel' in json.dumps(primary[1]['body']['messages']),
        'native_bash_output_in_context': len(primary) >= 3 + int(music) and '/work' in json.dumps(primary[2 + int(music)]['body']['messages']),
        'required_media_tools': bool(primary) and {'s12_view_image', 's12_inspect_video', 's12_inspect_audio'} <=
            {t['function']['name'] for t in primary[0]['body'].get('tools', [])},
        'provenance_all_requests': bool(requests) and all(r['headers'].get('x-ocrl-contract') == contract.sha256 for r in requests),
        'runtime_success': error is None and process.returncode == 0,
    }
    if editing:
        def tools_at(turn):
            return [m.get('content', '') for m in primary[turn]['body']['messages'] if m.get('role') == 'tool'] if len(primary) > turn else []
        observations = [json.loads(store.read_blob(e['data']['blob'])) for e in journal.events if e['kind'] == 'runtime_media_observation']
        errors = [e['data']['error'] for e in journal.events if e['kind'] == 'runtime_media_error']
        from .visual_budget import image_bytes
        pixel_sizes = {r['sha256']: r['width'] * r['height'] for o in observations for r in o['records']}
        request_images = [[p['image_url']['url'] for m in r['body']['messages']
            if isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url']
            for r in primary]
        max_images = max((len(urls) for urls in request_images), default=0)
        max_pixels = max((sum(pixel_sizes[hashlib.sha256(image_bytes(url)).hexdigest()] for url in urls)
            for urls in request_images), default=0)
        checks = {
            'runtime_success': error is None and process.returncode == 0,
            'all_editing_tools_exposed': bool(primary) and {'s12_contact_sheet', 's12_inspect_frames', 's12_analyze_audio', 's12_validate_media'} <=
                {t['function']['name'] for t in primary[0]['body'].get('tools', [])},
            'expected_primary_turns': len(primary) == len(editing_calls) + 1,
            'sheet_image_in_actual_next_request': len(primary) > 1 and has_image(primary[1]['body']['messages']) and 'actual_pts_s' in json.dumps(tools_at(1)),
            'precise_frames_in_next_request': 'frame_index' in json.dumps(tools_at(2)) and '0.5' in json.dumps(tools_at(2)),
            'native_read_budgeted': any(any(r['width'] == r['height'] == 896 for r in o['records']) and o['charged_units'] == 4 for o in observations),
            'sheet_source_units_preserved': bool(observations) and observations[0]['charged_units'] >= 5,
            'audio_evidence_in_next_request': 'integrated_lufs' in json.dumps(tools_at(4)),
            'qc_evidence_in_next_request': 'decode_ok' in json.dumps(tools_at(5)),
            'more_than_96_units_allowed': bool(observations) and observations[-1]['spent_units'] > 96
                and all(o['limit_units'] is None for o in observations),
            'no_legacy_episode_quota_error': not any('Visual budget exceeded' in e for e in errors),
            'invalid_video_request_is_reported': 'decoded video duration' in json.dumps(tools_at(12)),
            'tail_frame_reaches_actual_next_request': 'actual_time_s' in json.dumps(tools_at(13)) and '1.9' in json.dumps(tools_at(13)),
            'tool_error_is_recoverable': any('OCRL_AFTER_BUDGET_OK' == m or 'OCRL_AFTER_BUDGET_OK' in str(m) for m in tools_at(14)),
            'actual_request_exceeds_192_images': max_images > 192,
            'actual_request_exceeds_old_context_pixels': max_pixels > 192 * 448 * 448,
            'provenance_all_requests': all(r['headers'].get('x-ocrl-contract') == contract.sha256 for r in requests),
        }
    if music:
        checks['music_tool_exposed'] = bool(primary) and 's12_listen_music' in {
            t['function']['name'] for t in primary[0]['body'].get('tools', [])}
        checks['music_observation_in_next_context'] = len(primary) > 1 and all(
            text in json.dumps(primary[1]['body']['messages']) for text in ('precomputed_music_labels', 'brass', 'building'))
        checks['music_catalog_journaled'] = any(e['kind'] == 'runtime_music_catalog' for e in journal.events)
    if context_overflow:
        recoveries = [e for e in journal.events if e['kind'] == 'native_context_recovered']
        compacted = [r for r in requests if r['headers'].get('x-ocrl-agent') == 'compaction']
        checks.update(real_opencode_expected_primary_turns=len(primary) == 4,
            native_bash_output_in_context=len(primary) == 4 and '/work' in json.dumps(primary[3]['body']['messages']),
            native_compaction_requested=bool(compacted),
            summary_in_native_continuation=len(primary) >= 3 and 'OCRL_COMPACTION_SUMMARY' in json.dumps(primary[2]['body']['messages']),
            rejection_and_recovery_journaled=len(recoveries) == 1 and any(e['kind'] == 'native_context_overflow' for e in journal.events),
            same_native_session=len({r['headers']['x-ocrl-session'] for r in primary}) == 1)
    result = {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
              'error': error, 'scripted_policy': True, 'model_calls': 0, 'gpu_used': False,
              'training_parity_proven': False, 'request_count': len(requests), 'path': str(root),
              'elapsed_s': time.monotonic() - started, 'synthetic_music_catalog': music, 'editing_test': editing,
              'context_overflow_test': context_overflow}
    if editing:
        result.update(max_actual_context_images=max_images, max_actual_context_pixels=max_pixels)
    journal.seal(result['status'])
    atomic_bytes(root / 'result.json', canonical(result))
    print(json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--music', action='store_true')
    parser.add_argument('--editing', action='store_true')
    parser.add_argument('--context-overflow', action='store_true')
    options = parser.parse_args()
    if sum((options.music, options.editing, options.context_overflow)) > 1:
        parser.error('Choose one bounded smoke mode')
    asyncio.run(main(music=options.music, editing=options.editing, context_overflow=options.context_overflow))
