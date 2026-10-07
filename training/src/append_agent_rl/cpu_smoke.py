"""Real tool-only binary, MCP, media gate and broker; no LLM/GPU involved."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

from opencode_rl.contract import canonical, digest
from opencode_rl.runtime import task_prompt
from opencode_rl.store import TraceStore, atomic_bytes, atomic_json
from opencode_rl.editing_tools import EditingTools
from opencode_rl.music_observation import policy_sha256
from .contract import AppendContract, configuration, loop_source_sha256
from .controller import ToolSession


async def main():
    from PIL import Image
    if os.geteuid() != 0 or not Path('/opt/OCRL_SANDBOX_IMAGE').is_file():
        raise RuntimeError('Fresh isolated CPU qualification sandbox only')
    attempt = uuid.uuid4().hex
    root = Path('/root/append_smoke') / attempt
    root.mkdir(parents=True)
    for directory in ('/work/initial_state/clips', '/work/frames', '/work/output'):
        Path(directory).mkdir(parents=True, exist_ok=True)
    os.chmod('/work', 0o777)
    os.chmod('/work/output', 0o777)
    Image.new('RGB', (896, 896), 'blue').save('/work/frames/test.png')
    EditingTools.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=10:duration=2',
        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-map', '0:v', '-map', '1:a',
        '-c:v', 'ffv1', '-c:a', 'pcm_s16le', '/work/initial_state/clips/c01.mkv'])
    catalog = {'policy_sha256': policy_sha256(), 'assets': [{'music_id': 'music_0',
        'public_path': 'initial_state/music.wav', 'source_sha256': 'a'*64, 'windows': [
            {'index': 0, 'start_s': 0, 'duration_s': 15, 'labels': {'music_presence': 'present',
                'style': 'orchestral', 'vocals': 'none', 'tempo': 'fast', 'pulse': 'steady',
                'dynamics': 'building', 'instruments': ['brass', 'strings'], 'mood': ['tense'],
                'energy': ['medium', 'high', 'high']}}]}]}
    config = configuration(context=131072, output=16384, temperature=1., top_p=1., thinking=True, music=True)
    contract = AppendContract(**{key: '1'*64 for key in ('tokenizer_sha256', 'template_sha256',
        'processor_sha256', 'tools_sha256', 'sandbox_sha256', 'integration_sha256')},
        binary_sha256=hashlib.sha256(Path('/usr/local/bin/opencode-tools').read_bytes()).hexdigest(),
        config_sha256=digest(config), loop_sha256=loop_source_sha256(), max_context=131072, max_output=16384,
        thinking=True, temperature=1., top_p=1., top_k=-1, image_token_id=99, vision_patch_dim=6, spatial_merge_size=2)
    for name, value in [('contract.json', contract.to_dict()), ('config.json', config), ('music_catalog.json', catalog)]:
        atomic_json(root / name, value)
    atomic_bytes(root / 'task.txt', task_prompt('Inspect the provided local video and audio, then finish.').encode())
    journal = TraceStore(root / 'trace').new_request({'attempt_id': attempt, 'kind': 'real_tools_cpu_qualification'}, {})
    process = await asyncio.create_subprocess_exec('/usr/local/bin/python', '-m', 'append_agent_rl.broker',
        '--contract', str(root / 'contract.json'), '--config', str(root / 'config.json'),
        '--task', str(root / 'task.txt'), '--music-catalog', str(root / 'music_catalog.json'),
        '--attempt', attempt, '--stage', 's_4e', cwd='/work', env={**os.environ, 'PYTHONPATH': '/opt'},
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=100_000_000)
    adapter = SimpleNamespace(stdout=process.stdout, stdin=SimpleNamespace(write=process.stdin.write,
        write_eof=process.stdin.write_eof, drain=SimpleNamespace(aio=process.stdin.drain)),
        wait=SimpleNamespace(aio=process.wait))
    # Expose the subprocess's dynamic returncode like Modal's exec object.
    class ProcessAdapter:
        stdout, stdin, wait = adapter.stdout, adapter.stdin, adapter.wait
        @property
        def returncode(self): return process.returncode
    tools = ToolSession(ProcessAdapter(), journal)
    stderr = asyncio.create_task(process.stderr.read())
    checks, outputs, error = {}, [], None
    try:
        async with asyncio.timeout(240):
            profile = await tools.profile
            required = {'bash', 'read', 'write', 'edit', 'glob', 'grep', 'todowrite',
                's12_view_image', 's12_inspect_video', 's12_inspect_audio', 's12_contact_sheet',
                's12_inspect_frames', 's12_analyze_audio', 's12_validate_media', 's12_listen_music'}
            checks['native_and_media_catalogue'] = required <= {t['function']['name'] for t in profile['tools']}
            checks['tool_host_not_agent_loop'] = profile['native_agent_loop'] is False
            actions = [
                ('read', {'filePath': '/work/frames/test.png'}),
                ('s12_view_image', {'paths': ['frames/test.png']}),
                ('s12_contact_sheet', {'items': [{'path': 'initial_state/clips/c01.mkv', 'timestamp_s': .5},
                                               {'path': 'frames/test.png'}]}),
                ('s12_inspect_frames', {'path': 'initial_state/clips/c01.mkv', 'frame_indices': [5, 6], 'layout': 'individual'}),
                ('s12_inspect_video', {'path': 'initial_state/clips/c01.mkv', 'timestamps_s': [1.99, 0]}),
                ('s12_analyze_audio', {'path': 'initial_state/clips/c01.mkv', 'duration_s': 2}),
                ('s12_inspect_audio', {'path': 'initial_state/clips/c01.mkv', 'duration_s': 2, 'transcribe': False}),
                ('s12_validate_media', {'path': 'initial_state/clips/c01.mkv', 'expected': {'duration_s': 2, 'has_audio': True}}),
                ('s12_listen_music', {'music_id': 'music_0', 'window_index': 0}),
                ('bash', {'command': 'test ! -w "$OCRL_MUSIC_CATALOG" && test -z "$GOOGLE_AI_STUDIO_API_KEY" && printf APPEND_ISOLATED_OK',
                          'description': 'Verify read-only music and no service key'}),
                ('webfetch', {'url': 'http://127.0.0.1:1'}),
                ('read', {}),
            ]
            for i, (name, arguments) in enumerate(actions):
                output = await tools.step({'text': 'scripted qualification, not a policy trajectory',
                    'calls': [{'id': f'check-{i}', 'name': name, 'arguments': arguments}]})
                outputs.append(output['results'][0])
            checks['five_image_tools_return_normalized_images'] = all(
                row['attachments'] and all(p['mime'] == 'image/jpeg' for p in row['attachments']) for row in outputs[:5])
            checks['audio_analysis'] = 'integrated_lufs' in outputs[5]['output']
            checks['audio_inspection'] = not outputs[6]['is_error'] and 'duration' in outputs[6]['output']
            checks['media_validation'] = 'decode_ok' in outputs[7]['output']
            checks['read_only_music_labels'] = all(s in outputs[8]['output'] for s in ('precomputed_music_labels', 'brass'))
            checks['sandbox_isolation'] = 'APPEND_ISOLATED_OK' in outputs[9]['output']
            checks['disabled_network_tool'] = outputs[10]['is_error']
            checks['bad_arguments_are_tool_errors'] = outputs[11]['is_error']
            await tools.close(successful=True)
            checks['broker_completed'] = process.returncode == 0
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:2000]
        checks['broker_completed'] = False
        if process.returncode is None:
            process.terminate()
        tools.reader.cancel()
        await asyncio.gather(tools.reader, return_exceptions=True)
    finally:
        await process.wait()
        atomic_bytes(root / 'stderr.log', await stderr)
        atomic_json(root / 'tool_outputs.json', outputs)
        journal.seal('complete' if all(checks.values()) and error is None else 'failed')
    result = {'passed': all(checks.values()) and error is None, 'checks': checks, 'error': error,
        'path': str(root), 'runtime_attempt': attempt, 'gpu_used': False, 'model_calls': 0,
        'training_parity_proven': False, 'synthetic_music_catalog': True, 'binary_sha256': contract.binary_sha256}
    atomic_json(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
