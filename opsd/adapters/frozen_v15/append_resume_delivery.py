"""Hash-bound delivery scheduling around the unchanged frozen append controller.

Only the parent-side final-file/grade dependency changes. Train and validation
use this same owner, original broker/controller, inputs, tools, and output builder.
An early grade is never permission to admit a partially persisted trajectory.
"""
import asyncio
import json
from pathlib import Path
import time
import uuid

from opencode_rl.capacity import durable_io
from opencode_rl.store import TraceStore, atomic_json
from opencode_rl.task_bundle import validate_bundle_prompt
from opencode_rl.task_rollout import upload, helper, download
from append_agent_rl.contract import validate_runtime
from append_agent_rl.controller import run
from append_agent_rl.task_rollout import _capture_stderr

DELIVERY_POLICY = 'grade-inputs-before-optional-preview-v1'


async def rollout_task(*, app, image_id, bundle, destination, config, contract, backend_url,
                       commit, request_gate=None, timeout=7200, on_delivery=None):
    import modal
    validate_runtime(config, contract)
    root, bundle = Path(destination), Path(bundle)
    metadata = json.loads((bundle / 'bundle.json').read_bytes())
    validate_bundle_prompt(bundle, metadata)
    root.mkdir(parents=True, exist_ok=False)
    attempt = uuid.uuid4().hex
    for name, value in [('contract.json', contract.to_dict()), ('config.json', config)]:
        await durable_io(atomic_json, root / name, value)
    await durable_io(atomic_json, root / 'attempt.json', {'attempt_id': attempt, **metadata, 'started': time.time()})
    store = TraceStore(root / 'trace')
    journal = store.new_request({'attempt_id': attempt, 'kind': 'append_task_runtime'}, {})
    loop_journal = store.new_request({'attempt_id': attempt, 'kind': 'append_token_loop'}, {})
    sandbox = stderr_task = None
    policy_stopped = collected_complete = False
    started = time.monotonic()
    status = {'task': metadata['task'], 'stage': metadata['stage'], 'attempt_id': attempt,
              'harness': 'append-opencode-tools-v1', 'contract_sha256': contract.sha256}

    await durable_io(atomic_json, root / 'persistence_policy.json', {
        'format': 'append-core-trajectory-persistence-v1', 'delivery_policy': DELIVERY_POLICY,
        'workspace_archive': False, 'periodic_workspace_snapshot': False,
        'retained': ['tool_host_frames', 'loop_journal', 'engine_tokens_images_logprobs',
                     'append_packet', 'bounded_final_deliverables', 'grading_attempts', 'broker_stderr'],
        'omitted': ['unobserved_workspace_temporary_files', 'sandbox_runtime_database_backup'],
    })

    async def collect_outputs(result=None):
        nonlocal collected_complete
        collected = await helper(sandbox, 'collect', metadata['stage'])
        # Collection follows stop_policy and freezes root-owned immutable bytes.
        await durable_io(atomic_json, root / 'deliverables.json', collected)

        async def transfer(name, info):
            if not info['valid_file']:
                return
            transfer_started = time.time()
            complete = False
            try:
                await download(sandbox, info['path'], root / name, expected=info,
                               lane='control' if name.endswith('.json') else 'terminal')
                complete = True
            finally:
                await durable_io(journal.event, 'final_file_transfer', {
                    'name': name, 'bytes': info['bytes'], 'started_at': transfer_started,
                    'finished_at': time.time(), 'complete': complete})

        async def notify_grading():
            # Cleanup after a failed/cancelled controller must not launch grading.
            if result is None:
                return
            await durable_io(journal.event, 'grading_inputs_ready', {
                'time': time.time(), 'attempt_id': attempt, 'stage': metadata['stage'],
                'requires_preview': metadata['stage'] == 's_4e', 'policy': DELIVERY_POLICY})
            if on_delivery is not None:
                await on_delivery({**status, 'status': 'grading_ready',
                    'runtime': {k: v for k, v in result.items() if k != 'packet'}})

        for name, info in collected.items():
            if name != 'preview.mp4':
                await transfer(name, info)
        if metadata['stage'] != 's_4e':
            await notify_grading()
        if 'preview.mp4' in collected:
            await transfer('preview.mp4', collected['preview.mp4'])
        if metadata['stage'] == 's_4e':
            await notify_grading()
        collected_complete = True

    try:
        for retry in range(8):
            try:
                sandbox = await modal.Sandbox.create.aio(app=app, image=modal.Image.from_id(image_id),
                    block_network=True, cpu=(1, 4), memory=(4096, 8192), timeout=timeout + 900)
                break
            except Exception as exc:
                if retry == 7 or not any(word in str(exc).lower() for word in
                        ('resource_exhausted', 'quota', 'capacity', 'rate limit', 'too many')):
                    raise
                await durable_io(journal.event, 'sandbox_capacity_wait', {'retry': retry + 1, 'error': str(exc)[:500]})
                await commit()
                await asyncio.sleep(min(60, 5 * 2**retry))
        await durable_io(atomic_json, root / 'sandbox.json', {'id': sandbox.object_id, 'image': image_id,
            'block_network': True, 'secret_count': 0, 'volume_count': 0})
        await commit()
        for local, remote in [(bundle / 'input.tar', '/root/input.tar'), (bundle / 'task.txt', '/root/task.txt'),
                (bundle / 'music_catalog.json', '/root/music_catalog.json'),
                (root / 'config.json', '/root/config.json'), (root / 'contract.json', '/root/contract.json')]:
            await upload(sandbox, local, remote)
        await helper(sandbox, 'stage')
        proc = await sandbox.exec.aio('/usr/local/bin/python', '-m', 'append_agent_rl.broker',
            '--contract', '/root/contract.json', '--config', '/root/config.json', '--attempt', attempt,
            '--task', '/root/task.txt', '--stage', metadata['stage'], '--music-catalog', '/root/music_catalog.json',
            workdir='/work', env={'PYTHONPATH': '/opt'}, timeout=timeout)
        stderr_task = asyncio.create_task(_capture_stderr(proc.stderr, root / 'broker_stderr.log'))
        async with asyncio.timeout(timeout):
            result = await run(proc, backend_url, journal, loop_journal, attempt_id=attempt,
                               contract=contract, request_gate=request_gate)
        await durable_io(atomic_json, root / 'append_packet.json', result)
        await helper(sandbox, 'stop_policy')
        policy_stopped = True
        await collect_outputs(result)
        status.update(status='rollout_complete', runtime={k: v for k, v in result.items() if k != 'packet'})
        await durable_io(journal.event, 'delivery_ready', {'time': time.time(), 'attempt_id': attempt})
    except asyncio.CancelledError:
        status.update(status='cancelled', error_type='CancelledError', error='Administrative cancellation; never model zero')
        await durable_io(journal.event, 'attempt_cancelled', dict(status))
        raise
    except Exception as exc:
        status.update(status='infrastructure_error', error_type=type(exc).__name__, error=str(exc)[:1500])
        await durable_io(journal.event, 'attempt_error', dict(status))
    finally:
        try:
            if sandbox is not None:
                if not policy_stopped:
                    await helper(sandbox, 'stop_policy')
                    policy_stopped = True
                if not collected_complete:
                    # No new grade during recovery. Retain the same final bytes.
                    await collect_outputs()
        except Exception as exc:
            status['cleanup_error'] = type(exc).__name__ + ': ' + str(exc)[:500]
            if status.get('status') == 'rollout_complete':
                status['status'] = 'infrastructure_error'
        if stderr_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(stderr_task), timeout=15)
            except Exception as exc:
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
                status['stderr_error'] = type(exc).__name__ + ': ' + str(exc)[:500]
                if status.get('status') == 'rollout_complete':
                    status['status'] = 'infrastructure_error'
        retain_sandbox = sandbox is not None and not collected_complete
        if retain_sandbox:
            status['sandbox_retained'] = sandbox.object_id
            status['recovery_required'] = 'Final deliverable collection incomplete; no workspace backup exists'
        if not loop_journal.sealed:
            await durable_io(loop_journal.event, 'setup_or_controller_failure', dict(status))
            await durable_io(loop_journal.seal, 'failed')
        await durable_io(journal.seal, 'complete' if status.get('status') == 'rollout_complete' else 'failed')
        status['seconds'] = time.monotonic() - started
        await durable_io(atomic_json, root / 'result.json', status)
        await commit()
        if sandbox is not None and not retain_sandbox:
            await sandbox.terminate.aio()
    return status
