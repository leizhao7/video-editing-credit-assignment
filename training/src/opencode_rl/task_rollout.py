"""One real OpenCode attempt. Train and evaluation call this exact function."""
import asyncio
import hashlib
import json
import time
from pathlib import Path
from .contract import canonical, ParityError
from .relay_parent import forward_runtime
from .store import TraceStore, atomic_bytes, atomic_json
from .task_bundle import validate_bundle_prompt
from .artifact_capacity import artifact_capacity
from .capacity import durable_io, durable_file


async def upload(sandbox, local, remote):
    # This workspace's Modal CLI is 1.2.x. The platform has retired its FileIO
    # RPC; use the supported exec/stdio channel already used by our relay.
    # No SDK replacement, sandbox network access, or shared answer volume.
    local = Path(local)
    size = (await durable_io(local.stat)).st_size
    proc = await sandbox.exec.aio('/usr/local/bin/python', '-m', 'opencode_rl.sandbox_artifacts',
        'receive', remote, '--size', str(size), env={'PYTHONPATH': '/opt'}, timeout=600)
    sha = hashlib.sha256()
    async with durable_file(local, 'rb') as src:
        def read():
            chunk = src.read(1024 * 1024)
            sha.update(chunk)
            return chunk
        while chunk := await durable_io(read):
            proc.stdin.write(chunk)
            await proc.stdin.drain.aio()
    proc.stdin.write_eof()
    await proc.stdin.drain.aio()
    stdout, stderr = await asyncio.gather(proc.stdout.read.aio(), proc.stderr.read.aio())
    await proc.wait.aio()
    if proc.returncode or json.loads(stdout) != {'bytes': (await durable_io(local.stat)).st_size, 'sha256': sha.hexdigest()}:
        raise ParityError('Input transfer failed: ' + stderr[-1000:])


async def _download_stream(sandbox, remote, local, *, expected=None, offset=None, length=None):
    import base64
    proc = await sandbox.exec.aio('/usr/local/bin/python', '-m', 'opencode_rl.sandbox_artifacts',
        'send', remote, *([] if offset is None else ['--offset', str(offset), '--length', str(length)]),
        env={'PYTHONPATH': '/opt'}, timeout=600)
    local = Path(local)
    sha = hashlib.sha256()
    count = 0
    receipt, buffer = None, ''
    async with durable_file(local.with_suffix(local.suffix + '.partial'), 'wb') as dst:
        def write_chunk(encoded):
            chunk = base64.b64decode(encoded, validate=True)
            sha.update(chunk)
            dst.write(chunk)
            return len(chunk)
        async for text in proc.stdout:
            buffer += text
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                frame = await durable_io(json.loads, line)
                if 'data_b64' in frame:
                    count += await durable_io(write_chunk, frame['data_b64'])
                else:
                    receipt = frame
    stderr = await proc.stderr.read.aio()
    await proc.wait.aio()
    if buffer or proc.returncode or receipt != {'bytes': count, 'sha256': sha.hexdigest()}:
        raise ParityError(f'Artifact transfer truncated: bytes={count}, rc={proc.returncode}, '
                          f'receipt={receipt is not None}, trailing={len(buffer)}; ' + stderr[-1000:])
    if expected and (sha.hexdigest() != expected['sha256'] or count != expected['bytes']):
        raise ParityError('Artifact transfer integrity mismatch')
    await durable_io(local.with_suffix(local.suffix + '.partial').replace, local)
    return {'bytes': count, 'sha256': sha.hexdigest()}


async def download(sandbox, remote, local, *, expected=None, lane='terminal'):
    """Bounded exec stdout ranges; a transient cut retries only that range.

    Keep original bytes and verify the final full-file receipt. No policy
    regeneration, reduced trajectory content, or unbounded transfer fanout.
    """
    if lane not in ('control', 'terminal', 'periodic'):
        raise ValueError('Unknown artifact transfer lane')
    if lane == 'control' and (expected is None or type(expected.get('bytes')) is not int
                              or not 0 <= expected['bytes'] <= 2 * 1024**2):
        raise ValueError('Control lane requires a bounded small artifact')
    if expected is None:
        async with artifact_capacity().transfer(lane):
            return await _download_stream(sandbox, remote, local)
    local = Path(local)
    size = expected['bytes']
    if type(size) is not int or size < 0:
        raise ParityError('Invalid expected artifact size')
    partial = local.with_suffix(local.suffix + '.partial')
    part = local.with_suffix(local.suffix + '.range')
    sha, count, failures = hashlib.sha256(), 0, []
    try:
        async with durable_file(partial, 'wb') as dst:
            def append_range():
                body = part.read_bytes()
                sha.update(body)
                dst.write(body)
                return len(body)
            for offset in range(0, size, 8 * 1024**2):
                length = min(8 * 1024**2, size - offset)
                for retry in range(3):
                    try:
                        async with artifact_capacity().transfer(lane):
                            receipt = await _download_stream(sandbox, remote, part, offset=offset, length=length)
                        if receipt['bytes'] != length:
                            raise ParityError('Artifact range length mismatch')
                        break
                    except Exception as exc:
                        failures.append({'offset': offset, 'length': length, 'retry': retry,
                                         'type': type(exc).__name__, 'error': str(exc)[:1000]})
                        if retry == 2:
                            raise
                        await asyncio.sleep(1 + retry)
                count += await durable_io(append_range)
        if count != size or sha.hexdigest() != expected['sha256']:
            raise ParityError('Artifact transfer integrity mismatch')
        await durable_io(partial.replace, local)
        await durable_io(part.unlink, missing_ok=True)  # validated temporary range, full file is retained
        return {'bytes': count, 'sha256': sha.hexdigest()}
    finally:
        if failures:
            # Separate observer receipt; failed policy/trajectory attempts stay
            # in their normal archives. Only the temporary range is replaced.
            await durable_io(atomic_bytes, local.with_suffix(local.suffix + '.transfer_attempts.json'), canonical(failures))


async def helper(sandbox, action, identity=''):
    proc = await sandbox.exec.aio('/usr/local/bin/python', '-m', 'opencode_rl.sandbox_artifacts',
        action, *([identity] if identity else []), env={'PYTHONPATH': '/opt'}, timeout=300)
    stdout, stderr = await asyncio.gather(proc.stdout.read.aio(), proc.stderr.read.aio())
    await proc.wait.aio()
    if proc.returncode:
        raise ParityError(f'Sandbox artifact {action} failed: {stderr[-1000:]}')
    return json.loads(stdout) if stdout.strip() else None


async def archive_snapshot(sandbox, attempt, root, label, journal, commit):
    """Cancel queued background saves; drain any actually-started snapshot.

    Final/failed archives keep exactly the same contents, range retries, hash
    check and durable barrier. Never overwrite the shared remote archive path
    while a previous snapshot exec or download can still be using it.
    """
    if label not in ('latest_snapshot', 'complete_trajectory', 'failed_trajectory'):
        raise ValueError('Unknown trajectory archive label')
    queued = time.monotonic()
    periodic = label == 'latest_snapshot'
    admission = artifact_capacity().periodic_snapshots
    await durable_io(journal.event, 'archive_queued', {'label': label, 'time': time.time()})
    if periodic:
        # Deliberately outside the shield: no remote/local archive has started.
        await admission.acquire()

    async def persist():
        started = time.monotonic()
        try:
            await durable_io(journal.event, 'archive_started', {
                'label': label, 'queue_seconds': started - queued, 'time': time.time()})
            info = await helper(sandbox, 'snapshot', attempt)
            snapshot_done = time.monotonic()
            transferred = await download(sandbox, info['path'], root / f'{label}.tar.gz', expected=info,
                                         lane='periodic' if periodic else 'terminal')
            await durable_io(atomic_bytes, root / f'{label}.json', canonical(transferred))
            await durable_io(journal.event, 'archive_transferred', {
                'label': label, 'bytes': transferred['bytes'], 'time': time.time(),
                'snapshot_seconds': snapshot_done - started,
                'transfer_seconds': time.monotonic() - snapshot_done})
        finally:
            if periodic:
                # Volume commit is shared/coalesced and may be slow. It need
                # not occupy a remote snapshot/transfer slot for other tasks.
                admission.release()
        committing = time.monotonic()
        await commit()
        await durable_io(journal.event, 'archive_committed', {
            'label': label, 'commit_seconds': time.monotonic() - committing, 'time': time.time()})

    job = asyncio.create_task(persist())
    try:
        await asyncio.shield(job)
    except asyncio.CancelledError:
        # A remote exec survives observer cancellation. Finish that accepted
        # snapshot before rollout_task is allowed to create its final archive.
        await asyncio.gather(job, return_exceptions=True)
        raise


async def rollout_task(*, app, image_id, bundle, destination, config, contract, backend_url,
                       commit, api_key=None, api_transport=None, request_gate=None, timeout=7200,
                       on_delivery=None):
    import modal
    import uuid
    from .gemini_transport import GeminiTransport
    if api_key is not None and api_transport is not None:
        raise ParityError('Specify one API transport, not two')
    if api_key is not None:
        api_transport = GeminiTransport(api_key)
    root, bundle = Path(destination), Path(bundle)
    metadata = json.loads((bundle / 'bundle.json').read_bytes())
    validate_bundle_prompt(bundle, metadata)
    root.mkdir(parents=True, exist_ok=False)
    attempt = uuid.uuid4().hex
    for name, value in [('contract.json', contract.to_dict()), ('config.json', config)]:
        atomic_bytes(root / name, canonical(value))
    atomic_bytes(root / 'attempt.json', canonical({'attempt_id': attempt, **metadata, 'started': time.time()}))
    journal = TraceStore(root / 'trace').new_request({'attempt_id': attempt, 'kind': 'opencode_task_runtime'}, {})
    sandbox = None
    periodic = None
    started = time.monotonic()
    status = {'task': metadata['task'], 'stage': metadata['stage'], 'attempt_id': attempt}

    async def archive(label):
        await archive_snapshot(sandbox, attempt, root, label, journal, commit)

    async def checkpoint():
        while True:
            await asyncio.sleep(120)
            try:
                await archive('latest_snapshot')
            except Exception as exc:
                await durable_io(journal.event, 'snapshot_error', {'type': type(exc).__name__, 'message': str(exc)[:500]})
                await commit()

    try:
        # A capacity refusal is infrastructure, not a zero-reward model answer.
        # Retain the allocation journal even if creation never succeeds.
        for allocation_retry in range(8):
            try:
                sandbox = await modal.Sandbox.create.aio(app=app, image=modal.Image.from_id(image_id),
                    block_network=True, cpu=(1, 4), memory=(4096, 8192), timeout=timeout + 900)
                break
            except Exception as exc:
                message = str(exc).lower()
                if allocation_retry == 7 or not any(word in message for word in
                    ('resource_exhausted', 'quota', 'capacity', 'rate limit', 'too many')):
                    raise
                await durable_io(journal.event, 'sandbox_capacity_wait', {'retry': allocation_retry + 1, 'error': str(exc)[:500]})
                await commit()
                await asyncio.sleep(min(60, 5 * 2 ** allocation_retry))
        await durable_io(atomic_json, root / 'sandbox.json', {'id': sandbox.object_id, 'image': image_id,
            'block_network': True, 'secret_count': 0, 'volume_count': 0})
        await commit()
        for local, remote in [(bundle / 'input.tar', '/root/input.tar'), (bundle / 'task.txt', '/root/task.txt'),
            (bundle / 'music_catalog.json', '/root/music_catalog.json'),
            (root / 'config.json', '/root/config.json'), (root / 'contract.json', '/root/contract.json')]:
            await upload(sandbox, local, remote)
        await helper(sandbox, 'stage')
        proc = await sandbox.exec.aio('/usr/local/bin/python', '-m', 'opencode_rl.stdio_relay',
            '--contract', '/root/contract.json', '--config', '/root/config.json', '--attempt', attempt,
            '--task', '/root/task.txt', '--stage', metadata['stage'], '--music-catalog', '/root/music_catalog.json',
            workdir='/work', env={'PYTHONPATH': '/opt'}, timeout=timeout)
        periodic = asyncio.create_task(checkpoint())
        stderr_task = asyncio.create_task(proc.stderr.read.aio())
        try:
            result = await forward_runtime(proc, backend_url, journal, attempt_id=attempt,
                contract_sha256=contract.sha256, timeout=timeout,
                api_transport=api_transport, request_gate=request_gate)
            await proc.wait.aio()
            await durable_io(atomic_bytes, root / 'relay_stderr.log', (await stderr_task).encode())
        finally:
            if not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)
        if proc.returncode != 0:
            raise ParityError('Relay process failed')
        await helper(sandbox, 'stop_policy')
        collected = await helper(sandbox, 'collect', metadata['stage'])
        for name, info in collected.items():
            if info['valid_file']:
                await download(sandbox, info['path'], root / name, expected=info,
                               lane='control' if name.endswith('.json') else 'terminal')
        await durable_io(atomic_json, root / 'deliverables.json', collected)
        status.update(status='rollout_complete', runtime=result)
        await durable_io(journal.event, 'delivery_ready', {'time': time.time(), 'attempt_id': attempt})
        if on_delivery is not None:
            # Frozen, hash-checked delivery is ready. Scoring may overlap the
            # full archive, but training admission still waits for this
            # function's final archive + commit + successful return below.
            await on_delivery(dict(status))
    except asyncio.CancelledError:
        status.update(status='cancelled', error_type='CancelledError', error='Administrative cancellation; never model zero')
        await durable_io(journal.event, 'attempt_cancelled', dict(status))
        raise
    except Exception as exc:
        status.update(status='infrastructure_error', error_type=type(exc).__name__, error=str(exc)[:1500])
        await durable_io(journal.event, 'attempt_error', dict(status))
    finally:
        if periodic:
            periodic.cancel()
            await asyncio.gather(periodic, return_exceptions=True)
        try:
            if sandbox is not None:
                await helper(sandbox, 'stop_policy')
                await archive('complete_trajectory' if status.get('status') == 'rollout_complete' else 'failed_trajectory')
        except Exception as exc:
            status.update(status='infrastructure_error', archive_error=type(exc).__name__ + ': ' + str(exc)[:500])
        await durable_io(journal.seal, 'complete' if status.get('status') == 'rollout_complete' else 'failed')
        status['seconds'] = time.monotonic() - started
        await durable_io(atomic_json, root / 'result.json', status)
        await commit()
        if sandbox is not None:
            await sandbox.terminate.aio()
    return status
