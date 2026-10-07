"""Checkpoint-only supervisor; preserves the frozen v5 four-plus-four topology."""
import asyncio
import os
from pathlib import Path
import subprocess
import time

from append_resume_guard import canonical, read_json, require, verify_plan


async def supervise(plan, contract, argv, commit):
    import aiohttp
    from append_agent_rl.contract import validate_runtime
    from append_agent_rl.launch import inference_command
    from opencode_rl.store import atomic_bytes
    from opencode_rl.train_capacity import sample_capacity
    root = Path(plan['run_root'])
    verify_plan(plan, read_json(root/'resume_source_plan.json'), read_json(root/'resume_audit.json'))
    validate_runtime(plan['config'], contract)
    require(argv[:3] == ['/opt/SkyRL/.venv/bin/python', '-m', 'append_resume_entrypoint'],
            'Explicit checkpoint resume entrypoint required')
    require(read_json(root/'plan.json') == plan, 'Immutable resume plan changed')
    atomic_bytes(root/'contract.json', canonical(contract.to_dict()))
    await commit()
    children, streams, periodic_error = [], [], []
    env = {**os.environ, 'PYTHONPATH': '/patch:/opt:/opt/SkyRL', 'SKYRL_PYTHONPATH_EXPORT': '1',
        'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0', 'SKYRL_FREEZE_VISION': '1', 'SKYRL_VL_DUMMY': '1',
        'SKYRL_TRIM_MICROBATCH': '1', 'VLLM_HTTP_TIMEOUT_KEEP_ALIVE': '30',
        'S12_PRM_ENABLED': '0', 'S12_TOOLS_DIR': '/opt/s12tools', 'OCRL_PACK_BLOBS': '1',
        'NCCL_CUMEM_ENABLE': '0', 'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_FILE': str(root/'nccl.%h.%p.log')}
    env.pop('RAY_RUNTIME_ENV_HOOK', None)

    async def persist():
        while True:
            await asyncio.sleep(30)
            try:
                sample = {'time': time.time(), 'engines': await sample_capacity(plan['engines'])}
                await asyncio.to_thread(atomic_bytes, root/'capacity'/f'{time.time_ns()}.json', canonical(sample))
                await commit()
            except Exception as exc:
                periodic_error.append(exc)
                return

    periodic = asyncio.create_task(persist())
    try:
        for i, engine in enumerate(plan['engines']):
            log = (root/f'inference_{i}.log').open('ab', buffering=0)
            streams.append(log)
            command = inference_command(plan['model_path'], root/'contract.json', engine['trace_root'],
                                        int(engine['url'].rsplit(':', 1)[1]), contract)
            children.append(subprocess.Popen(command, env={**env, 'CUDA_VISIBLE_DEVICES': str(i)},
                stdout=log, stderr=subprocess.STDOUT, cwd='/opt/SkyRL'))
        deadline = time.monotonic()+1800
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as http:
            for engine in plan['engines']:
                while True:
                    if periodic_error:
                        raise periodic_error[0]
                    require(all(c.poll() is None for c in children), 'Inference initialization exited')
                    try:
                        async with http.get(engine['url']+'/ocrl/status') as response:
                            response.raise_for_status()
                            state = await response.json()
                            require(state['phase'] == 'open' and state['version'] == f'{contract.model_revision}:0',
                                    'Unexpected initial inference version')
                            break
                    except (aiohttp.ClientError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Inference initialization deadline')
                        await asyncio.sleep(1)
        log = (root/'training.log').open('ab', buffering=0)
        streams.append(log)
        trainer = subprocess.Popen(argv, env={**env, 'CUDA_VISIBLE_DEVICES': '4,5,6,7'},
            stdout=log, stderr=subprocess.STDOUT, cwd='/opt/SkyRL')
        children.append(trainer)
        atomic_bytes(root/'started.json', canonical({'training_pid': trainer.pid, 'time': time.time(),
            'mode': 'train', 'resume_step': plan['resume']['step'],
            'inference_gpus': [0, 1, 2, 3], 'training_gpus': [4, 5, 6, 7]}))
        await commit()
        while trainer.poll() is None:
            if periodic_error:
                raise periodic_error[0]
            require(all(c.poll() is None for c in children[:-1]), 'Inference engine exited during training')
            await asyncio.sleep(1)
        require(trainer.returncode == 0, f'Trainer failed with code {trainer.returncode}; logs retained')
        finished = root/'checkpoints'/f'{plan["requested_steps"]:06d}.json'
        require(finished.is_file() and read_json(finished)['step'] == plan['requested_steps'],
                'Trainer exited without the final completed-step checkpoint')
        atomic_bytes(root/'completed.json', canonical({'returncode': 0, 'time': time.time(),
            'mode': 'train', 'resumed_from': plan['resume']['step']}))
    except BaseException as exc:
        atomic_bytes(root/'failure.json', canonical({'type': type(exc).__name__, 'message': str(exc)[:1500]}))
        raise
    finally:
        periodic.cancel()
        await asyncio.gather(periodic, return_exceptions=True)
        # Only this invocation's children; never broad pkill or watchdog signals.
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                await asyncio.to_thread(child.wait, timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                await asyncio.to_thread(child.wait)
        for stream in streams:
            stream.close()
        await commit()
