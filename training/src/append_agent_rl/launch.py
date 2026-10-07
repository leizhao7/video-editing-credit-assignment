"""Independent launch commands/supervisor; old native launcher stays intact."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from opencode_rl.contract import ParityError, canonical
from opencode_rl.store import atomic_bytes
from opencode_rl.train_capacity import sample_capacity
from opencode_rl.train_launch import inference_command as native_inference_command
from opencode_rl.train_launch import training_command as native_training_command
from .contract import AppendContract, validate_runtime


def inference_command(model, contract_path, trace, port, contract):
    if not isinstance(contract, AppendContract):
        raise ParityError('Cannot launch append inference with a native-loop contract')
    command = native_inference_command(model, contract_path, trace, port, contract)
    if command[2] != 'opencode_rl.async_server':
        raise ParityError('Upstream inference entrypoint changed')
    command[2] = 'append_agent_rl.server'
    return command


def training_command(base_argv, plan_path, plan, *, steps, resume=''):
    contract = AppendContract(**plan['contract'])
    validate_runtime(plan['config'], contract)
    if plan.get('harness') != 'append-opencode-tools-v1' or plan.get('causal_prefix') is not True:
        raise ParityError('Distinct append harness plan required')
    # Retain the original strict checkpoint/data/grader receipt validation.
    # A V11 checkpoint cannot be resumed by changing a path or receipt hash.
    command = native_training_command(base_argv, plan_path, plan, steps=steps, resume=resume)
    if command[2] != 'opencode_rl.train_entrypoint':
        raise ParityError('Upstream training entrypoint changed')
    command[2] = 'append_agent_rl.train_entrypoint'
    if plan.get('process_reward') is not None:
        from prm_v2.live_config_v5 import config_for_plan
        process = config_for_plan(plan)
        command = [a for a in command if not a.startswith(('generator.ocrl_process_weight=',
                    'generator.ocrl_packed_prm_enabled='))]
        command.extend(['generator.ocrl_process_weight=' + str(process['weight']),
                        'generator.ocrl_packed_prm_enabled=true'])
    from prm_v2.live_config_v5 import gradient_config_for_plan
    gradient = gradient_config_for_plan(plan)
    # Strip a caller-supplied override: the immutable plan owns this schedule.
    command = [a for a in command if not a.startswith('generator.ocrl_gradient_probe_interval=')]
    if gradient is not None:
        command.append('generator.ocrl_gradient_probe_interval=' + str(gradient['interval']))
    return command


async def supervise(plan, contract, argv, commit):
    """Only terminates this invocation's children; no watchdog signals to siblings."""
    import aiohttp
    validate_runtime(plan['config'], contract)
    if plan.get('harness') != 'append-opencode-tools-v1' or argv[2] != 'append_agent_rl.train_entrypoint':
        raise ParityError('Explicit append-only training entrypoint required')
    root = Path(plan['run_root'])
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_bytes()) != plan:
        raise ParityError('Refusing to overwrite another run plan; use a distinct run directory')
    atomic_bytes(plan_path, canonical(plan))
    atomic_bytes(root / 'contract.json', canonical(contract.to_dict()))
    await commit()
    children, streams = [], []
    base_env = {**os.environ, 'PYTHONPATH': '/patch:/opt:/opt/SkyRL', 'SKYRL_PYTHONPATH_EXPORT': '1',
        'RAY_ENABLE_UV_RUN_RUNTIME_ENV': '0', 'SKYRL_FREEZE_VISION': '1', 'SKYRL_VL_DUMMY': '1',
        'SKYRL_TRIM_MICROBATCH': '1', 'VLLM_HTTP_TIMEOUT_KEEP_ALIVE': '30',
        'S12_PRM_ENABLED': '0', 'S12_TOOLS_DIR': '/opt/s12tools', 'OCRL_PACK_BLOBS': '1',
        # SkyRL injects 0 into Ray training workers; standalone inference does
        # not inherit that runtime_env. Mixed cuMem/legacy IPC fails at
        # ncclCommInitRank (p2p.cc:299 invalid argument), reproduced on 4i+4t.
        # Set this BEFORE either side imports CUDA/NCCL, never at sync time.
        'NCCL_CUMEM_ENABLE': '0', 'NCCL_DEBUG': 'INFO',
        'NCCL_DEBUG_FILE': str(root / 'nccl.%h.%p.log')}
    base_env.pop('RAY_RUNTIME_ENV_HOOK', None)
    periodic_error = []

    async def persist():
        while True:
            await asyncio.sleep(30)
            try:
                sample = {'time': time.time(), 'engines': await sample_capacity(plan['engines'])}
                await asyncio.to_thread(atomic_bytes, root / 'capacity' / f'{time.time_ns()}.json', canonical(sample))
                await commit()
            except Exception as exc:
                periodic_error.append(exc)
                return

    periodic = asyncio.create_task(persist())
    try:
        for index, engine in enumerate(plan['engines']):
            log = (root / f'inference_{index}.log').open('ab', buffering=0)
            streams.append(log)
            port = int(engine['url'].rsplit(':', 1)[1])
            command = inference_command(plan['model_path'], root / 'contract.json', engine['trace_root'], port, contract)
            children.append(subprocess.Popen(command, env={**base_env, 'CUDA_VISIBLE_DEVICES': str(index)},
                stdout=log, stderr=subprocess.STDOUT, cwd='/opt/SkyRL'))
        deadline = time.monotonic() + 1800
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as http:
            for engine in plan['engines']:
                while True:
                    if periodic_error:
                        raise periodic_error[0]
                    if any(child.poll() is not None for child in children):
                        raise RuntimeError('Inference initialization exited; see retained per-engine logs')
                    try:
                        async with http.get(engine['url'] + '/ocrl/status') as response:
                            response.raise_for_status()
                            state = await response.json()
                            if state['phase'] != 'open' or state['version'] != f'{contract.model_revision}:0':
                                raise ParityError('Unexpected initial inference policy')
                            break
                    except (aiohttp.ClientError, TimeoutError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Inference startup deadline')
                        await asyncio.sleep(1)
        log = (root / 'training.log').open('ab', buffering=0)
        streams.append(log)
        # Ray sees only the four training GPUs; inference is four private
        # subprocesses on 0..3. All control/weight URLs remain host-local.
        from prm_v2.live_config_v5 import training_numerics_env
        trainer_env = {**base_env, **training_numerics_env(plan), 'CUDA_VISIBLE_DEVICES': '4,5,6,7'}
        if plan.get('process_reward') is not None:
            # Ray otherwise installs uvloop in every worker. The pinned 0.22.1
            # has uvloop#740's cancelled-socket double-close bug, reproduced in
            # this exact image. Set BEFORE Ray workers start, not after a broken
            # FD has poisoned the trainer. Leave standalone vLLM loops alone.
            trainer_env['RAY_USE_UVLOOP'] = '0'
        trainer = subprocess.Popen(argv, env=trainer_env,
            stdout=log, stderr=subprocess.STDOUT, cwd='/opt/SkyRL')
        children.append(trainer)
        atomic_bytes(root / 'started.json', canonical({'training_pid': trainer.pid, 'time': time.time(),
            'mode': plan['mode'], 'inference_gpus': [0, 1, 2, 3], 'training_gpus': [4, 5, 6, 7],
            'prm_ray_use_uvloop': trainer_env.get('RAY_USE_UVLOOP') if plan.get('process_reward') is not None else None}))
        await commit()
        while trainer.poll() is None:
            if periodic_error:
                raise periodic_error[0]
            if any(child.poll() is not None for child in children[:-1]):
                raise RuntimeError('Inference engine exited during training')
            await asyncio.sleep(1)
        if trainer.returncode:
            raise RuntimeError(f'Trainer failed with code {trainer.returncode}; logs and attempts retained')
        requested = int(plan['requested_steps'])
        finished = root / 'checkpoints' / f'{requested:06d}.json'
        if not finished.exists() or json.loads(finished.read_bytes())['step'] != requested:
            raise RuntimeError('Process exited without the requested completed-step checkpoint; not a successful training run')
        atomic_bytes(root / 'completed.json', canonical({'returncode': 0, 'time': time.time(), 'mode': plan['mode']}))
    except BaseException as exc:
        atomic_bytes(root / 'failure.json', canonical({'type': type(exc).__name__, 'message': str(exc)[:1500]}))
        raise
    finally:
        periodic.cancel()
        await asyncio.gather(periodic, return_exceptions=True)
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
