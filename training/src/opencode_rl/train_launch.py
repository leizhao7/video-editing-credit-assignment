"""Single-node 4 inference + 4 training supervisor. Importing never starts work."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from .contract import ParityError, canonical, digest
from .store import atomic_bytes
from .train_capacity import training_capacity, scheduler_args, sample_capacity

PYTHON = '/opt/SkyRL/.venv/bin/python'


def inference_command(model, contract_path, trace, port, contract):
    capacity = training_capacity()
    return [PYTHON, '-m', 'opencode_rl.async_server', '--contract', str(contract_path),
        '--trace-root', str(trace), '--active-requests', str(capacity['engine_active_requests']),
        '--reserved-tokens', str(capacity['engine_reserved_tokens']),
        '--feature-preparation-concurrency', str(capacity['engine_feature_preparation_concurrency']),
        '--', '--model', model, '--served-model-name', 'qwen35-9b-policy',
        '--host', '127.0.0.1', '--port', str(port), '--tensor-parallel-size', '1',
        '--max-model-len', str(contract.max_context), *scheduler_args(capacity), '--dtype', 'bfloat16',
        '--generation-config', 'vllm', '--override-generation-config', '{"top_k":-1}',
        '--enable-prefix-caching', '--logprobs-mode', 'raw_logprobs', '--enable-auto-tool-choice',
        '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3',
        '--default-chat-template-kwargs', json.dumps({'enable_thinking': contract.thinking}),
        '--weight-transfer-config', '{"backend":"nccl"}', '--worker-extension-cls',
        'skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap',
        '--no-enable-log-requests']


def training_command(base_argv, plan_path, plan, *, steps, resume=''):
    capacity = training_capacity()
    if plan['mode'] == 'smoke' and not 1 <= steps <= 2:
        raise ParityError('Smoke cannot become an unbounded training run')
    if steps < 1 or (resume and not Path(resume).is_absolute()):
        raise ParityError('Invalid steps/resume')
    eval_interval = plan.get('eval_interval', -1)
    if eval_interval not in (-1, 10) or (plan['mode'] == 'train' and eval_interval != 10):
        raise ParityError('Formal training requires validation every ten steps')
    if eval_interval == 10 and not plan.get('validation'):
        raise ParityError('Validation requires a frozen prepared split')
    overrides = {arg.split('=', 1)[0]: arg.split('=', 1)[1] for arg in base_argv[3:] if '=' in arg}
    for key in ('generator.tasks_root', 'generator.val_tasks_root', 'generator.tell_n'):
        overrides.pop(key, None)
    overrides.update({
        'generator.ocrl_plan': str(plan_path), 'generator.ocrl_process_weight': '0.0',
        'generator.ocrl_causal_prefix': str(bool(plan.get('causal_prefix', False))).lower(),
        'generator.ocrl_contribution_dir': str(Path(plan['run_root']) / 'contributions'),
        'generator.step_wise_trajectories': 'true', 'generator.merge_stepwise_output': 'false',
        'generator.inference_engine.served_model_name': 'qwen35-9b-policy',
        'generator.inference_engine.run_engines_locally': 'false',
        'generator.inference_engine.external_proxy_url': plan['engines'][0]['url'],
        'generator.inference_engine.external_server_urls': json.dumps([e['url'] for e in plan['engines']]),
        'generator.inference_engine.max_num_seqs': str(capacity['max_num_seqs']),
        'generator.inference_engine.max_num_batched_tokens': str(capacity['max_num_batched_tokens']),
        'generator.inference_engine.gpu_memory_utilization': str(capacity['gpu_memory_utilization']),
        'trainer.algorithm.temperature': '1.0', 'trainer.algorithm.loss_reduction': 'token_mean',
        'trainer.algorithm.advantage_batch_normalize': 'false',
        'trainer.algorithm.max_seq_len': str(plan['contract']['max_context']),
        'trainer.eval_before_train': 'false', 'trainer.eval_interval': str(eval_interval),
        'generator.eval_n_samples_per_prompt': '1',
        'generator.eval_sampling_params.temperature': str(plan['contract']['temperature']),
        'generator.eval_sampling_params.top_p': str(plan['contract']['top_p']),
        'generator.eval_sampling_params.top_k': str(plan['contract']['top_k']),
        'generator.eval_sampling_params.max_generate_length': str(plan['contract']['max_output']),
        'trainer.ckpt_interval': '2', 'trainer.max_ckpts_to_keep': '2',
        'trainer.ckpt_path': str(Path(plan['checkpoint_root']) / 'rolling'),
        'trainer.export_path': str(Path(plan['checkpoint_root']) / 'exports'),
        'trainer.max_training_steps': str(steps), 'trainer.epochs': str(steps),
        'trainer.fully_async.simulate_training': 'false',
        'data.dataloader.num_workers': '0', 'data.dataloader.persistent_workers': 'false',
        'trainer.resume_mode': 'from_path' if resume else 'none',
        'trainer.resume_path': resume,
    })
    if resume:
        receipt = json.loads((Path(resume) / 'ocrl_resume.json').read_bytes())
        if (receipt['contract_sha256'] != digest(plan['contract']) or
                receipt['prepared_sha256'] != plan['prepared_sha256'] or receipt['grader_hashes'] != plan['grader_hashes']):
            raise ParityError('Resume changes runtime/dataset/grader; explicit migration required')
        if receipt['step'] >= steps:
            raise ParityError('Checkpoint already reached requested final step')
    return [base_argv[0], '-m', 'opencode_rl.train_entrypoint'] + [f'{k}={v}' for k, v in overrides.items()]


async def supervise(plan, contract, argv, commit):
    """Only terminates this invocation's children; no watchdog signals to siblings."""
    import aiohttp
    root = Path(plan['run_root'])
    plan_path = root / 'plan.json'
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
        trainer = subprocess.Popen(argv, env={**base_env, 'CUDA_VISIBLE_DEVICES': '4,5,6,7'},
            stdout=log, stderr=subprocess.STDOUT, cwd='/opt/SkyRL')
        children.append(trainer)
        atomic_bytes(root / 'started.json', canonical({'training_pid': trainer.pid, 'time': time.time(),
            'mode': plan['mode'], 'inference_gpus': [0, 1, 2, 3], 'training_gpus': [4, 5, 6, 7]}))
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
