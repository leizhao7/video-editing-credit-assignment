"""Public training deployment. Importing it does not reserve GPUs or train."""
from pathlib import Path
import asyncio
import json
import os
import sys

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from configuration import load_config
from environment.images import build_images
from launcher import bounded_run

CONFIG = load_config()
app = modal.App(CONFIG['app'])
if modal.is_local():
    IMAGES = build_images(CONFIG)
else:
    IMAGES = {name: modal.Image.debian_slim(python_version='3.12') for name in ('sandbox', 'cpu', 'gpu')}
logs = modal.Volume.from_name(CONFIG['logs_volume'], create_if_missing=True, version=2)
checkpoints = modal.Volume.from_name(CONFIG['checkpoint_volume'], create_if_missing=True)
release = modal.Volume.from_name(CONFIG['release_volume'])
models = modal.Volume.from_name(CONFIG['model_volume'], create_if_missing=True)
asr = modal.Volume.from_name(CONFIG['asr_cache'], create_if_missing=True)
judge_secrets = [modal.Secret.from_name(CONFIG['judge_secret'])]
training_secrets = [modal.Secret.from_name(name) for name in
                    dict.fromkeys(CONFIG[k] for k in ('process_secret', 'wandb_secret') if CONFIG[k])]
THREAD_ENV = {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
              'NUMEXPR_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'RAYON_NUM_THREADS': '2'}
PLAN_VOLUMES = {'/logs': logs, '/checkpoints': checkpoints, '/release': release, '/models': models}


@app.function(image=IMAGES['sandbox'], cpu=.25, memory=256, timeout=60, include_source=False)
def sandbox_asset():
    return {'purpose': 'sandbox image registration', 'model_calls': 0}


@app.function(image=IMAGES['cpu'], cpu=4, memory=16384, timeout=7200, max_containers=32,
              volumes={'/logs': logs, '/data': release, '/asr-cache': asr},
              secrets=judge_secrets, retries=0, include_source=False)
@modal.concurrent(max_inputs=1)
def grade(payload):
    import grader_service as service
    from opencode_rl.stage_grading import grade_stage
    import uuid
    from opencode_rl.store import atomic_json
    # The volume includes private corpus/ alongside public prepared/. The agent
    # receives neither this mount nor this function's API credentials.
    service.logs_volume = logs
    service._task_source = task_source
    asr.reload()
    result = grade_stage(payload, service)
    atomic_json(Path('/logs/trusted_grades') / uuid.uuid4().hex / 'result.json', result)
    logs.commit()
    asr.commit()
    return result


def task_source(payload):
    from opencode_rl.task_bundle import release_tasks, validate_source
    from opencode_rl.contract import ParityError
    split = {'_tt_train': 'train', '_tt_val': 'val'}.get(payload.get('source_root'))
    if split is None:
        raise ParityError('Unrecognized trusted corpus split')
    receipt = json.loads(Path('/data/release_receipt.json').read_bytes())
    rows = release_tasks('/data/release.json', split, expected_sha256=receipt['release_sha256'])
    found = [r for r in rows if r['task'] == payload.get('task')]
    if len(found) != 1:
        raise ParityError('Task not in the hash-bound release')
    source = Path('/data/corpus') / ('_tt_' + split) / found[0]['task']
    validate_source(source, found[0])
    return str(source)


@app.function(image=IMAGES['cpu'], cpu=2, memory=8192, timeout=300, retries=0, include_source=False)
def grader_contract():
    import grader_service as service
    from opencode_rl.stage_grading import service_contract
    from prm_training_source import mounted_manifest, digest
    manifest = mounted_manifest()
    return {**service_contract(service), 'integration_sha256': manifest['package_hashes']['opencode_rl'],
            'loop_sha256': manifest['package_hashes']['append_agent_rl'], 'process_source_sha256': digest(manifest)}


async def bind_plan(name, image_id, mode, steps, readiness='', direction='forward'):
    from launcher import create_plan
    plan, contract = create_plan(CONFIG, name=name, image_id=image_id, mode=mode, steps=steps,
                                 readiness=readiness, direction=direction)
    service = await grader_contract.remote.aio()
    for key, expected in (('integration_sha256', contract.integration_sha256),
                          ('loop_sha256', contract.loop_sha256),
                          ('process_source_sha256', plan['process_source_sha256'])):
        if service[key] != expected:
            raise ValueError('CPU grader and training image sources differ: ' + key)
    if service['review_policy'] != 'judge_only' or service['submission_validation'] != 'enabled':
        raise ValueError('Grading must retain the explicit validation/judge policy')
    plan['grader_hashes'] = service['grader_sha256']
    return plan, contract


@app.function(image=IMAGES['cpu'], cpu=2, memory=8192, timeout=1800,
              volumes=PLAN_VOLUMES, retries=0, include_source=False)
async def inspect_plan(name, image_id, mode='smoke', steps=2, readiness='', direction='forward'):
    from launcher import command, require_qualification
    plan, contract = await bind_plan(name, image_id, mode, steps, readiness, direction)
    if mode == 'train':
        require_qualification(plan, contract)
    return {'plan': plan, 'command': command(plan, logger=CONFIG['logger']),
            'contract_sha256': contract.sha256, 'gpu_allocated': False, 'training_started': False}


@app.function(image=IMAGES['gpu'], cpu=2, memory=16384, timeout=900,
              volumes=PLAN_VOLUMES, retries=0, include_source=False)
async def preflight(name, image_id):
    import subprocess
    from opencode_rl.store import atomic_json
    from opencode_rl.contract import canonical
    plan, _ = await bind_plan(name, image_id, 'smoke', 2)
    probe = subprocess.run(['/opt/SkyRL/.venv/bin/python', '/opt/editing-release/training/environment/preflight.py'],
        input=canonical(plan), capture_output=True, timeout=840,
        env={**os.environ, **THREAD_ENV, 'CUDA_VISIBLE_DEVICES': '',
             'PYTHONPATH': '/patch:/opt:/opt/SkyRL:/opt/editing-release/training'})
    result = {'returncode': probe.returncode, 'stdout': probe.stdout.decode()[-40000:],
              'stderr': probe.stderr.decode()[-16000:], 'gpu_allocated': False,
              'training_qualified': False, 'scope': 'source/import preflight only'}
    atomic_json(Path('/logs/preflight') / (name + '.json'), result)
    await logs.commit.aio()
    if probe.returncode:
        raise RuntimeError('Import preflight failed; see /logs/preflight/' + name + '.json')
    return result


@app.function(image=IMAGES['cpu'].env({'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}), cpu=4, memory=16384, timeout=7200,
              volumes={'/models': models}, retries=0, include_source=False)
def prepare_model():
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import snapshot_download
    from opencode_rl.contract import MODEL_REVISION
    destination = Path('/models') / MODEL_REVISION
    snapshot_download('Qwen/Qwen3.5-9B', revision=MODEL_REVISION, local_dir=str(destination))
    models.commit()
    return {'model': 'Qwen/Qwen3.5-9B', 'revision': MODEL_REVISION, 'path': str(destination), 'gpu_allocated': False}


@app.function(image=IMAGES['gpu'], gpu='H200:8', cpu=64, memory=524288, timeout=86400,
              single_use_containers=True, max_containers=1, retries=0, include_source=False,
              volumes=PLAN_VOLUMES, secrets=training_secrets)
async def train(name, image_id, mode, steps, readiness='', direction='forward', resume=''):
    from launcher import command, require_qualification
    from opencode_rl.capacity import CoalescedCommit
    from opencode_rl.store import atomic_json
    import csv
    import subprocess
    os.environ.update(THREAD_ENV)
    plan, contract = await bind_plan(name, image_id, mode, steps, readiness, direction)
    if mode == 'train':
        require_qualification(plan, contract)
    if resume and (not Path(resume).is_absolute() or not Path(resume).is_relative_to('/checkpoints')):
        raise ValueError('Resume requires an explicit checkpoint in the mounted checkpoint volume')
    argv = command(plan, logger=CONFIG['logger'], resume=resume)
    root = Path(plan['run_root'])
    root.mkdir(parents=True, exist_ok=False)
    rows = list(csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                    '--format=csv,noheader,nounits'], text=True, timeout=30).splitlines()))
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader,nounits'], text=True, timeout=30).strip()
    clean = len(rows) == len({r[1] for r in rows}) == 8 and not processes and all(int(r[2]) < 2048 for r in rows)
    atomic_json(root / 'initial_gpu_state.json', {'gpus': rows, 'processes': processes, 'clean': clean})
    atomic_json(root / 'command.json', argv)
    await logs.commit.aio()
    if not clean:
        raise RuntimeError('Allocation is not empty; no unrelated process was terminated')
    if 'opsd' in plan:
        from examples.train.answer_self_distillation.supervisor import supervise
    else:
        from append_agent_rl.launch import supervise
    try:
        await supervise(plan, contract, argv, CoalescedCommit(logs.commit.aio))
    finally:
        await checkpoints.commit.aio()
    return {'completed': True, 'run': name, 'steps': steps, 'profile': CONFIG['profile'], 'mode': mode}


@app.cls(image=IMAGES['cpu'], cpu=1, memory=512, max_containers=1, timeout=60, retries=0, include_source=False)
@modal.concurrent(max_inputs=128)
class DemonstrationQuota:
    @modal.enter()
    async def setup(self):
        import time
        from opsd_demo_quota import Admission, TokenBucket
        self.admission = Admission(bucket=TokenBucket(now=time.monotonic(),
                                  rate=CONFIG['demonstrator_tokens_per_minute'] / 60))

    @modal.method()
    async def acquire(self, count):
        return await self.admission.acquire(count)

    @modal.method()
    async def observe(self, lease, status, actual, retry_after):
        self.admission.bucket.observe(lease, status, actual, retry_after, self.admission.clock())
        return self.admission.describe()


@app.function(image=IMAGES['cpu'], cpu=1, memory=2048, timeout=330, max_containers=8,
              secrets=training_secrets, retries=0, include_source=False)
@modal.concurrent(max_inputs=8)
async def generate(request):
    if CONFIG['profile'] != 'opsd_suggestions':
        raise ValueError('The demonstrator is enabled only in the turn-suggestion profile')
    import httpx
    from examples.train.answer_self_distillation.turn_demonstrations import check_service_request
    from opsd_demo_admitted_request import generate_once
    body = check_service_request(request)
    gate = modal.Cls.from_name(CONFIG['app'], 'DemonstrationQuota')()
    class Client:
        async def acquire(self, count):
            return await gate.acquire.remote.aio(count)
        async def observe(self, *args):
            return await gate.observe.remote.aio(*args)
    async with httpx.AsyncClient(timeout=request['timeout_seconds'], follow_redirects=False) as client:
        return await generate_once(request, body, client, Client(),
            {'x-goog-api-key': os.environ['GOOGLE_AI_STUDIO_API_KEY'], 'Content-Type': 'application/json'})


@app.local_entrypoint()
def main(action='check', name='smoke-001', steps: int = 2, readiness='', direction='forward', resume=''):
    if action == 'prepare-model':
        print(json.dumps(prepare_model.remote(), indent=2))
        return
    if action not in ('check', 'preflight', 'smoke', 'train'):
        raise ValueError('Choose check, preflight, prepare-model, smoke, or train')
    bounded_run(name, 'train' if action == 'train' else 'smoke', steps)
    image_id = IMAGES['sandbox'].object_id
    if action == 'preflight':
        result = preflight.remote(name, image_id)
    else:
        mode = 'train' if action == 'train' else 'smoke'
        result = inspect_plan.remote(name, image_id, mode, steps, readiness, direction)
        if action in ('smoke', 'train'):
            call = train.spawn(name, image_id, mode, steps, readiness, direction, resume)
            result = {'call_id': call.object_id, 'detached': True, 'run': name,
                      'gpus': 8, 'profile': CONFIG['profile'], 'steps': steps}
    print(json.dumps(result, indent=2))
