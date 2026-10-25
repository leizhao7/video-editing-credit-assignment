"""Public Modal image recipes; image construction never starts training.

The GPU image retains the recovered CUDA/SkyRL dependency stack. Sandboxes have
no corpus, answers, model weights, cloud volume or secrets baked into them.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
SKYRL = '59d4daedee24c7b1a79d857bbf322a6d195c3792'
OPENCODE = 'd7a6e1daaf2271bcd0b611fbb27d4956806e475a'
IGNORE = ['**/__pycache__/**', '**/.pytest_cache/**', '**/.DS_Store', '**/node_modules/**']


def sources(image, profile):
    return (image
        .add_local_dir(str(ROOT / 'training'), '/opt/editing-release/training', copy=True, ignore=IGNORE)
        .add_local_dir(str(ROOT / 'opsd'), '/opt/editing-release/opsd', copy=True, ignore=IGNORE)
        .add_local_dir(str(ROOT / 'evaluation'), '/opt/s12tools', copy=True, ignore=IGNORE)
        .env({'EDITING_TRAINING_PROFILE': profile, 'OCRL_PACK_BLOBS': '1',
              'PYTHONPATH': '/opt:/opt/s12tools:/opt/SkyRL',
              'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}))


def native_host(image):
    return image.run_commands(
        'curl -fsSL https://bun.sh/install -o /tmp/install-bun.sh',
        'BUN_INSTALL=/opt/bun bash /tmp/install-bun.sh bun-v1.3.14',
        'git init /opt/opencode-source',
        'git -C /opt/opencode-source remote add origin https://github.com/anomalyco/opencode.git',
        f'git -C /opt/opencode-source fetch --depth 1 origin {OPENCODE}',
        'git -C /opt/opencode-source checkout --detach FETCH_HEAD',
        'python /opt/editing-release/training/environment/prepare_opencode.py /opt/opencode-source',
        'cd /opt/opencode-source && /opt/bun/bin/bun install --frozen-lockfile --ignore-scripts',
        'cd /opt/opencode-source/packages/opencode && /opt/bun/bin/bun run script/build_append_host.ts bun-linux-x64',
        'install -m 755 /opt/opencode-source/packages/opencode/dist/append-tools/bun-linux-x64/opencode-tools /usr/local/bin/opencode-tools',
        'cp /opt/opencode-source/packages/opencode/dist/append-tools/bun-linux-x64/bundle-metafile.json /opt/opencode-tools-metafile.json',
        'PYTHONPATH=/opt/editing-release/training/src python -m opencode_rl.install_binary',
        '/usr/local/bin/opencode-tools --version')


def install_cpu_runtime():
    """Image-build callback: assemble the same bytes as the GPU source profile."""
    import os
    import shutil
    import sys
    sys.path.insert(0, '/opt/editing-release/training/scripts')
    from materialize import materialize
    profile = os.environ['EDITING_TRAINING_PROFILE']
    variant = 'opsd' if profile.startswith('opsd_') else ('decay' if profile == 'prm_decay' else 'fixed')
    runtime = materialize(variant, '/tmp/editing-runtime')
    for package in ('append_agent_rl', 'opencode_rl', 'prm_v2'):
        shutil.copytree(runtime / package, Path('/opt') / package)
    support = Path('/opt/editing-release/training/environment/support')
    Path('/opt/SkyRL').mkdir(exist_ok=True)
    shutil.copyfile('/opt/s12tools/s12_task.py', '/opt/SkyRL/s12_task.py')
    for name in ('prm_training_source.py',):
        shutil.copyfile(runtime / name, Path('/opt') / name)
    for name in ('modal_append_prm_train.py', 'grader_service.py', 'opsd_demo_quota.py',
                 'opsd_demo_retry_after.py', 'opsd_demo_admitted_request.py'):
        shutil.copyfile(support / name, Path('/opt') / name)
    tools = Path('/opt/s12-tools')
    tools.mkdir(exist_ok=True)
    for src, dest in (('observation_worker.py', 'worker.py'), ('tool_schema.py', 'schema.py')):
        shutil.copyfile(support / src, tools / dest)
    owner = Path('/opt/append_timeout_bin/bash')
    owner.parent.mkdir(exist_ok=True)
    shutil.copyfile(support / 'command_owner.py', owner)
    owner.chmod(0o555)
    sys.path.insert(0, '/opt')
    from modal_append_prm_train import build_manifest
    import prm_training_source
    prm_training_source.install_manifest(build_manifest('/opt/editing-release/training/provenance.json'))
    if profile.startswith('opsd_'):
        recipe = Path('/opt/editing-release/opsd') / ('recipes/reference_answers' if profile == 'opsd_reference'
                                                    else 'examples/train/answer_self_distillation')
        shutil.copytree(recipe, '/opt/SkyRL/examples/train/answer_self_distillation')


def build_images(config):
    import modal
    profile = config['profile']
    env = {'EDITING_TRAINING_APP': config['app'], 'OPSD_DEMONSTRATOR_APP': config['app'],
           'EDITING_TRAINING_CONFIG_JSON': json.dumps(config),
           'S12_TRAINING_JUDGE_ONLY': '1', 'S12_TRAINING_SUBMISSION_CHECKS': '1'}
    cpu = sources(modal.Image.debian_slim(python_version='3.12'), profile)
    cpu = (cpu.apt_install('git', 'curl', 'ca-certificates', 'unzip', 'ffmpeg', 'ripgrep', 'util-linux', 'libgomp1')
        .pip_install('faster-whisper==1.2.1', 'librosa==0.11.0', 'soundfile==0.13.1',
                     'pillow==11.3.0', 'numpy==2.2.6', 'aiohttp==3.13.3',
                     'httpx==0.28.1', 'jsonschema==4.26.0', 'huggingface-hub==0.36.2')
        .env(env).run_commands('python /opt/editing-release/training/environment/images.py --install-cpu-runtime')
        .run_commands('HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from faster_whisper.utils import download_model; download_model(\'small.en\', output_dir=\'/opt/s12-whisper-small.en\'); download_model(\'small.en\', output_dir=\'/opt/s12-asr-cache/hub/whisper-small.en\')"')
        .env({'HF_HOME': '/opt/s12-asr-cache', 'HF_HUB_CACHE': '/opt/s12-asr-cache/hub'}))
    if profile.startswith('opsd_'):
        cpu = cpu.pip_install('torch==2.11.0', index_url='https://download.pytorch.org/whl/cpu')
    # The grader's WhisperModel('small.en') lookup must be cached before offline use.
    cpu = cpu.run_commands('HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from faster_whisper import WhisperModel; WhisperModel(\'small.en\', device=\'cpu\', compute_type=\'int8\')"')
    cpu = native_host(cpu)
    sandbox = cpu.run_commands(
        'useradd --create-home --shell /bin/bash rollout',
        'mkdir -p /work /var/lib/ocrl', 'touch /opt/OCRL_SANDBOX_IMAGE').workdir('/work')
    gpu = sources(modal.Image.from_registry('nvidia/cuda:13.1.0-devel-ubuntu24.04', add_python='3.12'), profile)
    gpu = (gpu.apt_install('git', 'curl', 'ca-certificates', 'unzip', 'build-essential', 'ninja-build', 'ffmpeg', 'libgomp1')
        .pip_install('uv==0.11.23')
        .run_commands('git init /opt/SkyRL',
            'git -C /opt/SkyRL remote add origin https://github.com/NovaSky-AI/SkyRL.git',
            f'git -C /opt/SkyRL fetch --depth 1 origin {SKYRL}',
            'git -C /opt/SkyRL checkout --detach FETCH_HEAD',
            'cp /opt/s12tools/s12_task.py /opt/SkyRL/s12_task.py',
            'cp /opt/editing-release/training/environment/uv.lock /opt/SkyRL/uv.lock',
            'cd /opt/SkyRL && uv sync --frozen --extra fsdp --no-install-project',
            'uv pip install --python /opt/SkyRL/.venv/bin/python modal==1.5.4 aiohttp==3.13.3 httpx==0.28.1 jsonschema==4.26.0')
        .env({**env, 'OCRL_ISOLATED_IMAGE_BUILD': '1'})
        .run_commands('PYTHONPATH=/opt/editing-release/training/scripts /opt/SkyRL/.venv/bin/python /opt/editing-release/training/environment/reconstruct.py --skyrl-root /opt/SkyRL --profile ' + profile + ' --install-runtime'))
    gpu = native_host(gpu).workdir('/opt/SkyRL')
    return {'sandbox': sandbox, 'cpu': cpu, 'gpu': gpu}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-cpu-runtime', action='store_true', required=True)
    parser.parse_args()
    install_cpu_runtime()
