"""Reconstruct the full reviewed SkyRL source patch, then install runtime code.

Source-only patch verification is safe on an external pinned checkout. Installing
the runtime is restricted to an explicitly selected isolated image build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY / 'training/scripts'))
from materialize import materialize
from source_integrity import source_file


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply_source_patch(root, patch, entries, patch_sha256):
    root, patch = Path(root).resolve(), Path(patch)
    if sha(patch) != patch_sha256:
        raise ValueError('Source patch differs from its reviewed manifest')
    actual = {name: sha(source_file(root, name)) for name in entries}
    if all(actual[name] == row['after_sha256'] for name, row in entries.items()):
        return {'changed': False, 'files': len(entries)}
    if not all(actual[name] == row['before_sha256'] for name, row in entries.items()):
        raise ValueError('Checkout is neither the complete pinned preimage nor the reviewed postimage')
    subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=root, check=True)
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=root, check=True)
    if any(sha(source_file(root, name)) != row['after_sha256'] for name, row in entries.items()):
        raise ValueError('Applied source bytes differ from the reviewed postimage')
    return {'changed': True, 'files': len(entries)}


def reconstruct_skyrl(root, *, decay=False):
    manifest = tomllib.loads((HERE / 'manifest.toml').read_text())
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != manifest['skyrl_commit']:
        raise ValueError('Use the pinned SkyRL commit in the source manifest')
    if decay:
        path = REPOSITORY / 'training/patches/skyrl'
        native = json.loads((path / 'manifest.json').read_bytes())
        combined = {name: row['after_sha256'] for name, row in manifest['skyrl_files'].items()}
        combined.update({row['path']: row['after_sha256'] for row in native['files']})
        if (sha(HERE / 'skyrl-full.patch') == manifest['patch_sha256'] and
                sha(path / 'native-startup.patch') == native['patch_sha256'] and
                all(sha(source_file(root, name)) == expected for name, expected in combined.items())):
            return {'changed': False, 'files': len(combined), 'decay': {'changed': False}}
    result = apply_source_patch(root, HERE / 'skyrl-full.patch',
                                manifest['skyrl_files'], manifest['patch_sha256'])
    if decay:
        path = REPOSITORY / 'training/patches/skyrl'
        native = json.loads((path / 'manifest.json').read_bytes())
        result['decay'] = apply_source_patch(root, path / 'native-startup.patch',
            {row['path']: row for row in native['files']}, native['patch_sha256'])
    return result


def install_runtime(profile):
    if os.environ.get('OCRL_ISOLATED_IMAGE_BUILD') != '1' or sys.platform != 'linux':
        raise ValueError('Runtime installation requires an explicitly selected isolated Linux image build')
    if profile not in ('grpo', 'prm_fixed', 'prm_decay', 'opsd_reference', 'opsd_suggestions'):
        raise ValueError('Unknown training profile')
    destination = Path('/opt')
    with tempfile.TemporaryDirectory(prefix='editing-image-runtime-') as temp:
        variant = 'opsd' if profile.startswith('opsd_') else ('decay' if profile == 'prm_decay' else 'fixed')
        source = materialize(variant, Path(temp) / 'runtime')
        for package in ('append_agent_rl', 'opencode_rl', 'prm_v2'):
            if (destination / package).exists():
                raise ValueError('Runtime package already exists in the image: ' + package)
            shutil.copytree(source / package, destination / package)
        shutil.copyfile(source / 'prm_training_source.py', destination / 'prm_training_source.py')
    shutil.copyfile(HERE / 'support/modal_append_prm_train.py', destination / 'modal_append_prm_train.py')
    for patch in (HERE / 'patches').glob('*.py'):
        target = Path('/patch') / patch.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(patch, target)
    owner = destination / 'append_timeout_bin/bash'
    owner.parent.mkdir(exist_ok=True)
    shutil.copyfile(HERE / 'support/command_owner.py', owner)
    owner.chmod(0o555)
    subprocess.run([sys.executable, str(destination / 'modal_append_prm_train.py'),
                    '--provenance', str(REPOSITORY / 'training/provenance.json')], check=True)
    # Only these patches target installed dependency packages. SkyRL itself was
    # reconstructed from the complete hash-checked patch, not moving source.
    for name in ('vlm_device_fix.py', 'mm_cache_miss_fix.py', 'vllm_nccl_drain_fix.py'):
        subprocess.run([sys.executable, str(Path('/patch') / name)], check=True)
    env = {**os.environ, 'PYTHONPATH': '/opt:/opt/SkyRL'}
    for module in ('opencode_rl.install_training_fixes', 'opencode_rl.install_startup_fixes'):
        subprocess.run([sys.executable, '-m', module], env=env, check=True)
    if profile.startswith('opsd_'):
        recipe = REPOSITORY / ('opsd/recipes/reference_answers' if profile == 'opsd_reference'
                               else 'opsd/examples/train/answer_self_distillation')
        target = Path('/opt/SkyRL/examples/train/answer_self_distillation')
        if target.exists():
            raise ValueError('OPSD recipe already exists in the image')
        shutil.copytree(recipe, target)
        subprocess.run([sys.executable, '-m', 'examples.train.answer_self_distillation.install'],
                       env=env, check=True)
    return {'profile': profile, 'installed': True, 'gpu_qualified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skyrl-root', type=Path, required=True)
    parser.add_argument('--profile', choices=('grpo', 'prm_fixed', 'prm_decay', 'opsd_reference', 'opsd_suggestions'), default='grpo')
    parser.add_argument('--install-runtime', action='store_true')
    args = parser.parse_args(argv)
    result = reconstruct_skyrl(args.skyrl_root, decay=args.profile == 'prm_decay')
    if args.install_runtime:
        if args.skyrl_root.resolve() != Path('/opt/SkyRL'):
            raise ValueError('Image runtime uses the explicit /opt/SkyRL layout')
        result['runtime'] = install_runtime(args.profile)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
