"""Trusted public-only staging shared by OpenCode rollout callers (train/eval)."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import tarfile
import tempfile
from .contract import ParityError, canonical, digest
from .runtime import task_prompt
from .store import atomic_bytes

RELEASE_SHA256 = "a278ef9f59eab8fcab96dda59f72dc82cbd3bc4350392611bc526a86d5cb3c16"


def public_file(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and '..' not in path.parts and
            (name in ('task.json', 'task_prompt.md') or
             (len(path.parts) > 1 and path.parts[0] == 'initial_state')))


def validate_bundle_prompt(bundle, metadata):
    """Fail closed on old cached environment instructions; never rewrite them.

    Protocol preflights are synthetic non-dataset inputs with their own prompt.
    Real train/eval tasks must carry the current shared task_prompt suffix.
    """
    raw = (Path(bundle) / 'task.txt').read_bytes()
    if metadata.get('task') == 'protocol_preflight':
        return
    if hashlib.sha256(raw).hexdigest() != metadata.get('prompt_sha256'):
        raise ParityError('Prepared task prompt differs from its recorded hash')
    suffix = task_prompt('').encode()
    if not raw.endswith(suffix) or (metadata.get('environment_prompt_sha256') is not None and
            metadata['environment_prompt_sha256'] != hashlib.sha256(suffix).hexdigest()):
        raise ParityError('Stale prepared environment prompt; prepare a new bundle/run, preserving the old evaluation')


def release_tasks(path, split, *, expected_sha256=None):
    expected_sha256 = RELEASE_SHA256 if expected_sha256 is None else expected_sha256
    if not isinstance(expected_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ParityError('An explicit release SHA-256 is required')
    if split not in ('train', 'val'):
        raise ParityError('Unknown release split')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ParityError('Dataset release changed')
    all_rows = json.loads(raw)['tasks']
    rows = [x for x in all_rows if x['probe']['split'] == split]
    if len({r['task'] for r in all_rows}) != len(all_rows):
        raise ParityError('Duplicate task identity')
    return rows


def validate_source(source, row):
    source = Path(source)
    for name, expected in row['files'].items():
        path = source / name
        if '..' in PurePosixPath(name).parts or PurePosixPath(name).is_absolute():
            raise ParityError('Unsafe corpus path')
        if path.is_symlink() or not path.is_file():
            raise ParityError(f'Missing regular corpus file: {name}')
        with path.open('rb') as handle:
            sha = hashlib.file_digest(handle, 'sha256').hexdigest()
        if sha != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ParityError(f'Corpus corruption: {row["task"]}/{name}')


def music_manifest(source):
    source = Path(source)
    man = json.loads((source / 'initial_state/manifest.json').read_bytes())
    music = man.get('music')
    # Only the manifest's explicit original music entry is authorized. Never
    # treat dialogue files or arbitrary .wav files as music.
    if not music:
        return {'assets': []}
    if not isinstance(music, dict) or not isinstance(music.get('file'), str):
        raise ParityError('Unreviewed music manifest schema')
    rel = PurePosixPath('initial_state') / music['file']
    if not public_file(str(rel)):
        raise ParityError('Music path escaped original inputs')
    # Modal corpus roots can be symlinked release pointers. Resolve the trusted
    # source after validating its bytes, not the policy's output directory.
    # The music preparer still rejects any unresolved symlink in its input.
    path = (source / rel).resolve(strict=True)
    if not path.is_relative_to(source.resolve(strict=True)):
        raise ParityError('Music source escaped trusted task')
    with path.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'assets': [{'music_id': 'music_0', 'source_path': str(path),
        'source_sha256': sha, 'public_path': str(rel)}]}


def prepare_bundle(source, row, destination, *, release_sha256=None):
    release_sha256 = RELEASE_SHA256 if release_sha256 is None else release_sha256
    if not isinstance(release_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', release_sha256):
        raise ParityError('An explicit release SHA-256 is required')
    import s12_task
    source, destination = Path(source), Path(destination)
    validate_source(source, row)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ocrl-public-') as temp:
        root = Path(temp)
        (root / 'initial_state').symlink_to(source / 'initial_state', target_is_directory=True)
        man = json.loads((source / 'initial_state/manifest.json').read_bytes())
        items = s12_task.visual_items(str(root), row['stage'], man)
        made = s12_task.extract_frames(str(root), items, s12_task.n_frames(row['stage']))
        public = {n: meta for n, meta in row['files'].items() if public_file(n)}
        with tarfile.open(destination / 'input.tar', 'w') as archive:
            for name in sorted(public):
                archive.add(source / name, arcname=name, recursive=False)
            archive.add(root / 'frames', arcname='frames')
        # No old loop's extra quotas or hidden gold-count access. This prompt
        # and the same task_bundle function are the new RL caller's entrypoint.
        prompt = task_prompt((source / 'task_prompt.md').read_text() +
            f'\n\nWork in /work. frames/ contains {len(made)} prepared overview frames. '
            'Full original media is in initial_state/. Write the required files under output/.')
        atomic_bytes(destination / 'task.txt', prompt.encode())
        metadata = {'task': row['task'], 'stage': row['stage'], 'source_root': '_tt_' + row['probe']['split'],
            'release_sha256': release_sha256, 'public_files': public, 'frames': made,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'environment_prompt_sha256': hashlib.sha256(task_prompt('').encode()).hexdigest(),
            'private_files_staged': False}
        atomic_bytes(destination / 'bundle.json', canonical(metadata))
    return metadata
