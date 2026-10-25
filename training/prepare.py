"""Bind reviewed task folders to a private, immutable training release.

All generated material stays outside the repository. Only task.json,
task_prompt.md, and initial_state/ may enter an agent's public input bundle;
verifier material remains in the separate trusted corpus.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
STAGES = {'s1': 's_1', 's_1': 's_1', 's2': 's_2', 's_2': 's_2',
          's3': 's_3', 's_3': 's_3', 's4e': 's_4e', 's_4e': 's_4e'}
COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,159}')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def file_identity(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': Path(path).stat().st_size}


def create_release(accepted_index, destination, *, validation_films=()):
    index = Path(accepted_index).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if destination == REPOSITORY or destination.is_relative_to(REPOSITORY):
        raise ValueError('Training data must be prepared outside the repository')
    if destination.exists():
        raise ValueError('Use a new release directory; existing releases are immutable')
    paths = json.loads(index.read_bytes())
    if not isinstance(paths, list) or not paths or not all(isinstance(x, str) for x in paths):
        raise ValueError('Expected a nonempty accepted_tasks.json list of absolute task paths')
    if any(not Path(p).is_absolute() for p in paths):
        raise ValueError('Accepted task paths must be absolute')
    validation_films = set(validation_films)
    sources, rows, films = {}, [], set()
    for name in paths:
        source = Path(name)
        if source.is_symlink() or not source.is_dir():
            raise ValueError('Accepted task must be a regular directory')
        source = source.resolve(strict=True)
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError('Release destination overlaps an input task')
        film = source.parent.name
        if not COMPONENT.fullmatch(film) or not COMPONENT.fullmatch(source.name):
            raise ValueError('Film and task names must be simple directory components')
        uid = film + '__' + source.name
        if not COMPONENT.fullmatch(uid) or uid in sources:
            raise ValueError('Duplicate or overly long task identity')
        task = json.loads((source / 'task.json').read_bytes())
        stage = STAGES.get(str(task.get('stage', '')).split('/')[0].lower())
        if stage is None:
            raise ValueError('Task has an unsupported editing stage')
        required = ('task.json', 'task_prompt.md', 'probe.json',
                    'initial_state/manifest.json', 'verifier/answer.json')
        for relative in required:
            if not (source / relative).is_file():
                raise ValueError('Missing trusted task input: ' + relative)
        selected = []
        for root_name in ('task.json', 'task_prompt.md', 'probe.json', 'initial_state', 'verifier'):
            root = source / root_name
            candidates = [root, *root.rglob('*')] if root.is_dir() else [root]
            for path in candidates:
                if path.is_symlink():
                    raise ValueError('Task sources may not contain symlinks')
                if path.is_file():
                    selected.append(path)
        files = {p.relative_to(source).as_posix(): file_identity(p) for p in sorted(selected)}
        split = 'val' if film in validation_films else 'train'
        rows.append({'task': uid, 'stage': stage, 'probe': {'split': split, 'film': film},
                     'files': files})
        sources[uid] = source
        films.add(film)
    if validation_films - films:
        raise ValueError('Validation film names do not occur in the accepted task index')
    train_count = sum(r['probe']['split'] == 'train' for r in rows)
    if not train_count:
        raise ValueError('At least one training task is required')
    rows.sort(key=lambda row: row['task'])
    release = {'format': 'video-editing-task-release-v1', 'train': train_count,
               'val': len(rows) - train_count, 'tasks': rows}
    # Validate all paths and identities before creating any output.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for row in rows:
            root = destination / 'corpus' / ('_tt_' + row['probe']['split']) / row['task']
            for name, expected in row['files'].items():
                source = sources[row['task']] / name
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target, follow_symlinks=False)
                if target.is_symlink() or file_identity(target) != expected:
                    raise ValueError('Task source changed while preparing the release')
        raw = canonical(release)
        (destination / 'release.json').write_bytes(raw)
        result = {'release_sha256': hashlib.sha256(raw).hexdigest(),
                  'accepted_index_sha256': file_identity(index)['sha256'],
                  'train_tasks': train_count, 'validation_tasks': release['val'],
                  'validation_films': sorted(validation_films), 'public_inputs_prepared': False}
        (destination / 'release_receipt.json').write_bytes(canonical(result))
    except BaseException:
        shutil.rmtree(destination)
        raise
    return result


async def prepare_inputs(directory, *, reference_answers=False):
    """Prepare image/music inputs on the trusted host, never in an agent sandbox."""
    directory = Path(directory).expanduser().resolve(strict=True)
    if directory == REPOSITORY or directory.is_relative_to(REPOSITORY):
        raise ValueError('Training data must stay outside the repository')
    sys.path[:0] = [str(REPOSITORY / 'training/src'), str(REPOSITORY / 'evaluation')]
    from opencode_rl.contract import canonical as runtime_canonical
    from opencode_rl.task_bundle import prepare_bundle, release_tasks, music_manifest, validate_source
    from opencode_rl.train_dataset import freeze_prepared, PreparedDataset, BUNDLE_FILES
    from opencode_rl.music_observation import policy_sha256, validate_catalog
    from opencode_rl.prepare_music import GeminiMusicPreparer
    import os
    import aiohttp
    release_path = directory / 'release.json'
    receipt = json.loads((directory / 'release_receipt.json').read_bytes())
    identity = receipt['release_sha256']
    counts = {}
    for split in ('train', 'val'):
        rows = release_tasks(release_path, split, expected_sha256=identity)
        if not rows:
            continue
        bundles = directory / 'prepared' / split / 'tasks'
        for row in rows:
            source = directory / 'corpus' / ('_tt_' + split) / row['task']
            validate_source(source, row)
            bundle = bundles / row['task']
            ready = bundle / 'ready.json'
            if ready.exists():
                saved = json.loads(ready.read_bytes())
                if set(saved) != set(BUNDLE_FILES) or any(
                        file_identity(bundle / name)['sha256'] != sha for name, sha in saved.items()):
                    raise ValueError('Previously prepared inputs changed')
                continue
            prepare_bundle(source, row, bundle, release_sha256=identity)
            music = music_manifest(source)
            if music['assets']:
                key = os.environ.get('GOOGLE_AI_STUDIO_API_KEY')
                if not key:
                    raise ValueError('Music preparation requires GOOGLE_AI_STUDIO_API_KEY')
                preparer = GeminiMusicPreparer(key, str(directory / 'private_music_cache' / row['task']), concurrency=4)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as client:
                    catalog = await preparer.prepare(client, music)
            else:
                catalog = validate_catalog({'policy_sha256': policy_sha256(), 'assets': []})
            (bundle / 'music_catalog.json').write_bytes(runtime_canonical(catalog))
            ready.write_bytes(runtime_canonical({name: file_identity(bundle / name)['sha256'] for name in BUNDLE_FILES}))
        manifest = directory / 'prepared' / split / 'manifest.json'
        freeze_prepared(release_path, bundles, manifest, split,
                        expected_release_sha256=identity)
        dataset = PreparedDataset(manifest, bundles, release_path, split,
                                  expected_release_sha256=identity)
        for task in dataset.by_task:
            dataset.verify_task(task)
        counts[split] = {'task_count': len(dataset), 'sha256': dataset.sha256}
    if reference_answers:
        sys.path.insert(0, str(REPOSITORY / 'opsd'))
        from examples.train.answer_self_distillation.answers import build_catalog
        private = directory / 'private'
        private.mkdir(exist_ok=True)
        catalog = build_catalog(release_path, directory / 'corpus')
        target = private / 'reference_answers.json'
        raw = runtime_canonical(catalog)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('Reference answers changed; create a new release')
        target.write_bytes(raw)
        counts['reference_answers_sha256'] = hashlib.sha256(raw).hexdigest()
    result = {**receipt, 'public_inputs_prepared': True, 'prepared': counts}
    (directory / 'preparation_receipt.json').write_bytes(runtime_canonical(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accepted-tasks', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--validation-film', action='append', default=[])
    parser.add_argument('--prepare-inputs', action='store_true',
                        help='Extract frames and prepare music; music may call Gemini')
    parser.add_argument('--reference-answers', action='store_true',
                        help='Prepare a separate teacher-only training answer catalog')
    args = parser.parse_args(argv)
    destination = Path(args.destination).expanduser().resolve()
    if destination.exists():
        receipt = json.loads((destination / 'release_receipt.json').read_bytes())
        if file_identity(Path(args.accepted_tasks).expanduser())['sha256'] != receipt['accepted_index_sha256']:
            raise ValueError('Accepted task index changed; create a new release')
        if sorted(set(args.validation_film)) != receipt['validation_films']:
            raise ValueError('Validation split changed; create a new release')
        result = receipt
    else:
        result = create_release(args.accepted_tasks, args.destination,
                                validation_films=args.validation_film)
    if args.prepare_inputs:
        result = asyncio.run(prepare_inputs(destination, reference_answers=args.reference_answers))
    elif args.reference_answers:
        parser.error('--reference-answers requires --prepare-inputs')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
