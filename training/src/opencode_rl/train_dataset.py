"""Frozen public bundles, stable dataset UIDs and private grader input hashes."""
import hashlib
import json
import threading
from pathlib import Path

from .contract import ParityError, canonical, digest
from .task_bundle import release_tasks, validate_bundle_prompt, RELEASE_SHA256
from .music_observation import validate_catalog
from .store import atomic_bytes

BUNDLE_FILES = ('bundle.json', 'task.txt', 'input.tar', 'music_catalog.json')
SOURCE_FILES = ('initial_state/manifest.json', 'task_prompt.md', 'probe.json', 'verifier/answer.json')


def sha_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def freeze_prepared(release, bundles, destination, split='train'):
    """Called only by trusted preparation, never by a policy sandbox."""
    if split not in ('train', 'val'):
        raise ParityError('Unknown prepared split')
    rows = release_tasks(release, split)
    records = []
    for row in rows:
        root = Path(bundles) / row['task']
        meta = json.loads((root / 'bundle.json').read_bytes())
        if (meta['task'] != row['task'] or meta['stage'] != row['stage'] or
                meta['source_root'] != '_tt_' + split or meta['release_sha256'] != RELEASE_SHA256 or
                meta.get('private_files_staged') is not False):
            raise ParityError('Training bundle identity/leakage mismatch')
        validate_bundle_prompt(root, meta)
        validate_catalog(json.loads((root / 'music_catalog.json').read_bytes()))
        records.append({'task': row['task'], 'stage': row['stage'],
            'files': {name: sha_file(root / name) for name in BUNDLE_FILES},
            'source_hashes': {name: row['files'][name]['sha256'] for name in SOURCE_FILES if name in row['files']}})
    record = {'format': f'ocrl-prepared-{split}-v1', 'release_sha256': RELEASE_SHA256, 'tasks': records}
    path = Path(destination)
    body = canonical(record)
    if path.exists() and path.read_bytes() != body:
        raise ParityError('Prepared release is immutable; use a new preparation directory')
    atomic_bytes(path, body)
    return record


class PreparedDataset:
    def __init__(self, manifest, bundles, release, split='train'):
        if split not in ('train', 'val'):
            raise ParityError('Unknown prepared split')
        self.split, self.source_root = split, '_tt_' + split
        self.bundles = Path(bundles)
        self.manifest = json.loads(Path(manifest).read_bytes())
        self.sha256 = digest(self.manifest)
        expected = release_tasks(release, split)
        self.rows = self.manifest['tasks']
        if (self.manifest.get('format') != f'ocrl-prepared-{split}-v1' or
                self.manifest.get('release_sha256') != RELEASE_SHA256 or
                [r['task'] for r in self.rows] != [r['task'] for r in expected]):
            raise ParityError('Prepared release/order changed or includes the other split')
        for actual, source in zip(self.rows, expected):
            expected_sources = {name: source['files'][name]['sha256'] for name in SOURCE_FILES if name in source['files']}
            if actual['stage'] != source['stage'] or actual['source_hashes'] != expected_sources:
                raise ParityError('Prepared private source binding changed')
        self.by_task = {r['task']: r for r in self.rows}
        self.verified = set()
        self._verification_locks = {task: threading.Lock() for task in self.by_task}

    def verify_task(self, task):
        # All eight siblings start concurrently. Verify immutable bundle bytes
        # once per task, without serializing unrelated tasks behind one lock.
        with self._verification_locks[task]:
            return self._verify_task(task)

    def _verify_task(self, task):
        row = self.by_task[task]
        if task not in self.verified:
            root = self.bundles / task
            if set(row['files']) != set(BUNDLE_FILES) or any(sha_file(root / n) != sha for n, sha in row['files'].items()):
                raise ParityError('Prepared task bytes changed')
            meta = json.loads((root / 'bundle.json').read_bytes())
            validate_bundle_prompt(root, meta)
            if (meta['task'] != task or meta['source_root'] != self.source_root or meta['stage'] != row['stage'] or
                    meta.get('private_files_staged') is not False or meta.get('release_sha256') != RELEASE_SHA256):
                raise ParityError('Prepared task identity changed')
            self.verified.add(task)
        return {**row, 'source_root': self.source_root}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (row['task'], 'opencode', {'task': row['task'], 'stage': row['stage']}, row['task'])

    def collate_fn(self, items):
        return [{'prompt': p, 'env_class': e, 'env_extras': x, 'uid': u} for p, e, x, u in items]
