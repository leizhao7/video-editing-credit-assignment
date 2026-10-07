"""Root-only staging/archiving helper inside the isolated sandbox; no grading."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sqlite3
import tarfile
import tempfile


def transfer_path(raw):
    path = Path(raw)
    if path.parent != Path('/root') or (path.name not in ('input.tar', 'task.txt', 'music_catalog.json',
        'config.json', 'contract.json', 'trajectory.tar.gz') and not path.name.startswith('submitted_')):
        raise ValueError('Transfer path outside trusted sandbox artifacts')
    if path.is_symlink():
        raise ValueError('Symlink artifact')
    return path


def receive(raw, size):
    import sys
    path = transfer_path(raw)
    if not 0 <= size <= 4 * 1024**3:
        raise ValueError('Invalid transfer length')
    sha = hashlib.sha256()
    left = size
    with path.open('xb') as handle:
        while left:
            chunk = sys.stdin.buffer.read(min(left, 1024**2))
            if not chunk:
                raise ValueError('Truncated input transfer')
            handle.write(chunk)
            sha.update(chunk)
            left -= len(chunk)
    print(json.dumps({'bytes': size, 'sha256': sha.hexdigest()}))


def send(raw, offset=0, length=None):
    import base64
    sha, count = hashlib.sha256(), 0
    path = transfer_path(raw)
    size = path.stat().st_size
    if offset < 0 or offset > size or (length is not None and not 0 <= length <= 8 * 1024**2):
        raise ValueError('Invalid artifact range')
    left = size - offset if length is None else min(length, size - offset)
    with path.open('rb') as handle:
        handle.seek(offset)
        while left and (chunk := handle.read(min(192 * 1024, left))):
            sha.update(chunk)
            count += len(chunk)
            left -= len(chunk)
            print(json.dumps({'data_b64': base64.b64encode(chunk).decode()}), flush=True)
    print(json.dumps({'bytes': count, 'sha256': sha.hexdigest()}), flush=True)


def stop_policy():
    import pwd
    import signal
    import time
    uid = pwd.getpwnam('rollout').pw_uid
    if uid == 0:
        raise RuntimeError('Policy must not be root')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for path in Path('/proc').glob('[0-9]*/status'):
            try:
                owner = next(line for line in path.read_text().splitlines() if line.startswith('Uid:'))
                if int(owner.split()[1]) == uid:
                    os.kill(int(path.parent.name), sig)
            except (OSError, StopIteration):
                pass
        if sig == signal.SIGTERM:
            time.sleep(.2)


def stage():
    with tarfile.open('/root/input.tar') as archive:
        for member in archive:
            parts = Path(member.name).parts
            if (member.name.startswith('/') or '..' in parts or not parts or
                parts[0] not in ('initial_state', 'frames', 'task.json', 'task_prompt.md') or
                member.issym() or member.islnk() or not (member.isfile() or member.isdir())):
                raise ValueError('Non-public or unsafe staging archive member')
        archive.extractall('/work', filter='data')
    Path('/work/output').mkdir(exist_ok=True)
    import pwd
    user = pwd.getpwnam('rollout')
    os.chown('/work', user.pw_uid, user.pw_gid)
    os.chown('/work/output', user.pw_uid, user.pw_gid)
    for path in Path('/work').rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)


def snapshot(attempt):
    # Keep artifacts inert. Do not dereference policy-created symlinks or run
    # anything from /work on a privileged host. DB backup includes live WAL.
    from .contract import ID
    if not ID.fullmatch(attempt):
        raise ValueError('Invalid attempt')
    target = Path('/root/trajectory.tar.gz')
    with tempfile.TemporaryDirectory(prefix='ocrl-db-') as temp:
        databases = []
        runtime = Path('/var/lib/ocrl') / attempt
        for db in runtime.rglob('*.db'):
            if db.is_symlink():
                continue
            dest = Path(temp) / str(len(databases))
            try:
                with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as src, sqlite3.connect(dest) as dst:
                    src.backup(dst)
                databases.append((dest, 'sqlite_snapshots/' + str(db.relative_to(runtime))))
            except sqlite3.Error:
                pass  # Original DB and WAL are still archived below.
        with tarfile.open(target, 'w:gz', compresslevel=1, dereference=False) as archive:
            if runtime.exists():
                archive.add(runtime, arcname='runtime')
            for child in Path('/work').iterdir() if Path('/work').exists() else []:
                if child.name not in ('initial_state', 'frames'):
                    archive.add(child, arcname='work/' + child.name)
            for db, name in databases:
                archive.add(db, arcname=name)
    with target.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    print(json.dumps({'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha}))


def collect(stage_name):
    names = {'s_1': 'selection.json', 's_2': 'order.json', 's_3': 'cuts.json', 's_4e': 'edit.json'}
    result = {}
    for name, limit in [(names[stage_name], 2 * 1024**2), ('preview.mp4', 128 * 1024**2)]:
        path = Path('/work/output') / name
        if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
            result[name] = {'valid_file': False, 'reason': 'missing or symlink'}
        elif not 0 < path.stat().st_size <= limit:
            result[name] = {'valid_file': False, 'reason': 'size out of bound'}
        else:
            # Freeze bytes while the policy is no longer running, then return
            # only this root-owned copy for grading. Archive always keeps original.
            import shutil
            dest = Path('/root') / ('submitted_' + name)
            shutil.copyfile(path, dest)
            with dest.open('rb') as handle:
                sha = hashlib.file_digest(handle, 'sha256').hexdigest()
            result[name] = {'valid_file': True, 'path': str(dest), 'bytes': dest.stat().st_size, 'sha256': sha}
    print(json.dumps(result))


if __name__ == '__main__':
    if os.geteuid() != 0 or not Path('/opt/OCRL_SANDBOX_IMAGE').exists():
        raise RuntimeError('Isolated sandbox root only')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'snapshot', 'collect', 'stop_policy', 'receive', 'send'])
    parser.add_argument('identity', nargs='?')
    parser.add_argument('--size', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--length', type=int)
    args = parser.parse_args()
    {'stage': lambda: stage(), 'snapshot': lambda: snapshot(args.identity), 'stop_policy': stop_policy,
     'receive': lambda: receive(args.identity, args.size), 'send': lambda: send(args.identity, args.offset, args.length),
     'collect': lambda: collect(args.identity)}[args.action]()
