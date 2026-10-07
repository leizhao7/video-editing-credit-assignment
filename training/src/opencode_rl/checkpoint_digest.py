"""Candidate checkpoint-only SHA256 isolation; not installed in any trainer.

The SHA algorithm and full-file byte coverage are unchanged. A fixed, verified
coreutils child process performs the digest so the trainer's Python GIL cannot
serialize every 256 KiB read with rollout preparation. No checkpoint publication,
copy, deletion, runtime injection or policy/optimizer operation lives here.
"""
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess

BACKEND = Path('/usr/bin/sha256sum')
BACKEND_SHA256 = 'e484c36c0613879fbd11058f7a2e194783bced59d4805d64c3d4b0d7f664094d'


def verify_backend():
    if (not BACKEND.is_file() or BACKEND.is_symlink()
            or hashlib.sha256(BACKEND.read_bytes()).hexdigest() != BACKEND_SHA256):
        raise ValueError('Checkpoint SHA backend differs from the qualified image')


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def checkpoint_sha_file(path):
    """Hash all bytes, preserving exact filename binding and rejecting mutation.

    Call only after native checkpoint writers are finalized. Parent directory
    and complete model/optimizer/sampler checks remain owned by persistence.
    No new arbitrary deadline, retries, fallback digest or concurrent hash pool.
    """
    verify_backend()
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Checkpoint digest requires an ordinary file')
    result = subprocess.run([str(BACKEND), '--zero', '--', str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    raw = result.stdout
    # --zero suppresses filename escaping, including for spaces/newlines. Match
    # the entire returned filename, not just a digest-looking first token.
    if (not isinstance(raw, bytes) or len(raw) != 67 + len(os.fsencode(path))
            or re.fullmatch(rb'[0-9a-f]{64}', raw[:64]) is None
            or raw[64:] != b'  ' + os.fsencode(path) + b'\0'):
        raise ValueError('Checkpoint digest output/filename binding differs')
    after = path.lstat()
    if not stat.S_ISREG(after.st_mode) or _identity(before) != _identity(after):
        raise ValueError('Checkpoint changed while hashing')
    return raw[:64].decode('ascii')
