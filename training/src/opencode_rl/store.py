"""Durable request journal and content-addressed input tensors; no pickle."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from .contract import HASH, ID, ParityError, canonical, digest


def atomic_bytes(path: Path, body: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value):
    """Serialization belongs in the same off-loop durable operation as I/O."""
    atomic_bytes(path, canonical(value))


class TraceStore:
    def __init__(self, root, *, packed=None, read_cache_bytes=0):
        if type(read_cache_bytes) is not int or read_cache_bytes < 0:
            raise ValueError('Nonnegative read cache byte budget required')
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.packed = os.environ.get('OCRL_PACK_BLOBS') == '1' if packed is None else packed
        self._blob_lock = threading.RLock()
        self.mm_index_lock = threading.RLock()
        self._db = None
        # Attempt-local, byte-bounded cache of already hash-verified immutable
        # bytes. Never cache mutable tensors or share across trace roots.
        self._read_cache = OrderedDict()
        self._read_cache_limit = read_cache_bytes
        self._read_cache_size = 0

    def clear_read_cache(self):
        with self._blob_lock:
            self._read_cache.clear()
            self._read_cache_size = 0

    def _packed_db(self):
        if self._db is None:
            self._db = sqlite3.connect(self.root / 'blobs.sqlite3', check_same_thread=False, timeout=60)
            # Image tensors are multi-MB: 4K pages caused thousands of tiny
            # filesystem writes while holding the shared blob lock. Set this
            # BEFORE entering WAL for new databases. Existing databases retain
            # their original page size; never VACUUM/migrate a live trace.
            self._db.execute('PRAGMA page_size=65536')
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('PRAGMA synchronous=FULL')
            # Preserve the roughly 4MB auto-checkpoint threshold, not 1000
            # enlarged pages (~64MB). Every returned blob is still FULL-synced.
            page_size = self._db.execute('PRAGMA page_size').fetchone()[0]
            if page_size == 65536:
                self._db.execute('PRAGMA wal_autocheckpoint=64')
            self._db.execute('CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY, body BLOB NOT NULL)')
            self._db.commit()
        return self._db

    def blob(self, body: bytes) -> dict:
        sha = hashlib.sha256(body).hexdigest()
        if self.packed:
            with self._blob_lock:
                db = self._packed_db()
                old = db.execute('SELECT body FROM blobs WHERE sha256=?', (sha,)).fetchone()
                if old is not None and old[0] != body:
                    raise ParityError('Corrupted packed content-addressed blob')
                if old is None:
                    db.execute('INSERT INTO blobs VALUES (?,?)', (sha, body))
                    db.commit()
            return {'sha256': sha, 'bytes': len(body), 'storage': 'sqlite-v1'}
        path = self.root / "blobs" / sha
        if path.exists():
            if path.read_bytes() != body:
                raise ParityError("Corrupted content-addressed blob")
        else:
            atomic_bytes(path, body)
        return {"sha256": sha, "bytes": len(body)}

    def read_blob(self, ref):
        if not isinstance(ref, dict) or not HASH.fullmatch(str(ref.get("sha256", ""))):
            raise ParityError("Invalid blob reference")
        if type(ref.get('bytes')) is not int or ref['bytes'] < 0:
            raise ParityError('Invalid blob size')
        key = (ref['sha256'], ref['bytes'], ref.get('storage'))
        with self._blob_lock:
            if key in self._read_cache:
                self._read_cache.move_to_end(key)
                return self._read_cache[key]
        if ref.get('storage') == 'sqlite-v1':
            from contextlib import closing
            with closing(sqlite3.connect(f'file:{self.root / "blobs.sqlite3"}?mode=ro', uri=True, timeout=60)) as db:
                row = db.execute('SELECT body FROM blobs WHERE sha256=?', (ref['sha256'],)).fetchone()
            if row is None:
                raise ParityError('Missing packed blob')
            body = row[0]
        elif 'storage' in ref:
            raise ParityError('Unknown blob storage format')
        else:
            body = (self.root / "blobs" / ref["sha256"]).read_bytes()
        if len(body) != ref["bytes"] or hashlib.sha256(body).hexdigest() != ref["sha256"]:
            raise ParityError("Blob integrity mismatch")
        if 0 < len(body) <= self._read_cache_limit:
            with self._blob_lock:
                if key not in self._read_cache:
                    while self._read_cache_size + len(body) > self._read_cache_limit:
                        _, evicted = self._read_cache.popitem(last=False)
                        self._read_cache_size -= len(evicted)
                    self._read_cache[key] = body
                    self._read_cache_size += len(body)
        return body

    def tensor(self, value):
        import numpy as np
        if type(value).__module__.startswith("torch"):
            import torch
            tensor = value.detach().cpu().contiguous()
            # Raw byte view preserves bfloat16 without rounding through float32.
            body = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return {"kind": "torch", "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape), "blob": self.blob(body)}
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biuf":
            raise ParityError("Unsupported tensor dtype")
        array = np.ascontiguousarray(array)
        return {"kind": "numpy", "dtype": array.dtype.str, "shape": list(array.shape),
                "blob": self.blob(array.tobytes())}

    def load_tensor(self, ref):
        import numpy as np
        body = self.read_blob(ref["blob"])
        shape = ref["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ParityError("Invalid tensor shape")
        if ref["kind"] == "torch":
            import torch
            allowed = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"}
            if ref["dtype"] not in allowed:
                raise ParityError("Unsupported torch dtype")
            dtype = getattr(torch, ref["dtype"])
            expected = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()
            if len(body) != expected:
                raise ParityError("Tensor byte/shape mismatch")
            if not body:
                return torch.empty(shape, dtype=dtype)
            return torch.frombuffer(bytearray(body), dtype=dtype).reshape(shape).clone()
        dtype = np.dtype(ref["dtype"])
        if dtype.hasobject or dtype.kind not in "biuf":
            raise ParityError("Unsupported numpy dtype")
        if len(body) != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
            raise ParityError("Tensor byte/shape mismatch")
        return np.frombuffer(body, dtype=dtype).reshape(shape).copy()

    def new_request(self, metadata, body):
        attempt = metadata.get("attempt_id", "")
        if not ID.fullmatch(attempt):
            raise ParityError("Invalid attempt ID")
        request_id = uuid.uuid4().hex
        path = self.root / "requests" / request_id
        path.mkdir(parents=True, exist_ok=False)
        journal = RequestJournal(self, path)
        journal.event("request", {"metadata": metadata, "body": body,
                                  "wire_request_sha256": digest(body)})
        atomic_bytes(self.root / "attempts" / attempt / "requests" / f"{request_id}.json",
                     canonical({"request_id": request_id, "metadata": metadata}))
        return journal

    def attempt_requests(self, attempt_id):
        if not ID.fullmatch(attempt_id):
            raise ParityError("Invalid attempt ID")
        return [json.loads(path.read_bytes()) for path in sorted(
            (self.root / "attempts" / attempt_id / "requests").glob("*.json"))]

    def read_request(self, request_id):
        if not ID.fullmatch(request_id):
            raise ParityError("Invalid request ID")
        path = self.root / "requests" / request_id
        try:
            result = json.loads((path / "sealed.json").read_bytes())
            events = [json.loads(line) for line in (path / "events.jsonl").read_bytes().splitlines()]
        except (OSError, ValueError) as exc:
            raise ParityError("Incomplete/corrupt request journal") from exc
        if result["events_sha256"] != digest(events):
            raise ParityError("Request journal changed after seal")
        return events, result


class RequestJournal:
    def __init__(self, store, path):
        self.store = store
        self.path = path
        self.events = []
        self.lock = threading.RLock()
        self.sealed = False

    @property
    def request_id(self):
        return self.path.name

    def event(self, kind, data):
        with self.lock:
            if self.sealed:
                raise ParityError("Cannot append to sealed request")
            entry = {"seq": len(self.events), "kind": kind, "data": data}
            with (self.path / "events.jsonl").open("ab") as handle:
                handle.write(canonical(entry) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.events.append(entry)

    def seal(self, status):
        with self.lock:
            if self.sealed:
                raise ParityError("Request already sealed")
            atomic_bytes(self.path / "sealed.json", canonical({
                "status": status, "events_sha256": digest(self.events),
                "request_id": self.request_id,
            }))
            self.sealed = True
