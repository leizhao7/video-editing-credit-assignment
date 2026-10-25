"""Patch vLLM 0.26.0 to drain every packed NCCL producer stream.

v0.26.0 exits ``packed_nccl_broadcast_producer`` as soon as the iterator is
empty.  With double buffering that only synchronizes the *next* slot; the most
recent broadcast can still be reading ``chunk.packed_tensor`` when the local
``in_flight`` references are destroyed.  Drain all slots before returning so
CUDA/NCCL owns no buffer whose Python lifetime has ended.
"""

import os
from pathlib import Path


TARGET = Path(os.environ.get(
    "VLLM_PACKED_TENSOR_FILE",
    "/opt/SkyRL/.venv/lib/python3.12/site-packages/vllm/"
    "distributed/weight_transfer/packed_tensor.py",
))

src = TARGET.read_text()
fixed = """            buffer_idx = (buffer_idx + 1) % num_buffers

    # Drain every slot before returning.  The last broadcast may be queued on
    # a different stream than the slot synchronized at iterator exhaustion.
    for stream in streams:
        stream.synchronize()
    in_flight.clear()


def packed_nccl_broadcast_consumer("""

if fixed in src:
    print("[vllm-nccl-drain] already patched")
else:
    old = """            buffer_idx = (buffer_idx + 1) % num_buffers


def packed_nccl_broadcast_consumer("""
    if src.count(old) != 1:
        raise RuntimeError(
            "vLLM packed NCCL producer source changed; refusing an unverified patch"
        )
    TARGET.write_text(src.replace(old, fixed))
    print(f"[vllm-nccl-drain] patched {TARGET}")
