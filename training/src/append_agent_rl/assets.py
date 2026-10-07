"""Create a new contract from actual assets; never edit old run receipts."""
import hashlib
from pathlib import Path

from opencode_rl.assets import from_assets as native_assets
from .contract import AppendContract, loop_source_sha256


def from_assets(*args, binary_path='/usr/local/bin/opencode-tools', max_turns=100, **kwargs):
    # Shared tokenizer/template/processor/tool and integration hashes. The old
    # native binary remains installed as the fallback asset; this contract
    # explicitly names the new binary and new loop source instead.
    native = native_assets(*args, **kwargs).to_dict()
    native.update(schema='append-opencode-rl-contract-v1', loop_sha256=loop_source_sha256(),
        binary_sha256=hashlib.sha256(Path(binary_path).read_bytes()).hexdigest(), max_turns=max_turns)
    return AppendContract(**native)
