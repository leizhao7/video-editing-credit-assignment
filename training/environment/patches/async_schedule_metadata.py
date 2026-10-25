"""Stamp fully-async rollout metadata after acquiring its capacity slot.

SkyRL prepares ``GeneratorInput`` before awaiting the staleness manager.  A
worker can wait across a completed optimizer step and weight sync, so the
``BatchMetadata.global_step`` captured during preparation may describe the old
policy even though generation starts on the new one.  Staleness accounting is
not affected (it separately captures ``self.global_step`` after the await), but
trajectory persistence consumes ``BatchMetadata`` and therefore files the full
rollout under the wrong step.

Refresh the mutable per-group metadata at the same point used for staleness
accounting.  There is no await between the slot acquisition, the version
snapshot, and the metadata write, so all three describe the policy version that
the generator is about to use.
"""

from __future__ import annotations

import py_compile
import sys


TARGET = "/opt/SkyRL/skyrl/train/fully_async_trainer.py"
MARKER = "S12_ASYNC_SCHEDULE_METADATA"

OLD = '''                # 3. Generate one rollout group
                global_step_at_start = self.global_step  # for staleness control

                group_start_time = time.monotonic()
'''

NEW = '''                # 3. Generate one rollout group.  Preparation happens before
                # acquire_submission_slot() and may wait across a completed train/sync
                # step.  Refresh the persisted batch metadata from the same atomic
                # post-acquire snapshot used by staleness accounting.
                # S12_ASYNC_SCHEDULE_METADATA
                global_step_at_start = int(self.global_step)
                batch_metadata = generator_input.get("batch_metadata")
                if batch_metadata is None:
                    raise RuntimeError("Fully-async generation requires BatchMetadata")
                batch_metadata.global_step = global_step_at_start

                group_start_time = time.monotonic()
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[async-schedule-metadata] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[async-schedule-metadata] already patched", flush=True)
        return 0
    count = source.count(OLD)
    if count != 1:
        print(
            f"[async-schedule-metadata] anchor matched {count}; refusing",
            flush=True,
        )
        return 1
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(OLD, NEW))
    py_compile.compile(TARGET, doraise=True)
    print(f"[async-schedule-metadata] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
