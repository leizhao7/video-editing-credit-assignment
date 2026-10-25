"""Fail fast if any response-level training tensor is token-misaligned."""
from __future__ import annotations

import py_compile
import sys


TARGET = "/opt/SkyRL/skyrl/train/dataset/preprocess.py"
MARKER = "S12_RESPONSE_ALIGNMENT_GUARD"

OLD = '''    _verify_inputs(prompts, responses, rewards, loss_masks)

    prompt_token_lens = [len(p) for p in prompts]
'''

NEW = '''    _verify_inputs(prompts, responses, rewards, loss_masks)

    # S12_RESPONSE_ALIGNMENT_GUARD: these tensors are right-aligned below.
    # A one-token length mismatch would otherwise silently shift supervision
    # and behavior logprobs instead of failing at the producer boundary.
    response_fields = {
        "rewards": rewards,
        "loss_masks": loss_masks,
    }
    if logprobs is not None:
        response_fields["rollout_logprobs"] = logprobs
    for sample_index, response in enumerate(responses):
        expected = len(response)
        for field_name, field_values in response_fields.items():
            actual = len(field_values[sample_index])
            if actual != expected:
                raise ValueError(
                    f"Response alignment mismatch at sample {sample_index}: "
                    f"len(response_ids)={expected}, len({field_name})={actual}"
                )

    prompt_token_lens = [len(p) for p in prompts]
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[response-alignment] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[response-alignment] already patched", flush=True)
        return 0
    count = source.count(OLD)
    if count != 1:
        print(f"[response-alignment] anchor matched {count}; refusing", flush=True)
        return 1
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(OLD, NEW))
    py_compile.compile(TARGET, doraise=True)
    print(f"[response-alignment] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
