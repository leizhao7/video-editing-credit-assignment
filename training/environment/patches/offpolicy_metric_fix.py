"""Compute importance-ratio diagnostics only on active response tokens."""
from __future__ import annotations

import py_compile
import sys


TARGET = (
    "/opt/SkyRL/skyrl/backends/skyrl_train/utils/"
    "off_policy_correction_utils.py"
)
MARKER = "S12_OFFPOLICY_ACTIVE_TOKEN_METRICS"

OLD = '''    is_ratio = safe_exp_delta(old_log_probs - rollout_logprobs, clip=20.0, out_dtype=old_log_probs.dtype)
    metrics = {}
    metrics["is_ratio_mean"] = masked_mean(is_ratio, loss_mask).mean().detach().item()
    metrics["is_ratio_std"] = (is_ratio * loss_mask).std().detach().item()
    metrics["is_ratio_max"] = (is_ratio * loss_mask).max().detach().item()
    metrics["is_ratio_min"] = (is_ratio * loss_mask).min().detach().item()
'''

NEW = '''    is_ratio = safe_exp_delta(old_log_probs - rollout_logprobs, clip=20.0, out_dtype=old_log_probs.dtype)
    metrics = {}
    # S12_OFFPOLICY_ACTIVE_TOKEN_METRICS: multiplying by the mask inserts
    # padding zeros, which forces min=0 and distorts std.  Select active tokens
    # exactly; these are diagnostics only and do not change the correction.
    active_is_ratio = is_ratio[loss_mask.bool()]
    if active_is_ratio.numel() > 0:
        metrics["is_ratio_mean"] = active_is_ratio.mean().detach().item()
        metrics["is_ratio_std"] = active_is_ratio.std(unbiased=False).detach().item()
        metrics["is_ratio_max"] = active_is_ratio.max().detach().item()
        metrics["is_ratio_min"] = active_is_ratio.min().detach().item()
    else:
        metrics["is_ratio_mean"] = 0.0
        metrics["is_ratio_std"] = 0.0
        metrics["is_ratio_max"] = 0.0
        metrics["is_ratio_min"] = 0.0
'''


def main() -> int:
    try:
        with open(TARGET, encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        print(f"[offpolicy-metrics] missing {TARGET}", flush=True)
        return 1
    if MARKER in source:
        print("[offpolicy-metrics] already patched", flush=True)
        return 0
    count = source.count(OLD)
    if count != 1:
        print(f"[offpolicy-metrics] anchor matched {count}; refusing", flush=True)
        return 1
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(source.replace(OLD, NEW))
    py_compile.compile(TARGET, doraise=True)
    print(f"[offpolicy-metrics] patched and compiled: {TARGET}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
