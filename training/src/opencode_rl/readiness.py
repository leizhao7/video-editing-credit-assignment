"""No production training until source contracts AND real parity probes pass."""
from .contract import ParityError

REQUIRED = (
    "source_pins", "actual_opencode_s1", "actual_opencode_s2", "actual_opencode_s3", "actual_opencode_s4e",
    "image_tensors", "thinking_sse", "compaction_boundary", "stream_cancel", "sandbox_isolation",
    "actual_trainer_tokens", "same_checkpoint_logprobs", "prm_contributions", "async_weight_lease",
    "dataset_release_797",
)

TRAINING_REQUIRED = ('actual_optimizer_step', 'all_four_inference_weight_sync',
                     'checkpoint_resume', 'durable_trajectories')


def require_ready(report, contract, *, process_enabled=True):
    if report.get("contract_sha256") != contract.sha256:
        raise ParityError("Readiness report belongs to another runtime")
    required = [key for key in REQUIRED if process_enabled or key != 'prm_contributions']
    missing = [key for key in required if report.get("checks", {}).get(key) is not True]
    if missing:
        raise ParityError("Training is not enabled; missing real checks: " + ", ".join(missing))
    if report.get("token_mismatches") != 0 or report.get("image_mismatches") != 0:
        raise ParityError("Non-zero context mismatches")
    if not report.get("evidence_paths"):
        raise ParityError("No durable parity evidence")


def require_training_ready(report, contract, *, process_enabled=False):
    require_ready(report, contract, process_enabled=process_enabled)
    missing = [key for key in TRAINING_REQUIRED if report.get('checks', {}).get(key) is not True]
    if missing:
        raise ParityError('Full RL still requires real hardware acceptance: ' + ', '.join(missing))


def require_direct_start_authorization(authorization, contract):
    """User may waive another prelaunch experiment, never runtime parity checks.

This is deliberately NOT a readiness report and must never be presented as
passed optimizer/numerical qualification of the new integration snapshot.
"""
    if (not isinstance(authorization, dict) or
            authorization.get('kind') != 'user_requested_direct_start' or
            authorization.get('instruction') != '不用了直接开跑' or
            authorization.get('waived') != ['additional_prelaunch_qualification'] or
            authorization.get('contract_sha256') != contract.sha256 or
            authorization.get('qualification_complete') is not False or
            authorization.get('runtime_parity_guards_enabled') is not True):
        raise ParityError('Direct start requires explicit, snapshot-bound user authorization')
