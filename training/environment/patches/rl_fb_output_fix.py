"""Stop materialising unused per-token Python outputs after PPO backward.

SkyRL's ``return_per_token_outputs`` flag is honoured by the SFT branch but is
accidentally ignored by the RL branch.  The regular PPO trainer also leaves the
flag at its public-API default even though it consumes only scalar metrics.
For long trajectories this converts millions of GPU values to Python floats,
retains them on every worker, and sends them through Ray after every update.

Patch the pinned SkyRL source at image-build time so:

1. the internal PPO trainer explicitly requests no per-token outputs; and
2. the RL worker branch honours the already documented flag.

Public/Tinker callers keep the default ``True`` and therefore keep their
per-token results.  The model forward, backward, optimizer, and all scalar
metrics are unchanged.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("/opt/SkyRL")
TRAINER = ROOT / "skyrl/train/trainer.py"
WORKER = ROOT / "skyrl/backends/skyrl_train/workers/worker.py"


def _patch_trainer(text: str) -> str:
    old = """                status = self.dispatch.forward_backward_from_staged(model, chunk_refs)"""
    new = """                if model == \"policy\":
                    # The internal PPO trainer consumes scalar metrics only.  Avoid
                    # returning per-token Python lists through Ray after backward.
                    status = self.dispatch.forward_backward_from_staged(
                        model, chunk_refs, return_per_token_outputs=False
                    )
                else:
                    status = self.dispatch.forward_backward_from_staged(model, chunk_refs)"""
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"trainer patch anchor count is {count}, expected 1")
    return text.replace(old, new)


def _patch_worker(text: str) -> str:
    start_marker = "            # Build per-sequence loss_fn_outputs with logprobs.\n"
    end_marker = "\n\n            status = {\n"
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError("worker RL output start marker not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError("worker RL output end marker not found")
    if text.find(start_marker, start + 1) >= 0:
        raise RuntimeError("worker RL output marker is ambiguous")

    replacement = """            # The public/Tinker path needs per-token logprobs, but the
            # internal PPO trainer consumes scalar metrics only.  Honour the
            # documented flag so long trajectories are not converted into
            # millions of Python floats and returned through Ray unnecessarily.
            if return_per_token_outputs:
                batch_size = action_log_probs.shape[0]
                seq_len = action_log_probs.shape[1]

                if response_mask is not None:
                    valid_lens = response_mask.sum(dim=1).int().tolist()
                elif loss_mask is not None:
                    valid_lens = (loss_mask > 0).sum(dim=1).int().tolist()
                else:
                    valid_lens = [seq_len] * batch_size

                detached_log_probs = action_log_probs.detach().cpu()
                loss_fn_outputs = []
                for i, valid_len in enumerate(valid_lens):
                    loss_fn_outputs.append(
                        {
                            \"logprobs\": detached_log_probs[i, -valid_len:].tolist()
                            if valid_len > 0
                            else [],
                        }
                    )
            else:
                loss_fn_outputs = [{} for _ in range(action_log_probs.shape[0])]"""
    return text[:start] + replacement + text[end:]


def main() -> None:
    trainer_before = TRAINER.read_text(encoding="utf-8")
    worker_before = WORKER.read_text(encoding="utf-8")
    trainer_after = _patch_trainer(trainer_before)
    worker_after = _patch_worker(worker_before)
    ast.parse(trainer_after, filename=str(TRAINER))
    ast.parse(worker_after, filename=str(WORKER))
    TRAINER.write_text(trainer_after, encoding="utf-8")
    WORKER.write_text(worker_after, encoding="utf-8")
    print("[rl-fb-output] trainer requests scalar-only PPO results", flush=True)
    print("[rl-fb-output] RL worker now honours return_per_token_outputs", flush=True)


if __name__ == "__main__":
    main()
