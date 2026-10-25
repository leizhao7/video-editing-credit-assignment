"""Exclude fully masked isolation rows from GRPO group statistics.

The S12 agent loop preserves a failed trajectory as a one-token sentinel with
``loss_mask=[0]`` and ``reward=0``. SkyRL prevents that row from
backpropagating, but its GRPO estimator still includes the zero reward in the
group mean/std. The failed row can therefore change every healthy sibling's
advantage despite being described as isolated.

Pass the training loss mask to the estimator and compute group statistics from
live trajectories only. Dead rows retain zero advantage; a live singleton has
zero group-relative advantage (and the fully-async collector separately drops
groups with fewer than two live trajectories).
"""

from __future__ import annotations

import py_compile
import sys


PPO_TARGET = "/opt/SkyRL/skyrl/backends/skyrl_train/utils/ppo_utils.py"
TRAINER_TARGET = "/opt/SkyRL/skyrl/train/trainer.py"
MARKER = "S12_GRPO_MASKED_ADVANTAGE"

OLD_SIGNATURE = '''    grpo_norm_by_std: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).
'''

NEW_SIGNATURE = '''    grpo_norm_by_std: bool = True,
    loss_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).
'''

OLD_STATS = '''    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
'''

NEW_STATS = '''    with torch.no_grad():
        # S12_GRPO_MASKED_ADVANTAGE: all-zero loss-mask rows are persisted
        # diagnostics, not members of the GRPO comparison population.
        bsz = scores.shape[0]
        live_sequences = (
            torch.ones(bsz, dtype=torch.bool, device=scores.device)
            if loss_mask is None
            else loss_mask.to(scores.device).sum(dim=-1) > 0
        )
        for i in range(bsz):
            if bool(live_sequences[i].item()):
                id2score[index[i]].append(scores[i])
        all_indices = index.tolist() if hasattr(index, "tolist") else list(index)
        for idx in set(all_indices):
            live_scores = id2score.get(idx, [])
            if len(live_scores) == 0:
                id2mean[idx] = scores.new_tensor(0.0)
                id2std[idx] = scores.new_tensor(1.0)
            elif len(live_scores) == 1:
                # A singleton has no group-relative signal.
                id2mean[idx] = live_scores[0]
                id2std[idx] = scores.new_tensor(1.0)
            else:
                stacked = torch.stack(live_scores)
                id2mean[idx] = torch.mean(stacked)
                id2std[idx] = torch.std(stacked)
        for i in range(bsz):
            if not bool(live_sequences[i].item()):
                scores[i] = 0.0
            elif grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
'''

OLD_CALL = '''                grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            )
        data["returns"] = returns
'''

NEW_CALL = '''                grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
                # S12_GRPO_MASKED_ADVANTAGE: fatal-isolation rows have
                # loss_mask=0 and must not alter healthy siblings' group stats.
                loss_mask=data["loss_mask"],
            )
        data["returns"] = returns
'''


def _replace_once(source: str, old: str, new: str, name: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{name} anchor matched {count}; refusing")
    return source.replace(old, new)


def main() -> int:
    try:
        with open(PPO_TARGET, encoding="utf-8") as fh:
            ppo_source = fh.read()
        with open(TRAINER_TARGET, encoding="utf-8") as fh:
            trainer_source = fh.read()
    except FileNotFoundError as exc:
        print(f"[grpo-masked-advantage] missing {exc.filename}", flush=True)
        return 1
    if MARKER in ppo_source and MARKER in trainer_source:
        print("[grpo-masked-advantage] already patched", flush=True)
        return 0
    try:
        ppo_source = _replace_once(
            ppo_source, OLD_SIGNATURE, NEW_SIGNATURE, "signature"
        )
        ppo_source = _replace_once(ppo_source, OLD_STATS, NEW_STATS, "stats")
        trainer_source = _replace_once(
            trainer_source, OLD_CALL, NEW_CALL, "trainer-call"
        )
    except RuntimeError as exc:
        print(f"[grpo-masked-advantage] {exc}", flush=True)
        return 1
    with open(PPO_TARGET, "w", encoding="utf-8") as fh:
        fh.write(ppo_source)
    with open(TRAINER_TARGET, "w", encoding="utf-8") as fh:
        fh.write(trainer_source)
    py_compile.compile(PPO_TARGET, doraise=True)
    py_compile.compile(TRAINER_TARGET, doraise=True)
    print("[grpo-masked-advantage] patched and compiled", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
