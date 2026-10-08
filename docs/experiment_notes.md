# Experiment notes

This page records the experimental setting behind the paper. These are configuration records and research findings, not benchmark outputs or an executable reproduction recipe. See [release status](release_status.md) for missing runtime components.

## Compared learning signals

| Setting | Feedback | Where to inspect it |
| :--- | :--- | :--- |
| Outcome-only GRPO | Final task outcome, applied through a trajectory-level advantage. | [`configs/grpo.json`](../configs/grpo.json) |
| GRPO+PRM (fixed λ) | Outcome feedback plus an LLM judge's turn-level process signal. | [`configs/prm_fixed.json`](../configs/prm_fixed.json), [training guide](training.md) |
| GRPO+PRM (decayed λ) | The same process signal with its coefficient reduced to zero. | [`configs/prm_decay.json`](../configs/prm_decay.json), [training guide](training.md) |
| OPSD | Token-level forward KL against a frozen teacher with additional information. | [`configs/opsd.json`](../configs/opsd.json), [OPSD guide](opsd.md) |

The main paper comparisons use Qwen3.5-9B as the student. OPSD's frozen teacher receives reference answers or turn suggestions that are not exposed to the student. The complete original reference-answer training snapshot remains to be recovered; the retained turn-suggestion recipe must not be represented as that snapshot.

Additional experiments used Qwen3.6-35B-A3B with outcome-only GRPO and fixed-weight GRPO+PRM. These are separate from the paper's main Qwen3.5-9B comparison; they do not establish OPSD or decayed-PRM results for the larger model. The configurations and source contracts currently organized in this repository target Qwen3.5-9B.

## Training and evaluation conventions

- Reward-based training uses 16 tasks per update, 8 rollouts per task, seed 42, and 50 optimizer updates. OPSD uses one rollout per task.
- Training losses use token-mean reduction over valid assistant tokens.
- The fixed process-reward coefficient is 0.5. The decayed condition branches from the fixed-PRM checkpoint at update 20 and reduces the coefficient cubically to zero over updates 20–30, using the consumer optimizer's global step. The coefficient remains zero afterward.
- Main results average four evaluation repetitions of each final checkpoint. These are evaluation repetitions, not independent training runs.
- The original experimental dataset contained 797 tasks from 49 films, with 716 training tasks and 81 validation tasks. The dataset and its task instances are not distributed.

## Interpreting the findings

The task comparison changes editing demands, evaluation criteria, and interaction horizon together. Larger gains on Fine Cut therefore do not establish that horizon alone determines the value of intermediate feedback, and absolute scores should not be read as a common measure of task difficulty.

The reward-hacking observation applies to the tested GRPO+PRM setup with LLM-judged turn-level rewards. GiGPO and Turn-PPO are discussed as related work, not evaluated baselines. OPSD gradient and disagreement statistics describe failure patterns; they do not establish that gradient magnitude or signal dilution caused the observed performance changes.

## Illustrations

The [README](../README.md) opens with **Figure 1 from the paper**, rendered without content changes as [`paper-teaser.png`](assets/paper-teaser.png). It contains illustrative film thumbnails and aggregate experimental results, not a distributable task package or the underlying media and result files.

- Panel (a) shows independent task schematics ordered by base-policy horizon. Turns and scores are task-type averages, not example-specific results; evaluation metrics average four repetitions. Fine Cut keeps picture and timing fixed.
- Panel (b) reports final score differences from outcome-only GRPO.
- Panel (c) shows fixed-weight process and outcome rewards, with decayed-weight outcome rewards available from update 21. Faint lines connect recorded observations; thick lines show centered five-observation means. The aligned strip shows fixed and cubic-decay weights λ, with the update 20–30 decay window shaded.

The source figure is `teaser-editable-v6.pdf` (SHA-256: `0178cea1dcc1514c48fb13ce5dbdb36d46bf398cbc27f3853387584a8d51af5d`). It was rendered with Poppler at 2,600 pixels wide, preserving its aspect ratio. Only this reviewed PNG is permitted by the release check; its exact path and SHA-256 are pinned in [`check_release.py`](../scripts/check_release.py), and its bytes are also tracked in the source manifest.

The separate [`learning-signals.svg`](assets/learning-signals.svg) is a hand-authored conceptual diagram of feedback granularity. It contains no film frames or measured results. The earlier [`editing-tasks.svg`](assets/editing-tasks.svg) is retained as an editable conceptual alternative.
