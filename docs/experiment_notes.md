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

The following settings describe the main Qwen3.5-9B experiments.

- Reward-based training uses 16 tasks per update, 8 rollouts per task, seed 42, and 50 optimizer updates. OPSD uses one rollout per task.
- Training losses use token-mean reduction over valid assistant tokens.
- The fixed process-reward coefficient is 0.5. The decayed condition branches from the fixed-PRM checkpoint at update 20 and reduces the coefficient cubically to zero over updates 20–30, using the consumer optimizer's global step. The coefficient remains zero afterward.
- Main results average four evaluation repetitions of each final checkpoint. These are evaluation repetitions, not independent training runs.
- The original experimental dataset contained 797 tasks from 49 films, with 716 training tasks and 81 validation tasks. The dataset and its task instances are not distributed.

## Reported validation scores

Main **Qwen3.5-9B** validation results. Scores average four evaluation repetitions of each final checkpoint (**avg@4**); higher is better. Best values in each column are bold.

| Method | CS | NS | RC | FC | Overall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Base model | 0.657 | 0.303 | 0.205 | 0.152 | 0.284 |
| Outcome-only GRPO | **0.821** | 0.500 | **0.356** | 0.201 | 0.413 |
| GRPO+PRM (fixed λ) | 0.482 | 0.457 | 0.276 | 0.239 | 0.334 |
| GRPO+PRM (decayed λ) | 0.804 | **0.552** | 0.348 | **0.293** | **0.447** |
| OPSD (reference answers) | 0.606 | 0.267 | 0.219 | 0.162 | 0.277 |
| OPSD (turn suggestions) | 0.129 | 0.063 | 0.032 | 0.000 | 0.044 |

<details>
<summary>How to read these scores</summary>

- **avg@4** averages four outcome scores for each task, rather than selecting the best attempt. These are evaluation repetitions, not four independent training seeds.
- **Overall** averages equally over all 81 validation tasks: 14 CS, 15 NS, 26 RC, and 26 FC. It is not the unweighted mean of the four task-type columns.
- Task types have different scoring criteria. Their absolute scores are not a shared measure of task difficulty.
- These are reported paper results, not a benchmark rerun from this repository. See the [experimental conventions](#training-and-evaluation-conventions) and [reproduction status](release_status.md).

</details>

## Interpreting the findings

The task comparison changes editing demands, evaluation criteria, and interaction horizon together. Larger gains on Fine Cut therefore do not establish that horizon alone determines the value of intermediate feedback, and absolute scores should not be read as a common measure of task difficulty.

The reward-hacking observation applies to the tested GRPO+PRM setup with LLM-judged turn-level rewards. GiGPO and Turn-PPO are discussed as related work, not evaluated baselines. OPSD gradient and disagreement statistics describe failure patterns; they do not establish that gradient magnitude or signal dilution caused the observed performance changes.

## Illustrations

The [README](../README.md) highlights three findings in **one [combined SVG figure](assets/findings-overview.svg)** redrawn from the paper's aggregate results. It contains no task instances, film media, private grading references, or rollout exports.

- **Panel (a), benefits on more complex tasks:** relative score changes from outcome-only GRPO to GRPO+PRM with decayed λ. Percentages use the audited four-decimal evaluation means, in the paper figure's task order (NS, RC, CS, FC): outcome-only scores `0.5000, 0.3557, 0.8206, 0.2015`; decayed-PRM scores `0.5518, 0.3477, 0.8037, 0.2929`. Thus Fine Cut gains 45.4%, and the other three changes are +10.4%, −2.2%, and −2.1%. These are relative changes, not percentage-point changes or a common measure of task difficulty. Fine Cut combines more editing operations and has the longest mean interaction horizon under both the base model and outcome-only GRPO. Complexity and horizon are not independently controlled.
- **Panel (b), reward hacking and mitigation:** the same observations and centered five-observation means as paper Figure 1(c), including the decayed outcome curve from update 21. Lines connect available observations. The aligned strip shows λ held at 0.5 or decayed cubically from 0.5 to zero over updates 20–30. Shading marks this decay window. The large percentages are **held-out validation truncation rates** from the model-behavior table, separate from the **training rewards** in the curves. These are individual training runs, not independent seed averages.
- **Panel (c), desirable behavioral patterns:** decayed-PRM behavior relative to outcome-only GRPO, from the model-behavior table. Mean frames seen per rollout increase from `84.2` to `129.8` (1.54×); mean reasoning characters per completed turn increase from `618` to `1,035` (1.67×); and completed turns with explicit self-correction markers increase from `19.0%` to `36.0%` (1.89×). The dashed vertical line marks the GRPO baseline at 1×. The overall validation score also rises from `0.413` to `0.447`. These are behavior frequencies and lengths, not judgments of reasoning quality or verified corrections; they do not by themselves ensure successful completion.

All held-out scores use the main Qwen3.5-9B comparison and avg@4. No additional experiments, uncertainty estimates, or datasets are introduced by these charts.

**[Figure 1 from the paper](assets/paper-teaser.png)** remains available as an unchanged rendering. It contains illustrative film thumbnails and aggregate experimental results, not a distributable task package or the underlying media and result files.

- Panel (a) shows independent task schematics ordered by base-policy horizon. Turns and scores are task-type averages, not example-specific results; evaluation metrics average four repetitions. Fine Cut keeps picture and timing fixed.
- Panel (b) reports final score differences from outcome-only GRPO.
- Panel (c) shows fixed-weight process and outcome rewards, with decayed-weight outcome rewards available from update 21. Faint lines connect recorded observations; thick lines show centered five-observation means. The aligned strip shows fixed and cubic-decay weights λ, with the update 20–30 decay window shaded.

The source figure is `teaser-editable-v6.pdf` (SHA-256: `0178cea1dcc1514c48fb13ce5dbdb36d46bf398cbc27f3853387584a8d51af5d`). It was rendered with Poppler at 2,600 pixels wide, preserving its aspect ratio. The reviewed paper PNG and the README poster banner have their exact paths and SHA-256 hashes pinned in [`check_release.py`](../scripts/check_release.py), and their bytes are also tracked in the source manifest.

The header [`film-poster-collage.png`](assets/film-poster-collage.png) contains one original poster for each of the 49 films, arranged for visual balance. It illustrates the source collection rather than task outputs or experimental results. [Poster credits](poster_credits.md) identify the public image sources; the artwork is excluded from the code license.

The task-generation section uses [`editing-tasks.svg`](assets/editing-tasks.svg) to explain the four independently constructed task types. This editable conceptual diagram contains no film frames or measured results. The paper figure, including its third-party film imagery, is not covered by the project's code license.
