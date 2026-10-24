<div align="center">

<a href="docs/assets/film-poster-collage.png"><img src="docs/assets/film-poster-collage.png" width="100%" alt="All 49 films used in the study, shown as a three-row collage of their original posters."></a>
<p><sub>49 real films · <a href="docs/poster_credits.md">Poster credits</a></sub></p>

<h1>When Fine-Grained Credit Assignment<br>Helps and Hurts</h1>
<p><strong>Multimodal Agentic Reinforcement Learning through Video Editing</strong></p>

<p>
<a href="mailto:leizhao7@upenn.edu">Lei Zhao</a><sup>1</sup> · <a href="mailto:tom@philolabs.ai">Zongheng Cao</a><sup>2</sup> · <a href="mailto:christine@philolabs.ai">Xinyu Hu</a><sup>2</sup>
</p>
<p>
<a href="https://www.upenn.edu/"><img src="https://www.google.com/s2/favicons?domain=www.upenn.edu&sz=64" height="18" alt="Penn logo"></a> <sup>1</sup>University of Pennsylvania &nbsp;&nbsp; <a href="https://philolabs.ai/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://philolabs.ai/brand/philo_icon_cream.svg"><img src="https://philolabs.ai/brand/philo_icon_black.svg" height="18" alt="Philo Labs logo"></picture></a> <sup>2</sup>Philo Labs
</p>
<p>
<img src="https://img.shields.io/badge/arXiv-Coming_soon-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv: coming soon" title="Paper link coming soon">
<img src="https://img.shields.io/badge/Project_Web-Coming_soon-2563EB?style=for-the-badge" alt="Project Web: coming soon" title="Project website coming soon">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0A8AA0?style=for-the-badge&logo=opensourceinitiative&logoColor=white" alt="MIT license"></a>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
</p>
<p>
<a href="#task-generation">🎬 Task generation</a> &nbsp;•&nbsp;
<a href="#-findings">🔍 Findings</a> &nbsp;•&nbsp;
<a href="#-results">📊 Results</a> &nbsp;•&nbsp;
<a href="#-quick-start">🚀 Quick start</a> &nbsp;•&nbsp;
<a href="#-guides">🧭 Guides</a> &nbsp;•&nbsp;
<a href="#-reproduction-status">🔁 Reproducibility</a> &nbsp;•&nbsp;
<a href="#-citation">📌 Citation</a>
</p>

</div>

**Teaching AI to Edit Video with 49 Real Films.** We turn movies and their trailers into video-editing tasks, then train multimodal agents to inspect footage, reason about editing decisions, and use editing tools. Evaluation on held-out films shows when finer-grained feedback helps or hurts task performance. This repository includes our scalable task-generation pipeline, RL training code, and outcome graders.

## 🎉 News

- **[2026-10-21]** English task-generation and scoring interfaces, refreshed training setup documentation, and additional CPU checks.
- **[2026-10-20]** Added a resumable project workflow for generating tasks from a movie and trailer, reviewing their quality, and scoring submitted edits.

<a id="task-generation"></a>

## 🎬 Turn Any Movie–Trailer Pair into RL Tasks

**Four editing task types, with grading included.** Start with a movie and its matching trailer. Our scalable pipeline aligns the footage, builds task briefs and media inputs, attaches outcome graders, and filters candidates through quality review.

<p align="center">
<a href="docs/assets/editing-tasks.svg"><img src="docs/assets/editing-tasks.svg" width="100%" alt="Four independent editing task types: CS selects relevant shots, NS orders them, RC sets clip boundaries, and FC coordinates dialogue, music, and transitions over fixed picture timing."></a>
</p>

*The four task types are constructed independently. Fine Cut keeps picture and timing fixed.*

**Generate tasks from your own media.** Complete the [one-time setup](docs/workflow.md#1-set-up-once), replace the media paths below, and run from the repository root. Use a new project directory outside the checkout.

```bash
python video_editing.py init \
  --project "$HOME/video-editing-demo" \
  --movie '/path/to/movie.mkv' --trailer '/path/to/trailer.mp4' \
  --name myfilm --backend claude --model sonnet

python video_editing.py doctor --project "$HOME/video-editing-demo"
python video_editing.py run --project "$HOME/video-editing-demo"
```

**Output:** task folders in `training_task/myfilm/` and an **`accepted_tasks.json`** list of candidates that passed quality review, both inside your project. Each task includes its brief, media inputs, and private grading references.

Rerunning `run` reuses completed work. It can incur Modal and model usage. [Full walkthrough →](docs/workflow.md)

<details>
<summary>Backend options, progress, and smaller first reviews</summary>

- **Annotation backends:** Claude Code, Codex CLI, or OpenCode 1.x. Choose `--backend` and a model available to your account with `--model` at initialization; OpenCode uses `provider/model`. [Backend setup →](docs/agent_setup.md)
- **Preview or track progress:** use `run --plan` or `status` with the same `--project` path. The project's `README.md` links to stage logs.
- **Review a smaller batch:** append `--review-limit 1` to `run` to generate all candidates but review only one. Use `review --review-limit 5` for up to five additional candidates.
- **Stop before review:** `run --until tasks` builds candidates without Gemini quality review. `run --until probe` performs only a local media check.

</details>

## 🔍 Findings

**1. Fine-grained feedback helps most on Fine Cut.** Decaying process rewards gives the largest improvement over outcome-only GRPO on Fine Cut. Outcome-only GRPO remains competitive on the three shorter-horizon task types. Task demands, scoring criteria, and horizon vary together, so this comparison does not isolate a causal effect of horizon.

**2. More process reward can hurt task completion.** With a fixed process-reward weight, judge-preferred intermediate behaviors increase while completion and outcome scores decline. Reducing the weight to zero mitigates this pattern and yields the highest overall score in our comparison.

**3. Dense token-level supervision is not sufficient.** OPSD is sensitive to the teacher's privileged information. Reference answers yield little change from the base model, while turn suggestions accompany a collapse in performance. Gradient and disagreement statistics characterize these patterns; their causes and effective remedies remain open questions.

These findings concern the tested methods and tasks. In particular, the process-reward results apply to **LLM-judged turn-level rewards in GRPO+PRM**, not to untested methods such as GiGPO or Turn-PPO.

## 📊 Results

<p align="center">
<a href="docs/assets/paper-teaser.png"><img src="docs/assets/paper-teaser.png" width="100%" alt="Paper Figure 1: four editing tasks ordered by base-policy horizon, score gains over outcome-only GRPO, and process/outcome rewards with the fixed and decayed weight schedules."></a>
</p>

*Four editing tasks, their gains from intermediate feedback, and the reward-hacking pattern during training. [Figure details](docs/experiment_notes.md#illustrations).*

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
- These are reported paper results, not a benchmark rerun from this repository. See the [experimental conventions](docs/experiment_notes.md#training-and-evaluation-conventions) and [reproduction status](#-reproduction-status).

</details>

## 🚀 Quick start

### 🧪 Inspect the training code

The [training guide](docs/training.md#run-the-offline-cpu-tests) covers CPU setup and fixed/decayed PRM tests. [OPSD](docs/opsd.md#selected-cpu-verification) has separate numerical and checkpoint checks. **A portable GPU training launcher is still pending**, and generated tasks need a trainer preparation adapter before they can be used for RL.

<details>
<summary>Source-only checks and repository layout</summary>

These checks require only Python 3.12 and do not start training or call a model:

```bash
python scripts/check_release.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests
```

```text
video_editing.py     Project setup, generation, review, and scoring
data_construction/   Task-generation stages and annotation backends
training/            GRPO/PRM runtime, decay overlay, and CPU tests
opsd/                Distillation objectives, teacher inputs, and adapters
evaluation/          Outcome graders and submission scoring
configs/             Recorded experiment settings and source identities
docs/                Setup guides, experiment notes, and paper figure
scripts/             Source-release verification
tests/               Offline source and interface regression checks
```

</details>

## 🧭 Guides

| Goal | Start here |
| :--- | :--- |
| Generate and review tasks from a movie and trailer | [Project workflow](docs/workflow.md) · [Pipeline design](docs/data_construction.md) |
| Choose and set up an annotation CLI | [Agent setup](docs/agent_setup.md) |
| Review quality and sample tasks across films | [Quality review and sampling](docs/data_construction.md#quality-review-and-sampling) |
| Validate and score an agent's edit | [Submission scoring](docs/workflow.md#5-check-and-score-an-agents-edit) · [Evaluation details](docs/evaluation.md) |
| Inspect GRPO/PRM or token-level distillation | [Training](docs/training.md) · [OPSD](docs/opsd.md) |
| Find experiment settings and interpretation limits | [Experiment notes](docs/experiment_notes.md) · [Recorded configurations](configs/) |
| Check what has been verified | [Release status](docs/release_status.md) |

## 🔁 Reproduction status

**Checked:** synthetic CPU tests, local workflow commands, and component-level construction, review, and scoring on real media. The full-film construction check involved debugging and resuming completed stages.

**Still unverified:** a fresh complete-film run through the final unified workflow, full GPU training, and end-to-end reproduction of the paper's benchmark. The training build, public launch services, custom-task adapter, and original reference-answer OPSD snapshot still have gaps. [Detailed validation record →](docs/release_status.md)

**Data:** we provide code, the paper figure, and the poster banner. The dataset, underlying film media, task instances, reference answers, rollouts, judge caches, and model checkpoints are not distributed.

## 📌 Citation

To cite the software:

```bibtex
@software{zhao2026videoediting,
  title  = {When Fine-Grained Credit Assignment Helps and Hurts: Multimodal Agentic Reinforcement Learning through Video Editing},
  author = {Zhao, Lei and Cao, Zongheng and Hu, Xinyu},
  year   = {2026},
  url    = {https://github.com/leizhao7/video-editing-credit-assignment}
}
```

Repository metadata is also available in [`CITATION.cff`](CITATION.cff). A verified paper link and formal paper citation will be added when available.

## 🙏 Acknowledgments and license

This implementation builds on **SkyRL**, **vLLM**, **OpenCode**, and **Qwen**. Recorded dependency identities are listed in [`configs/dependency_pins.json`](configs/dependency_pins.json).

Project-authored code is licensed under [MIT](LICENSE). Third-party components retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). The paper figure, [poster artwork](docs/poster_credits.md), external media, and model weights are outside this code license.

Part of this work was completed while Lei Zhao was an intern at Philo Labs.

Contact: [Lei Zhao](mailto:leizhao7@upenn.edu) · [Zongheng Cao](mailto:tom@philolabs.ai) · [Xinyu Hu](mailto:christine@philolabs.ai).
