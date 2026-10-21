<div align="center">

# When Fine-Grained Credit Assignment<br>Helps and Hurts
### Multimodal Agentic Reinforcement Learning through Video Editing

**[Lei Zhao](mailto:leizhao7@upenn.edu)<sup>1</sup> · [Zongheng Cao](mailto:tom@philolabs.ai)<sup>2</sup> · [Xinyu Hu](mailto:christine@philolabs.ai)<sup>2</sup>**

<sup>1</sup> University of Pennsylvania &nbsp;&nbsp; <sup>2</sup> Philo Labs

[Overview](#overview) · [Findings](#what-we-learn) · [Quick start](#quick-start) · [Agent setup](#set-up-an-annotation-agent) · [Documentation](#documentation) · [Citation](#citation) · [License](#license)

</div>

We build a **scalable pipeline for generating video-editing tasks** from real films and their released trailers. The tasks reflect real editing workflows and support training and evaluation of multimodal agents.

**When does feedback on intermediate actions help—and when does it hurt task completion?** Using these tasks, we compare trajectory-, turn-, and token-level learning signals for agents that combine visual inspection, reasoning, and tool use.

<p align="center">
  <a href="docs/assets/paper-teaser.png"><img src="docs/assets/paper-teaser.png" width="100%" alt="Paper Figure 1: four editing tasks ordered by base-policy interaction horizon; score differences from outcome-only GRPO; and process/outcome reward curves with the fixed and decayed process-reward weights."></a>
</p>

**Figure 1 · When process rewards help and hurt.** (a) Independent editing tasks, ordered by base-policy horizon; turns and base scores are task-type averages. (b) Final score differences from outcome-only GRPO. (c) Training rewards and the process-reward weight λ; shading marks the decay window over updates 20–30. [View full resolution](docs/assets/paper-teaser.png) · [Figure details](docs/experiment_notes.md#illustrations)

> **Try your own film:** The [project workflow](docs/workflow.md) connects task generation, quality review, and submission scoring. Its components have real-media checks; the new unified wrapper has not yet been rerun on a complete film. Full GPU training and benchmark reproduction remain unverified. [Validation scope →](docs/release_status.md)

## Overview

**Multimodal agentic RL.** Our experiments build on **SkyRL** for training, **vLLM** for generation, and an **OpenCode-based agent harness** for multi-turn visual inspection and video-editing tool use.

With **Qwen3.5-9B**, the paper compares **outcome-only GRPO**, **GRPO+PRM**, and **on-policy self-distillation (OPSD)** with either reference answers or turn suggestions as teacher-only information. We also ran **outcome-only GRPO** and **GRPO+PRM** with **Qwen3.6-35B-A3B** in additional experiments. [Training framework](docs/training.md) · [OPSD implementation](docs/opsd.md)

**Scalable data generation.** We develop a **data generation framework** that aligns trailer shots with source-film footage, divides trailers into narrative chunks, and generates editorial briefs. Each chunk yields four task types by varying the supplied media and required editing operation. These tasks are **constructed independently**: one task's output is not used to initialize the next.

Using our framework, we generated **2,352 video-editing tasks from 49 films**. Filtering and sampling retained **797 tasks: 716 for training and 81 for validation**, with no films shared between the two splits. Training on these tasks improved video-editing performance on held-out films. [Explore the task-generation pipeline →](docs/data_construction.md)

| Task | Editing decision |
| :--- | :--- |
| **CS** · Content Selection | Select relevant shots from a candidate pool. |
| **NS** · Narrative Sequencing | Arrange selected shots into a coherent sequence. |
| **RC** · Rough Cut | Set each clip's in- and out-points within a fixed sequence. |
| **FC** · Fine Cut | Select and trim dialogue, place music, choose transitions, and render the result. Picture and timing stay fixed. |

## What we learn

The main **Qwen3.5-9B** experiments show three patterns:

- **When it helps.** Decaying process rewards provides the largest benefit on Fine Cut. Outcome-only GRPO remains competitive on the three shorter-horizon task types. Task demands, scoring criteria, and horizon vary together, so this comparison does not isolate a causal effect of horizon.
- **When it hurts.** With a fixed process-reward weight, judge-preferred intermediate behaviors can increase while task completion and outcome scores decline. Reducing that weight to zero mitigates this pattern in our GRPO+PRM experiments.
- **Dense feedback is not sufficient.** OPSD is sensitive to the teacher's privileged information. Teacher–student disagreement and gradient statistics reveal contrasting failure patterns; effective remedies remain an open question.

These process-reward findings concern **LLM-judged turn-level rewards in our GRPO+PRM setup**. They do not establish the behavior of untested methods such as GiGPO or Turn-PPO. The full experimental context and implementation limits are summarized in the [experiment notes](docs/experiment_notes.md).

## Quick start

**[Start with the setup guide →](docs/workflow.md)** You need Python 3.12, local media tools, Modal, one authenticated annotation CLI, and a Gemini API key for quality review. The guide covers installation and account setup. No local GPU is required.

From the installed repository checkout, give the workflow a movie, its matching trailer, and a **new private project directory**:

```sh
python video_editing.py init \
  --project "$HOME/video-editing-demo" \
  --movie '/path/to/movie.mkv' --trailer '/path/to/trailer.mp4' \
  --name myfilm --backend claude --model sonnet

python video_editing.py doctor --project "$HOME/video-editing-demo"
python video_editing.py run --project "$HOME/video-editing-demo" --plan

# Paid preprocessing/annotation; review only one candidate on this first run.
python video_editing.py run --project "$HOME/video-editing-demo" --review-limit 1
python video_editing.py status --project "$HOME/video-editing-demo"
```

Replace the paths and choose an annotation model available to your account. `init`, `doctor`, `status`, and `run --plan` make no cloud/model calls. `run` performs preprocessing, annotation, four task builders, scoring preparation, and Gemini quality review; it can incur Modal and model usage. Use `run --until probe` for a local media check, or `run --until tasks` to stop before Gemini review.

Progress and log links appear in the private project's `README.md`. Rerunning construction or review resumes saved work. `accepted_tasks.json` contains candidates that passed review; `review --review-limit 5` reviews up to five additional tasks. Task generation includes scoring rules, but **an agent must submit its edit before it can receive a score**. `score` checks submissions locally; `score --execute` starts a new evaluation, using Gemini for RC/FC.

For source-only verification, run `python scripts/check_release.py` and `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_release_tools.py'`. These standard-library checks do not exercise the full training or media pipeline.

### Set up an annotation agent

Task generation supports **Claude Code, Codex CLI, and OpenCode 1.x**. Install and sign in to one CLI, then pass `--backend` and `--model` to `video_editing.py init`. The project saves those choices, so it does not require sourcing a shell profile. OpenCode uses a `provider/model` ID; `--cli` selects a particular executable.

For the [manual individual-stage workflow](docs/one_film.md), `python data_construction/setup_agent.py` remains available to create sourceable backend profiles. The optional [live CLI check](docs/agent_setup.md#3-verify-before-processing-media) tests file/image access and JSON output with a real model call.

**[Installation, login, model selection, and troubleshooting →](docs/agent_setup.md)**

## Documentation

| I want to… | Guide | Source |
| :--- | :--- | :--- |
| Generate and review tasks from my movie/trailer | [Recommended project workflow](docs/workflow.md) | [`video_editing.py`](video_editing.py) |
| Set up an annotation CLI | [Agent setup](docs/agent_setup.md) | [`setup_agent.py`](data_construction/setup_agent.py) |
| Run individual construction stages | [Manual one-film reference](docs/one_film.md) | [`pipeline.py`](data_construction/pipeline.py) |
| Explore the scalable task-generation pipeline | [Task generation](docs/data_construction.md) | [`data_construction/`](data_construction/) |
| Review and sample generated tasks | [Quality review](docs/data_construction.md#quality-review-and-sampling) | [`task_quality.py`](data_construction/task_quality.py) · [`quality_selection.py`](data_construction/quality_selection.py) |
| Inspect GRPO and fixed/decayed process rewards | [Training](docs/training.md) | [`training/`](training/) |
| Inspect token-level self-distillation and teacher inputs | [OPSD](docs/opsd.md) | [`opsd/`](opsd/) |
| Check and score completed edits | [Project scoring](docs/workflow.md#5-check-and-score-an-agents-edit) · [Evaluation details](docs/evaluation.md) | [`video_editing.py`](video_editing.py) · [`task_pipeline.py`](evaluation/task_pipeline.py) |
| Find recorded settings and dependency identities | [Experiment notes](docs/experiment_notes.md) | [`configs/`](configs/) |
| Check reproduction gaps and release boundaries | [Release status](docs/release_status.md) | [`scripts/check_release.py`](scripts/check_release.py) |

<details>
<summary><strong>Repository map</strong></summary>

```text
video_editing.py     Project setup, resumable generation, review, and scoring
data_construction/   Task-generation pipeline, adapters, and generic prompts
training/            Fixed-PRM source, separate decay overlay, and CPU tests
opsd/                Forward-KL recipe, teacher-input code, and adapters
evaluation/          Outcome graders and a separate decay overlay
configs/             Recorded settings, dependency pins, and source hashes
docs/                Component guides and paper figure
scripts/             Dependency-free source verification
tests/               Synthetic tests for release tools
```

</details>

## Data and reproducibility

The dataset is not distributed. The README includes the paper's Figure 1, with illustrative film thumbnails and aggregate results. Underlying film media, generated task instances, reference answers, cached trajectories, model checkpoints, and raw benchmark result files are not distributed.

The paper's four evaluation repetitions are **not four independent training seeds**. Recorded configurations and upstream commit IDs also do not reconstruct the original modified training environment. The final deterministic sampling procedure has been recovered; new semantic audits do not replay the historical task-by-task author repairs. Remaining environment, launch-service, evaluation-version and reference-answer OPSD gaps are tracked in [release status](docs/release_status.md).

## Citation

Author and software metadata are available in [`CITATION.cff`](CITATION.cff). A formal paper link and publication citation will be added when available.

Part of this work was completed while Lei Zhao was an intern at Philo Labs. For questions, contact [Lei Zhao](mailto:leizhao7@upenn.edu), [Zongheng Cao](mailto:tom@philolabs.ai), or [Xinyu Hu](mailto:christine@philolabs.ai).

## Acknowledgments

The implementation builds on OpenCode, SkyRL, vLLM, and Qwen3.5-9B. Exact identities are recorded in [`configs/dependency_pins.json`](configs/dependency_pins.json); local modifications must be considered alongside those pins.

## License

Project-authored code is licensed under the [MIT License](LICENSE). Third-party components retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). The paper figure, external media, and model weights are not covered by this code license.
