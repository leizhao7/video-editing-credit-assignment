# When Fine-Grained Credit Assignment Helps and Hurts: Multimodal Agentic Reinforcement Learning through Video Editing

Code for studying outcome-level, turn-level and token-level learning signals in movie-trailer editing. The four task types are Content Selection (CS), Narrative Sequencing (NS), Rough Cut (RC), and Fine Cut (FC).

**Release preparation draft.** This tree collects recovered source and portable data-construction code. It is not yet a verified, end-to-end reproduction package. In particular, the original experiments used modified training dependencies and private execution images. The remaining work is recorded in [release status](docs/release_status.md); copying a historical Modal launcher is not sufficient to reconstruct that environment.

We intend to release code and the data-construction procedure. The dataset, film media, generated task instances, reference answers, cached trajectories, and model checkpoints are not included.

This source-only tree also excludes generated audit/test reports, subtitles, annotation records, result tables and caches. Personal filesystem paths, original film/task examples and historical cloud deployment identifiers have been removed or replaced with operator-supplied configuration. Generic source code, configuration templates, prompt templates, licenses and code-hash manifests remain. Data-construction inputs and outputs are selected through `VIDEO_EDITING_DATA_ROOT`; optional Modal resources are configured in `data_construction/environment.example`.

## Authors and contact

- Lei Zhao — [leizhao7@upenn.edu](mailto:leizhao7@upenn.edu)
- Zongheng Cao — [tom@philolabs.ai](mailto:tom@philolabs.ai)
- Xinyu Hu — [christine@philolabs.ai](mailto:christine@philolabs.ai)

Citation metadata is available in [CITATION.cff](CITATION.cff).

## Repository layout

| Path | Contents |
|---|---|
| `data_construction/` | Portable copies of task-building utilities and release prompt templates |
| `training/` | Recovered fixed-PRM source and separately identified decay changes |
| `opsd/` | Recovered forward-KL distillation recipe and supporting adapters |
| `evaluation/` | Outcome grader source; decay transport changes remain a separate overlay |
| `configs/` | Recorded method settings, dependency identities, and source hashes |
| `scripts/` | Local source verification; no training or cloud deployment on import |
| `docs/` | Construction procedure, provenance, and unresolved reproduction requirements |

The internal task-stage names map to CS=`s_1`, NS=`s_2`, RC=`s_3`, and FC=`s_4e`. The earlier `s_4` task is not the paper's Fine Cut task.

## Local verification

Python 3.12 is used by the recovered training recipe. Start with the dependency-free source check:

```sh
python scripts/check_release.py
```

This checks syntax, recorded source hashes, symlinks, and excluded artifact types. It does not run training, make judge requests, or establish that all imports and GPU dependencies are installed. See [evaluation](docs/evaluation.md) for the local transport regressions and the distinction between those checks and a full benchmark run.

Dependency-free tests for release validation, source assembly and patch verification run with:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_release_tools.py'
```

Source assembly rejects malformed manifests and changed source files, applies the decay overlay after the base layer, and cleans up incomplete destinations on failure. The preprocessing and evaluation guides document additional synthetic tests for their error handling. All test fixtures are created temporarily; no benchmark examples or generated reports are shipped.

## Experiment configuration

The reward-based settings use 16 tasks per update, 8 rollouts per task, seed 42, and 50 optimizer updates. OPSD uses one rollout per task. Training uses token-mean loss reduction. The recorded settings in `configs/` distinguish the methods and preserve implementation details; they are reference configuration records, not drop-in cloud launch files.

For PRM, the fixed coefficient is 0.5. The decayed condition branches from the fixed-PRM checkpoint at update 20 and reduces the coefficient cubically to zero over updates 20–30, measured by the consumer optimizer's global step. After update 30, the coefficient remains zero.

The reported main results use four evaluation repetitions. These are not four independent training seeds. The original dataset contained 797 tasks from 49 films, with 716 training and 81 validation tasks. Rebuilding tasks from independently supplied media will not reproduce the original dataset or numerical results exactly.

## Dependencies and attribution

The implementation uses OpenCode, SkyRL, vLLM, and Qwen3.5-9B. Commit and model identities are in `configs/dependency_pins.json`. Those upstream commit IDs must be combined with the recorded local changes; they do not imply unmodified upstream installations. Upstream licenses and build requirements must be retained when integrating dependencies. The project license and formal paper link remain to be added.
