# OPSD

Both OPSD recipes, their coherent source runtime, teacher-input construction, sparse targets, environment sources and public Modal launch adapters are included. GPU execution of the public build remains unqualified; see the [training commands](training.md#configure-and-deploy). No dataset, reference-answer catalog, film/audio material, cached teacher response, model weight, or cloud credential is included.

## Paper settings

Both experiments use a frozen initial Qwen3.5-9B teacher, forward KL on the teacher's top 64 vocabulary tokens plus one aggregate tail class, full-vocabulary normalization, and token-mean reduction over valid assistant tokens. The settings are 16 tasks per batch, one rollout per task, 50 updates, and zero outcome-reward loss weight. Tool observations, context, padding, and private hints do not contribute to student loss.

| Setting | Teacher-only information |
|---|---|
| Reference answers | A private reference card inserted into the teacher system context; `answers.py` constructs and validates the catalog. |
| Turn suggestions | A Gemini proposal for the current next action, appended after the original turn context for the fixed Qwen teacher; previous or future proposals are excluded. |

`topk.py`, `topk_worker.py`, `topk_trainer.py`, `trainer.py`, `teacher_inputs.py`, and `turn_teacher.py` are unchanged from the retained source snapshot. They retain full-vocabulary softmax normalization rather than renormalizing only the top-64 entries. The aggregate tail does not preserve relative preferences within that tail.

The original recipe API also supports reverse KL and contains a legacy sampled objective for comparison. The public launcher explicitly selects forward KL and top-64 targets by default; formal paper settings use `--action train --steps 50`. The underlying recipe API retains its original defaults, so direct callers must specify `direction="forward"`. The n8 and turn-reference experiments are excluded.

## Source provenance

| Component | Pin |
|---|---|
| SkyRL | `59d4daedee24c7b1a79d857bbf322a6d195c3792` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| OpenCode | `d7a6e1daaf2271bcd0b611fbb27d4956806e475a` |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

The retained final turn-suggestion recipe has SHA256 `aa076cdab138da59f584e1fb3edc9fc485fca81e54b7b792bbccb39881df0cab`. The original reference-answer recipe was recovered from its historical training image on 2026-10-24. All 17 files reproduce the recorded aggregate SHA256 `43262b0844c3b778d3d9b76607c1ec0f1e298ca02781a16bcfa05a4258f45018`. Its algorithm is published separately under `opsd/recipes/reference_answers/`. The private deployment file is replaced by the common public launcher, and the supervisor also accepts the explicitly selected source profile; the public recipe therefore has a new hash.

The public draft has a new source hash because deployment coupling and test fixture paths were changed. `opsd/source_map.json` records original and public SHA256 values and the exact change category for every copied file. No historical hash is reassigned to this draft.

The original turn-suggestion run used reviewed transport and admission repairs after updates 8 and 14, followed by an unchanged-recipe full-state continuation after update 20. Run-specific migration adapters are excluded. `check_resume` still accepts matching objective/source identities and rejects mismatched identities; this public draft does not replay the archived migration history.

## Configuration and deployment boundary

The private `deployment.py` files are replaced by `training/launcher.py`, `training/modal_train.py` and `training/cloud.example.toml`. The profiles are `opsd_reference` and `opsd_suggestions`. Both use the coherent recovered OPSD harness overlay, fixed initial teacher, one rollout per task, and fresh source-bound contracts. The public supervisor accepts `runtime_profile="source"`; historical frozen-profile checks are retained and are not waived or impersonated.

Prepare reviewed tasks with `training/prepare.py --prepare-inputs`. Reference-answer OPSD also requires `--reference-answers`; its catalog stays under the private release directory. Turn suggestions use the same app's `generate` service and a single shared `DemonstrationQuota` class. `demonstrator_tokens_per_minute` is an explicit operator setting; request validation, CountTokens admission, raw response retention and bounded training-side retries are preserved. The new deployment does not assume the authors' API quota or credentials.

The recovered `training/environment/uv.lock`, installed package record, complete SkyRL patch and 21 dependency/runtime helpers supply the previously missing environment sources. The OPSD installer adds its checked validity-only admission and top-K worker/head paths on top of that reconstructed base. No private image identifier is used as a public build recipe. The older files in `opsd/adapters/frozen_v15/` remain historical utilities; the public source profile does not depend on an incomplete frozen adapter tree.

Users must prepare their own authorized media and task artifacts. Exact score replication requires the original evaluation data, which is not distributed.

## Selected CPU verification

For checks that do not need a SkyRL checkout, run from the repository root:

```bash
# Standard-library checkpoint identity checks; no package installation needed.
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_opsd_resume.py'

# After installing training/requirements-cpu.txt in a separate environment:
PYTHONPATH=opsd PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider opsd/tests/test_paper_top64.py
```

On 2026-10-21 these commands passed 9 resume tests and 3 K=64 numerical tests, respectively. They check matching checkpoint identities, rejection of changed objectives/data/contracts/graders, valid resume steps, and the paper's sparse KL calculation. They do not exercise the complete trainer or multimodal harness. The broader runner below still needs the matching SkyRL and harness sources; installing the PRM CPU requirements alone is insufficient.

In the earlier environment with matching SkyRL and harness sources, the reviewed selection completed **82 passed, 16 skipped, 0 failed**. It covers an independent dense reference at exactly K=64 with 65 probability classes, chunked KL loss/gradients, full-vocabulary normalization, token mean under unequal simulated data-parallel partitions, token/media masks, teacher-only input construction, and bounded quota/retry utilities using synthetic clients. No GPU, real model, real dataset, service startup, or cloud call was used. This broader selection was not rerun successfully in the fresh PRM-only environment: collection stops at the missing SkyRL dependency.

The 16 skips are two production-source inspections and 14 parameterized production-trimmer checks whose explicit source fixtures were not provided. No dependency error occurred in the completed selection. Tests used an existing CPU environment with Python 3.12, PyTorch 2.14.0, pytest 9.1.1, and SciPy 1.18.1, plus the recorded SkyRL checkout and existing harness source on `PYTHONPATH`. These are test-environment versions, not the recovered production training lockfile. The shared environment was not modified.

After making the pinned SkyRL and matching harness packages importable, run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python opsd/tests/run_cpu.py
```

The runner adds the OPSD package/runtime paths, selects only the reviewed synthetic CPU tests, and disables CUDA visibility. It does not install dependencies. Missing SkyRL, harness, Torch, SciPy, pytest, httpx, or jsonschema dependencies must be resolved in the user's own environment first.

Optional source-inspection fixtures are explicit:

| Variable | Input |
|---|---|
| `OPSD_TEST_SKYRL_ROOT` | Matching SkyRL source checkout |
| `OPSD_TEST_FUSED_CE_SOURCE` | Matching production fused-head Python source |
| `OPSD_TEST_FROZEN_SOURCE_DIR` | Hash-matched production trimmer/worker Python sources |
| `OPSD_TEST_FROZEN_ADAPTER_DIR` | Matching frozen adapter sources for broader optional tests |
| `OPSD_TEST_BAD_JSON_DIR` | Optional retained-output fixtures for a test outside the selected runner |

Absent optional fixtures cause explicit skips; configured but missing files cause failures. No historical JSON fixture or answer/media asset is bundled. The full recipe test suite, distributed training, GPU checkpoint restore, and end-to-end launch remain unverified for this public draft.
