# OPSD

The OPSD draft contains the training objective, teacher-input construction, sparse target transport, selected runtime adapters, and CPU tests. A complete public training environment and launcher are still being assembled. No dataset, reference-answer catalog, film/audio material, cached teacher response, model weight, or cloud credential is included.

## Paper settings

Both experiments use a frozen initial Qwen3.5-9B teacher, forward KL on the teacher's top 64 vocabulary tokens plus one aggregate tail class, full-vocabulary normalization, and token-mean reduction over valid assistant tokens. The settings are 16 tasks per batch, one rollout per task, 50 updates, and zero outcome-reward loss weight. Tool observations, context, padding, and private hints do not contribute to student loss.

| Setting | Teacher-only information |
|---|---|
| Reference answers | A private reference card inserted into the teacher system context; `answers.py` constructs and validates the catalog. |
| Turn suggestions | A Gemini proposal for the current next action, appended after the original turn context for the fixed Qwen teacher; previous or future proposals are excluded. |

`topk.py`, `topk_worker.py`, `topk_trainer.py`, `trainer.py`, `teacher_inputs.py`, and `turn_teacher.py` are unchanged from the retained source snapshot. They retain full-vocabulary softmax normalization rather than renormalizing only the top-64 entries. The aggregate tail does not preserve relative preferences within that tail.

The original API also supports reverse KL and contains a legacy sampled objective for comparison. A paper configuration must explicitly set `direction="forward"`, `top_k=64`, `mode="train"`, and `steps=50`; do not rely on the original API defaults. The n8 and turn-reference experiments are excluded.

## Source provenance

| Component | Pin |
|---|---|
| SkyRL | `59d4daedee24c7b1a79d857bbf322a6d195c3792` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| OpenCode | `d7a6e1daaf2271bcd0b611fbb27d4956806e475a` |
| Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |

The retained final turn-suggestion recipe has SHA256 `aa076cdab138da59f584e1fb3edc9fc485fca81e54b7b792bbccb39881df0cab`. The original reference-answer recipe has SHA256 `43262b0844c3b778d3d9b76607c1ec0f1e298ca02781a16bcfa05a4258f45018`; that complete historical snapshot has not yet been recovered. The current code contains reference-answer mode, but it is not represented as the byte-identical original answer experiment snapshot.

The public draft has a new source hash because deployment coupling and test fixture paths were changed. `opsd/source_map.json` records original and public SHA256 values and the exact change category for every copied file. No historical hash is reassigned to this draft.

The original turn-suggestion run used reviewed transport and admission repairs after updates 8 and 14, followed by an unchanged-recipe full-state continuation after update 20. Run-specific migration adapters are excluded. `check_resume` still accepts matching objective/source identities and rejects mismatched identities; this public draft does not replay the archived migration history.

## Configuration and deployment boundary

`deployment.py` is omitted because it embeds historical run paths, a private catalog checksum, and cloud deployment metadata. No included module imports it. `launch.new_plan` and `launch.training_command` remain plan/command construction functions; they require a correctly prepared harness configuration and do not constitute a standalone launcher.

Application names can be supplied through:

- `OPSD_DEMONSTRATOR_APP`: the user's demonstration service application; the neutral default is `opsd-turn-demonstrator`.
- `OPSD_QUOTA_APP`: the user's quota service application; the neutral default is `opsd-demo-quota`.

These variables change service routing only. The Gemini prompt/schema, teacher history boundary, model identity, retry limits, and quota arithmetic are preserved. No service is created by the CPU test runner. The service core receives authentication headers from its caller; credentials must stay outside the repository.

For a public build, restore the matching `append_agent_rl`, `opencode_rl`, and `/patch` sources and the production source preimages required by the hash-checked installer. The included seven adapter files are a subset, not a replacement for that complete dependency tree. `install.py` expects the documented container layout (`/opt/SkyRL`, `/patch`, and `/opt` receipts), so a future public Dockerfile must create that layout or expose a separately reviewed path configuration. A private cloud image identifier is not a build specification.

A new public CLI should take model, prepared-task, private-catalog, run-output, and checkpoint paths from user configuration, construct fresh source-bound identities, and select the two paper presets explicitly. Optimizer, sampling, microbatch, staleness, tool, and validation settings must be recovered from the recorded configuration; they must not be guessed from later experiments. The exact CUDA/PyTorch/Ray training dependency manifest is still missing. Keep upstream license/notice files when assembling the full build.

The data-construction code and private-catalog builder may be released without the original dataset. Users must prepare their own authorized media and generated task artifacts. This supports reuse of the workflow; exact score replication still requires the original evaluation data, which is not distributed.

## Selected CPU verification

The reviewed selection completed **82 passed, 16 skipped, 0 failed**. It covers an independent dense reference at exactly K=64 with 65 probability classes, chunked KL loss/gradients, full-vocabulary normalization, token mean under unequal simulated data-parallel partitions, token/media masks, teacher-only input construction, and bounded quota/retry utilities using synthetic clients. No GPU, real model, real dataset, service startup, or cloud call was used.

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
