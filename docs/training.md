# Training code

This draft contains the fixed-PRM training runtime and the changes used for the decayed-PRM run. The core algorithms are available for inspection, and the synthetic CPU tests pass. **A portable GPU training build has not yet been qualified.** The historical deployment launchers and private cloud resource identifiers are excluded.

## Layout and provenance

- `training/src/`: fixed-PRM runtime, including the append-only agent harness, native OpenCode tool host, GRPO/PRM advantage construction, causal packing and integrity checks.
- `training/overlays/decay/`: eight changed files to apply over the fixed runtime. These preserve the decayed run's scheduling, transport, collection and persistence changes.
- `training/patches/skyrl/`: the two native SkyRL startup changes as a patch, with required before/after hashes. Whole modified SkyRL files are not vendored.
- `training/provenance.json`: per-file source/public hashes and public portability and source-only adaptations.
- `training/third_party/`: preserved upstream licenses, package metadata and lockfiles. These are provenance artifacts, not a newly resolved training environment.
- `training/tests/`: synthetic examples only; no film assets, task corpus or reference answers.

The internal archive matches all 114 files in the fixed runtime manifest and all 116 files in the decay runtime manifest. The public fixed source includes 113 files: the private historical launcher is excluded, and the Azure transport configuration and offline calibration inputs are adapted as described below. The decay variant adds eight replacement files and the two-file SkyRL patch. This is a PRM runtime snapshot; it does not independently establish an exact final outcome-only GRPO snapshot. See `opsd/` for separately organized distillation code.

## Code map

| Component | Main files |
| --- | --- |
| Append-only agent and native editing tools | `src/append_agent_rl/loop.py`, `native/tool_host.ts`, `task_rollout.py` |
| Capture and validate exact model inputs/actions | `src/opencode_rl/records.py`, `causal_prefix.py`, `prefix_transport.py` |
| Construct outcome and turn-level process advantages | `src/opencode_rl/training.py` |
| Integrate with the native SkyRL trainer | `src/opencode_rl/skyrl_adapter.py`, `prefix_backward.py`, `append_agent_rl/train_entrypoint.py` |
| Process judge protocol, evidence and collection | `src/prm_v2/judge_v5.py`, `live_packet_v5.py`, `live_judge_v5.py` |
| Runtime source and gradient checks | `src/prm_training_source.py`, `src/prm_v2/gradient_source_guard_v5.py` |
| Optimizer-step decay schedule | `overlays/decay/prm_v2/live_config_v5.py` |

Paths in this table are relative to `training/`, except that `native/tool_host.ts` and `task_rollout.py` share the first row's `src/append_agent_rl/` directory.

The trainer requires `loss_reduction=token_mean` and rejects additional batch-level advantage normalization. Outcome rewards are normalized across trajectories for a task. Process scores are normalized among eligible trajectories at the same ordinal turn; the process coefficient is applied **after** this normalization. Prompt/tool gaps, masked action tokens and data-parallel padding are excluded from policy loss. These checks remain in the public code.

Fixed PRM uses a coefficient of 0.5. The decay overlay uses the consumer optimizer's global step: 0.5 through step 20, then `0.5 * (1 - (step - 20) / 10)^3`, reaching zero at step 30 and remaining zero. This is a cubic schedule, not a linear ramp. The global step is preserved on resume. The overlay stops collecting new process rewards at the zero boundary and preserves outcome learning.

The judge configuration records ten-turn windows, three votes and `gemini-3.8-flash`. A failed process assessment zeros only that process contribution, retaining the outcome reward. Provider availability and credentials are deployment inputs; importing the configuration does not call a provider.

## Run the offline CPU tests

Use a Python environment with the packages listed in `training/requirements-cpu.txt`. This list covers tests only and is not a GPU training lockfile. From the repository root:

```bash
python training/scripts/run_cpu_tests.py
# Or select one variant:
python training/scripts/run_cpu_tests.py --variant decay
```

The runner verifies the public hashes, assembles each variant in a separate temporary directory, disables outbound socket connections, hides CUDA devices and runs each variant in its own process. It needs no credentials, Modal account, media files or model checkpoints. Directly running all test directories in one Python process can mix fixed and decay imports; use this runner.

The assembly helper validates manifest paths, duplicate entries, file hashes and symlinks before creating a runtime. Decay files override the base layer regardless of manifest ordering. It writes the bytes that were verified, preserves executable permissions, refuses existing destinations and removes its new destination if writing fails. The dependency-free release-tool tests in `tests/test_release_tools.py` cover these boundaries and patch verification. These helpers do not alter training losses or reward calculations.

Validated on 2026-10-07: **68 fixed tests plus 3 subtests, and 24 decay tests passed**. Coverage includes judge protocol validation, synthetic evidence serialization, causal packing, original-action gradients, advantage decomposition, process coefficients after normalization, padding masks, restart step handling and the zero-weight boundary. These tests do not measure judge quality, model performance, native GPU numerical parity or a complete training run.

To inspect an assembled CPU runtime without starting anything:

```bash
python training/scripts/materialize.py --variant decay --destination /tmp/editing-decay-runtime
```

The destination must not already exist. This command does not install or patch SkyRL, create cloud resources or satisfy the historical training qualification checks.

## Dependencies and configuration changes

The recorded upstream versions are OpenCode `d7a6e1daaf2271bcd0b611fbb27d4956806e475a`, SkyRL `59d4daedee24c7b1a79d857bbf322a6d195c3792`, vLLM `0.26.0` / `568afb3a13806beb53bb2e6bd518269357b237c0`, and Bun `1.3.14`. SkyRL and OpenCode licenses are preserved. The OpenCode lockfile contains a historical floating `ghostty-web` dependency whose frozen installation still needs resolution; the saved lockfile is not evidence of a successful fresh build.

In the public fixed copy, `azure_gpt55_transport.py` is adapted: an account-specific endpoint and deployment default are replaced with required configuration. Set `AZURE_OPENAI_CHAT_COMPLETIONS_URL` and `AZURE_OPENAI_DEPLOYMENT` before importing the Azure evaluation transport. The endpoint can alternatively be passed explicitly to its constructor. Deployment metadata uses the same configured value as the request. Sampling checks and request/context mapping are unchanged; no Azure request is made by the CPU test suite. The original and public hashes are recorded in `training/provenance.json`.

The offline calibration adapter `prm_v2/dataset.py` requires explicit `--source` and `--output` arguments. Optional `--challenge STAGE=TASK_ID` arguments replace corpus-specific task IDs. Sources, outputs and task choices belong to the operator; no original rollout archive or fixed challenge list is bundled. The ranking calculation is unchanged. Synthetic test fixtures use generic tool-markup descriptions, and generated test reports remain outside this repository.

Gemini transports read `GOOGLE_AI_STUDIO_API_KEY`. Cloud image, application, volume, checkpoint and dataset locations are supplied by a training plan. No working private plan is included. Container paths such as `/opt/SkyRL` describe the original runtime contract and remain in source; they are not a portable installation recipe.

The SkyRL patch applies only to the already-qualified parent files identified by its manifest, not automatically to a pristine checkout of the upstream commit. Before using it, run `python training/scripts/verify_skyrl_patch.py before <SkyRL-root>`; after applying the patch, use the same command with `after`. A hash mismatch must be resolved by reconstructing the intended parent build, not by removing or updating the guard to match an unrelated file.

## Remaining build work

1. Reconstruct the complete dependency/image build and public deployment wrapper, including media tools, model files and task-bundle locations; resolve the OpenCode frozen-build issue.
2. Requalify source, tokenizer/template, tools, numerical parity and gradient checks for the new build and issue a matching contract. The historical source guards still require the excluded launcher and original hashes; they intentionally do not certify this adapted public copy.
3. Connect the released grading components under `evaluation/` to a public service wrapper and plan. Their core sources are present; the private historical Modal service wrapper is not part of the portable recipe.
4. Complete dependency notices/licenses and GPU smoke/numerical tests, then execute an end-to-end training and evaluation run before claiming full reproducibility.

The dataset and film assets are not distributed. The data-construction code documents how task inputs are produced. Evaluation repetitions are separate from training seeds; four evaluation repetitions must not be presented as four independent training runs.
