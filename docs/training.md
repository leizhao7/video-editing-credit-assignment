# Training code

This draft contains the fixed-PRM training runtime and the changes used for the decayed-PRM run. The core algorithms are available for inspection, and the synthetic CPU tests pass. **A portable GPU training build has not yet been qualified.** The historical deployment launchers and private cloud resource identifiers are excluded.

## What can I run now?

| Goal | Current support |
| --- | --- |
| Inspect GRPO/PRM losses and run synthetic CPU checks | Supported by the instructions below; no media, model, or cloud account is needed. |
| Train on tasks from `video_editing.py` | Not connected yet. The trainer still requires the historical dataset identity; see [using your own tasks](#using-your-own-tasks). |
| Launch the paper's GPU runs from this checkout | A public environment build, service deployment, and launch plan are still required. Files in `configs/` record settings; they are not executable presets. |
| Run OPSD checks or training | Separate dependencies and remaining gaps are documented in [OPSD](opsd.md). The PRM CPU requirements do not provide an OPSD environment. |

The retained supervisor uses four inference GPUs and four training GPUs. It does not provide a validated single-GPU or smaller smoke configuration. Installing an annotation CLI for data generation does not install the pinned OpenCode training harness, SkyRL, or vLLM.

## Layout and provenance

- `training/src/`: fixed-PRM runtime, including the append-only agent harness, native OpenCode tool host, GRPO/PRM advantage construction, causal packing and integrity checks.
- `training/overlays/decay/`: eight changed files to apply over the fixed runtime. These preserve the decayed run's scheduling, transport, collection and persistence changes.
- `training/patches/skyrl/`: the two native SkyRL startup changes as a patch, with required before/after hashes. Whole modified SkyRL files are not vendored.
- `training/provenance.json`: per-file source/public hashes and public portability and source-only adaptations.
- `training/third_party/`: preserved upstream licenses, package metadata and lockfiles. These are provenance artifacts, not a newly resolved training environment.
- `training/tests/`: synthetic examples only; no film assets, task corpus or reference answers.

The internal archive matches all 114 files in the fixed runtime manifest and all 116 files in the decay runtime manifest. The public fixed source includes 113 files: the private historical launcher is excluded, and Azure transport configuration, offline calibration inputs, and provider-error compatibility are adapted as described below. The decay variant adds eight replacement files and the two-file SkyRL patch. This is a PRM runtime snapshot; it does not independently establish an exact final outcome-only GRPO snapshot. See `opsd/` for separately organized distillation code.

## Code map

| Component | Main files |
| --- | --- |
| Append-only agent and native editing tools | [loop.py](../training/src/append_agent_rl/loop.py), [native/tool_host.ts](../training/src/append_agent_rl/native/tool_host.ts), [task_rollout.py](../training/src/append_agent_rl/task_rollout.py) |
| Capture and validate exact model inputs/actions | [records.py](../training/src/opencode_rl/records.py), [causal_prefix.py](../training/src/opencode_rl/causal_prefix.py), [prefix_transport.py](../training/src/opencode_rl/prefix_transport.py) |
| Construct outcome and turn-level process advantages | [training.py](../training/src/opencode_rl/training.py) |
| Integrate with the native SkyRL trainer | [skyrl_adapter.py](../training/src/opencode_rl/skyrl_adapter.py), [prefix_backward.py](../training/src/opencode_rl/prefix_backward.py), [train_entrypoint.py](../training/src/append_agent_rl/train_entrypoint.py) |
| Process judge protocol, evidence and collection | [judge_v5.py](../training/src/prm_v2/judge_v5.py), [live_packet_v5.py](../training/src/prm_v2/live_packet_v5.py), [live_judge_v5.py](../training/src/prm_v2/live_judge_v5.py) |
| Runtime source and gradient checks | [prm_training_source.py](../training/src/prm_training_source.py), [gradient_source_guard_v5.py](../training/src/prm_v2/gradient_source_guard_v5.py) |
| Optimizer-step decay schedule | [live_config_v5.py](../training/overlays/decay/prm_v2/live_config_v5.py) |

The trainer requires `loss_reduction=token_mean` and rejects additional batch-level advantage normalization. Outcome rewards are normalized across trajectories for a task. Process scores are normalized among eligible trajectories at the same ordinal turn; the process coefficient is applied **after** this normalization. Prompt/tool gaps, masked action tokens and data-parallel padding are excluded from policy loss. These checks remain in the public code.

Fixed PRM uses a coefficient of 0.5. The decay overlay uses the consumer optimizer's global step: 0.5 through step 20, then `0.5 * (1 - (step - 20) / 10)^3`, reaching zero at step 30 and remaining zero. This is a cubic schedule, not a linear ramp. The global step is preserved on resume. The overlay stops collecting new process rewards at the zero boundary and preserves outcome learning.

The recorded decay experiment branches from the fixed-PRM checkpoint after update 20, including optimizer state; it does not restart from the initial model. See [the recorded decay settings](../configs/prm_decay.json).

The judge configuration records ten-turn windows, three votes and `gemini-3.8-flash`. A failed process assessment zeros only that process contribution, retaining the outcome reward. Provider availability and credentials are deployment inputs; importing the configuration does not call a provider.

## Run the offline CPU tests

From the repository root, create a Python 3.12 environment outside the checkout and install the CPU dependencies. Package installation uses the network; the tests themselves run offline. This requirements file covers tests only, has unpinned dependencies, and is not a GPU training lockfile.

```bash
python3.12 -m venv "$HOME/.venvs/video-editing-training-cpu"
source "$HOME/.venvs/video-editing-training-cpu/bin/activate"
python -m pip install -r training/requirements-cpu.txt
export PYTHONDONTWRITEBYTECODE=1
python training/scripts/run_cpu_tests.py
# Or select one variant:
python training/scripts/run_cpu_tests.py --variant decay
# Source assembly, checkpoint identity, and grading-error compatibility:
python -m unittest discover -s tests
```

The runner verifies the public hashes, assembles each variant in a separate temporary directory, disables outbound socket connections, hides CUDA devices and runs each variant in its own process. It needs no credentials, Modal account, media files or model checkpoints. Directly running all test directories in one Python process can mix fixed and decay imports; use this runner.

The assembly helper validates manifest paths, duplicate entries, file hashes and symlinks before creating a runtime. Decay files override the base layer regardless of manifest ordering. It writes the bytes that were verified, preserves executable permissions, refuses existing destinations and removes its new destination if writing fails. The dependency-free release-tool tests in `tests/test_release_tools.py` cover these boundaries and patch verification. These helpers do not alter training losses or reward calculations.

Revalidated on 2026-10-21 in a fresh CPU environment: **68 fixed tests plus 3 subtests, and 24 decay tests passed**, with both variant exit codes equal to zero. This check used Python 3.12.14, PyTorch 2.14.1, pytest 9.1.1, NumPy 2.5.4, and SciPy 1.18.1. These are test-environment versions, not a recovered production lockfile. Coverage includes judge protocol validation, synthetic evidence serialization, causal packing, original-action gradients, advantage decomposition, process coefficients after normalization, padding masks, restart step handling and the zero-weight boundary. These tests do not measure judge quality, model performance, native GPU numerical parity or a complete training run.

To inspect an assembled CPU runtime without starting anything:

```bash
python training/scripts/materialize.py --variant decay --destination /tmp/editing-decay-runtime
```

The destination must not already exist. This command does not install or patch SkyRL, create cloud resources or satisfy the historical training qualification checks.

## Using your own tasks

`accepted_tasks.json` from the project workflow is not yet a trainer input. [task_bundle.py](../training/src/opencode_rl/task_bundle.py) binds preparation to the original dataset release SHA-256, [train_dataset.py](../training/src/opencode_rl/train_dataset.py) enforces that identity and task ordering, and [train_entrypoint.py](../training/src/opencode_rl/train_entrypoint.py) requires 81 validation tasks when validation is enabled. Changing dataset paths alone cannot admit a new collection.

A public adapter must turn reviewed tasks into a new, validated release manifest and prepared bundles, separate public task inputs from private scoring material, and bind their identities into a new training plan. The original checks must remain intact for historical runs. The `prm_v2/dataset.py` adapter described below prepares offline judge-calibration inputs; it does not connect generated tasks to the RL trainer.

Some retained training environment/tool prompts are Chinese, including those in `runtime.py`, `mcp_tools.py`, and `submission_policy.py`. They are model inputs, not just interface text. An English training prompt set needs a new prompt/source identity and validation; the English data-generation interface does not change these historical training inputs.

## Dependencies and configuration changes

The recorded upstream versions are OpenCode `d7a6e1daaf2271bcd0b611fbb27d4956806e475a`, SkyRL `59d4daedee24c7b1a79d857bbf322a6d195c3792`, vLLM `0.26.0` / `568afb3a13806beb53bb2e6bd518269357b237c0`, and Bun `1.3.14`. SkyRL and OpenCode licenses are preserved. The OpenCode lockfile contains a historical floating `ghostty-web` dependency whose frozen installation still needs resolution; the saved lockfile is not evidence of a successful fresh build.

In the public fixed copy, `azure_gpt55_transport.py` is adapted: an account-specific endpoint and deployment default are replaced with required configuration. Set `AZURE_OPENAI_CHAT_COMPLETIONS_URL` and `AZURE_OPENAI_DEPLOYMENT` before importing the Azure evaluation transport. The endpoint can alternatively be passed explicitly to its constructor. Deployment metadata uses the same configured value as the request. Sampling checks and request/context mapping are unchanged; no Azure request is made by the CPU test suite. The original and public hashes are recorded in `training/provenance.json`.

The offline calibration adapter `prm_v2/dataset.py` requires explicit `--source` and `--output` arguments. Optional `--challenge STAGE=TASK_ID` arguments replace corpus-specific task IDs. Sources, outputs and task choices belong to the operator; no original rollout archive or fixed challenge list is bundled. The ranking calculation is unchanged. Synthetic test fixtures use generic tool-markup descriptions, and generated test reports remain outside this repository.

Both grading-client variants recognize provider-refusal errors from the current English evaluation adapter as well as their historical Chinese forms. This compatibility change preserves terminal refusal handling and does not convert a provider failure into a policy score. Original and public source identities remain separately recorded.

Gemini transports read `GOOGLE_AI_STUDIO_API_KEY`. Cloud image, application, volume, checkpoint and dataset locations are supplied by a training plan. No working private plan is included. Container paths such as `/opt/SkyRL` describe the original runtime contract and remain in source; they are not a portable installation recipe.

The SkyRL patch applies only to the already-qualified parent files identified by its manifest, not automatically to a pristine checkout of the upstream commit. Before using it, run `python training/scripts/verify_skyrl_patch.py before <SkyRL-root>`; after applying the patch, use the same command with `after`. A hash mismatch must be resolved by reconstructing the intended parent build, not by removing or updating the guard to match an unrelated file.

## Remaining build work

1. Reconstruct the complete dependency/image build and public deployment wrapper, including media tools, model files and task-bundle locations; resolve the OpenCode frozen-build issue.
2. Requalify source, tokenizer/template, tools, numerical parity and gradient checks for the new build and issue a matching contract. The historical source guards still require the excluded launcher and original hashes; they intentionally do not certify this adapted public copy.
3. Connect the released grading components under `evaluation/` to a public service wrapper and plan. Their core sources are present; the private historical Modal service wrapper is not part of the portable recipe.
4. Add the reviewed-task adapter and public dataset/plan validation described above, with explicit GPU allocation and model presets.
5. Complete dependency notices/licenses and GPU smoke/numerical tests, then execute an end-to-end training and evaluation run before claiming full reproducibility.

The dataset and film assets are not distributed. The data-construction code documents how task inputs are produced. Evaluation repetitions are separate from training seeds; four evaluation repetitions must not be presented as four independent training runs.
