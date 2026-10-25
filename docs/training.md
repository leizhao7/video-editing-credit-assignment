# Training

The repository now includes recovered training sources, the full SkyRL patch and environment lock, reviewed-task preparation, and public Modal image/service/launch adapters. **The new public GPU deployment has not completed an optimizer smoke run.** Source recovery, CPU checks and native tool-host builds are verified separately below.

## What can I run now?

| Goal | Entry point | Verification |
| --- | --- | --- |
| GRPO/PRM algorithm checks | `training/scripts/run_cpu_tests.py` | Fixed and decay CPU suites |
| Reviewed tasks → frozen trainer inputs | `training/prepare.py` | Real FFmpeg preparation, film holdout, input integrity and private-answer separation tests |
| Rebuild the patched SkyRL sources | `training/environment/reconstruct.py` | All 22 files match recovered postimages; decay adds two checked changes |
| Rebuild the native OpenCode tool host | `training/environment/images.py` | Fresh Linux CPU build and offline binary startup passed |
| Deploy graders and inspect a training plan | `training/modal_train.py` | Configuration/import checked; complete public service deployment pending |
| GPU optimization | `training/modal_train.py --action smoke` | Public 4-inference + 4-training GPU configuration supplied; GPU execution pending |
| Formal training | `--action train` | Requires real, current-source qualification evidence; no receipt is fabricated |

The launch adapter currently targets **Qwen3.5-9B on one H200:8 node**, with four independent inference engines and four FSDP policy/reference GPUs. It does not offer a validated single-GPU or Qwen3.6-35B-A3B preset. The larger model's reported results remain documented in [experiment notes](experiment_notes.md).

## Using your own tasks

First generate and review tasks with [the project workflow](workflow.md). Its `accepted_tasks.json` contains the accepted task directories. Prepare a new private release **outside this checkout**:

```bash
python3.12 training/prepare.py \
  --accepted-tasks /path/to/project/accepted_tasks.json \
  --destination /path/to/private/training-release \
  --validation-film held-out-film-slug \
  --prepare-inputs \
  --reference-answers
```

Install `training/requirements-cpu.txt` and FFmpeg/FFprobe first. Music preparation also needs `GOOGLE_AI_STUDIO_API_KEY`; it caches labels for the original music. `--reference-answers` is needed only for reference-answer OPSD and prepares a separate teacher-only catalog. No answers are sent to student rollout workers.

Supply each held-out film with another `--validation-film`. Every task from that film goes into validation. There must be at least one training task. Formal training and both OPSD profiles require a nonempty validation split. With no held-out film, a train-only release can be used for a bounded GRPO/PRM smoke check.

The output contains:

```text
training-release/
  release.json                 # SHA-bound task identities and source-file hashes
  release_receipt.json
  preparation_receipt.json
  corpus/_tt_train|_tt_val/     # Trusted media, task definitions and verifier material
  prepared/train|val/          # Public agent input bundles and frozen manifests
  private/reference_answers.json  # Optional, teacher-only training catalog
```

Agent archives contain only `task.json`, `task_prompt.md`, `initial_state/`, and extracted `frames/`. Private verifiers, answers and `probe.json` stay in the trusted corpus. Existing prepared outputs are checked before reuse; source changes, corrupted inputs and symlinks fail closed. Use a new destination after changing the accepted task collection or film split.

New releases have explicit SHA-256 identities. Omitting that identity still selects the historical release guard; historical runs are not silently admitted as a new collection. New validation counts are bound to the new manifest; the old default remains 81.

## Configure and deploy

Use a separate Python 3.12 environment with `modal==1.6.2`, authenticate to your own Modal workspace, and copy the template outside the repository:

```bash
cp training/cloud.example.toml /path/to/private/training.toml
export EDITING_TRAINING_CONFIG=/path/to/private/training.toml
python training/cloud.py validate --config "$EDITING_TRAINING_CONFIG"
```

Choose one profile in the TOML:

| Profile | Training feedback |
| --- | --- |
| `grpo` | Outcome-only GRPO |
| `prm_fixed` | Outcome reward plus fixed-weight online process rewards |
| `prm_decay` | Outcome reward plus process rewards that decay to zero |
| `opsd_reference` | Forward top-64 + tail KL, teacher conditioned on private reference answers |
| `opsd_suggestions` | Same forward KL, teacher conditioned on turn suggestions |

Use a distinct app name for each deployed profile. The release, models, logs and checkpoints use separate operator-owned volumes. The secret names are configuration inputs: the outcome judge needs `GOOGLE_AI_STUDIO_API_KEY`; PRM and turn-suggestion generation need a configured process secret with that variable. Console logging needs no W&B credential. The dataset, film media, model weights and credentials are never committed to GitHub.

Upload the prepared release to a **new, empty** release volume, then build/deploy the source images:

```bash
python training/cloud.py upload \
  --config "$EDITING_TRAINING_CONFIG" \
  --release-directory /path/to/private/training-release
python -m modal deploy training/modal_train.py
```

Deployment builds CUDA dependencies and downloads the pinned native host/ASR components. It does not submit an RL run or allocate the H200:8 training node. The GPU image rebuild itself remains unqualified; dependency or source mismatches stop the build rather than replacing a pin.

Download the fixed model revision into your model volume, then inspect the command and run an import/source preflight:

```bash
python -m modal run training/modal_train.py --action prepare-model
python -m modal run training/modal_train.py --action check --name smoke-001
python -m modal run training/modal_train.py --action preflight --name smoke-001
```

These commands allocate CPU containers only. The plan binds the actual model tokenizer/template, source packages, tool host, sandbox image, prepared inputs and grader hashes. It creates fresh run identities rather than impersonating the paper's private deployments.

## Start a bounded GPU smoke run

The following command explicitly requests **eight H200 GPUs**, uses at most two optimizer updates, and submits a detached job:

```bash
python -m modal run training/modal_train.py \
  --action smoke --name smoke-001 --steps 2
```

A GRPO/PRM smoke uses two tasks with two rollouts each and smaller microbatches; its results are infrastructure checks, not paper scores. OPSD preserves the required 16-task, one-rollout shape even during smoke checks. Model generation and RC/FC grading may call paid judge APIs. Run directories are immutable; use a new name for each attempt.

A successful process exit is insufficient: the supervisor checks the requested completed-step checkpoint, preserves failures/logs and terminates only its own children. Checkpoints and traces remain on separate volumes.

For formal training:

```bash
python -m modal run training/modal_train.py \
  --action train --name experiment-001 --steps 50 \
  --readiness /logs/qualification/current-runtime.json
```

The readiness file must contain **real evidence** for the current contract, dataset, grader, token/image/logprob/gradient parity, collectives, sandbox tools, optimizer step, all-engine weight sync, checkpoint resume and durable trajectories. A new release requires its own `release_sha256` and `checks.dataset_release`; historical `dataset_release_797` evidence is not accepted for another collection. The retained guards in `opencode_rl/readiness.py` and the training entrypoints enforce these requirements. PRM additionally requires its current all-rank gradient probe evidence. Import or CPU unit-test reports do not satisfy this gate.

`--resume /checkpoints/<prior-run>/rolling/<checkpoint>` creates a new named continuation only when checkpoint receipts match the contract, prepared inputs, grader and objective. It cannot convert GRPO optimizer state into OPSD or migrate fixed PRM to decay by changing a path. The paper's decay run branched from the fixed run at update 20; the public decay profile from base is a new run, and does not claim to reproduce that historical optimizer migration.

## Losses and runtime boundaries

The trainer requires `loss_reduction=token_mean` and rejects extra batch-level advantage normalization. Outcome rewards are normalized across task siblings. Process scores are normalized among eligible siblings at the same ordinal turn, with the process coefficient applied **after** normalization. Prompt/tool gaps, masked actions and data-parallel padding do not enter policy loss.

The fixed process coefficient is 0.5. Decay follows the consumer optimizer's global step: 0.5 through update 20, then `0.5 * (1 - (step - 20) / 10)^3`, reaching zero at update 30. It is cubic, not linear. Resume preserves the global step. The online process judge uses ten-turn windows, three votes and `gemini-3.8-flash`. Provider failures retain the existing infrastructure/exclusion handling rather than becoming a policy reward.

The public grader reconstructs sources from the trusted, hash-checked corpus. CS/NS use task verifiers; RC/FC run evidence generation, blind/reference judging and aggregation. Judge credentials and private references are never mounted into rollout sandboxes. The grader checks source hashes again; unavailable grading is not a model zero. The shared ASR cache uses a separate Modal Volume in the public adapter.

Historical Chinese model-facing environment prompts are retained. Translating them changes the model's inputs and needs a new source identity; the English task-generation interface does not silently rewrite them.

## Layout and provenance

- `training/src/`: GRPO/PRM runtime, append-only agent loop, native tool host and integrity guards.
- `training/overlays/decay/`: eight decayed-PRM replacements; apply over the base runtime.
- `training/overlays/opsd/`: differing files from the coherent reference-image harness; unchanged files are shared with the base layer. Do not mix its generator with the later PRM runtime.
- `training/environment/`: actual training `uv.lock`, 277 installed-package version records, complete SkyRL patch, 21 `/patch` helpers, native host build and trusted service support.
- `opsd/recipes/reference_answers/`: recovered reference-answer algorithm; `opsd/examples/train/answer_self_distillation/` contains the final turn-suggestion variant. See [OPSD](opsd.md).
- `training/provenance.json` and `opsd/source_map.json`: historical source hashes and separately recorded public adaptations.
- `training/third_party/`: preserved OpenCode/SkyRL licenses and upstream metadata.

The upstream pins are SkyRL `59d4daedee24c7b1a79d857bbf322a6d195c3792`, OpenCode `d7a6e1daaf2271bcd0b611fbb27d4956806e475a`, Bun 1.3.14 and vLLM 0.26.0. The recovered CUDA environment uses PyTorch 2.11.0+cu130, Transformers 5.8.0 and Ray 2.57.0. `installed-packages.txt` is an inspection record; the build uses the recovered lock and explicit runtime additions.

The native host pins the already-resolved ghostty commit from the recovered lock before frozen installation. A separate bundle-metafile check requires that ghostty is absent from the tool-only binary. The new build has its own binary hash; it is not relabeled as the original binary.

To reconstruct source only in an external pinned SkyRL checkout:

```bash
python3.12 training/environment/reconstruct.py \
  --skyrl-root /path/to/pinned/SkyRL --profile prm_fixed
# --profile prm_decay additionally applies the two-file decay patch.
```

This does not install dependencies or certify GPU behavior. Both complete preimages and already-complete postimages are accepted; partial or corrupted trees are rejected. Runtime installation is restricted to an explicitly selected isolated Linux image build. The old two-file patch alone cannot rebuild the original base.

## Run the offline CPU tests

```bash
python3.12 -m venv "$HOME/.venvs/video-editing-training-cpu"
source "$HOME/.venvs/video-editing-training-cpu/bin/activate"
python -m pip install -r training/requirements-cpu.txt
export PYTHONDONTWRITEBYTECODE=1
python training/scripts/run_cpu_tests.py
python -m unittest discover -s tests
PYTHONPATH=opsd python -m pytest -q -p no:cacheprovider opsd/tests/test_paper_top64.py
```

The algorithm runner verifies source hashes, assembles variants in separate temporary directories, disables outbound sockets and hides CUDA devices. CPU requirements are test dependencies, not the GPU environment lock. FFmpeg/FFprobe enable the real synthetic-media preparation regression.

On 2026-10-24, source reconstruction and eight new release/launch-boundary tests passed. The full root suite passed 46 tests; GRPO/PRM passed 68 fixed tests plus 3 subtests and 24 decay tests, and the top-64 KL check passed 3 tests. The restored demonstrator quota/retry helpers passed 38 support tests; one additional client test requires full SkyRL fixtures/dependencies and was excluded from that isolated helper run. The fresh native host built from the pinned source and frozen dependencies, reported 2,767 bundle inputs with no ghostty dependency, and started in a network-blocked CPU sandbox. No GPU was allocated. These checks do not establish full training, all-rank numerical parity, service-level grading integration or benchmark reproduction. See [release status](release_status.md) for verification limits.
