# Generate tasks from one film

**For a new run, use the [recommended project workflow](workflow.md).** Its `video_editing.py` entry point handles the commands below in order, saves progress, and connects quality review and scoring. This page remains the manual reference for inspecting or running individual stages; use a separate private root rather than mixing manual commands into a managed project.

Start with one full film and its matching trailer in a new private directory. This walkthrough builds candidate CS, NS, RC, and FC tasks, attaches the released scoring configuration, and runs semantic quality review. The resulting accepted-task list can be passed directly to the scoring entry point. It does not reproduce the paper's private dataset or every historical repair.

You need Python 3.12, FFmpeg/FFprobe, Tesseract with English language data, a Modal account, and one authenticated annotation CLI. Use English media with English as the default audio track: transcription requests English, and several clip-extraction steps use the default track. You do not need a local GPU or DaVinci Resolve: local commands prepare and assemble media; Modal runs TransNetV2, Whisper, and Demucs.

## 1. Install in a separate environment

From the repository root:

```sh
python3.12 -m venv "$HOME/.venvs/video-editing"
source "$HOME/.venvs/video-editing/bin/activate"
python -m pip install -r data_construction/requirements.txt \
  -r data_construction/requirements-modal.txt \
  -r evaluation/requirements.txt

python --version
ffmpeg -version
ffprobe -version
tesseract --list-langs
modal setup
```

Check that Tesseract lists `eng`. On macOS, Homebrew provides `ffmpeg` and `tesseract`; on Ubuntu, the corresponding packages are `ffmpeg`, `tesseract-ocr`, and `tesseract-ocr-eng`. Install a current Python separately if `python3` resolves to an older system Python. These Python requirements are direct dependencies, not the historical experiment environment.

Use the requirements file rather than installing transcription packages individually. It constrains PyAV below 19 for faster-whisper 1.2.1: newer PyAV removed an argument that this release still uses. Otherwise audio validation can fail before transcription starts. [Upstream compatibility issue](https://github.com/SYSTRAN/faster-whisper/issues/1589).

Install and authenticate one CLI through [Agent setup](agent_setup.md); create its run-specific configuration in the next step. Select a model available to your account; an installed CLI alone does not establish model access.

## 2. Supply exactly one source and one trailer

```sh
export VIDEO_EDITING_DATA_ROOT="$HOME/video-editing-demo"
mkdir -p "$VIDEO_EDITING_DATA_ROOT/sources/MyFilm"
cp '/path/to/full-movie.mkv' "$VIDEO_EDITING_DATA_ROOT/sources/MyFilm/source.mkv"
cp '/path/to/matching-trailer.mp4' "$VIDEO_EDITING_DATA_ROOT/sources/MyFilm/trailer.mp4"

# Example using Claude; use your own available backend/model if different.
python data_construction/setup_agent.py \
  --backend claude --model sonnet --effort high \
  --data-root "$VIDEO_EDITING_DATA_ROOT" \
  --output "$VIDEO_EDITING_DATA_ROOT/agent.env" --check
source "$VIDEO_EDITING_DATA_ROOT/agent.env"
```

Replace the two input paths with your media. Filenames must contain `source` and `trailer`, respectively. Keep unrelated videos out of this directory. A source-file symlink can avoid duplicating a large film; keep the containing directory writable for the generated proxy. Reserve space for the proxy, matching cache, clips, and task bundles as well as the original inputs.

The `--check` call uses your model on small generated inputs and may incur usage; it checks file reading, image viewing, and JSON writing before the long job. Inspect its trace for actual image-tool use. For Codex or OpenCode, change both backend and model, and omit or replace the Claude-specific effort value as described in [Agent setup](agent_setup.md). When resuming, source the saved profile instead of recreating it.

`MyFilm` becomes the export slug `myfilm`. Start with the inexpensive local check:

```sh
python data_construction/pipeline.py --film MyFilm --only probe
cat "$VIDEO_EDITING_DATA_ROOT/exports/myfilm/probe.json"
```

Verify the chosen filenames, durations, frame rates, and crop. Use a matching edition of the film and trailer. Poor matching later can indicate different cuts, crops, or source footage; it is not a reason to invent matches.

## 3. Run media preprocessing

Create a new volume for this run. Replace the example name if it already exists in your account.

```sh
export VIDEO_EDITING_MODAL_VOLUME=my-video-editing-demo
modal volume create "$VIDEO_EDITING_MODAL_VOLUME"
python data_construction/pipeline.py --film MyFilm
```

This command uploads a video proxy and the trailer and runs paid Modal jobs. The first run also obtains model weights. It then downloads reference clips, aligns them to the source film, measures audio and picture boundaries, and writes a local report. Use the same volume and private root when resuming the same run. Use a new root and volume when changing media: completion flags and caches are not an input-versioning system.

The matcher uses corresponding frame pairs. Color-only fallback candidates are marked `REVIEW(colour)` by default because the old acceptance threshold is not calibrated for the corrected score; usable unmatched footage can remain trailer-only. Inspect the match/refinement CSVs, especially disconnected near peaks and candidates needing review. New source in-points and RC tolerances carry explicit alignment sampling units. See [alignment behavior](data_construction.md#alignment-quality-limitation). Existing cached artifacts are not migrated automatically; use a new private root when comparing the corrected pipeline with older outputs.

macOS uses VideoToolbox for the proxy by default; other platforms use `libx264`. Set `VIDEO_EDITING_PROXY_ENCODER=libx264` explicitly to use software encoding on macOS. Cropping, output width, and the full-duration check are shared.

Each stage prints its start and a private log path under `exports/myfilm/logs/`. A long encode or model download can take time. On failure, read that stage's full log, fix the cause, and rerun the same command; successful stages are skipped. `--repair` only reconstructs flags from existing files and does not repair missing work.

## 4. Prepare annotations and dialogue

Run the inspection helpers against this single-film root:

```sh
# Optional 30-minute limit for each shot/S4 annotation or translation invocation.
export VIDEO_EDITING_ANNOTATION_TIMEOUT_S=1800
python data_construction/run_script_queue.py --film myfilm --prep-only
python data_construction/shot_luma.py --all
python data_construction/verify_matches.py --all
python data_construction/card_text.py --film myfilm
python data_construction/run_script_queue.py --film myfilm
python data_construction/translate_script.py --film myfilm
python data_construction/run_s4_queue.py --film myfilm
```

The two annotation queues and translation invoke your selected model. The shot queue checks descriptions and beat/chunk coverage; the S4 queue checks cut annotations. Claude S4 also checks successful image access as described below. These checks do not establish annotation accuracy. Uniformly sampling a fast-cut trailer can miss short shots. Translation supplies the English fields consumed by task builders, even when the first annotation already uses English. Logs and generated annotations stay under the private root. Inspect visual descriptions, chunk boundaries, subtitle cards, and the generated report before treating them as reviewed data.

With Claude, the shot queue uses two phases. Visual annotation runs in separate calls of at most 12 shots, each with its own labeled contact sheet and inputs. Shots shorter than 0.75 seconds require a separate view of their larger individual frame. Python copies dialogue directly from the supplied inputs and validates each completed batch before adding it to the saved shot log. A final independent call generates the logline, beats, and chunks; Python combines these with the unchanged shot descriptions and validates the full result. Private drafts and images live in `edit/task_<slug>/work/script_visuals/`; the run log reports saved description counts. These checks establish the data contract, not annotation accuracy. A still image does not establish motion, so ambiguous shots need further inspection.

Claude S4 uses separate calls of at most 12 boundaries. Private strips under `edit/task_<slug>/work/cut_visuals/` show consecutive frames near each supplied reference boundary. The queue accepts a batch only after every strip has a matched successful image-read result and the output passes validation. This proves image access, not that the model interpreted the pictures correctly. `nominal BEFORE/AFTER` labels refer to the supplied numerical boundary: the first nominal after frame may still show the preceding scene. Gray `NO FRAME` tiles are placeholders, not video black frames. Inspect the source video for ambiguous motion or transition timing; `no_change` still requires measured continuity support.

Allow time for full-trailer annotation; the small CLI setup check does not predict its runtime. Queues and translation save provider events in `edit/runs/<slug>/`. Both Claude queues strictly recheck the saved ordered prefix against the current inputs before resuming; invalid prefixes are rejected, and unfinished or rejected batches must be repeated. To archive the previous draft and request fresh inspection, add `--redo` to the relevant command:

```sh
python data_construction/run_script_queue.py --film myfilm --redo
python data_construction/run_s4_queue.py --film myfilm --redo
```

Drafts never count as completed annotations. The optional total limit above applies separately to each batch, the script structure call, and translation; active status events do not reset it. Failed translation restores the original annotation. On timeout, inspect the logs and deliberately retry, adjust the limit, or change backend. Unset this variable to disable it. Codex and OpenCode use their existing single-session annotation workflow.

FC also needs dialogue candidates extracted from the **source film**. The main preprocessing command only transcribes the trailer; it does not perform this separate sequence:

```sh
python data_construction/modal/modal_film_transcribe.py --extract --film myfilm
python data_construction/modal/modal_film_transcribe.py --upload --film myfilm
modal run data_construction/modal/modal_film_transcribe.py::main --only myfilm
python data_construction/modal/modal_film_transcribe.py --fetch --film myfilm
python data_construction/locate_lines.py --film myfilm
python data_construction/cut_takes.py --film myfilm
```

Full-film Whisper runs on Modal. The final take check uses `small.en` locally and may download its weights. Inspect `exports/myfilm/takes.json` and listen to candidate takes: a successful command or a transcript match does not establish dialogue accuracy. Missing `takes.json` is an incomplete setup, not evidence that a task needs no dialogue.

## 5. Assemble and inspect candidate tasks

```sh
python data_construction/build_s1_tasks.py --film myfilm
python data_construction/build_s23_tasks.py --film myfilm
python data_construction/build_s4_tasks.py --film myfilm
python data_construction/build_s4_editorial.py --film myfilm
```

Outputs appear under `training_task/myfilm/`: `s_1` is CS, `s_2` is NS, `s_3` is RC, and `s_4e` is FC. `s_4` is an intermediate used to construct FC. Builders filter unsuitable chunks, so counts need not match across task types. Check that the intended stages contain tasks; exit status alone does not establish nonempty output.

Builders print the current chunk and total before processing it. This is progress through candidate chunks, not a count of accepted tasks; the final summary reports how many tasks were built. A [one-film check](release_status.md#one-film-construction-check) exercised this workflow, with the scope and remaining limitations recorded separately.

For each task, inspect `task.json`, `task_prompt.md`, `initial_state/manifest.json`, verifier metadata, and the referenced media. Every supplied clip, handle, dialogue take, and music file must exist and be readable. Cards explicitly marked for the agent to build need not have a supplied clip. Confirm that FC contains usable dialogue candidates where expected, rather than accepting an empty pool by accident.

These are candidate tasks. Their existence and valid JSON do not establish solvability, fair grading, or accurate descriptions. Continue with the separate scoring preparation and quality audit below.

## 6. Attach scoring and review task quality

Create an explicit task list for this film, excluding the historical `s_4` intermediate. The example keeps the entire film in a single demo split; do not split chunks from the same film between training and validation.

```sh
python - <<'PYCODE'
import json
import os
from pathlib import Path
root = Path(os.environ["VIDEO_EDITING_DATA_ROOT"]).expanduser().resolve()
film = root / "training_task/myfilm"
tasks = [str(p.parent) for stage in ("s_1", "s_2", "s_3", "s_4e")
         for p in sorted((film / stage).glob("c*/task.json"))]
if not tasks:
    raise SystemExit("No constructed tasks found")
(root / "candidate_tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
(root / "film_splits.json").write_text(json.dumps({"myfilm": "train"}) + "\n")
print(f"Listed {len(tasks)} candidate tasks")
PYCODE

python evaluation/task_pipeline.py \
  --tasks "$VIDEO_EDITING_DATA_ROOT/candidate_tasks.json" --prepare-only

# Bind complete inputs and the current scoring/tool contracts without model calls.
python data_construction/task_quality.py \
  --tasks "$VIDEO_EDITING_DATA_ROOT/candidate_tasks.json" \
  --film-splits "$VIDEO_EDITING_DATA_ROOT/film_splits.json" \
  --output "$VIDEO_EDITING_DATA_ROOT/quality-inputs-001" --prepare-only
```

Scoring preparation installs the released descriptors and preserves any existing CS/NS task-specific grader. Earlier generated descriptors are backed up privately. Complete this step before auditing so the review binds the scoring rules that will actually be used. Inspect the saved `input.json` files before starting paid calls.

Quality review uses the Gemini API directly, separately from the annotation CLI. Set `GOOGLE_AI_STUDIO_API_KEY` in the process environment. The default audit model is the recorded `gemini-3.8-flash`; set `VIDEO_EDITING_QUALITY_MODEL` or pass `--model` to choose a model available to your account. The selected model must support video and audio. This command submits one full-task audit per candidate and may incur substantial usage for a whole film:

```sh
python data_construction/task_quality.py \
  --tasks "$VIDEO_EDITING_DATA_ROOT/candidate_tasks.json" \
  --film-splits "$VIDEO_EDITING_DATA_ROOT/film_splits.json" \
  --output "$VIDEO_EDITING_DATA_ROOT/quality-review-001"
```

To try one task first, replace `--tasks` with `--task '/absolute/path/to/a/task'`; use a new output directory for each intentional run. The utility refuses to reuse an existing output directory and does not retry model requests automatically. `--timeout` is a limit per provider HTTP operation, not the entire film review. Changing the annotation CLI's timeout does not change this API setting.

Each task receives a complete public description, full media timelines, tool/runtime contract, and clearly marked private grading evidence. Larger video sets use a mapped full-timeline reel and separate original-speed embedded audio, rather than omitting candidate clips. Video sampling remains approximate. The [quality review guide](data_construction.md#quality-review-and-sampling) describes transport and evidence limitations.

Read `summary.json` and per-task `terminal.json`/`result.json`. Only complete `keep_candidate` decisions enter `accepted_tasks.json`. A concrete major defect can produce `filter_recommended`; `repair_needed` preserves a repair plan without applying it; missing evidence or API/schema failures stay outside the accepted list and are not called bad tasks. Original task files remain in place. Make repairs in separate copies and audit their updated inputs again. A model-reported pass is not human ground truth or proof of annotation accuracy.

No fixed retained count is required for a one-film test. If a larger dataset needs size balancing, `accepted_index.json` can feed the recovered [deterministic sampler](data_construction.md#quality-review-and-sampling) with explicit split/type targets and a seed. The historical 797 count is not a quota for new films, and size exclusions are not quality rejections.

## 7. Score completed task submissions

Quality review evaluates task construction. Outcome scoring evaluates an agent's completed edit. First run your agent or provide the task's requested deliverables in its `output/` directory: `selection.json` for CS, `order.json` for NS, `cuts.json` for RC, and `edit.json` plus the rendered `preview.mp4` for FC. Follow each task's public manifest and prompt.

For a nonempty accepted-task list:

```sh
# Validate submitted files without judge calls.
python evaluation/task_pipeline.py \
  --tasks "$VIDEO_EDITING_DATA_ROOT/quality-review-001/accepted_tasks.json" \
  --preflight-only \
  --summary "$VIDEO_EDITING_DATA_ROOT/submission-preflight.json"

# Score valid submissions. RC/FC make separate paid Gemini judge calls.
python evaluation/task_pipeline.py \
  --tasks "$VIDEO_EDITING_DATA_ROOT/quality-review-001/accepted_tasks.json" \
  --summary "$VIDEO_EDITING_DATA_ROOT/submission-scores.json"
```

Do not change a task or its scoring sources between quality review and scoring without reviewing the change. Missing or invalid submissions and infrastructure failures have distinct result states; inspect the summary rather than assuming every task has a valid score. See [evaluation](evaluation.md) for scoring configuration, aggregation, and failure behavior. These commands exercise the released integration and do not establish equivalence to every historical experiment.

## Finishing a run

Keep media, generated JSON, logs, caches, and task directories outside Git. Modal run apps stop when their entrypoints finish, but the volume persists. Download anything you need before deleting your own test volume in Modal. Do not remove a volume shared with other work.
