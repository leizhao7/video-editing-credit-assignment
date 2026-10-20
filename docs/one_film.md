# Generate tasks from one film

Start with one full film and its matching trailer in a new private directory. This walkthrough builds candidate CS, NS, RC, and FC tasks. Final quality review, dataset selection, and the paper's complete scoring setup are separate steps.

You need Python 3.12, FFmpeg/FFprobe, Tesseract with English language data, a Modal account, and one authenticated annotation CLI. Use English media with English as the default audio track: transcription requests English, and several clip-extraction steps use the default track. You do not need a local GPU or DaVinci Resolve: local commands prepare and assemble media; Modal runs TransNetV2, Whisper, and Demucs.

## 1. Install in a separate environment

From the repository root:

```sh
python3.12 -m venv "$HOME/.venvs/video-editing"
source "$HOME/.venvs/video-editing/bin/activate"
python -m pip install -r data_construction/requirements.txt \
  -r data_construction/requirements-modal.txt

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

macOS uses VideoToolbox for the proxy by default; other platforms use `libx264`. Set `VIDEO_EDITING_PROXY_ENCODER=libx264` explicitly to use software encoding on macOS. Cropping, output width, and the full-duration check are shared.

Each stage prints its start and a private log path under `exports/myfilm/logs/`. A long encode or model download can take time. On failure, read that stage's full log, fix the cause, and rerun the same command; successful stages are skipped. `--repair` only reconstructs flags from existing files and does not repair missing work.

## 4. Prepare annotations and dialogue

Run the inspection helpers against this single-film root:

```sh
python data_construction/run_script_queue.py --film myfilm --prep-only
python data_construction/shot_luma.py --all
python data_construction/verify_matches.py --all
python data_construction/card_text.py --film myfilm
python data_construction/run_script_queue.py --film myfilm
python data_construction/translate_script.py --film myfilm
python data_construction/run_s4_queue.py --film myfilm
```

The two annotation queues and translation invoke your selected model. The shot queue checks descriptions and beat/chunk coverage; the S4 queue checks cut annotations. These are structural checks: they do not prove that the model viewed every shot or described it correctly. Uniformly sampling a fast-cut trailer can miss short shots. Translation supplies the English fields consumed by task builders, even when the first annotation already uses English. Logs and generated annotations stay under the private root. Inspect visual descriptions, chunk boundaries, subtitle cards, and the generated report before treating them as reviewed data.

Allow time for full-trailer annotation; the small CLI setup check does not predict its runtime. Queues save provider events in `edit/runs/<slug>/` and validate a film-level result at completion. They do not provide per-shot annotation checkpoints. Inspect the current run's logs before restarting; a retry can repeat substantial model work.

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

For each task, inspect `task.json`, `task_prompt.md`, `initial_state/manifest.json`, verifier metadata, and the referenced media. Every supplied clip, handle, dialogue take, and music file must exist and be readable. Cards explicitly marked for the agent to build need not have a supplied clip. Confirm that FC contains usable dialogue candidates where expected, rather than accepting an empty pool by accident.

These are candidate tasks. Review the [known temporal-alignment limitation](data_construction.md#alignment-quality-limitation) before accepting source matches. CS's final quality policy requires genuine review evidence; NS/RC's historical automatic rubric writers are not bundled; FC supplies a rubric seed. See [scoring and reconstruction limits](data_construction.md#scoring-and-reconstruction-limits) before training or reporting benchmark results.

## Finishing a run

Keep media, generated JSON, logs, caches, and task directories outside Git. Modal run apps stop when their entrypoints finish, but the volume persists. Download anything you need before deleting your own test volume in Modal. Do not remove a volume shared with other work.
