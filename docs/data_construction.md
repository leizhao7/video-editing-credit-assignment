# Data construction release draft

This guide describes generic construction utilities and prompt templates. The dataset, source films, reference edits, clips, task instances, answer JSON, transcripts, film lists, audit decisions, and model outputs are not distributed. Supply your own private media and keep every generated artifact outside the code tree.

The code preserves the matching, trimming, selection, and task-assembly calculations while separating code paths from private data paths. Corpus-specific examples and measurements have been removed from comments and documentation. Three generated descriptions/warnings were rewritten to remove historical corpus statistics. Release adaptations also validate configuration and paths, propagate subprocess failures, and clean up temporary resources; normal processing parameters and successful output schemas are preserved. It has not been validated end to end on new media or cloud infrastructure. Historical environment versions and final dataset selection are not reconstructed. Data-construction provenance maps and release-audit results are kept outside the public code package.

## Setup and private layout

Use Python 3.10 or newer and install `data_construction/requirements.txt` in your own environment. These are direct import dependencies, not the historical experiment lockfile. Install `ffmpeg` and `ffprobe` separately; `card_text.py` also needs Tesseract with English language data. The optional `cut_takes.py` transcript check loads faster-whisper `small.en` on explicit invocation and may require the model weights to be available in its normal cache. The inherited proxy stage uses macOS VideoToolbox; an equivalent Linux encoder configuration has not been verified. Explicit cloud stages use the bundled adapters under `data_construction/modal/`, a configured Modal CLI, and an explicitly supplied user-owned volume. Install `data_construction/requirements-modal.txt` for the local CLI dependency; the adapters declare their GPU image dependencies separately. Explicit annotation stages use a separately installed and authenticated Claude-compatible CLI accepting the flags in the queue source; set the model yourself.

All construction artifacts resolve under `VIDEO_EDITING_DATA_ROOT`, defaulting to `data/private` under the current working directory. Code and bundled resources resolve beside their Python files. Set this variable before invoking any script; the scripts do not read an `.env` file automatically. Configuration names are listed in `data_construction/environment.example`; no credentials are provided or embedded.

```sh
export VIDEO_EDITING_DATA_ROOT="$PWD/data/private"
python data_construction/pipeline.py --help
python data_construction/run_script_queue.py --help
```

The tree below describes private artifacts created or consumed by the code; it is not a released dataset or a synthetic fixture.

```text
data/private/
  sources/<input-directory>/         # user-supplied source film and reference edit
  exports/<slug>/                    # probe/state, shot manifests, matches, transcript, stems
  edit/task_<slug>/
    s3_units.json                    # locked measured units
    reference/golden.mp4             # private reference link
    stages/                         # input annotations and measured cut metadata
    cards/                          # private extracted card stills
  edit/runs/                        # private prompts, provider logs, reports
  training_task/<slug>/s_1/                 # CS task instances
  training_task/<slug>/s_2/                 # NS task instances
  training_task/<slug>/s_3/                 # RC task instances
  training_task/<slug>/s_4/                 # historical intermediate, not paper FC
  training_task/<slug>/s_4e/                # FC editorial task instances
  .clipcache/                       # private clip cache
```

`probe.py` identifies source/reference files from supplied media and writes `probe.json`; inspect its choices before continuing. The slug is derived from the input directory. The task stages use the inherited `s_1`, `s_2`, `s_3`, and `s_4e` names for CS, NS, RC, and FC respectively.

## Pipeline and explicit entry points

All commands below are opt-in commands. Replace `<slug>` with your own input slug. Imports and `--help` do not start preprocessing, annotation, training, or cloud work. Queue scripts require a target (`--film`, `--films` where available, or explicit `--all`).

Whitespace-only optional environment settings use their documented defaults. Worker counts and `PROXY_LANES` must be positive; stage names and target selection are validated before processing. Ambiguous film selectors, mismatched probe slugs and colliding export names are rejected. `--gc` retains its separate cache-cleanup behavior.

| Step | Entry point | Inputs and outputs |
| --- | --- | --- |
| Media preparation | `pipeline.py` | Orchestrates probe → proxy → upload → shots → fetch → matching → cards → audio → units → cuts → measured audio → report. Tracks stages in private `exports/<slug>/state.json`. |
| Source/reference matching | `p2_match.py`, `p2_refine.py`, `verify_matches.py` | Shot manifests and private media → matching CSV and verification evidence. |
| Card/unit preparation | `classify_cards.py`, `build_s3.py`, `build_cutlist.py`, `shot_luma.py` | Matches and clips → classification, locked units, cut list, luminance evidence. |
| Measured audio/continuity | `audio_cuts.py`, `picture_cont.py` | Timed transcript, music stem, and media → per-boundary measurements. |
| Shot/beat/chunk annotation | `run_script_queue.py` | Measured facts and reference → `stages/script_inputs.json` and `stages/script.json`; validates coverage and required fields. |
| Optional language completion | `translate_script.py` | Adds English text fields without intentionally changing existing annotation fields. Invokes an external model explicitly. |
| CS assembly | `build_s1_tasks.py` | Annotations and private quality evidence → CS candidate/target materials and verifier metadata. |
| NS/RC assembly | `build_s23_tasks.py` | Same measured/annotated inputs → NS and RC task directories. |
| FC annotation | `run_s4_queue.py` | Locked units/cuts plus measured audio/continuity → `stages/s4_cuts.json`; also prepares continuity and a report. |
| FC intermediate | `build_s4_tasks.py` | Uses `s4_cuts.json` and the existing per-chunk tasks to create historical `s_4` inputs. |
| FC editorial assembly | `build_s4_editorial.py` | Transforms the intermediate to `s_4e` editing tasks and private `_rubric_seed.json`. Final scoring assembly is not included here. |
| Inspection utilities | `build_report.py`, `card_text.py`, `locate_lines.py`, `cut_takes.py` | Local report/card text and optional transcript-aligned take preparation. Inspect each command's inputs before invoking it. |

A local-only starting step is:

```sh
python data_construction/pipeline.py --film '<slug>' --only probe
```

Run subsequent local stages only after their prerequisites exist. `pipeline.py --repair` reconstructs stage flags from existing artifacts; it does not generate missing artifacts. The complete pipeline still requires cloud preprocessing. Four generic adapters are now included under `data_construction/modal/`; no historical deployment ID, secret, media, model weights or volume contents are bundled. The pipeline defaults to this directory, with `--modal-dir` or `VIDEO_EDITING_MODAL_DIR` available for an explicit override.

| Bundled adapter | Preserved processing contract |
| --- | --- |
| `modal_tn_generic.py::detect` | TransNetV2 shot detection and clip export; the pipeline uses thresholds 0.30 for source and 0.20 for reference. |
| `modal_transcribe.py::run` | Reference transcription with Whisper `large-v3`, English, beam size 5, VAD and word timestamps. |
| `modal_stems.py::split` | `htdemucs` separation from 44.1 kHz stereo, then 16 kHz mono stems; music sums drums, bass and other with `normalize=0`. |
| `modal_film_transcribe.py::main` | Source-film transcription after explicit local extraction/upload, supporting `locate_lines.py` and `cut_takes.py`. |

Set `VIDEO_EDITING_MODAL_VOLUME` to an existing volume in your own account before cloud commands; the adapters do not create a volume automatically. App names and GPU class are configurable through `environment.example`, with neutral app names and the historical A10G default. Imports only define image/function specifications and lazy volume references; they do not look up remote state, allocate a GPU, download weights or process media. Missing volume configuration is rejected before processing. Temporary audio and stem files use a separate directory for each call.

The first three adapters retain the pipeline's function names and CLI arguments. Their output names include `golden_clips/manifest.csv`, `source_clips/manifest.csv`, `golden_transcript.json`, and `golden_stems/music.wav` on the volume; the fetch stage places these under the private exports root, including `source_manifest.csv` and `stems/music.wav`. A transcript contains `segments` with `start`, `end`, `text`, and `words`; each word uses `s`, `e`, `w`, and probability `p`. Source-film transcription emits the same word schema as `source_transcript.json`.

The optional source-film path is separate from the main pipeline:

```sh
# These commands create/process private artifacts and explicitly invoke cloud work.
export VIDEO_EDITING_MODAL_VOLUME='<your-existing-volume>'
python data_construction/modal/modal_film_transcribe.py --extract
python data_construction/modal/modal_film_transcribe.py --upload
modal run data_construction/modal/modal_film_transcribe.py::main --only '<slug>'
python data_construction/modal/modal_film_transcribe.py --fetch
python data_construction/locate_lines.py --film '<slug>'
python data_construction/cut_takes.py --film '<slug>'
```

The local extraction/upload/fetch helpers enumerate private inputs; the remote `--only` filter limits the transcription call. Inspect the private root first. The historical extraction helpers reuse existing outputs based on file-size checks, so use separate roots or remove stale generated artifacts when inputs or settings change.

The adapters reject failed decoders, empty video and incomplete frames, validate shot exports and stem outputs, and publish JSON metadata atomically. Single-shot export avoids passing an empty segment list to FFmpeg. Empty transcripts remain valid. Local action conflicts and failed batch items produce a nonzero exit status. These branches are tested with temporary synthetic inputs and mocked subprocesses, models and Modal definitions:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider data_construction/modal/tests
```

These tests require `pytest` and `numpy`; they do not load a real Modal SDK or model, run FFmpeg, or access cloud services. They are not GPU execution or end-to-end processing checks on new media.

The image recipes currently use unpinned TransNetV2/PyTorch, faster-whisper/CTranslate2 and Demucs dependencies; the cuDNN constraint specifies only its major version. Exact package/model revisions, weight hashes, FFmpeg codec build and CUDA compatibility still need qualification. Third-party code/binary notices and model-weight redistribution terms must be collected before bundling those dependencies; this source package includes none of their model weights or generated outputs. The new adapters preserve preprocessing settings and output schemas but are not a verified historical environment lockfile.

After preprocessing, a typical explicit sequence is:

```sh
python data_construction/shot_luma.py --all
python data_construction/verify_matches.py --all
python data_construction/run_script_queue.py --film '<slug>' --prep-only
# Set VIDEO_EDITING_ANNOTATION_MODEL before these annotation calls:
python data_construction/run_script_queue.py --film '<slug>'
python data_construction/translate_script.py --film '<slug>'
python data_construction/build_s1_tasks.py --film '<slug>'
python data_construction/build_s23_tasks.py --film '<slug>'
python data_construction/run_s4_queue.py --film '<slug>'
python data_construction/build_s4_tasks.py --film '<slug>'
python data_construction/build_s4_editorial.py --film '<slug>'
```

`shot_luma.py` and `verify_matches.py` inspect all supplied exports under the private root; unlike per-film builders they support `--all`, not `--film`. Some CS protections require additional private review evidence described below. This sequence documents the dependency order, not a claim that an arbitrary input passes every quality gate. The inherited scripts can replace generated files; keep experiments in separate private roots.

`build_s3.py` produces unit metadata/media without an editor connection. Exporting a Resolve project is separately opt-in with `--export-resolve` and `VIDEO_EDITING_RESOLVE_MCP_ROOT`; that external integration is not bundled. The two queue prompts under `data_construction/prompts/` are release rewrites. Their field contracts were checked against the inherited validators. They intentionally omit real annotations and corpus-derived vocabulary. The original heuristics (including chunk size and S4 vocabulary-diversity checks) remain, so a passing JSON file still needs human review.

Queue failures now propagate to the command's exit status. Annotation stderr goes to a separate private log to avoid blocking its stdout stream; unfinished attempts cannot inherit a previous success flag. The translation step checks all original fields and entries and restores the original file if the process fails or changes those fields. Local regressions cover these cases, redo handling, paths containing spaces, subprocess cleanup and download failures:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s data_construction/tests
```

These dependency-free tests use temporary synthetic metadata and local child processes, without invoking an annotation provider or processing media.

## Scoring and reconstruction limits

The current paper scoring code is maintained under `evaluation/`. The `grade_s1.py`, `grade_s2.py`, and `grade_s3.py` files in this directory are historical construction dependencies; they are not a declaration that every generated task has the paper's final verifier attached. In particular, the old automatic S2/S3 rubric writers are omitted and disabled by default. The optional `VIDEO_EDITING_LEGACY_RUBRIC_WRITER_S2`/`S3` hooks only support deliberate legacy reconstruction with a user-provided writer; they must not be used to infer current paper scoring. The FC builder writes a seed for later rubric assembly. End-to-end grader assembly and the reported aggregation weights remain to be verified; no weights are asserted by this release guide. Generated tasks are **not yet certified for the paper's official scoring**.

The local source archive is insufficient to regenerate the final reported dataset:

- Private source/reference media, manifests, annotations, answer files, film lists, and manual overrides are intentionally excluded.
- CS quality helpers consume user-supplied evidence under `review_evidence/s1/` beneath the private root. The normalized layout uses `inputs/` for bound review inputs, `model_results.json` and `blind_model_results.json` for review results, and `classification/` for generated classification checks. Only readers, validators, and policies are included; no per-task evidence or human decisions are provided. Missing evidence must not be replaced with fabricated approval.
- The human/model audit that reduced candidates to the reviewed 925 tasks is not reproduced by a complete public pipeline here.
- A historical record describes the final 797 selection as stratified by split/stage/film with stable SHA256 ordering. Its reusable sampler and complete inputs were not found; the exact 797 selection is not implemented in this release.
- The old split script and old corpus summaries describe different task coverage and counts; they are intentionally excluded rather than presented as the final split procedure.
- Historical annotation prompts are replaced with generic templates. Annotation outputs and task-build fingerprints will not be byte-identical to the paper's runs. The build fingerprint is computed from the included construction code and supplied private measurement files; it is a change detector, not a proof of experimental equivalence.
- Dependency versions, provider/model revision, all audit decisions, and the final scoring/aggregation assembly need further provenance before an end-to-end reproduction claim is appropriate.

Running the pipeline on new inputs produces private data and provider logs; those outputs are outside the public release scope. The public directory contains only implementation code, generic prompt templates, resources, configuration examples, and usage documentation.
