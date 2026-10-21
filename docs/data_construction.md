# Video-editing task generation pipeline

The pipeline is designed to scale task construction across films and their released trailers: align source and reference footage, extract narrative chunks, generate editorial briefs, and assemble four types of editing tasks. The underlying video and audio come from real media; AI models generate the textual task descriptions. This guide describes the construction utilities and prompt templates.

The dataset, source films, reference edits, clips, task instances, answer JSON, transcripts, film lists, audit decisions, and model outputs are not distributed. Supply your own private media and keep every generated artifact outside the code tree.

The release separates code paths from private data paths. Temporal matching now uses corresponding frame pairs and records its sampling units; uncalibrated color-only matches require review. These corrections are described below and do not retroactively change the paper's recorded data. Corpus-specific examples and measurements have been removed from comments and documentation. Three generated descriptions/warnings were rewritten to remove historical corpus statistics. Release adaptations also validate configuration and paths, propagate subprocess failures, and clean up temporary resources; other inherited processing parameters and successful annotation schemas are preserved. One private run from raw media exercised macOS orchestration, Modal preprocessing, Codex annotation, and all four candidate-task builders; see [verification scope](release_status.md#one-film-construction-check). Historical environment versions and the complete task-by-task repair history are not reconstructed. The final 925-to-797 size sampler has been recovered and checked against the private release index. Data-construction provenance maps and release-audit results are kept outside the public code package.

For a first run, follow the [one-film walkthrough](one_film.md), including the separate source-dialogue preparation for FC.

## Setup and private layout

Use Python 3.10 or newer and install `data_construction/requirements.txt` in your own environment. These are direct import dependencies, not the historical experiment lockfile. Install `ffmpeg` and `ffprobe` separately; `card_text.py` also needs Tesseract with English language data. The optional `cut_takes.py` transcript check loads faster-whisper `small.en` on explicit invocation and may require the model weights to be available in its normal cache. The proxy stage defaults to VideoToolbox on macOS and `libx264` elsewhere. `VIDEO_EDITING_PROXY_ENCODER=libx264` also selects software encoding on macOS. Both preserve cropping, output width, bitrate, and the full-duration check; the Linux platform branch is covered by tests, while full-film runtime qualification is recorded separately. Explicit cloud stages use the bundled adapters under `data_construction/modal/`, a configured Modal CLI, and an explicitly supplied user-owned volume. Install `data_construction/requirements-modal.txt` for the local CLI dependency; the adapters declare their GPU image dependencies separately. Explicit annotation stages use a separately installed and authenticated Claude, Codex, or OpenCode 1.x CLI, configured as described below.

All construction artifacts resolve under `VIDEO_EDITING_DATA_ROOT`, defaulting to `data/private` under the current working directory. Code and bundled resources resolve beside their Python files. Set this variable before invoking any script; the scripts do not read an `.env` file automatically. Configuration names are listed in `data_construction/environment.example`; no credentials are provided or embedded.

```sh
export VIDEO_EDITING_DATA_ROOT="$HOME/video-editing-data"
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
    work/                           # visual aids, validated prefixes, translation patches
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

Whitespace-only optional text settings use their documented defaults. Worker counts and `PROXY_LANES` must be positive; stage names and target selection are validated before processing. Ambiguous film selectors, mismatched probe slugs and colliding export names are rejected. `--gc` retains its separate cache-cleanup behavior.

| Step | Entry point | Inputs and outputs |
| --- | --- | --- |
| Media preparation | `pipeline.py` | Orchestrates probe → proxy → upload → shots → fetch → matching → cards → audio → units → cuts → measured audio → report. Tracks stages in private `exports/<slug>/state.json`. |
| Source/reference matching | `p2_match.py`, `p2_refine.py`, `verify_matches.py` | Shot manifests and private media → matching CSV and verification evidence. |
| Card/unit preparation | `classify_cards.py`, `build_s3.py`, `build_cutlist.py`, `shot_luma.py` | Matches and clips → classification, locked units, cut list, luminance evidence. |
| Measured audio/continuity | `audio_cuts.py`, `picture_cont.py` | Timed transcript, music stem, and media → per-boundary measurements. |
| Shot/beat/chunk annotation | `run_script_queue.py` | Measured facts and reference → `stages/script_inputs.json` and `stages/script.json`; validates coverage and required fields. |
| Optional language completion | `translate_script.py` | Adds English text fields without intentionally changing existing annotation fields. Invokes an external model explicitly. |
| CS assembly | `build_s1_tasks.py` | Annotations and measured inputs → CS candidate/target materials and verifier metadata. Semantic quality requires separate review evidence. |
| NS/RC assembly | `build_s23_tasks.py` | Same measured/annotated inputs → NS and RC task directories. |
| FC annotation | `run_s4_queue.py` | Locked units/cuts plus measured audio/continuity → `stages/s4_cuts.json`; also prepares continuity and a report. |
| FC intermediate | `build_s4_tasks.py` | Uses `s4_cuts.json` and the existing per-chunk tasks to create historical `s_4` inputs. |
| FC editorial assembly | `build_s4_editorial.py` | Transforms the intermediate to `s_4e` editing tasks and private `_rubric_seed.json`. Attach the released scoring configuration with `evaluation/task_pipeline.py --prepare-only`. |
| Semantic quality review | `task_quality.py` | Complete task/media and fallible private grading references → kept, filtered, repair-needed, or held records; emits a task list for scoring. |
| Dataset size sampling | `quality_selection.py` | Reviewed candidates and explicit split/type targets → deterministic film-stratified subset; does not make quality judgments. |
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

Source-film transcription is separate from the main pipeline. Complete this path before assembling FC dialogue candidates:

```sh
# These commands create/process private artifacts and explicitly invoke cloud work.
export VIDEO_EDITING_MODAL_VOLUME='<your-existing-volume>'
python data_construction/modal/modal_film_transcribe.py --extract --film '<slug>'
python data_construction/modal/modal_film_transcribe.py --upload --film '<slug>'
modal run data_construction/modal/modal_film_transcribe.py::main --only '<slug>'
python data_construction/modal/modal_film_transcribe.py --fetch --film '<slug>'
python data_construction/locate_lines.py --film '<slug>'
python data_construction/cut_takes.py --film '<slug>'
```

Local extraction/upload/fetch require an exact `--film` slug or explicit `--all`; remote `--only` also matches an exact slug. Source audio is selected from the file recorded by `probe.py`, including when that input is a symlink. The historical extraction helpers reuse existing outputs based on file-size checks, so use separate roots or remove stale generated artifacts when inputs or settings change.

The adapters reject failed decoders, empty video and incomplete frames, validate shot exports and stem outputs, and publish JSON metadata atomically. Single-shot export avoids passing an empty segment list to FFmpeg. Empty transcripts remain valid. Local action conflicts and failed batch items produce a nonzero exit status. These branches are tested with temporary synthetic inputs and mocked subprocesses, models and Modal definitions:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider data_construction/modal/tests
```

These tests require `pytest` and `numpy`; they do not load a real Modal SDK or model, run FFmpeg, or access cloud services. They are not GPU execution or end-to-end processing checks on new media.

The transcription recipes pin faster-whisper to 1.2.1 and constrain PyAV to `>=11,<19` for compatibility. TransNetV2/PyTorch, CTranslate2 and Demucs are not fully pinned; the cuDNN constraint specifies only its major version. Exact package/model revisions, weight hashes, FFmpeg codec build and CUDA compatibility still need qualification. Third-party code/binary notices and model-weight redistribution terms must be collected before bundling those dependencies; this source package includes none of their model weights or generated outputs. The new adapters preserve preprocessing settings and output schemas but are not a verified historical environment lockfile.

## Annotation CLI configuration

For first-time configuration, follow [Agent setup](agent_setup.md) or run `python3 data_construction/setup_agent.py`. It generates a backend-specific profile outside the repository and prints the command to activate it. The environment variables below remain available for manual setup.

`run_script_queue.py`, `run_s4_queue.py`, and `translate_script.py` share these settings. Authentication and provider credentials stay in the selected CLI's native configuration.

| Variable | Meaning |
| --- | --- |
| `VIDEO_EDITING_ANNOTATION_BACKEND` | `claude` (default), `codex`, or `opencode`. OpenCode support targets the 1.x CLI; 2.x compatibility is not established. |
| `VIDEO_EDITING_ANNOTATION_CLI` | Optional executable path or name; blank uses the backend name. This is one executable argument, not a shell command containing flags. |
| `VIDEO_EDITING_ANNOTATION_MODEL` | Required for annotation. Use the selected CLI's model identifier; OpenCode requires `provider/model`. |
| `VIDEO_EDITING_ANNOTATION_EFFORT` | Optional backend-native effort/variant: Claude `--effort`, Codex `model_reasoning_effort`, or OpenCode `--variant`. When unset, Claude keeps `max` for both queues and `low` for translation; Codex and OpenCode keep their native defaults. |
| `VIDEO_EDITING_ANNOTATION_TIMEOUT_S` | Optional total seconds per invocation in the shot and S4 annotation queues, translation, and CLI smoke checks. Must be finite and positive; leave unset to disable (blank is invalid). Active model output does not reset this deadline. Timeout terminates that invocation's process group and reports failure. |
| `VIDEO_EDITING_MCP_CONFIG` | Optional user-owned MCP config for Claude S4 only. S4 rejects this setting with another backend; unset it and use that CLI's native MCP configuration. |

The Claude adapter preserves the existing tool allowlist, `Agent,Task` denylist, `acceptEdits` permission mode, task directory, and stream-JSON options. Codex uses `exec --json --sandbox workspace-write --skip-git-repo-check -c approval_policy="never" --disable multi_agent --cd <task-directory>`. OpenCode uses `run --dir <absolute-task-directory> --format json --model <provider/model>` with native authentication and permissions; it does not pass `--auto` or override version-specific permission configuration. Permission questions in its noninteractive run are denied automatically.

These controls are not interchangeable: Codex's workspace-write sandbox is not Claude's tool allowlist, and OpenCode's permissions are not an OS sandbox. Select a model and tool configuration that supports image input and local image inspection. The task workspace must be writable and its symlinked reference media readable; external source media need not be writable.

For example, select a backend before running the same entry points:

```sh
export VIDEO_EDITING_ANNOTATION_BACKEND=codex
export VIDEO_EDITING_ANNOTATION_MODEL='<your-model-id>'
unset VIDEO_EDITING_ANNOTATION_CLI  # Clear any previous executable override.
# Optional: export VIDEO_EDITING_ANNOTATION_CLI='/path/to/codex'
```

Backend selection preserves the output schemas, validators, retry rules, and checks that outputs were written by the current attempt. Provider-reported cost is recorded when available; unknown cost remains `null`, not zero. Tool-event counts reflect each CLI's event format and are not directly comparable across backends. A direct API/SDK backend would also need a tool-execution loop; changing an endpoint URL alone would not replace these agent CLIs.

Changing the backend or model does not automatically rerun completed annotations. Use separate private data roots for backend comparisons. Both `run_script_queue.py` and `run_s4_queue.py` support `--redo`; translation skips annotations that already contain all required English fields.

Translation requests only the English fields in a small JSON patch inside the task's private `work/` directory; it does not ask the model to reproduce the full shot log. Python validates the patch's fields and ordered IDs, checks that the source annotation was not modified, and merges the English fields into the original document. Text already in English is copied to its corresponding `_en` field. Private prompt, session, progress, and stderr logs are saved under `edit/runs/<slug>/`. Successful CLI completion and output validation are both required. A failed or timed-out translation restores the original annotation bytes; it does not retry automatically.

Claude shot annotation uses separate sessions for visual descriptions and narrative structure. The queue prepares private contact sheets containing one representative frame per shot, labeled by ID and requested midpoint. Each visual session receives at most 12 shots, its own input and manifest subset, and a separate output file. Python inserts dialogue from the original input rather than trusting a model-generated copy, then validates the successful batch before appending it to the saved draft. A final fresh session generates only the logline, beats, and chunks from the frozen shot log. Python preserves the shot entries and validates the whole result before publishing it.

Claude S4 annotation similarly uses independent batches of at most 12 cut boundaries, each with its own frozen units, measured audio/continuity, and visual manifest. The queue requires a matched successful native image-read result for every listed boundary strip, as well as successful CLI completion and valid output. Python checks exact cut identities and order, finite nonnegative numeric parameters, and measured continuity support for `no_change` before appending a batch. Image-read coverage establishes access to the pictures, not the correctness of transition labels or notes. The whole-film review heuristic and report run after all batches pass.

Both Claude queues preserve accepted batches and strictly revalidate the saved ordered prefix against the current inputs before resuming. An incomplete batch is repeated; an invalid prefix is rejected without silently changing it. `--redo` archives the previous draft and starts fresh inspection. Each call uses the configured model and effort, its own logs, and the optional per-invocation deadline. Drafts alone never establish completed annotations or visual accuracy. Codex and OpenCode retain their single-session annotation workflow.

Individual frames retain up to 1024 pixels on their longest edge without upscaling; contact sheets contain smaller previews. The shot prompt requires a separate individual-frame read for every shot shorter than 0.75 seconds. The frame window ends at the midpoint; millisecond-rounded starts allow at most 0.5 ms of earlier numerical tolerance to retain single-frame shots. The visual manifest records these limits. Cache identity covers input bytes and video path/size/mtime, not a video content hash. Contact sheets are inspection aids, not generated annotations, and one still cannot establish motion.

S4 visual aids live in `edit/task_<slug>/work/cut_visuals/`. Each strip shows up to three consecutive decoded frames on each side of a supplied reference `timeline_s` boundary, with recorded PTS; six strips share a navigation sheet. `nominal BEFORE/AFTER` describes the numerical sampling side, including the declared rounding tolerance, not a verified visual edit or shot identity. Gray `NO FRAME` tiles are empty slots, not black video frames. If either side has no decodable frame in its bounded window, preparation fails. Inspect the source video when the strips cannot establish a transition or its duration, and retain the supplied measurements rather than treating stills as continuity or audio evidence.

### Real CLI checks

On 2026-10-20, each backend passed live file-read/write and image-tool checks through the shared adapter. Tests used newly generated JSON and a two-color PNG in temporary directories with spaces in their paths. They verified the copied nonce, computed sum, identified colors, unchanged inputs, zero exit status, and successful completion events. Tool logs were also inspected to confirm actual image viewing, rather than inferring it from a correct color answer.

| CLI version | Model used in the live check | Result |
| --- | --- | --- |
| Claude Code 2.1.185 | `sonnet` (resolved to `claude-sonnet-4-6`) | Passed |
| Codex CLI 0.144.6 | `gpt-5.6-sol` | Passed |
| OpenCode 1.19.3 | `anthropic/claude-haiku-4-5` | Passed |

These are tested configurations, not default model choices or guarantees of account access. For example, the local Codex default `gpt-6.1-sol` was rejected by the tested CLI account; explicitly selecting its available `gpt-5.6-sol` model succeeded. The adapter does not silently substitute models. OpenCode used native Anthropic environment authentication and a temporary native permission config allowing local read/edit tools. Codex retained its workspace-write sandbox. No credentials, raw provider logs, or generated test files are included in the repository.

To check your own installed and authenticated CLI, run this opt-in command, which makes a real model call and may incur usage:

```sh
python data_construction/check_annotation_cli.py \
  --backend codex --model '<available-model-id>' --timeout 180
```

Use `--backend claude` or `--backend opencode` with the appropriate model ID. Optional `--cli` selects an executable and `--effort` selects its native effort/variant. The checker prints the external temporary directory containing inputs and logs; it never writes test artifacts into the repository. A passing result confirms the small task's output and completion status. Inspect the native tool trace as well to establish image-tool use; some Codex versions omit image-tool events from the JSONL progress stream. `--help` and the unit tests do not invoke a provider.

These checks validate CLI interoperability, not annotation quality or complete video preprocessing, FFmpeg execution, MCP integrations, or reproduction of the paper's dataset. Those still require validation on your own media and environment.

After preprocessing, a typical explicit sequence is:

```sh
python data_construction/run_script_queue.py --film '<slug>' --prep-only
python data_construction/shot_luma.py --all
python data_construction/verify_matches.py --all
python data_construction/card_text.py --film '<slug>'
# Set VIDEO_EDITING_ANNOTATION_MODEL before these annotation calls:
python data_construction/run_script_queue.py --film '<slug>'
python data_construction/translate_script.py --film '<slug>'
python data_construction/build_s1_tasks.py --film '<slug>'
python data_construction/build_s23_tasks.py --film '<slug>'
python data_construction/run_s4_queue.py --film '<slug>'
python data_construction/build_s4_tasks.py --film '<slug>'
# Complete source-film transcription, locate_lines.py, and cut_takes.py above first.
python data_construction/build_s4_editorial.py --film '<slug>'
```

`shot_luma.py` and `verify_matches.py` inspect all supplied exports under the private root; unlike per-film builders they support `--all`, not `--film`. The inspection helpers require the `script_inputs.json` produced by `--prep-only`. Some CS protections require additional private review evidence described below. This sequence documents the dependency order, not a claim that an arbitrary input passes every quality gate. The inherited scripts can replace generated files; keep experiments in separate private roots.

`build_s3.py` produces unit metadata/media without an editor connection. Exporting a Resolve project is separately opt-in with `--export-resolve` and `VIDEO_EDITING_RESOLVE_MCP_ROOT`; that external integration is not bundled. The queue prompts under `data_construction/prompts/` and the Claude stage prompts in `annotation_script_stages.py` and `annotation_s4_stages.py` are release rewrites. Their field contracts were checked against the inherited validators. They intentionally omit real annotations and corpus-derived vocabulary. The original heuristics (including chunk size and S4 vocabulary-diversity checks) remain. Passing their JSON checks establishes structure, not semantic accuracy; use the separate quality audit before accepting generated tasks.

Queue failures now propagate to the command's exit status. Annotation stderr goes to a separate private log to avoid blocking its stdout stream; unfinished attempts cannot inherit a previous success flag. The translation step checks all original fields and entries and restores the original file if the process fails or changes those fields. Local regressions cover these cases, redo handling, paths containing spaces, subprocess cleanup and download failures:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s data_construction/tests
```

These tests use temporary synthetic metadata and local child processes, without invoking an annotation provider. The proxy test also encodes a short synthetic video when FFmpeg/FFprobe are available; this is separate from the full-film check.

## Alignment quality limitation

Temporal matching now averages **corresponding frame descriptors at each offset**. The inherited expression `C[o:o+k, np.arange(k)].mean()` averaged a `k × k` set of cross-frame pairs; it could prefer a static average image to the correct ordered sequence. `temporal_alignment.py` supplies the corrected paired-frame score to both `p2_match.py` and `p2_refine.py`. Refinement also accounts for the actual rounded decoder seek origins and the full-shot fallback for short queries.

The old color-fallback threshold `0.482` was defined for the previous score and is not transferred to the corrected score. Color-only candidates therefore receive `REVIEW(colour)` by default, with candidate location and score retained as evidence but no accepted source in-point. Downstream classification can still retain their footage as trailer-only `golden_only` units. Geometric `MATCH` candidates retain their existing acceptance rule. Operators who have validated a paired-score threshold on their own material can pass `p2_match.py --paired-colour-threshold VALUE`; the pipeline does not choose a replacement threshold automatically.

`p2_refined.csv` records `alignment_score_kind`, `alignment_fps`, the contiguous near-peak interval, and disconnected near peaks. `plateau_frames` now means the width of the contiguous band around the selected maximum, not the total number of near-maximum offsets. The default `--peak-band 0.02` is a heuristic score band, not a calibrated confidence interval. RC construction converts that width to seconds using the alignment sampling rate while retaining the six-native-source-frame minimum tolerance. Legacy rows keep a legacy provenance label instead of being presented as corrected measurements.

Inspect source/reference pictures and timing on new media, particularly color-only candidates and disconnected peaks. These corrections do not prove visual correspondence or establish their effect on historical paper results. Existing completion flags, matching CSVs, units, annotations, and built tasks are not migrated automatically. Use a separate private root to validate a rebuilt pipeline rather than mixing old measurements with newly assembled tasks.

## Quality review and sampling

`task_quality.py` adapts the recovered Gemini quality-audit rules into a local entry point. It reads the complete public task, every supplied media asset, the released runtime prompt and tool contract, and explicitly marked fallible private grading evidence. Stage-relevant released scoring and aggregation sources are included from fixed code locations and hashed. Superseded descriptors under `verifier/.evaluation-backups/` are excluded. Run the scoring preparation step **before** binding audit inputs:

```sh
python -m pip install -r evaluation/requirements.txt
python evaluation/task_pipeline.py --task '/private/path/to/task' --prepare-only

# Offline input/contract binding; this does not call a model.
python data_construction/task_quality.py \
  --task '/private/path/to/task' --stage s_1 \
  --output '/private/path/to/new-audit-inputs' --prepare-only

# Paid semantic audit. Use an available Gemini model; no silent substitution.
python data_construction/task_quality.py \
  --task '/private/path/to/task' --stage s_1 \
  --output '/private/path/to/new-audit-run' \
  --model '<available-gemini-model-id>'
```

Set `GOOGLE_AI_STUDIO_API_KEY` in the process environment for a real audit. `--model` defaults to `VIDEO_EDITING_QUALITY_MODEL`, then to the recorded audit model `gemini-3.8-flash`; temperature is `0` and thinking level is `high`. Model availability is account-dependent. `--timeout` defaults to 600 seconds per provider HTTP operation, not the entire batch. There are no automatic model-request retries. The audit uses the API directly, independently of the Claude/Codex/OpenCode annotation CLI.

Use `--tasks FILE.json` for a JSON array of task-directory paths. Stage is inferred from the `film/stage/chunk` directory layout unless explicitly supplied. The bundled tool contract describes the recovered fixed harness, not every possible deployed agent configuration; use `--tools-json FILE.json` to provide a different runtime's function-schema array. Input evidence records the exact contract and source hashes.

Up to ten videos are supplied directly. Larger sets use a full-timeline labeled reel with an original-path/time mapping, plus every original embedded audio stream separately; external dialogue and music remain included. The reel is resized to 640 pixels and encoded at 12 fps, with short clips slowed according to the mapping. Gemini receives videos at 8 fps. This preserves timelines for review but cannot establish native-frame timing, exact loudness, or stereo properties. Files too large for inline transport use the provider's Files API; no clips are shortened to meet the inline limit.

| Result | Effect |
| --- | --- |
| `keep_candidate` | Requires reported coverage of every supplied asset and no unresolved major defect, missing evidence, scope change, or repair. Added to the accepted candidate list. |
| `filter_recommended` | Requires a major or critical issue tied to supplied evidence. Excluded from the candidate list; original files remain. |
| `repair_needed` | Preserves the model's repair plan. No text, answers, media, or grader code is automatically changed. |
| `needs_more_evidence` | Remains outside the accepted list pending additional evidence. |
| `held` | Records preparation, transport, API, non-STOP, schema, or source-integrity failure. It is neither a quality rejection nor a pass. |

The private output directory retains bound inputs, source identity, request, raw response when available, validated decision, and terminal status. Successful acceptance requires the expected model identity, a complete `STOP` response, valid evidence references, and unchanged input and scoring-source hashes. `source_integrity_verified` reports the completed hash check; `module_did_not_modify_sources` describes the audit utility's behavior. Model-reported media coverage is not independent proof of correct perception.

`accepted_tasks.json` is a path array accepted by `evaluation/task_pipeline.py --tasks`. `accepted_index.json` adds task identities and evidence hashes. For later size sampling, pass an explicit film-to-existing-split JSON mapping with `--film-splits`; the audit does not choose train/validation assignments. A one-film run should stay in one split. Repair tasks in separate copies, prepare their scoring descriptors, and audit the revised bytes in a new output directory. Do not reuse an earlier approval after changing the task or grader.

Size sampling is separate from quality filtering:

```sh
python data_construction/quality_selection.py \
  --candidates '/private/audit/accepted_index.json' \
  --targets '/private/desired-counts.json' --seed 'my-release-seed' \
  --output '/private/new-selection'
```

The targets file maps each existing split to every task type it contains and an explicit retained count, for example `{"train":{"s_1":8,"s_2":8}}` for a candidate index containing only those two groups. Counts must not exceed available candidates. The sampler rejects duplicate task identities and films spanning splits. It uses largest-remainder allocation within each split/type, retains at least one item per represented film when the target permits, and ranks items by seeded SHA256. Unselected candidates are marked as size exclusions, not bad tasks.

The paper's final size reduction used 925 reviewed/repaired candidates and retained 797: CS 136, NS 147, RC 257, and FC 257; train 716 and validation 81. The recovered seed and allocation algorithm reproduce the exact private 797-task identity set and film allocations. `--paper-797` requires the SHA256-bound historical 925-task index and rejects seed or target overrides. Those private inputs are not distributed, and this mode is not a target quota for newly generated datasets.

## Scoring and reconstruction limits

Use [the evaluation guide](evaluation.md) and `evaluation/task_pipeline.py` to attach the released scoring configuration and score task submissions. Existing CS/NS task-specific graders are preserved; new tasks can receive the bundled defaults. RC and FC use the released fixed scoring chain. The old S2/S3 rubric-writer hooks remain only for explicit legacy reconstruction and are not required by this workflow. Preparing a rubric is not a task-quality approval.

The public source is still insufficient to regenerate every historical data-construction decision:

- Private source/reference media, manifests, annotations, answers, film lists, and per-task overrides are intentionally excluded.
- CS quality helpers consume actual evidence under `review_evidence/s1/` beneath the private root. Their readers and policies are included, but private findings are not. Missing evidence must not be replaced with invented approval.
- The original 925 candidates had mixed review histories: some had additional independent-context checks, others a single audit or a repair-author judgment. They were not all double-reviewed or certified as human ground truth. The portable audit adapts the recovered S234 rules and adds a CS rule; it does not replay every historical repair or guarantee the same model decisions.
- The 925-to-797 sampler is recovered and parity-checked, but its exact private parent index is not shipped. Its 128 size exclusions were not new quality rejections.
- Old split scripts and corpus summaries describe other task coverage. For new datasets, supply a deliberate film-disjoint split rather than treating an old file as the final paper split.
- Generic annotation prompts, corrected temporal alignment, model revisions, and task-builder changes can produce different annotations and tasks. Build hashes detect changes; they do not prove experimental equivalence.
- Historical environment locks and full distributed training/service integration remain separate reproduction work.

Generated media, task JSON, audit evidence, selection indexes, and provider logs stay outside the public code package.
