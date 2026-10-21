# Data construction

See [the construction guide](../docs/data_construction.md) for entry points, private artifact contracts, dependencies, and reconstruction limits. This directory contains implementation code, generic prompt templates, and resources; it contains no dataset.

For a first run, use the [project workflow](../docs/workflow.md). The unified `video_editing.py` command handles source-film dialogue preparation, task construction, quality review, and scoring setup. The [one-film walkthrough](../docs/one_film.md) remains the manual command reference.

The main pipeline uses the generic adapters in `modal/` for shot detection, reference-edit transcription, and audio separation. `--modal-dir` or `VIDEO_EDITING_MODAL_DIR` can select a different adapter directory. Set `VIDEO_EDITING_MODAL_VOLUME` to your own volume before explicitly running cloud stages. Importing modules or requesting `--help` does not run those stages.

Start with [agent setup](../docs/agent_setup.md): `python3 data_construction/setup_agent.py` guides backend/model selection and writes a shell configuration outside the repository. Source the printed path to activate it. Installation and login use the selected CLI's native tools.

The annotation entry points `run_script_queue.py`, `run_s4_queue.py`, and `translate_script.py` share `VIDEO_EDITING_ANNOTATION_BACKEND=claude|codex|opencode` (default: `claude`) and require `VIDEO_EDITING_ANNOTATION_MODEL`. OpenCode uses its 1.x CLI and a `provider/model` identifier. An optional `VIDEO_EDITING_ANNOTATION_CLI` selects an executable, not a shell command with flags. See [annotation configuration](../docs/data_construction.md#annotation-cli-configuration) and `environment.example` for effort, MCP, and tool-access differences. All three adapters have passed real CLI file/image checks and synthetic regression tests. A [one-film construction check](../docs/release_status.md#one-film-construction-check) completed with Codex, including local assembly and Modal preprocessing; this does not qualify the full pipeline under every backend or approve candidate-task quality.

Full-film transcription is a separate adapter entry point included by the unified workflow; the lower-level `pipeline.py` only transcribes the trailer. `task_quality.py` produces the accepted-task list, `quality_selection.py` optionally samples across films, and `evaluation/task_pipeline.py` prepares scoring rules and scores submitted edits. Historical author repairs are not automatically replayed.

User-facing instructions, reports, logs, and newly authored annotation prose use English. Supplied dialogue, visible card text, IDs, and proper names retain their source form. Chinese recognition patterns remain only where needed to read historical annotations; Unicode fixtures test that source text is preserved.
