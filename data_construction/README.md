# Data construction

See [the construction guide](../docs/data_construction.md) for entry points, private artifact contracts, dependencies, and reconstruction limits. This directory contains implementation code, generic prompt templates, and resources; it contains no dataset.

The main pipeline uses the generic adapters in `modal/` for shot detection, reference-edit transcription, and audio separation. `--modal-dir` or `VIDEO_EDITING_MODAL_DIR` can select a different adapter directory. Set `VIDEO_EDITING_MODAL_VOLUME` to your own volume before explicitly running cloud stages. Importing modules or requesting `--help` does not run those stages.

The annotation entry points `run_script_queue.py`, `run_s4_queue.py`, and `translate_script.py` share `VIDEO_EDITING_ANNOTATION_BACKEND=claude|codex|opencode` (default: `claude`) and require `VIDEO_EDITING_ANNOTATION_MODEL`. Install and authenticate the selected CLI separately; OpenCode uses its 1.x CLI and a `provider/model` identifier. An optional `VIDEO_EDITING_ANNOTATION_CLI` selects an executable, not a shell command with flags. See [annotation configuration](../docs/data_construction.md#annotation-cli-configuration) and `environment.example` for effort, MCP, and tool-access differences. The adapters have fake-CLI regression coverage, not real-provider or media end-to-end validation.

Full-film transcription is a separate adapter entry point; the main pipeline does not call it automatically. The final dataset selection and final task-grader assembly remain outside this construction draft.
