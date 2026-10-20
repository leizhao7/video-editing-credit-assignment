# Data construction

See [the construction guide](../docs/data_construction.md) for entry points, private artifact contracts, dependencies, and reconstruction limits. This directory contains implementation code, generic prompt templates, and resources; it contains no dataset.

For a first run, use the [one-film walkthrough](../docs/one_film.md). It includes the source-film dialogue preparation required for FC.

The main pipeline uses the generic adapters in `modal/` for shot detection, reference-edit transcription, and audio separation. `--modal-dir` or `VIDEO_EDITING_MODAL_DIR` can select a different adapter directory. Set `VIDEO_EDITING_MODAL_VOLUME` to your own volume before explicitly running cloud stages. Importing modules or requesting `--help` does not run those stages.

Start with [agent setup](../docs/agent_setup.md): `python3 data_construction/setup_agent.py` guides backend/model selection and writes a shell configuration outside the repository. Source the printed path to activate it. Installation and login use the selected CLI's native tools.

The annotation entry points `run_script_queue.py`, `run_s4_queue.py`, and `translate_script.py` share `VIDEO_EDITING_ANNOTATION_BACKEND=claude|codex|opencode` (default: `claude`) and require `VIDEO_EDITING_ANNOTATION_MODEL`. OpenCode uses its 1.x CLI and a `provider/model` identifier. An optional `VIDEO_EDITING_ANNOTATION_CLI` selects an executable, not a shell command with flags. See [annotation configuration](../docs/data_construction.md#annotation-cli-configuration) and `environment.example` for effort, MCP, and tool-access differences. All three adapters have passed real CLI file/image checks and synthetic regression tests; full media processing remains unverified.

Full-film transcription is a separate adapter entry point; the main pipeline does not call it automatically. The final dataset selection and final task-grader assembly remain outside this construction draft.
