# Data construction

See [the construction guide](../docs/data_construction.md) for entry points, private artifact contracts, dependencies, and reconstruction limits. This directory contains implementation code, generic prompt templates, and resources; it contains no dataset.

The main pipeline uses the generic adapters in `modal/` for shot detection, reference-edit transcription, and audio separation. `--modal-dir` or `VIDEO_EDITING_MODAL_DIR` can select a different adapter directory. Set `VIDEO_EDITING_MODAL_VOLUME` to your own volume before explicitly running cloud stages. Importing modules or requesting `--help` does not run those stages.

Full-film transcription is a separate adapter entry point; the main pipeline does not call it automatically. The final dataset selection and final task-grader assembly remain outside this construction draft.
