# Outcome evaluation

`evaluation/` contains source recovered by SHA-256 from the final fixed-PRM run's grader manifest. Four CLI adapters have their historical Azure endpoint/deployment defaults replaced with environment variables. Original and release hashes are distinguished in `configs/evaluation_sources.json`. `evaluation/decay_overlay/` holds the files that differ in the final decay run, including response reuse and retry handling. Do not put both directories on `PYTHONPATH` and depend on accidental import precedence.

Release cleanup also removes film/task identifiers and historical measurement results from comments and documentation. Thresholds, weights, media extraction and score calculations are unchanged. One executable explanatory string changed: `s3_evidence.py` now states the picture-only task rule without a corpus-specific speech-cut statistic. This is a judge-input explanatory text change to remove a corpus statistic, so the released judge prompt is not byte-identical to the recorded run.

The public adapters additionally reject verifier failures and non-finite or out-of-range scores. Aggregation validates weights before running verifiers and atomically replaces each attempted result, including a null final score when required components are missing. Incomplete scoring exits with a nonzero status instead of leaving a previous successful score in place. Successful score formulas, default weights and result fields remain unchanged; these failure-handling adaptations are distinguished from historical source in the provenance map.

The fixed snapshot uses the `judge_only` grading policy with submission validation enabled. The judge defaults are Gemini 3.1 Pro Preview, temperature 0, seed 42. The source also retains older joint-review paths; their presence does not mean the paper's runs used them. Fine Cut is `s_4e`.

The Python modules inspect task manifests and submitted artifacts, prepare media evidence, call the judge, and aggregate rubric results. They expect a private task directory with a task prompt, initial-state manifest and media, verifier material where applicable, and model outputs. No such task instances are shipped here.

The complete historical remote service wrapper and the earlier outcome-only `grading_video.py` have not yet been recovered into this draft. Therefore, this is a grader source collection, not a verified command for reproducing all four task types on the original benchmark. Preserve the distinct versions until their runtime changes have been reviewed.

## CPU transport regression tests

These existing tests send requests only to a loopback HTTP server and use synthetic responses. They exercise request timeouts, bounded retries, error receipts, and deadline handling. They neither call Gemini nor require film media.

```sh
python -m pip install pytest pytest-asyncio aiohttp numpy scipy
python -m pytest evaluation/test_judge_deadline.py -q
```

Synthetic verifier and aggregation regressions run separately without judge requests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider evaluation/test_score_failures.py
```

The verifier has a configurable timeout (`S12_DETERMINISTIC_TIMEOUT_S`, default 180 seconds), bounded by `S12_GRADING_DEADLINE_UNIX` when supplied. `S12_GRADER_PYTHON` can select its Python executable. Tests exercise both fixed and decay adapters, successful legacy verifier responses, failed subprocesses, invalid weights, incomplete components and preservation of other tasks' results. They do not establish benchmark or model performance.

Running the real grader additionally needs FFmpeg/ffprobe, Pillow, faster-whisper, librosa, and the verified task schema. Actual judge calls require the operator's `GOOGLE_AI_STUDIO_API_KEY` in the environment. The private execution image's complete dependency lock and the remote service entry point remain to be reconstructed; no unverified full training installation command is provided.
