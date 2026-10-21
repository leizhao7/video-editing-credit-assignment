# Outcome evaluation

## Score a constructed task

`evaluation/task_pipeline.py` connects constructed CS, NS, RC and FC tasks to the released fixed-snapshot graders. It prepares rubric descriptors, validates submissions, extracts evidence, runs the required judges, and writes `evidence/task_result.json`. The [one-film walkthrough](one_film.md) connects task construction, quality review and scoring.

Use Python 3.12, FFmpeg and ffprobe. Install the local scoring dependencies in your virtual environment:

```sh
python -m pip install -r evaluation/requirements.txt
```

Keep tasks and results outside the source checkout. `--tasks` accepts a JSON array of absolute task-directory paths, including `accepted_tasks.json` from quality review or sampling. Preparing rubrics does not require model outputs or call a judge:

```sh
python evaluation/task_pipeline.py --tasks /private/candidates.json --prepare-only
```

After an agent has written the required files under each task's `output/`, validate them without making judge requests, then score them:

```sh
python evaluation/task_pipeline.py --tasks /private/quality/accepted_tasks.json --preflight-only
python evaluation/task_pipeline.py --tasks /private/quality/accepted_tasks.json \
  --summary /private/scoring/results.json
```

For one task, use `--task /private/tasks/film/s_3/chunk`. Stage inference uses the task layout and metadata; `--stage CS`, `NS`, `RC` or `FC` is available for other layouts. Conflicting stage information is rejected. FC needs both the edit JSON and its rendered preview. Preflight checks the files and media contract; it does not assess editing quality.

CS and NS use the task's existing `verifier/grade.py`, preserving task-specific repairs and recording its hash. A released default is installed only when the grader is absent. RC uses the bundled deterministic grader and two judge channels. FC uses the two judge channels. The weights are:

| Task | Deterministic | Reference-aware judge | Blind judge |
| :--- | ---: | ---: | ---: |
| CS / NS | 1.0 | 0 | 0 |
| RC | 0.4 | 0.4 | 0.2 |
| FC | 0 | 0.4 | 0.6 |

The generated `verifier/rubric.json` includes the actual rubric definitions and aggregation rules; `verifier/evaluation.json` records the scoring source hashes. Changed descriptors are backed up under `verifier/.evaluation-backups/`. Preparation does not approve task quality or modify reference answers.

RC and FC require `GOOGLE_AI_STUDIO_API_KEY` in the environment. Defaults are Gemini `gemini-3.1-pro-preview`, temperature 0, seed 42. `S12_GEMINI_MODEL`, `S12_JUDGE_TEMP` and `S12_JUDGE_SEED` explicitly override these settings. The entry point selects direct independent judging with submission checks; it does not mix in the decay overlay. Invalid model submissions receive zero. Infrastructure failures, absent judge components, and partial rubric responses remain unscored (`final: null`); the command exits nonzero. Each attempt replaces the previous task result so a failed attempt cannot leave a stale successful score.

## Source provenance and scope

`evaluation/` contains source recovered by SHA-256 from the final fixed-PRM run's grader manifest. Four CLI adapters have their historical Azure endpoint/deployment defaults replaced with environment variables. Original and release hashes are distinguished in `configs/evaluation_sources.json`. `evaluation/decay_overlay/` holds the files that differ in the final decay run, including response reuse and retry handling. Do not put both directories on `PYTHONPATH` and depend on accidental import precedence.

Release cleanup also removes film/task identifiers and historical measurement results from comments and documentation. Thresholds, weights, media extraction and score calculations are unchanged. One executable explanatory string changed: `s3_evidence.py` now states the picture-only task rule without a corpus-specific speech-cut statistic. This is a judge-input explanatory text change to remove a corpus statistic, so the released judge prompt is not byte-identical to the recorded run.

The public adapters additionally reject verifier failures and non-finite or out-of-range scores. Aggregation validates weights before running verifiers and atomically replaces each attempted result, including a null final score when required components are missing. Incomplete scoring exits with a nonzero status instead of leaving a previous successful score in place. Successful score formulas, default weights and result fields remain unchanged; these failure-handling adaptations are distinguished from historical source in the provenance map.

The fixed snapshot uses the `judge_only` grading policy with submission validation enabled. The judge defaults are Gemini 3.1 Pro Preview, temperature 0, seed 42. The source also retains older joint-review paths; their presence does not mean the paper's runs used them. Fine Cut is `s_4e`.

The Python modules inspect task manifests and submitted artifacts, prepare media evidence, call the judge, and aggregate rubric results. They expect a private task directory with a task prompt, initial-state manifest and media, verifier material where applicable, and model outputs. No such task instances are shipped here.

The local entry point is a new integration helper, not the historical remote service. A real check on four private candidate tasks completed all four scoring paths, including Gemini requests for RC and FC. CS/NS/RC used reference-matching submissions; FC used a deliberately simple silent hard-cut submission. This verifies execution and aggregation, not trained-policy performance, task quality, or reproduction of the paper's tables. The earlier outcome-only `grading_video.py` and the complete historical environment remain missing. Preserve the distinct versions until their runtime changes have been reviewed.

## CPU transport regression tests

These existing tests send requests only to a loopback HTTP server and use synthetic responses. They exercise request timeouts, bounded retries, error receipts, and deadline handling. They neither call Gemini nor require film media.

```sh
python -m pip install pytest pytest-asyncio aiohttp numpy scipy
python -m pytest evaluation/test_judge_deadline.py -q
```

Synthetic verifier and aggregation regressions run separately without judge requests:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider evaluation/test_score_failures.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s evaluation -p 'test_task_pipeline.py'
```

The verifier has a configurable timeout (`S12_DETERMINISTIC_TIMEOUT_S`, default 180 seconds), bounded by `S12_GRADING_DEADLINE_UNIX` when supplied. `S12_GRADER_PYTHON` can select its Python executable. Tests exercise both fixed and decay adapters, successful legacy verifier responses, failed subprocesses, invalid weights, incomplete components and preservation of other tasks' results. They do not establish benchmark or model performance.

`evaluation/requirements.txt` covers the local scoring entry point, not a complete historical dependency lock or a GPU training installation. The private execution image and the remote service entry point remain separate reconstruction work.
