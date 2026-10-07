"""Synthetic local verifier/CLI regressions; no media or provider calls."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def make_task(tmp_path, prefix, *, blind=.2, referenced=.8, deterministic=.5):
    task = tmp_path / "task"
    (task / "evidence").mkdir(parents=True)
    (task / "output").mkdir()
    (task / "verifier").mkdir()
    for suffix, score in (("", blind), ("_ref", referenced)):
        (task / f"evidence/{prefix}_score{suffix}.json").write_text(
            json.dumps({"score": score, "partial": False}))
    verifier = task / "verifier/grade.py"
    verifier.write_text(f"print({json.dumps({'score': deterministic})!r})\n")
    return task, verifier


def cli(prefix, task, verifier, *extra):
    # Keep provider credentials and deployment settings out of this local test.
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT") if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", S12_TRAINING_JUDGE_ONLY="1",
               S12_TRAINING_SUBMISSION_CHECKS="0", S12_S3_DETERMINISTIC_GRADER=str(verifier))
    return subprocess.run([sys.executable, "-B", str(ROOT / f"{prefix}_score.py"),
                           "--task", str(task), "--quiet", *extra],
                          env=env, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("prefix,expected", [("s3", .56), ("s4e", .44)])
@pytest.mark.parametrize("zero", [False, True])
def test_default_scores_unchanged_and_zero_is_valid(tmp_path, prefix, expected, zero):
    values = {"blind": 0., "referenced": 0., "deterministic": 0.} if zero else {}
    task, verifier = make_task(tmp_path, prefix, **values)
    result = cli(prefix, task, verifier)
    assert result.returncode == 0, result.stderr
    assert json.loads((task / f"evidence/{prefix}_final.json").read_text())["final"] == (0. if zero else expected)


@pytest.mark.parametrize("prefix", ["s3", "s4e"])
@pytest.mark.parametrize("damage", ["missing", "malformed", "nonobject", "null", "boolean",
                                    "nan", "infinity", "negative", "above_one"])
def test_bad_component_invalidates_only_current_task_final(tmp_path, prefix, damage):
    task, verifier = make_task(tmp_path, prefix)
    final = task / f"evidence/{prefix}_final.json"
    final.write_text('{"final":0.99,"previous":true}')
    other = tmp_path / "other-task/evidence"
    other.mkdir(parents=True)
    other_final = other / f"{prefix}_final.json"
    other_final.write_text('{"final":0.75}')
    reference = task / f"evidence/{prefix}_score_ref.json"
    if damage == "missing":
        reference.unlink()
    else:
        reference.write_text({"malformed": "{", "nonobject": "[]", "null": '{"score":null}',
                              "boolean": '{"score":true}', "nan": '{"score":NaN}',
                              "infinity": '{"score":1e309}', "negative": '{"score":-0.1}',
                              "above_one": '{"score":1.1}'}[damage])
    result = cli(prefix, task, verifier)
    assert result.returncode == 1
    current = json.loads(final.read_text())
    assert current["final"] is None and current["missing"]
    assert "previous" not in current
    assert "Incomplete scoring" in result.stderr
    assert other_final.read_text() == '{"final":0.75}'


@pytest.mark.parametrize("prefix", ["s3", "s4e"])
@pytest.mark.parametrize("invalid", ["negative", "nan", "infinity", "sum"])
def test_invalid_weights_rejected_before_starting_task(tmp_path, prefix, invalid):
    task, verifier = make_task(tmp_path, prefix)
    launched = tmp_path / "verifier-launched"
    verifier.write_text(f"from pathlib import Path\nPath({str(launched)!r}).touch()\nprint('{{\"score\":0.5}}')\n")
    final = task / f"evidence/{prefix}_final.json"
    final.write_text('{"final":0.75,"previous":true}')
    if invalid == "negative":
        extra = ["--w-blind", "-1", "--w-ref", "1.6" if prefix == "s3" else "2"]
    else:
        extra = ["--w-blind", {"nan": "nan", "infinity": "inf", "sum": "0.9"}[invalid]]
    result = cli(prefix, task, verifier, *extra)
    assert result.returncode == 2 and "weights" in result.stderr
    assert final.read_text() == '{"final":0.75,"previous":true}'
    assert not launched.exists()


@pytest.mark.parametrize("damage", ["failure", "missing", "invalid"])
def test_s3_bad_deterministic_verifier_does_not_reuse_final(tmp_path, damage):
    task, verifier = make_task(tmp_path, "s3")
    final = task / "evidence/s3_final.json"
    final.write_text('{"final":0.9}')
    if damage == "failure":
        verifier.write_text('import sys\nprint(\'{"score":0.9}\')\nsys.exit(2)\n')
    elif damage == "missing":
        verifier.unlink()
    else:
        verifier.write_text('print(\'{"score":true}\')\n')
    result = cli("s3", task, verifier)
    current = json.loads(final.read_text())
    assert result.returncode == 1 and current["final"] is None
    assert "deterministic" in current["missing"]


@pytest.fixture(params=["s12_task.py", "decay_overlay/s12_task.py"])
def harness(request, monkeypatch):
    for key in ("S12_GRADER_PYTHON", "S12_GRADING_DEADLINE_UNIX", "S12_DETERMINISTIC_TIMEOUT_S"):
        monkeypatch.delenv(key, raising=False)
    spec = importlib.util.spec_from_file_location("test_s12_harness", ROOT / request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("output,exit_code", [('{"score":0.9}', 2), ('{"score":-0.1}', 0),
    ('{"score":1.1}', 0), ('{"score":true}', 0), ('{"score":null}', 0),
    ('{"score":NaN}', 0), ('{"score":1e309}', 0), ('{"score":1e3}', 0),
    ('not JSON; diagnostic score: 0.9', 0)])
def test_s12_rejects_failed_or_invalid_verifier(harness, tmp_path, output, exit_code):
    task, verifier = make_task(tmp_path, "s3")
    (task / "output/selection.json").write_text('{}')
    verifier.write_text(f"import sys\nprint({output!r})\nsys.exit({exit_code})\n")
    score, detail = harness.grade(str(task), "s_1")
    assert score is None and detail


@pytest.mark.parametrize("output,expected", [('{"score":0}', 0.), ('{"score":0.35}', .35),
    ('0.35 legacy detail', .35), ('1e-2 legacy detail', .01)])
def test_s12_valid_json_and_legacy_outputs(harness, tmp_path, output, expected):
    task, verifier = make_task(tmp_path, "s3")
    (task / "output/selection.json").write_text('{}')
    verifier.write_text(f"print({output!r})\n")
    assert harness.grade(str(task), "s_1")[0] == expected


def test_s12_timeout_and_missing_verifier_are_unscored(harness, tmp_path, monkeypatch):
    task, verifier = make_task(tmp_path, "s3")
    (task / "output/selection.json").write_text('{}')
    verifier.write_text('import time\ntime.sleep(2)\nprint(\'{"score":1}\')\n')
    monkeypatch.setenv("S12_DETERMINISTIC_TIMEOUT_S", "0.05")
    score, detail = harness.grade(str(task), "s_1")
    assert score is None and "TimeoutExpired" in detail
    verifier.unlink()
    score, detail = harness.grade(str(task), "s_1")
    assert score is None and "rc=" in detail
