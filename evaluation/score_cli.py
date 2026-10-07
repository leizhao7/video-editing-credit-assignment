"""Input validation and atomic result publication for score CLIs."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from grading_contract import valid_score


def add_task_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task")
    group.add_argument("--tasks")


def validate_weights(parser, weights):
    if any(not math.isfinite(value) or value < 0 for value in weights):
        parser.error("weights must be finite and nonnegative")
    if abs(sum(weights) - 1.0) > 1e-9:
        parser.error("weights must sum to 1")


def task_paths(parser, args):
    try:
        tasks = json.loads(Path(args.tasks).read_text()) if args.tasks else [args.task]
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read task list: {exc}")
    if (not isinstance(tasks, list) or not tasks
            or any(not isinstance(task, str) or not task.strip() for task in tasks)):
        parser.error("tasks must be a nonempty JSON array of directory paths")
    if any(not Path(task).is_dir() for task in tasks):
        parser.error("every task path must be an existing directory")
    return tasks


def read_component(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return {"score": None, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(value, dict):
        return {"score": None, "error": "score component must be a JSON object"}
    return value


def run_verifier(script, task, *, quiet=False, allow_legacy=False):
    """Accept scores only from a successful, bounded verifier process."""
    def failed(message):
        return None, {"error": message}, message

    try:
        timeout = float(os.environ.get("S12_DETERMINISTIC_TIMEOUT_S", "180"))
        if not math.isfinite(timeout) or timeout <= 0:
            return failed("S12_DETERMINISTIC_TIMEOUT_S must be finite and positive")
        deadline = os.environ.get("S12_GRADING_DEADLINE_UNIX")
        if deadline:
            deadline = float(deadline)
            if not math.isfinite(deadline):
                return failed("S12_GRADING_DEADLINE_UNIX must be finite")
            timeout = min(timeout, deadline - time.time())
        if timeout <= 0:
            return failed("verifier deadline exhausted")
        command = [os.environ.get("S12_GRADER_PYTHON", sys.executable), str(script), "--task", task]
        if quiet:
            command.append("--quiet")
        run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return failed(f"verifier failed: {type(exc).__name__}: {exc}")
    raw = run.stdout.strip()
    if run.returncode:
        diagnostics = "\n".join(x.strip() for x in (run.stdout, run.stderr) if x.strip())
        return failed(f"verifier rc={run.returncode}: {diagnostics[-1000:]}")
    try:
        detail = json.loads(raw)
    except ValueError:
        if not allow_legacy:
            return failed("verifier did not return a JSON score object")
        try:
            detail = {"score": float(raw.split(maxsplit=1)[0])}
        except (ValueError, IndexError):
            return failed("verifier did not return a JSON score or a leading numeric score")
    if allow_legacy and not isinstance(detail, dict):
        detail = {"score": detail}
    if not isinstance(detail, dict) or not valid_score(detail.get("score")):
        return failed("verifier score must be a finite number in [0, 1]")
    try:
        json.dumps(detail, allow_nan=False)
    except (ValueError, TypeError):
        return failed("verifier returned non-finite or invalid JSON metadata")
    return float(detail["score"]), detail, raw


def write_result(path, result):
    """Replace only this task's result; never expose a partially written JSON file."""
    path = Path(path)
    payload = json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def evaluate_tasks(tasks, combine, result_path):
    rows = []
    for task in tasks:
        path = result_path(task)
        # A crash or interruption after this point cannot leave a previous score
        # masquerading as the result of the current attempt.
        write_result(path, {"task": task, "final": None, "status": "in_progress",
                            "why": "grading has not completed"})
        try:
            row = combine(task)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
            row = {"task": task, "final": None, "status": "error",
                   "why": f"{type(exc).__name__}: {exc}"}
        if row.get("final") is not None and not valid_score(row["final"]):
            row = {"task": task, "final": None, "status": "error",
                   "why": "aggregate is not a finite score in [0, 1]"}
        try:
            write_result(path, row)
        except (TypeError, ValueError) as exc:
            row = {"task": task, "final": None, "status": "error",
                   "why": f"result serialization failed: {exc}"}
            write_result(path, row)
        rows.append(row)
    return rows
