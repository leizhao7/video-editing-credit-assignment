"""Prepare private task rubrics or run the released outcome-scoring chain.

CS/NS retain task-specific deterministic graders. RC/FC use the recovered fixed
snapshot, direct independent judging, and submission checks. This entry point is
a public integration helper, not a reconstruction of the historical cloud service.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
CONSTRUCTION = ROOT.parent / "data_construction"
STAGES = {"CS": "s_1", "NS": "s_2", "RC": "s_3", "FC": "s_4e"}
KINDS = {"json_set": "s_1", "json_sequence": "s_2", "json_cutlist": "s_3"}
WEIGHTS = {"s_1": (1.0, 0.0, 0.0), "s_2": (1.0, 0.0, 0.0),
           "s_3": (0.4, 0.4, 0.2), "s_4e": (0.0, 0.4, 0.6)}
VERSION = "released-task-scoring-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text(),
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {Path(path).name}")
    return value


def inside(task, relative):
    path = task / relative
    if not path.resolve().is_relative_to(task):
        raise ValueError(f"Task-owned path escapes task directory: {relative}")
    return path


def infer_stage(task, requested=None):
    requested = STAGES.get(requested, requested)
    clues = set()
    if task.parent.name in WEIGHTS:
        clues.add(task.parent.name)
    meta_path = task / "verifier/task_meta.json"
    if meta_path.exists():
        meta = read_object(meta_path)
        if meta.get("deliverable_kind") in KINDS:
            clues.add(KINDS[meta["deliverable_kind"]])
        if str(meta.get("stage", "")).startswith("s4e/"):
            clues.add("s_4e")
    if requested:
        clues.add(requested)
    if len(clues) != 1 or next(iter(clues)) not in WEIGHTS:
        raise ValueError("Cannot infer an unambiguous task stage; supply --stage CS/NS/RC/FC")
    return clues.pop()


def publish_generated(task, path, value):
    """Back up an earlier descriptor, then atomically publish the new one."""
    from score_cli import write_result
    encoded = (json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != encoded:
        backup = inside(task, "verifier/.evaluation-backups")
        backup.mkdir(exist_ok=True)
        destination = backup / f"{path.name}.{digest(path)}"
        if not destination.exists():
            shutil.copyfile(path, destination)
    write_result(path, value)


def rubric_descriptor(stage):
    from rubrics_video import S3, S3_REFERENCE, S4, S4_REFERENCE
    det, ref, blind = WEIGHTS[stage]
    descriptor = {
        "format": VERSION, "stage": stage,
        "weights": {"deterministic": det, "referenced": ref, "blind": blind},
        "policy": "direct independent judging with submission checks",
        "implementation": "evaluation/task_pipeline.py",
        "scope": "Scoring configuration; this is not a task quality approval.",
        "metadata_precedence": "For this entry point, this active scoring chain and its weights supersede legacy construction-only reward, verifier_class, and scoring_status notes in task_meta.json. Answer keys and existing CS/NS task graders remain the scoring inputs specified here.",
    }
    if stage in ("s_1", "s_2"):
        descriptor["deterministic"] = {
            "runner": "verifier/grade.py",
            "default_metric": ("selected-set F1" if stage == "s_1" else "(1-ND) * LIS * ADJ"),
            "task_specific_repairs": "Existing task graders are preserved; inspect their recorded hash.",
        }
    else:
        descriptor["blind"] = deepcopy(S3 if stage == "s_3" else S4)
        descriptor["referenced"] = deepcopy(S3_REFERENCE if stage == "s_3" else S4_REFERENCE)
        descriptor["channel_aggregation"] = {
            "band_range": [0, 4], "zero_band_multiplier": 0.75,
            "formula": "weighted band mean / 4 * 0.75 ** number_of_zero_bands",
        }
        if stage == "s_3":
            for row in descriptor["referenced"]:
                row["weight"] *= 0.8
            descriptor["referenced"].append({
                "id": "duration", "weight": 0.2,
                "implementation": "s3_judge_ref.py:score_duration (mechanical band)",
            })
            descriptor["deterministic"] = {
                "runner": "evaluation/grade_s3.py",
                "metric": "fraction within reference in-point AND duration tolerance",
                "tolerances": "Read from this task's verifier/answer.json; seconds are not recomputed.",
            }
        descriptor["judge"] = {"provider": "gemini", "model_environment": "S12_GEMINI_MODEL",
                               "default_model": "gemini-3.1-pro-preview",
                               "default_temperature": 0, "default_seed": 42}
    return descriptor


def prepare(task, stage):
    """Attach the released scoring description without replacing repaired graders."""
    task = Path(task).resolve(strict=True)
    if not task.is_dir():
        raise ValueError("Task must be a directory")
    verifier = inside(task, "verifier")
    verifier.mkdir(exist_ok=True)
    read_object(inside(task, "initial_state/manifest.json"))
    read_object(inside(task, "verifier/answer.json"))
    sources = {p.name: digest(p) for p in sorted(ROOT.glob("*.py"))
               if not p.name.startswith("test_")}
    grader_info = None
    if stage in ("s_1", "s_2"):
        source = CONSTRUCTION / ("grade_s1.py" if stage == "s_1" else "grade_s2.py")
        grader = inside(task, "verifier/grade.py")
        if not grader.exists():
            shutil.copyfile(source, grader)
        if not grader.is_file():
            raise ValueError("verifier/grade.py must be a regular trusted file")
        grader_info = {"sha256": digest(grader), "matches_released_default": digest(grader) == digest(source),
                       "default_sha256": digest(source), "existing_grader_preserved": True}
    description = rubric_descriptor(stage)
    if grader_info:
        description["deterministic"]["grader"] = grader_info
    publish_generated(task, inside(task, "verifier/rubric.json"), description)
    identity = {"format": VERSION, "stage": stage, "evaluation_sources": sources,
                "task_grader": grader_info, "rubric_sha256": digest(verifier / "rubric.json")}
    publish_generated(task, inside(task, "verifier/evaluation.json"), identity)
    return identity


@contextmanager
def scoring_environment():
    values = {"S12_TRAINING_JUDGE_ONLY": "1", "S12_TRAINING_SUBMISSION_CHECKS": "1",
              "S12_TOOLS_DIR": str(ROOT), "S12_S3_DETERMINISTIC_GRADER": str(ROOT / "grade_s3.py")}
    defaults = {"S12_JUDGE_PROVIDER": "gemini", "S12_GEMINI_MODEL": "gemini-3.1-pro-preview",
                "S12_JUDGE_TEMP": "0", "S12_JUDGE_SEED": "42"}
    values.update({k: os.environ.get(k, v) for k, v in defaults.items()})
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def judge_component_paths(task, stage):
    prefix = "s3" if stage == "s_3" else "s4e"
    return {"blind": inside(task, f"evidence/{prefix}_score.json"),
            "referenced": inside(task, f"evidence/{prefix}_score_ref.json")}


def incomplete_judge_components(task, stage):
    """Require all fixed-snapshot rubrics, independently of the core aggregate.

    The historical direct-judge path can aggregate a partially answered channel.
    This public wrapper preserves its evidence but does not publish that result
    as a complete score. Call only after removing old channel files and grading.
    """
    from grading_contract import valid_score
    rubric = rubric_descriptor(stage)
    incomplete = {}
    for channel, path in judge_component_paths(task, stage).items():
        try:
            component = read_object(path)
        except (OSError, ValueError, TypeError) as exc:
            incomplete[channel] = [f"Missing or unreadable component: {type(exc).__name__}"]
            continue
        reasons = []
        if not valid_score(component.get("score")):
            reasons.append("No valid component score")
        if component.get("partial"):
            reasons.append("Component is marked partial")
        if component.get("missing"):
            reasons.append("Component reports missing rubrics")
        if component.get("required_images_missing"):
            reasons.append("Component reports missing required images")
        bands = component.get("per_rubric")
        expected = {row["id"] for row in rubric[channel]}
        if not isinstance(bands, dict) or set(bands) != expected:
            reasons.append("Rubric IDs do not cover the complete fixed snapshot")
        elif any(isinstance(band, bool) or not isinstance(band, (int, float))
                 or not math.isfinite(band) or not 0 <= band <= 4 for band in bands.values()):
            reasons.append("Rubric bands must be finite numbers from 0 through 4")
        if reasons:
            incomplete[channel] = reasons
    return incomplete


def score(task, stage, preflight_only=False):
    """Use the recovered scoring chain; infrastructure failures stay unscored."""
    import s12_task
    from grading_contract import InvalidSubmission, validate_task, valid_score
    from score_cli import run_verifier
    task = Path(task).resolve(strict=True)
    incomplete = {}
    with scoring_environment() as settings:
        deliverable = inside(task, "output/" + s12_task.out_name(stage))
        try:
            if not deliverable.is_file():
                raise InvalidSubmission(f"Missing {deliverable.name}")
            try:
                submission = s12_task.load_and_validate_deliverable(str(deliverable), stage)
            except (ValueError, TypeError) as exc:
                raise InvalidSubmission(str(exc)) from exc
            if stage in ("s_3", "s_4e"):
                # Read the on-disk delivery so FC also validates its preview.
                validate_task(task, stage)
            else:
                manifest = read_object(task / "initial_state/manifest.json")
                known = {row["id"] for row in manifest["candidates" if stage == "s_1" else "clips"]}
                chosen = submission["selected" if stage == "s_1" else "order"]
                if len(chosen) != len(set(chosen)) or set(chosen) - known:
                    raise InvalidSubmission("Duplicate or unknown clip IDs")
                if stage == "s_2" and set(chosen) != known:
                    raise InvalidSubmission("Order must contain all supplied clips once")
        except InvalidSubmission as exc:
            return {"final": 0.0, "status": "invalid_submission", "reason": str(exc)}
        if preflight_only:
            return {"final": None, "status": "preflight_passed", "scored": False}
        if stage in ("s_1", "s_2"):
            value, detail, _ = run_verifier(task / "verifier/grade.py", str(task), quiet=True, allow_legacy=True)
        else:
            if settings["S12_JUDGE_PROVIDER"] != "gemini":
                raise ValueError("This continuous-video workflow requires the Gemini provider")
            if not os.environ.get("GOOGLE_AI_STUDIO_API_KEY"):
                raise ValueError("Set GOOGLE_AI_STUDIO_API_KEY for RC/FC scoring")
            for folder in ("output", "evidence"):
                inside(task, folder)
            # The core normally clears these too. Clear them at this boundary so
            # an early core return cannot let earlier channel files satisfy the
            # completeness check for this invocation.
            for path in judge_component_paths(task, stage).values():
                path.unlink(missing_ok=True)
            # The module may already be imported by an offline preflight.
            s12_task.TOOLS = str(ROOT)
            value, detail = s12_task.grade(str(task), stage)
            try:
                detail = json.loads(detail)
            except (ValueError, TypeError):
                pass
            incomplete = incomplete_judge_components(task, stage)
        complete = valid_score(value) and not incomplete
        result = {"final": value if complete else None,
                "status": "scored" if complete else "incomplete", "detail": detail,
                "judge_settings": {k: v for k, v in settings.items() if k in
                                   ("S12_JUDGE_PROVIDER", "S12_GEMINI_MODEL", "S12_JUDGE_TEMP", "S12_JUDGE_SEED")}}
        if incomplete:
            result.update(incomplete_components=incomplete,
                          raw_aggregate=value if valid_score(value) else None)
        return result


def main(argv=None):
    from score_cli import add_task_arguments, task_paths, write_result
    parser = argparse.ArgumentParser(description=__doc__)
    add_task_arguments(parser)
    parser.add_argument("--stage", choices=list(STAGES) + list(WEIGHTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Attach scoring configuration; make no judge requests")
    mode.add_argument("--preflight-only", action="store_true", help="Validate submissions; make no judge requests")
    parser.add_argument("--summary", type=Path, help="Private batch result JSON")
    args = parser.parse_args(argv)
    paths = [Path(p).expanduser().resolve() for p in task_paths(parser, args)]
    if len(set(paths)) != len(paths):
        parser.error("Task paths must be unique")
    rows = []
    for task in paths:
        row = {"task": str(task), "final": None, "status": "in_progress"}
        result_path = None
        try:
            if not args.prepare_only:
                result_path = inside(task, "evidence/task_result.json")
                write_result(result_path, row)
            stage = infer_stage(task, args.stage)
            row["stage"] = stage
            identity = prepare(task, stage)
            row["preparation"] = {"format": identity["format"], "rubric_sha256": identity["rubric_sha256"]}
            row.update({"status": "prepared"} if args.prepare_only else score(task, stage, args.preflight_only))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            row.update(status="error", final=None, error=f"{type(exc).__name__}: {exc}")
        if result_path:
            write_result(result_path, row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("task", "stage", "status", "final", "error") if k in row}), flush=True)
    if args.summary:
        write_result(args.summary, {"format": VERSION, "tasks": rows})
    successful = {"prepared", "preflight_passed"} if args.preflight_only else {"prepared", "scored", "invalid_submission"}
    return int(any(row["status"] not in successful for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
