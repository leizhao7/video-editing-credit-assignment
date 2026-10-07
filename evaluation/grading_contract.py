"""Trusted, task-bound delivery validation shared by every v5 scoring entry.

InvalidSubmission is policy output (reward zero). Corpus/decoder failures are
RuntimeError and must not be converted into a policy score. No answer keys here.
"""
from functools import lru_cache
import json
import math
import os
import re
from pathlib import Path
import subprocess

try:
    # Standalone grader image copies the same source next to these scripts.
    from submission_policy import duration_tolerance
except ModuleNotFoundError as exc:
    if exc.name != 'submission_policy':
        raise
    # Local standalone scripts start with grader_v5, not the repository, on
    # sys.path. Resolve the shared source from this trusted file's directory.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from opencode_rl.submission_policy import duration_tolerance

VERSION = "s34-grading-v5"
TIME_EPS = 0.05  # one 24fps frame plus container rounding; not editing tolerance
DIRECT_JUDGE_CONTRACT = "s34-direct-judge-v1"


def judge_only():
    """Trusted deployment policy, never selected by model-authored artifacts."""
    return os.environ.get("S12_TRAINING_JUDGE_ONLY") == "1"


def submission_checks():
    """Submission validity and LLM consistency review are independent policies."""
    return not judge_only() or os.environ.get("S12_TRAINING_SUBMISSION_CHECKS") == "1"


def direct_band_response(value):
    """Read an unambiguous rubric score; explanation formatting is not a gate."""
    text = (value or "").strip()
    bands = re.findall(r"\bBAND\s*:\s*([0-4])(?![\w.])", text, re.IGNORECASE)
    if not bands and re.fullmatch(r"[0-4]", text):
        bands = [text]
    if not bands or len(set(bands)) != 1:
        return None, "", ""
    return int(bands[0]), "", text


class InvalidSubmission(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


@lru_cache(maxsize=4096)
def _probe(path, size, mtime_ns):
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                          capture_output=True, text=True, timeout=30)
    if proc.returncode:
        raise RuntimeError(f"trusted media probe failed: {path}: {proc.stderr[-300:]}")
    try:
        result = json.loads(proc.stdout)
        ends = []
        for stream in result["streams"]:
            if stream.get("codec_type") not in ("video", "audio"):
                continue
            duration = float(stream.get("duration", result.get("format", {}).get("duration")))
            start = float(stream.get("start_time", 0))
            if not math.isfinite(duration) or not 0 < duration <= 3600 or not math.isfinite(start):
                raise ValueError("duration")
            ends.append(start + duration)
        if not ends:
            raise ValueError("no audio/video stream")
        # Older ffprobe can include timed metadata/chapter DATA tracks in the
        # container duration (a 7s take can appear to be 7383s). Never let those
        # tracks define usable audio/video time.
        result["av_end_s"] = max(ends)
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError(f"unusable media duration: {path}") from exc
    return result


def media_info(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return _probe(str(path), stat.st_size, stat.st_mtime_ns)


def source_file(task, rel):
    root = (Path(task) / "initial_state").resolve(strict=True)
    path = (root / rel).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise RuntimeError("trusted manifest media path escaped initial_state")
    return path


def duration(task, rel, stream):
    info = media_info(source_file(task, rel))
    rows = [s for s in info["streams"] if s.get("codec_type") == stream]
    if not rows:
        raise RuntimeError(f"trusted source has no {stream}: {rel}")
    # Do not let a longer audio track extend the usable video range.
    raw = rows[0].get("duration", info["format"]["duration"])
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid {stream} duration: {rel}")
    return value


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise InvalidSubmission(f"{label} must be a finite number")
    return float(value)


def interval(row, limit, label):
    start = number(row.get("in_s"), label + ".in_s")
    end = number(row.get("out_s"), label + ".out_s")
    if start < 0 or end <= start or start >= limit or end > limit + TIME_EPS:
        raise InvalidSubmission(f"{label} range [{start}, {end}] outside media [0, {limit}]")


def ids(rows, key, label):
    if not isinstance(rows, list) or any(not isinstance(r, dict) or key not in r for r in rows):
        raise InvalidSubmission(f"{label} must be an array of objects containing {key}")
    result = [r[key] for r in rows]
    if any(not isinstance(x, (str, int)) or isinstance(x, bool) for x in result):
        raise InvalidSubmission(f"invalid {label} id")
    if len(result) != len(set(result)):
        raise InvalidSubmission(f"duplicate {label} id")
    return result


def validate_task(task, stage, submission=None):
    if stage not in ("s_3", "s_4e"):
        raise RuntimeError(f"unsupported v5 stage: {stage}")
    if not submission_checks():
        # Training judges assess the submitted edit. Do not veto their scores
        # with a second task-compliance/schema/duration review. File access and
        # bounded transport remain enforced at their actual I/O boundaries.
        return {"contract": DIRECT_JUDGE_CONTRACT, "valid": None,
                "submission_validation": "disabled_by_training_policy"}
    local_submission = submission is None
    name = "cuts.json" if stage == "s_3" else "edit.json"
    if submission is None:
        try:
            submission = read_json(Path(task) / "output" / name)
        except (ValueError, FileNotFoundError) as exc:
            raise InvalidSubmission(f"missing or malformed {name}") from exc
    if not isinstance(submission, dict):
        raise InvalidSubmission("deliverable must be an object")
    from s12_task import validate_deliverable_data
    try:
        validate_deliverable_data(submission, stage)
    except ValueError as exc:
        raise InvalidSubmission(str(exc)) from exc
    try:
        man = read_json(Path(task) / "initial_state/manifest.json")
    except (OSError, ValueError) as exc:
        raise RuntimeError("trusted task manifest is unreadable or malformed") from exc
    if stage == "s_3":
        expected = [c["id"] for c in man["clips"]]
        got = ids(submission.get("cuts"), "clip", "cuts")
        if not expected or got != expected:
            raise InvalidSubmission("cuts must contain every supplied clip exactly once, in supplied order")
        for cut, source in zip(submission["cuts"], man["clips"]):
            interval(cut, duration(task, source["file"], "video"), str(cut["clip"]))
        target = man.get("target_duration_s")
        if target is not None:
            tolerance = duration_tolerance(target)
            actual = sum(float(c["out_s"]) - float(c["in_s"]) for c in submission["cuts"])
            if abs(actual - target) > tolerance + 1e-6:
                raise InvalidSubmission(f"cuts duration {actual:.3f}s outside {target:.3f}s ±{tolerance:.3f}s")
    elif stage == "s_4e":
        shots = {s["id"]: s for s in man["shots"]}
        pool = {d["id"]: d for d in man.get("dialogue", [])}
        expected_joins = [j["join"] for j in man["joins"]]
        got = ids(submission.get("joins", []), "join", "joins")
        if set(got) != set(expected_joins):
            raise InvalidSubmission("joins must cover every supplied join exactly once")
        for join in submission.get("joins", []):
            if join.get("picture") not in ("cut", "dip_to_black", "dissolve", "other") or join.get("sound") not in ("hard", "under", "prelap", "carries_over"):
                raise InvalidSubmission("invalid join type")
        for row in submission.get("dialogue", []):
            if not isinstance(row, dict) or row.get("take") not in pool:
                raise InvalidSubmission("dialogue references an unknown take")
            d = pool[row["take"]]
            interval(row, duration(task, d["file"], "audio"), row["take"])
            number(row.get("at_s"), "dialogue.at_s")
        # Reusing a take at different times can be legitimate; duplicate sound
        # layers for one shot are also not rejected just for sharing a source.
        for row in submission.get("production_sound", []):
            if not isinstance(row, dict) or row.get("shot") not in shots:
                raise InvalidSubmission("production_sound references an unknown shot")
            if "keep" in row and not isinstance(row["keep"], bool):
                raise InvalidSubmission("production_sound.keep must be boolean")
            if row.get("keep") is False:
                continue
            s = shots[row["shot"]]
            if not s.get("has_sound"):
                raise InvalidSubmission("cannot keep nonexistent production sound")
            timed = [k in row for k in ("in_s", "out_s", "at_s")]
            if any(timed):
                if not all(timed) or not s.get("handle"):
                    raise InvalidSubmission("timed production sound requires a supplied handle and all time fields")
                interval(row, duration(task, f"handles/{s['id']}.m4a", "audio"), s["id"])
                number(row["at_s"], "production_sound.at_s")
        card_ids = ids(submission.get("cards", []), "shot", "cards")
        required = {s["id"] for s in shots.values() if s.get("build_it") or s.get("origin") == "to_build"}
        if not set(card_ids) <= set(shots) or not required <= set(card_ids):
            raise InvalidSubmission("cards must reference supplied shots and include every required title card")
        for card in submission.get("cards", []):
            if not isinstance(card.get("text"), str) or not card["text"].strip():
                raise InvalidSubmission("title card text is empty")
        music = submission.get("music")
        if music is not None:
            if not isinstance(music, dict):
                raise InvalidSubmission("music must be an object")
            if music.get("starts_at_s") is not None:
                start = number(music["starts_at_s"], "music.starts_at_s")
                if not 0 <= start < duration(task, "music.wav", "audio"):
                    raise InvalidSubmission("music starts beyond the supplied cue")
        if local_submission:
            preview = Path(task) / "output/preview.mp4"
            if not preview.is_file():
                raise InvalidSubmission("missing output/preview.mp4")
            expected = man.get("stretch_dur_s")
            if expected is None:
                expected = sum(float(s["dur_s"]) for s in shots.values())
            tolerance = duration_tolerance(expected)
            try:
                info = media_info(preview)
            except RuntimeError as exc:
                raise InvalidSubmission("submitted preview cannot be probed") from exc
            video = next((s for s in info["streams"] if s.get("codec_type") == "video"), None)
            if video is None:
                raise InvalidSubmission("submitted preview has no video stream")
            actual = info["av_end_s"] - float(video.get("start_time", 0))
            if abs(actual - expected) > tolerance + 1e-6:
                raise InvalidSubmission(f"preview duration {actual:.3f}s outside {expected:.3f}s ±{tolerance:.3f}s")
    else:
        raise RuntimeError(f"unsupported v5 stage: {stage}")
    return {"contract": VERSION, "valid": True, "media_time_epsilon_s": TIME_EPS}


def valid_score(value):
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value) and 0 <= value <= 1)
