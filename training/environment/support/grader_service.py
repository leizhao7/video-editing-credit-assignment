"""Trusted v5 grading core, recovered from the experiment service.

Deployment resources are supplied by modal_train.py. No policy-provided
verifier or source media is executed or accepted by this boundary.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

APP_NAME = os.environ.get("EDITING_TRAINING_APP", "editing-training")
logs_volume = None  # Bound to the trusted log volume by the deployment wrapper.

FUNCTION_NAME = "grade_judged"


ASR_MODEL = "small.en"


ALLOWED_ROOTS = {"_tt_train", "_tt_val"}


ALLOWED_STAGES = {"s_3", "s_4e"}


PINNED_JUDGE_PROVIDER = "gemini"


PINNED_JUDGE_MODEL = "gemini-3.1-pro-preview"


PINNED_JUDGE_TEMPERATURE = "0"


PINNED_JUDGE_SEED = "42"


EVIDENCE_TIMEOUT_S = 900


JUDGE_TIMEOUT_S = 390


MAX_DELIVERABLE_BYTES = 2 * 1024 * 1024


MAX_PREVIEW_BYTES = 128 * 1024 * 1024


JSON_ARTIFACTS = (
    "s3_evidence.json", "s3_score.json", "s3_score_ref.json",
    "s3_final.json", "s4e_evidence.json", "s4e_score.json",
    "s4e_score_ref.json", "s4e_final.json", "s3_consistency.json", "s4e_consistency.json",
)


_RUNTIME_CAPABILITIES = None


GRADER_PYTHON = "/usr/local/bin/python"


class InvalidSubmission(ValueError):
    """A policy-authored artifact is invalid and therefore earns reward zero."""


def _runtime_capabilities() -> dict[str, Any]:
    global _RUNTIME_CAPABILITIES
    if _RUNTIME_CAPABILITIES is not None:
        return _RUNTIME_CAPABILITIES
    required_modules = (
        "faster_whisper", "ctranslate2", "librosa", "numpy", "scipy", "soundfile"
    )
    binaries = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    probe = (
        "import importlib.metadata as m,importlib.util as u,json,sys;"
        f"mods={required_modules!r};"
        "missing=[x for x in mods if u.find_spec(x) is None];"
        "versions={};"
        "[(versions.__setitem__(x, "
        "m.version('faster-whisper' if x=='faster_whisper' else x))) "
        "for x in mods if x not in missing];"
        "print(json.dumps({'python':sys.executable,'missing':missing,'versions':versions}))"
    )
    checked = subprocess.run(
        [GRADER_PYTHON, "-c", probe], capture_output=True, text=True, timeout=60
    )
    try:
        probed = json.loads(checked.stdout) if checked.returncode == 0 else {}
    except json.JSONDecodeError:
        probed = {}
    missing = probed.get("missing", list(required_modules))
    if checked.returncode != 0 or missing or not all(binaries.values()):
        raise RuntimeError(
            "grader runtime incomplete: "
            f"python={GRADER_PYTHON!r}, rc={checked.returncode}, "
            f"missing_modules={missing}, binaries={binaries}, "
            f"stderr={checked.stderr[-300:]!r}"
        )
    _RUNTIME_CAPABILITIES = {
        "format": "s12-grader-runtime-v2",
        "modal_python": sys.executable,
        "grader_python": probed.get("python"),
        "modules": probed.get("versions", {}),
        "binaries": binaries,
        "asr_model": ASR_MODEL,
        "asr_failure_policy": "fail_closed",
        "required_evidence_health": "s12-evidence-health-v3",
        "s4e_asr_contract": "s12-asr-v5-render-clock",
    }
    return _RUNTIME_CAPABILITIES


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(raw: Any, *, label: str) -> str:
    value = str(raw or "")
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _task_source(payload: dict[str, Any]) -> str:
    root = _safe_component(payload.get("source_root"), label="source_root")
    task = _safe_component(payload.get("task"), label="task")
    if root not in ALLOWED_ROOTS:
        raise ValueError(f"source_root not allowed: {root!r}")
    source = os.path.realpath(f"/data/{root}/{task}")
    expected_root = os.path.realpath(f"/data/{root}")
    if os.path.commonpath((expected_root, source)) != expected_root:
        raise ValueError("task path escaped source root")
    if not os.path.isdir(source):
        raise FileNotFoundError(source)
    return source


def _canonical_task_stage(source: str) -> str:
    """Read the trusted corpus identity instead of trusting the caller's label."""
    path = os.path.join(source, "task.json")
    if not os.path.isfile(path):
        raise ValueError(f"trusted task is missing task.json: {source}")
    with open(path, encoding="utf-8") as handle:
        task = json.load(handle)
    raw = str(task.get("stage") or "").split("/", 1)[0].lower()
    aliases = {"s3": "s_3", "s_3": "s_3", "s4e": "s_4e", "s_4e": "s_4e"}
    stage = aliases.get(raw)
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"trusted task has unsupported stage: {task.get('stage')!r}")
    return stage


def _expected_preview_duration(source: str) -> float:
    path = os.path.join(source, "initial_state", "manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    value = manifest.get("stretch_dur_s")
    if value is None:
        value = sum(
            float(shot.get("dur_s") or 0)
            for shot in (manifest.get("shots") or [])
            if isinstance(shot, dict)
        )
    from grading_contract import duration_tolerance
    duration_tolerance(value)  # reject malformed trusted data, never model zero
    return float(value)


def _validate_preview(path: str, expected_duration_s: float) -> float:
    """Recheck the sanitized preview at the grading boundary.

    The rollout sandbox is still responsible for decoding/re-encoding untrusted
    media.  This second check makes offline regrades obey the same duration and
    video-stream contract rather than relying on a caller assertion.
    """
    from grading_contract import duration_tolerance
    tolerance = duration_tolerance(expected_duration_s)
    checked = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, text=True, timeout=60,
    )
    if checked.returncode != 0:
        raise InvalidSubmission(f"preview ffprobe failed: {checked.stderr[-300:]}")
    try:
        info = json.loads(checked.stdout)
        streams = info.get("streams") or []
        video = next((s for s in streams if s.get('codec_type') == 'video'), None)
        av = [s for s in streams if s.get('codec_type') in ('video', 'audio')]
        origin = float(video.get('start_time', 0)) if video else 0.
        ends = [float(s.get('start_time', 0)) + float(s.get('duration', (info.get('format') or {}).get('duration')))
                for s in av]
        if not ends or not all(math.isfinite(e) for e in ends) or not math.isfinite(origin):
            raise ValueError('nonfinite A/V timestamps')
        duration = max(ends) - origin
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InvalidSubmission("preview ffprobe returned no finite format duration") from exc
    if video is None or not math.isfinite(duration) or duration <= 0:
        raise InvalidSubmission(
            "preview must contain a decodable video stream and finite duration"
        )
    if abs(duration - expected_duration_s) > tolerance + 1e-6:
        raise InvalidSubmission(
            f"preview duration {duration:.3f}s is outside trusted task duration "
            f"{expected_duration_s:.3f}s±{tolerance:.3f}s"
        )
    decode = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", path,
                             "-map", "0:v:0", "-map", "0:a:0?", "-threads", "1",
                             "-f", "null", "-"], capture_output=True, text=True, timeout=120)
    if decode.returncode:
        raise InvalidSubmission("preview decode failed: " + decode.stderr[-300:])
    return duration


def _write_input(task_dir: str, source: str, payload: dict[str, Any]) -> str:
    from grading_contract import submission_checks
    stage = str(payload.get("stage") or "")
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"unsupported stage: {stage!r}")
    trusted_stage = _canonical_task_stage(source)
    if stage != trusted_stage:
        raise ValueError(
            f"payload stage {stage!r} does not match trusted task stage {trusted_stage!r}"
        )
    deliverable_name = "cuts.json" if stage == "s_3" else "edit.json"
    raw = payload.get("deliverable_json")
    if not isinstance(raw, str) or not raw.strip():
        raise InvalidSubmission("deliverable_json must be a non-empty string")
    encoded = raw.encode("utf-8")
    if len(encoded) > MAX_DELIVERABLE_BYTES:
        raise InvalidSubmission(f"deliverable too large: {len(encoded)}")
    # Parse before writing so malformed/non-standard JSON never reaches a
    # grader process.  Python's default decoder accepts NaN/Infinity even
    # though they are not JSON and can poison time arithmetic downstream.
    def reject_constant(value: str):
        raise InvalidSubmission(f"non-finite JSON constant: {value}")

    try:
        decoded = json.loads(raw, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise InvalidSubmission(f"malformed deliverable JSON: {exc}") from exc
    # ``s12_task`` is the same file used by the live Env.  Keeping the schema
    # there prevents remote regrades and rollouts from drifting apart.
    import s12_task
    if not isinstance(decoded, dict):
        raise InvalidSubmission("deliverable JSON must be readable as an object")
    if submission_checks():
        try:
            s12_task.validate_deliverable_data(decoded, stage)
        except ValueError as exc:
            raise InvalidSubmission(f"deliverable schema invalid: {exc}") from exc
    from grading_contract import validate_task, InvalidSubmission as ContractInvalid
    try:
        validate_task(source, stage, decoded)
    except ContractInvalid as exc:
        raise InvalidSubmission(str(exc)) from exc
    os.makedirs(os.path.join(task_dir, "output"), exist_ok=True)
    with open(os.path.join(task_dir, "output", deliverable_name), "wb") as handle:
        handle.write(encoded)

    preview = payload.get("preview_bytes")
    if stage == "s_4e":
        if not isinstance(preview, bytes) or not preview:
            raise InvalidSubmission("S4E requires non-empty sanitized preview bytes")
        if len(preview) > MAX_PREVIEW_BYTES:
            raise InvalidSubmission(f"preview too large: {len(preview)}")
        with open(os.path.join(task_dir, "output", "preview.mp4"), "wb") as handle:
            handle.write(preview)
        if submission_checks():
            _validate_preview(
                os.path.join(task_dir, "output", "preview.mp4"),
                _expected_preview_duration(source),
            )
    elif preview not in (None, b""):
        raise InvalidSubmission("S3 payload unexpectedly contains preview bytes")
    return deliverable_name


def _reconstruct_task(payload: dict[str, Any]) -> str:
    source = _task_source(payload)
    # One input per container makes a stable path safe.  The evidence JSON is
    # embedded in the LLM prompt and includes its task path; a random mkdtemp
    # suffix would otherwise perturb an irrelevant token on every repeat.
    task_dir = "/tmp/s12_remote_grade_task"
    shutil.rmtree(task_dir, ignore_errors=True)
    os.makedirs(task_dir)
    try:
        if os.path.isdir(os.path.join(source, "initial_state")):
            os.symlink(
                os.path.join(source, "initial_state"),
                os.path.join(task_dir, "initial_state"),
            )
        for filename in ("probe.json", "task_prompt.md", "task.json"):
            src = os.path.join(source, filename)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(task_dir, filename))
        verifier = os.path.join(source, "verifier")
        if os.path.isdir(verifier):
            shutil.copytree(verifier, os.path.join(task_dir, "verifier"))
        _write_input(task_dir, source, payload)
        return task_dir
    except BaseException:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise


def _service_contract() -> dict[str, Any]:
    from grading_contract import judge_only, submission_checks, DIRECT_JUDGE_CONTRACT
    graders = {'@deployment': _sha256('/opt/editing-release/training/modal_train.py')}
    for path in sorted(glob.glob("/opt/s12tools/*.py")):
        graders[os.path.basename(path)] = _sha256(path)
    service_path = os.path.realpath(__file__)
    return {
        "format": "s12-remote-grade-contract-v5",
        "grading_contract": "s34-grading-v5",
        "video_contract": "s34-v5-continuous-video-1",
        "consistency_contract": DIRECT_JUDGE_CONTRACT if judge_only() else "s34-joint-consistency-v3",
        "review_policy": "judge_only" if judge_only() else "reviewed",
        "submission_validation": "enabled" if submission_checks() else "disabled",
        "mode": "public-training-service",
        "app": APP_NAME,
        "function": FUNCTION_NAME,
        "service_sha256": _sha256(service_path),
        "grader_sha256": graders,
        "harness_sha256": _sha256("/opt/SkyRL/s12_task.py"),
        "runtime": _runtime_capabilities(),
        "transport": (
            "shared-schema-validated deliverable JSON plus sanitized preview bytes; "
            "stage and preview duration rebound to trusted release task"
        ),
        "judge": {
            "provider": PINNED_JUDGE_PROVIDER,
            "model": PINNED_JUDGE_MODEL,
            "temperature": PINNED_JUDGE_TEMPERATURE,
            "seed": PINNED_JUDGE_SEED,
        },
        "duration_tolerance_policy": "max(1s,min(3s,10% target))" if submission_checks() else None,
    }


def grade_judged(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    # Shared absolute deadline survives outer retries and Modal queueing.
    deadline = float(payload.get('grading_deadline_unix', time.time() + 600))
    if not math.isfinite(deadline):
        raise ValueError('Invalid grading deadline')
    os.environ['S12_GRADING_DEADLINE_UNIX'] = str(min(deadline, time.time() + 600))
    _runtime_capabilities()
    # Bound numerical libraries in the evidence process.  FFmpeg calls in the
    # checked-in scripts inherit the same four-vCPU function cgroup.
    requested = {
        "judge_provider": str(payload.get("judge_provider") or PINNED_JUDGE_PROVIDER),
        "judge_model": str(payload.get("judge_model") or PINNED_JUDGE_MODEL),
        "judge_temperature": str(
            payload.get("judge_temperature")
            if payload.get("judge_temperature") is not None
            else PINNED_JUDGE_TEMPERATURE
        ),
        "judge_seed": str(payload.get("judge_seed") or PINNED_JUDGE_SEED),
        "evidence_timeout_s": str(
            payload.get("evidence_timeout_s") or EVIDENCE_TIMEOUT_S
        ),
        "judge_timeout_s": str(payload.get("judge_timeout_s") or JUDGE_TIMEOUT_S),
    }
    pinned = {
        "judge_provider": PINNED_JUDGE_PROVIDER,
        "judge_model": PINNED_JUDGE_MODEL,
        "judge_temperature": PINNED_JUDGE_TEMPERATURE,
        "judge_seed": PINNED_JUDGE_SEED,
        "evidence_timeout_s": str(EVIDENCE_TIMEOUT_S),
        "judge_timeout_s": str(JUDGE_TIMEOUT_S),
    }
    mismatched = {key: (requested[key], expected) for key, expected in pinned.items()
                  if requested[key] != expected}
    if mismatched:
        raise ValueError(f"remote grader configuration is pinned; mismatches={mismatched}")

    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "S12_EVIDENCE_CONCURRENCY": "1",
        "S12_EVIDENCE_CPUS_PER_SLOT": "4",
        "S12_EVIDENCE_GATE_DIR": f"/tmp/evidence-gate-{os.getpid()}",
        "S12_ASR_CACHE_DIR": "/asr-cache",
        "S12_TOOLS_DIR": "/opt/s12tools",
        "S12_GRADER_PYTHON": GRADER_PYTHON,
        "S12_JUDGE_CONCURRENCY": "1",
        "S12_JUDGE_PROVIDER": PINNED_JUDGE_PROVIDER,
        "S12_GEMINI_MODEL": PINNED_JUDGE_MODEL,
        "S12_JUDGE_TEMP": PINNED_JUDGE_TEMPERATURE,
        "S12_JUDGE_SEED": PINNED_JUDGE_SEED,
        "S12_EVIDENCE_TIMEOUT_S": str(EVIDENCE_TIMEOUT_S),
        "S12_JUDGE_TIMEOUT_S": str(JUDGE_TIMEOUT_S),
    })
    sys.path.insert(0, "/opt/SkyRL")
    import s12_task

    import uuid
    audit_dir = "/logs/grader_v5/" + uuid.uuid4().hex
    os.makedirs(audit_dir, exist_ok=False)
    metadata = {k: payload.get(k) for k in ("task", "stage", "source_root")}
    raw = str(payload.get("deliverable_json") or "").encode()
    metadata["deliverable_sha256"] = hashlib.sha256(raw).hexdigest()
    if len(raw) <= MAX_DELIVERABLE_BYTES:
        Path(audit_dir, "submitted.json").write_bytes(raw)
    preview = payload.get("preview_bytes")
    if isinstance(preview, bytes):
        metadata["preview_sha256"] = hashlib.sha256(preview).hexdigest()
        if len(preview) <= MAX_PREVIEW_BYTES:
            Path(audit_dir, "submitted_preview.mp4").write_bytes(preview)
    source = _task_source(payload)
    metadata["source_sha256"] = {name: _sha256(os.path.join(source, name))
        for name in ("initial_state/manifest.json", "task_prompt.md", "probe.json", "verifier/answer.json")
        if os.path.isfile(os.path.join(source, name))}
    Path(audit_dir, "input_contract.json").write_text(json.dumps(metadata))
    logs_volume.commit()
    try:
        task_dir = _reconstruct_task(payload)
    except InvalidSubmission as exc:
        result = {
            "format": "s12-remote-grade-result-v1",
            "task": payload.get("task"),
            "stage": str(payload.get("stage") or ""),
            "score": 0.0,
            "detail": json.dumps({
                "final": 0.0,
                "invalid_submission": True,
                "reason": str(exc),
            }, ensure_ascii=False),
            "invalid_submission": True,
            "invalid_reason": str(exc),
            "seconds": time.monotonic() - started,
            "judge_files": {},
            "service_contract": _service_contract(),
            "input_contract": metadata,
            "audit_directory": audit_dir,
        }
        print(
            "S12_REMOTE_GRADE_INVALID="
            + json.dumps({
                "task": result["task"], "stage": result["stage"],
                "reason": result["invalid_reason"],
            }, sort_keys=True),
            flush=True,
        )
        Path(audit_dir, "result.json").write_text(json.dumps(result, ensure_ascii=False))
        logs_volume.commit()
        return result
    try:
        stage = str(payload["stage"])
        score, detail = s12_task._run_judge_chain(  # noqa: SLF001
            task_dir, *s12_task.JUDGED[stage]
        )
        artifacts = {}
        evidence_dir = os.path.join(task_dir, "evidence")
        for filename in JSON_ARTIFACTS:
            path = os.path.join(evidence_dir, filename)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as handle:
                    artifacts[filename] = json.load(handle)
        result = {
            "format": "s12-remote-grade-result-v1",
            "task": payload.get("task"),
            "stage": stage,
            "score": score,
            "detail": detail,
            "seconds": time.monotonic() - started,
            "judge_files": artifacts,
            "service_contract": _service_contract(),
            "input_contract": metadata,
            "audit_directory": audit_dir,
        }
        with open(os.path.join(task_dir, "result.json"), "w") as handle:
            json.dump(result, handle, ensure_ascii=False)
        print(
            "S12_REMOTE_GRADE="
            + json.dumps({
                "task": result["task"], "stage": stage, "score": score,
                "seconds": round(result["seconds"], 3),
                "artifact_count": len(artifacts),
            }, sort_keys=True),
            flush=True,
        )
        return result
    finally:
        # Persist successful AND failed attempt artifacts, including exact LLM
        # messages with inline audio/images, before deleting private scratch.
        for name in ("evidence", "output", "result.json", "task_prompt.md"):
            src = os.path.join(task_dir, name)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(audit_dir, name))
            elif os.path.isfile(src):
                shutil.copy2(src, os.path.join(audit_dir, name))
        logs_volume.commit()
        shutil.rmtree(task_dir, ignore_errors=True)
