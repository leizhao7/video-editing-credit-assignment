"""Local prerequisites for the one-film workflow; never authenticate or call a model.

Only the standard library is imported. Installed Python packages are located,
not imported, so this check can explain how to repair a fresh environment.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


SCOPES = {"probe", "build", "review", "preflight", "score", "all"}
BUILD_IMPORTS = ("numpy", "cv2", "librosa", "faster_whisper", "modal")
SCORE_IMPORTS = ("PIL", "numpy", "scipy", "librosa", "faster_whisper", "aiohttp", "soundfile")
PACKAGE_NAMES = {"cv2": "opencv-python", "PIL": "Pillow", "faster_whisper": "faster-whisper"}
BACKENDS = {"claude", "codex", "opencode"}


def _run(command, timeout=10):
    """Run a bounded local inspection without exposing raw tool output."""
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        return None, type(exc).__name__
    if result.returncode:
        return None, f"exit status {result.returncode}"
    return result, None


def _positive_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _duration(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _probe(path, executable):
    result, error = _run([executable, "-v", "error", "-show_entries",
                          "stream=codec_type,duration:format=duration", "-of", "json", str(path)], timeout=30)
    if error:
        return False, f"Local ffprobe inspection failed ({error})."
    try:
        document = json.loads(result.stdout)
        streams = document["streams"]
        if not isinstance(streams, list) or any(not isinstance(row, dict) for row in streams):
            raise ValueError("Invalid streams")
        kinds = {row.get("codec_type") for row in streams}
        duration = _duration(document.get("format", {}).get("duration"))
        if duration is None:
            duration = max((_duration(row.get("duration")) or 0 for row in streams), default=0)
    except (KeyError, TypeError, ValueError, AttributeError):
        return False, "Local ffprobe did not return valid stream metadata."
    missing = {"video", "audio"} - kinds
    if missing:
        return False, "Missing required " + " and ".join(sorted(missing)) + " stream(s)."
    if not duration:
        return False, "Media duration is missing or not finite and positive."
    return True, f"Video and audio streams found; duration {duration:.3f} seconds. English/default-track content is not verified."


def check(config, scope="all"):
    """Return ``name/ok/detail/fix`` records without cloud or model requests.

    ``probe`` checks only the local media-probe prerequisites; ``build`` inspects
    local media, annotation CLI and preprocessing dependencies;
    ``review`` checks quality-audit prerequisites; ``preflight`` needs only the
    standard library and local media tools; ``score`` checks outcome-scoring
    prerequisites. Keys and package discovery are not authentication checks.
    """
    if scope not in SCOPES:
        raise ValueError("scope must be probe, build, review, preflight, score, or all")
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    rows = []

    def add(name, ok, detail, fix=""):
        rows.append({"name": name, "ok": bool(ok), "detail": detail, "fix": "" if ok else fix})

    build = scope in {"build", "all"}
    probe = scope == "probe"
    review = scope in {"review", "all"}
    score = scope in {"score", "all"}
    add("python", sys.version_info >= (3, 12),
        f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}; requires 3.12 or newer.",
        "Create and activate a Python 3.12+ environment, then install the workflow requirements.")

    project = config.get("project")
    try:
        project_ok = bool(project) and Path(project).is_absolute() and Path(project).is_dir()
    except (TypeError, OSError, ValueError):
        project_ok = False
    add("project", project_ok, "Private project directory exists." if project_ok else "Private project directory is missing or invalid.",
        "Initialize a new private project directory outside the repository.")

    executables = {}
    for name in ("ffmpeg", "ffprobe") + (("tesseract",) if build else ()):
        executable = shutil.which(name)
        executables[name] = executable
        add(name, bool(executable), f"{name} is on PATH." if executable else f"{name} is not on PATH.",
            f"Install {name} and make its executable available on PATH.")

    if build:
        tesseract = executables.get("tesseract")
        result, error = _run([tesseract, "--list-langs"]) if tesseract else (None, "executable missing")
        languages = set((result.stdout + "\n" + result.stderr).split()) if result else set()
        add("tesseract_eng", "eng" in languages,
            "English OCR language data is available." if "eng" in languages else "English OCR language data could not be confirmed.",
            "Install Tesseract English language data (tesseract-ocr-eng on Ubuntu), then run tesseract --list-langs.")

        agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
        backend = agent.get("backend")
        model = agent.get("model")
        model_ok = isinstance(model, str) and bool(model.strip())
        if backend == "opencode" and model_ok:
            model_ok = "/" in model and all(model.split("/", 1))
        add("annotation_model", backend in BACKENDS and model_ok,
            "Annotation backend and model are configured; model access is unverified." if backend in BACKENDS and model_ok
            else "Choose a supported annotation backend and a model ID.",
            "Set agent.backend to claude, codex, or opencode and agent.model to an available model (provider/model for OpenCode).")
        cli = agent.get("cli") or backend
        executable = shutil.which(os.path.expanduser(cli)) if isinstance(cli, str) and cli else None
        if executable and backend in BACKENDS:
            result, error = _run([executable, "--version"])
            match = re.search(r"(?<![\w.])v?(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)(?![\w.])",
                              result.stdout + "\n" + result.stderr) if result else None
            version = match.group(1) if match else None
            compatible = result is not None and (backend != "opencode" or (version is not None and version.split(".")[0] == "1"))
            detail = (f"Local {backend} version: {version or 'not detected'}. Authentication and model access are unverified."
                      if result else f"Annotation CLI version inspection failed ({error}).")
        else:
            compatible, detail = False, "Annotation CLI executable is unavailable or its backend is unsupported."
        add("annotation_cli", compatible, detail,
            "Install the selected CLI and authenticate yourself. This adapter requires OpenCode 1.x; agent.cli must be an executable path/name, not a shell command.")
        add("modal_volume", isinstance(config.get("modal_volume"), str) and bool(config["modal_volume"].strip()),
            "Modal volume is configured; account access and volume existence are unverified." if config.get("modal_volume") else "Modal volume is not configured.",
            "Choose a dedicated Modal volume in the project configuration and initialize your Modal account.")
        timeout = config.get("annotation_timeout_s")
        add("annotation_timeout", timeout is None or _positive_number(timeout),
            "Annotation timeout is configured (or disabled with null)." if timeout is None or _positive_number(timeout) else "Annotation timeout must be positive or null.",
            "Set annotation_timeout_s to a positive number of seconds, or null to disable it.")

    dependencies = (set(BUILD_IMPORTS if build else ()) | set(SCORE_IMPORTS if score else ())
                    | ({"numpy", "cv2"} if probe else set()))
    requirements = ("data_construction/requirements.txt -r data_construction/requirements-modal.txt -r evaluation/requirements.txt"
                    if build else ("data_construction/requirements.txt" if probe else "evaluation/requirements.txt"))
    for module in sorted(dependencies):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            found = False
        package = PACKAGE_NAMES.get(module, module)
        add("dependency:" + module, found,
            f"{package} is discoverable; runtime loading has not been tested." if found else f"{package} is missing from this Python environment.",
            f"From the repository root run: python -m pip install -r {requirements}")

    inputs = config.get("inputs") if isinstance(config.get("inputs"), dict) else {}
    for name in ("movie", "trailer"):
        item = inputs.get(name) if isinstance(inputs.get(name), dict) else {}
        raw = item.get("path")
        try:
            path = Path(raw) if isinstance(raw, str) and raw else None
            readable = bool(path and path.is_absolute() and path.is_file() and path.stat().st_size > 0 and os.access(path, os.R_OK))
        except (OSError, ValueError):
            readable = False
        add("media:" + name, readable,
            f"{name.capitalize()} is a readable, nonempty local file." if readable else f"{name.capitalize()} is missing, empty, unreadable, or not an absolute local path.",
            "Initialize the project using existing local movie and trailer files; keep the source files available.")
        if (build or probe) and readable:
            if executables.get("ffprobe"):
                ok, detail = _probe(path, executables["ffprobe"])
            else:
                ok, detail = False, "Media metadata was not checked because ffprobe is missing."
            add("media:" + name + ":probe", ok, detail,
                "Check this file with ffprobe. Supply a playable movie/trailer with video and audio, preferably English on the default track.")

    if review or score:
        present = bool(os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "").strip())
        add("gemini_key", present,
            "GOOGLE_AI_STUDIO_API_KEY is present; validity, quota, and model access are unverified." if present else "GOOGLE_AI_STUDIO_API_KEY is not set.",
            "Set GOOGLE_AI_STUDIO_API_KEY in your shell; do not put the credential in project configuration or Git.")
    for active, name in ((review, "quality_model"), (score, "judge_model")):
        if active:
            value = config.get(name)
            ok = isinstance(value, str) and bool(value.strip())
            add(name, ok, f"{name} is configured; availability has not been checked." if ok else f"{name} is not configured.",
                f"Set {name} to a Gemini model available to your account that supports the required media.")
    if review:
        timeout = config.get("quality_timeout_s")
        add("quality_timeout", _positive_number(timeout),
            "Quality timeout is positive; it applies per HTTP operation." if _positive_number(timeout) else "Quality timeout must be a finite positive number.",
            "Set quality_timeout_s to a positive number of seconds.")

    add("authentication_scope", True,
        "Local checks only: no cloud/model requests or login commands were run. CLI/Modal authentication, Gemini credentials, model access, and annotation accuracy are not confirmed.")
    return rows
