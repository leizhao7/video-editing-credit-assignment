"""Filesystem and explicit optional-service configuration for the release draft.

Importing this module neither creates directories nor calls a model/provider.
"""
from pathlib import Path
from contextlib import contextmanager
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time


def env_text(name, default=""):
    """Treat unset and whitespace-only optional configuration consistently."""
    return os.environ.get(name, "").strip() or default


def env_positive_int(name, default):
    try:
        value = int(env_text(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


CODE_ROOT = str(Path(__file__).resolve().parent)
DATA_ROOT = str(Path(env_text("VIDEO_EDITING_DATA_ROOT", "data/private")).expanduser().resolve())
ROOT = DATA_ROOT


def require_target(parser, args):
    """Do not silently process all private inputs when no target was selected."""
    if getattr(args, "gc", False):
        return  # Cache cleanup has no film-selection or worker semantics.
    for key in ("film", "films"):
        value = getattr(args, key, None)
        if value is not None:
            if not value.strip():
                parser.error(f"--{key} must not be blank")
            setattr(args, key, value.strip())
    targets = sum(bool(getattr(args, key, False)) for key in ("film", "films", "all", "gc"))
    if not targets:
        parser.error("select --film, --films, or --all explicitly")
    if targets > 1:
        parser.error("select only one of --film, --films, --all, or --gc")
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be positive")


def validate_slug(value):
    """A slug is one directory component, never a relative or absolute path."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Invalid film slug: {value!r}")
    return value


def validate_data_slug(root, value):
    slug = validate_slug(value)
    root = Path(root).resolve()
    for relative in (f"exports/{slug}", f"edit/task_{slug}", f"edit/runs/{slug}"):
        try:
            (root / relative).resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Film output path escapes the private data root: {relative}") from exc
    return slug


def export_slug(probe, path):
    slug = validate_slug(probe.get("slug"))
    if Path(path).parent.name != slug:
        raise ValueError(f"Probe slug does not match its export directory: {path}")
    validate_data_slug(Path(path).parent.parent.parent, slug)
    return slug


def select_films(films, args):
    """Resolve one unambiguous film, an explicit slug list, or all supplied films."""
    if getattr(args, "films", None):
        requested = list(dict.fromkeys(line.strip() for line in
            Path(args.films).expanduser().read_text(encoding="utf-8").splitlines() if line.strip()))
        if not requested:
            raise ValueError("Film list is empty")
        for slug in requested:
            validate_slug(slug)
        missing = [slug for slug in requested if slug not in films]
        if missing:
            raise ValueError(f"Unknown film slugs: {missing}")
        return {slug: films[slug] for slug in requested}
    query = getattr(args, "film", None)
    if query:
        if query in films:
            return {query: films[query]}
        hits = {slug: name for slug, name in films.items() if query.lower() in name.lower()}
        if len(hits) != 1:
            raise ValueError(f"--film {query!r} must match exactly one film; found {len(hits)}")
        return hits
    return films


def annotation_backend():
    backend = env_text("VIDEO_EDITING_ANNOTATION_BACKEND", "claude").lower()
    if backend not in {"claude", "codex", "opencode"}:
        raise ValueError("VIDEO_EDITING_ANNOTATION_BACKEND must be claude, codex, or opencode")
    return backend


def annotation_cli():
    return os.path.expanduser(env_text("VIDEO_EDITING_ANNOTATION_CLI", annotation_backend()))


@contextmanager
def annotation_process(command, cwd, stderr_path):
    """Stream stdout, with an optional total deadline independent of CLI output."""
    timeout_name = "VIDEO_EDITING_ANNOTATION_TIMEOUT_S"
    timeout_text = os.environ.get(timeout_name)
    timeout = None
    if timeout_text is not None:
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise ValueError(f"{timeout_name} must be a finite positive number of seconds") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{timeout_name} must be a finite positive number of seconds")

    with open(stderr_path, "w", encoding="utf-8") as errors:
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=errors,
                                   text=True, bufsize=1, start_new_session=True)
        timed_out = threading.Event()
        cancelled = threading.Event()
        timer = None

        def signal_group(sig):
            # start_new_session gives this invocation its own group, including CLI children.
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        def watch_deadline():
            remaining = timeout
            # Event.wait has a platform-dependent maximum, even for finite floats.
            while remaining > 0:
                interval = min(remaining, threading.TIMEOUT_MAX)
                started = time.monotonic()
                if cancelled.wait(interval):
                    return
                remaining -= time.monotonic() - started
            if process.poll() is not None:
                # A CLI may exit while its child still owns the stdout pipe.
                # Only a vanished group means there is nothing left to stop.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    return
            timed_out.set()
            signal_group(signal.SIGTERM)
            # Give descendants a chance to exit too, even if the leader exits first.
            time.sleep(1)
            signal_group(signal.SIGKILL)

        try:
            if timeout is not None:
                timer = threading.Thread(target=watch_deadline,
                                         name=f"annotation-timeout-{process.pid}", daemon=True)
                timer.start()
            yield process
        finally:
            cancelled.set()
            if timer is not None and timer.ident is not None:
                timer.join()
            # The leader can exit before its children. Always clean this invocation's
            # group when leaving the context, including early caller errors.
            signal_group(signal.SIGKILL)
            process.wait()
            if process.stdout:
                process.stdout.close()
            if timed_out.is_set():
                raise TimeoutError(
                    f"Annotation CLI exceeded the {timeout:g}s total runtime limit "
                    f"({timeout_name}); its process group was terminated. "
                    f"See {stderr_path} and the local session log.")


def require_annotation_model():
    value = env_text("VIDEO_EDITING_ANNOTATION_MODEL")
    if not value:
        raise RuntimeError("Set VIDEO_EDITING_ANNOTATION_MODEL before invoking an annotation step.")
    return value


def cloud_volume():
    value = env_text("VIDEO_EDITING_MODAL_VOLUME")
    if not value:
        raise RuntimeError("Set VIDEO_EDITING_MODAL_VOLUME to your own preprocessing volume.")
    return value


def run_legacy_rubric_hook(stage):
    """Optional user-supplied legacy metadata hook; disabled by default.

    This is not the paper's final grading assembly. No historical writer or
    private configuration is bundled in this release draft.
    """
    variable = "VIDEO_EDITING_LEGACY_RUBRIC_WRITER_" + stage.upper()
    value = os.environ.get(variable, "").strip()
    if not value:
        return False
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{variable} does not point to a Python script")
    subprocess.run([sys.executable, str(path)], cwd=DATA_ROOT, check=True)
    return True
