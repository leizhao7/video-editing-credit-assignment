"""Opt-in real-CLI smoke test using only generated JSON and an RGB image.

Runs the selected authenticated CLI and may incur provider usage. No real media
or annotations are read. Temporary inputs and logs are retained outside the repo.
"""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import secrets
import signal
import struct
import tempfile
import threading
import time
import zlib

try:
    from .annotation_backend import AnnotationEvents, build_annotation_command
    from .release_config import annotation_process
except ImportError:
    from annotation_backend import AnnotationEvents, build_annotation_command
    from release_config import annotation_process


PALETTE = {"red": (255, 0, 0), "green": (0, 200, 0),
           "blue": (0, 0, 255), "yellow": (255, 255, 0)}


def two_color_png(left, right, width=128, height=64):
    """Generate an unlabelled two-panel image without an imaging dependency."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    row = b"\x00" + bytes(left) * (width // 2) + bytes(right) * (width - width // 2)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b""))


def temporary_run_directory():
    """TMPDIR may point into the repo; never put a smoke run there."""
    repo = Path(__file__).resolve().parents[1]
    for candidate in (Path(tempfile.gettempdir()), Path("/tmp")):
        candidate = candidate.resolve()
        if candidate == repo or repo in candidate.parents or not candidate.is_dir():
            continue
        try:
            return Path(tempfile.mkdtemp(prefix="annotation cli smoke ", dir=candidate)).resolve()
        except OSError:
            continue
    raise OSError("No writable temporary directory outside the repository")


@contextmanager
def backend_environment(backend, model, cli=None, effort=None):
    """Explicit smoke arguments take precedence over inherited annotation settings."""
    values = {"VIDEO_EDITING_ANNOTATION_BACKEND": backend,
              "VIDEO_EDITING_ANNOTATION_MODEL": model,
              "VIDEO_EDITING_ANNOTATION_CLI": cli or "",
              "VIDEO_EDITING_ANNOTATION_EFFORT": effort or ""}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_result(path, expected):
    try:
        if path.is_symlink():
            return ["result.json must be a regular file in the task directory"]
        if path.stat().st_size > 4096:
            return ["result.json exceeds the smoke-test size limit"]
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return ["result.json is missing or is not valid UTF-8 JSON"]
    if not isinstance(result, dict) or set(result) != set(expected):
        return ["result.json must contain exactly nonce, total, left_color, right_color"]
    if type(result["total"]) is not int:
        return ["result.json total must be an integer"]
    if result != expected:
        return ["result.json does not match the generated input and image"]
    return []


def run_check(backend, model, *, cli=None, effort=None, timeout=180):
    run_dir = temporary_run_directory()
    task_dir = run_dir / "task workspace"
    task_dir.mkdir()
    logs = run_dir / "logs"
    logs.mkdir()
    rng = secrets.SystemRandom()
    left, right = rng.sample(list(PALETTE), 2)
    payload = {"nonce": secrets.token_hex(16), "numbers": [rng.randrange(1, 100) for _ in range(6)]}
    expected = {"nonce": payload["nonce"], "total": sum(payload["numbers"]),
                "left_color": left, "right_color": right}
    input_path, image_path = task_dir / "input.json", task_dir / "colors.png"
    input_bytes = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    image_bytes = two_color_png(PALETTE[left], PALETTE[right])
    input_path.write_bytes(input_bytes)
    image_path.write_bytes(image_bytes)
    prompt = """Perform this small local tool-use check in the current task directory.
Read input.json. Use your image-viewing tool to visually inspect colors.png, which
has a solid left half and a solid right half. The color names are red, green,
blue, or yellow. Do not decode or analyze the PNG bytes or pixels with code as a
substitute for image viewing. If you cannot view the image, report that failure.
Write result.json with exactly these fields: nonce (copy from input.json), total
(integer sum of its numbers), left_color, right_color (lowercase color names).
Only read the two supplied input files and write result.json. Do not change the
inputs or any other files. Do not use the network, install tools, or delegate.
Complete the file, then stop. Do not merely print the JSON in your final reply.
"""
    (logs / "prompt.txt").write_text(prompt, encoding="utf-8")
    timed_out = threading.Event()
    timer = None
    returncode = None
    errors = []
    started = time.monotonic()
    events = AnnotationEvents(backend)
    try:
        with backend_environment(backend, model, cli, effort):
            command = build_annotation_command(prompt, task_dir, tools="Read,Write,Edit,Bash(python3:*)",
                                               effort="low")
        with annotation_process(command.argv, cwd=task_dir, stderr_path=logs / "stderr.log") as process:
            def terminate():
                timed_out.set()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            timer = threading.Timer(timeout, terminate)
            timer.daemon = True
            timer.start()
            try:
                with (logs / "session.jsonl").open("w", encoding="utf-8") as stream:
                    for line in process.stdout:
                        stream.write(line)
                        stream.flush()
                        events.feed(line)
                returncode = process.wait()
            finally:
                timer.cancel()
                timer.join()
        if timed_out.is_set():
            errors.append("total runtime limit exceeded")
        if returncode != 0:
            errors.append("CLI exited with a nonzero status")
        if events.failures():
            errors.append("CLI did not report a successful session; see local logs")
        errors.extend(validate_result(task_dir / "result.json", expected))
        if (input_path.is_symlink() or image_path.is_symlink()
                or input_path.read_bytes() != input_bytes or image_path.read_bytes() != image_bytes):
            errors.append("a supplied input was changed")
        if {path.name for path in task_dir.iterdir()} != {"input.json", "colors.png", "result.json"}:
            errors.append("unexpected files were created in the task directory")
    except Exception as exc:
        errors.append(f"checker could not complete ({type(exc).__name__}); see local logs")
        (logs / "checker-error.txt").write_text(repr(exc), encoding="utf-8")
    finally:
        if timer is not None:
            timer.cancel()
    summary = {"ok": not errors, "backend": backend, "model": model,
               "elapsed_seconds": round(time.monotonic() - started, 2),
               "returncode": returncode, "timed_out": timed_out.is_set(), "errors": errors}
    (logs / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, run_dir


def nonempty(value):
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("must not be blank")
    return value


def positive_seconds(value):
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from exc
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("claude", "codex", "opencode"), required=True)
    parser.add_argument("--model", type=nonempty, required=True, help="explicit model ID; provider/model for OpenCode")
    parser.add_argument("--cli", type=nonempty, help="executable path/name, without flags; defaults to backend name")
    parser.add_argument("--effort", type=nonempty, help="native effort/variant; default: Claude low, others native default")
    parser.add_argument("--timeout", type=positive_seconds, default=180, help="total runtime limit in seconds (default: 180)")
    args = parser.parse_args(argv)
    cli = args.cli
    if cli and ("/" in cli or os.sep in cli):
        cli = str(Path(cli).expanduser().resolve())
    summary, directory = run_check(args.backend, args.model, cli=cli,
                                   effort=args.effort, timeout=args.timeout)
    status = "PASS" if summary["ok"] else "FAIL"
    print(f"{status}: {args.backend} smoke check ({summary['elapsed_seconds']:.1f}s)")
    for error in summary["errors"]:
        print(f"  {error}")
    print(f"Local inputs and logs: {directory}")
    if summary["ok"]:
        print("Output validated. Inspect session.jsonl to confirm image-tool use.")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
