"""Create a private, sourceable annotation-CLI profile; real model calls are opt-in."""
import argparse
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

try:
    from .check_annotation_cli import nonempty, positive_seconds, run_check
except ImportError:
    from check_annotation_cli import nonempty, positive_seconds, run_check


BACKENDS = ("claude", "codex", "opencode")
GUIDES = {"claude": "https://code.claude.com/docs/en/setup",
          "codex": "https://developers.openai.com/codex/cli",
          "opencode": "https://opencode.ai/docs/"}
REPO = Path(__file__).resolve().parents[1]


def outside_repo(value, label, *, output=False):
    path = Path(value).expanduser().absolute()
    if output and path.is_symlink():
        raise ValueError("output must not be a symlink")
    resolved = path.resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise ValueError(f"{label} must be outside the repository")
    if not output and resolved.exists() and not resolved.is_dir():
        raise ValueError("data root must be a directory")
    return resolved


def save_profile(path, values, force=False):
    """Exclusive creation by default, with no symlink following even on overwrite."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    flags |= os.O_TRUNC if force else os.O_EXCL
    if path.is_symlink():
        raise ValueError("output must not be a symlink")
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write("# Annotation configuration only; credentials stay in the CLI's native store.\n")
        for key, value in values.items():
            stream.write(f"export {key}={shlex.quote(value)}\n")


def inspect_cli(backend, cli):
    executable = shutil.which(cli or backend)
    if not executable:
        print(f"CLI not found: {shlex.quote(cli or backend)}. The profile was still saved.")
        return False, False
    version = None
    try:
        result = subprocess.run([executable, "--version"], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            match = re.search(r"(?<![\w.])v?(\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?)(?![\w.])", result.stdout)
            version = match.group(1) if match else None
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        pass
    print(f"CLI: {backend}; version: {version or 'not detected'}.")
    incompatible = backend == "opencode" and version is not None and version.split(".")[0] != "1"
    if incompatible:
        print("Incompatible OpenCode version: this adapter supports the 1.x CLI only.")
    return True, incompatible


def interactive_values(args, parser):
    try:
        while args.backend not in BACKENDS:
            args.backend = input("Backend (claude/codex/opencode): ").strip().lower()
        while not args.model:
            args.model = input("Model ID (required; OpenCode uses provider/model): ").strip()
        args.data_root = input(f"Private data root [{args.data_root}]: ").strip() or args.data_root
        args.cli = input("CLI executable path/name [backend name]: ").strip() or None
        args.effort = input("Native effort/variant [existing defaults]: ").strip() or None
    except (EOFError, KeyboardInterrupt):
        parser.error("setup cancelled before all required values were supplied")


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--model", type=nonempty, help="required model ID; no model is selected automatically")
    parser.add_argument("--data-root", type=nonempty, default=str(Path.home() / "video-editing-data"),
                        help="private directory outside the repo (default: ~/video-editing-data)")
    parser.add_argument("--cli", type=nonempty, help="executable path/name only; blank profile value uses backend name")
    parser.add_argument("--effort", type=nonempty, help="optional backend-native effort/variant")
    parser.add_argument("--output", type=nonempty, help="default: ~/.config/video-editing-credit-assignment/<backend>.env")
    parser.add_argument("--force", action="store_true", help="explicitly overwrite an existing profile")
    parser.add_argument("--check", action="store_true", help="run a real model smoke check; may incur provider usage")
    parser.add_argument("--timeout", type=positive_seconds, default=180, help="seconds for --check only (default: 180)")
    args = parser.parse_args(arguments)
    if not arguments and sys.stdin.isatty():
        interactive_values(args, parser)
    if not args.backend or not args.model:
        parser.error("--backend and --model are required outside the no-argument interactive wizard")
    if args.backend == "opencode" and ("/" not in args.model or not all(args.model.split("/", 1))):
        parser.error("OpenCode requires a provider/model ID; list choices with 'opencode models'")
    cli = args.cli or ""
    if cli and "/" in cli:
        cli = str(Path(cli).expanduser().resolve())
    try:
        data_root = outside_repo(args.data_root, "data root")
        output = outside_repo(args.output or Path.home() / ".config/video-editing-credit-assignment" / f"{args.backend}.env",
                              "output", output=True)
        values = {"VIDEO_EDITING_DATA_ROOT": str(data_root),
                  "VIDEO_EDITING_ANNOTATION_BACKEND": args.backend,
                  "VIDEO_EDITING_ANNOTATION_CLI": cli,
                  "VIDEO_EDITING_ANNOTATION_MODEL": args.model,
                  "VIDEO_EDITING_ANNOTATION_EFFORT": args.effort or "",
                  "VIDEO_EDITING_MCP_CONFIG": ""}
        save_profile(output, values, args.force)
    except FileExistsError:
        parser.error("profile already exists; choose --output or explicitly use --force")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Saved private profile: {output} (mode 0600).")
    found, incompatible = inspect_cli(args.backend, cli)
    login = [cli or args.backend] + (["login"] if args.backend == "codex" else ["auth", "login"])
    print(f"Install/authentication guide: {GUIDES[args.backend]}")
    print(f"If needed, log in yourself: {shlex.join(login)}")
    print("Authentication and model availability have not been checked. No login was executed.")
    print("Load this profile in your shell (not run automatically):")
    print(f"  source {shlex.quote(str(output))}")
    check = ("python3 " + shlex.quote(str(REPO / "data_construction/check_annotation_cli.py"))
             + ' --backend "$VIDEO_EDITING_ANNOTATION_BACKEND" --model "$VIDEO_EDITING_ANNOTATION_MODEL"')
    if cli:
        check += ' --cli "$VIDEO_EDITING_ANNOTATION_CLI"'
    if args.effort:
        check += ' --effort "$VIDEO_EDITING_ANNOTATION_EFFORT"'
    print("Optional real model check (may incur provider usage):")
    print(f"  {check}")
    print("After preparing the required private inputs, prepare one annotation task:")
    print("  python3 " + shlex.quote(str(REPO / "data_construction/run_script_queue.py"))
          + " --film '<your-film-slug>' --prep-only")
    if args.check:
        if not found or incompatible:
            print("Real check was not started: install a supported CLI first.")
            return 1
        print("Running the requested real model check now; this may incur provider usage.", flush=True)
        summary, directory = run_check(args.backend, args.model, cli=cli or None,
                                       effort=args.effort, timeout=args.timeout)
        print(f"Real check: {'PASS' if summary['ok'] else 'FAIL'}. Local inputs and logs: {directory}")
        print("Inspect the session log to confirm image-tool use; this is not a full media-pipeline test.")
        return 0 if summary["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
