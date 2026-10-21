"""Resumable one-film workflow. Importing this module performs no work.

The existing builders and graders remain the implementations. This module owns
project setup, ordered execution, checkpoints and incremental review receipts.
Credentials stay in the environment or each CLI's native credential store.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import uuid

REPO = Path(__file__).resolve().parents[1]
FORMAT = "video-editing-project-v1"
CONFIG = "project.json"
STATE = "workflow/state.json"
STAGES = ("s_1", "s_2", "s_3", "s_4e")
LABELS = dict(zip(STAGES, ("CS", "NS", "RC", "FC")))
PHASES = ("probe", "media", "annotations", "dialogue", "tasks", "review")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def owned(project, relative):
    """Generated files may never escape this project's private directory."""
    path = project / relative
    if not path.resolve().is_relative_to(project):
        raise ValueError(f"Project output escapes its directory: {relative}")
    return path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_identity():
    """Documentation/test edits do not invalidate a project; implementation does."""
    paths = [REPO / "video_editing.py"]
    for folder in ("data_construction", "evaluation"):
        paths.extend(p for p in (REPO / folder).rglob("*") if p.is_file()
                     and (p.suffix == ".py" or p.name.startswith("requirements")
                          or "prompts" in p.parts or "assets" in p.parts)
                     and "tests" not in p.parts and not p.name.startswith("test_"))
    return hashlib.sha256(canonical({str(p.relative_to(REPO)): sha(p) for p in sorted(paths)})).hexdigest()


def init_project(args):
    project = args.project.expanduser().absolute()
    if project.exists() or project.is_symlink():
        raise ValueError("Project directory already exists. Choose a new directory; use run to resume an existing project.")
    project = project.resolve()
    if project.is_relative_to(REPO) or REPO.is_relative_to(project):
        raise ValueError("Choose a new private project directory outside the source checkout.")
    inputs = {}
    for kind, supplied in (("movie", args.movie), ("trailer", args.trailer)):
        path = supplied.expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".mkv", ".mov"):
            raise ValueError(f"{kind} must be an existing .mp4, .mkv or .mov file")
        if path.is_relative_to(project):
            raise ValueError("Input media must already exist outside the new project")
        print(f"Fingerprinting {kind}: {path.name}", flush=True)
        inputs[kind] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
    if os.path.samefile(inputs["movie"]["path"], inputs["trailer"]["path"]):
        raise ValueError("Movie and trailer must be different files")
    config = {"format": FORMAT, "project": str(project), "film": args.name,
              "inputs": inputs, "agent": {"backend": args.backend, "model": args.model,
                  "cli": str(Path(args.cli).expanduser().resolve()) if args.cli and "/" in args.cli else args.cli or "",
                  "effort": args.effort or ""},
              "modal_volume": f"ve-{args.name}-{uuid.uuid4().hex[:12]}",
              "quality_model": args.quality_model, "judge_model": args.judge_model,
              "annotation_timeout_s": args.annotation_timeout, "quality_timeout_s": args.quality_timeout}
    validate_config(config, project)
    project.mkdir(parents=True, mode=0o700)
    source = project / "sources" / args.name
    source.mkdir(parents=True)
    for kind, filename in (("movie", "source"), ("trailer", "trailer")):
        path = Path(inputs[kind]["path"])
        (source / (filename + path.suffix.lower())).symlink_to(path)
    save(project / CONFIG, config)
    state = {"format": FORMAT, "created_at": now(), "config_sha256": hashlib.sha256(canonical(config)).hexdigest(),
             "source_sha256": source_identity(), "steps": {}, "reviews": {}}
    save(project / STATE, state)
    print(f"Created {project}\nMedia are linked; original files were not copied or edited.")
    print(f"Modal volume (created only by run): {config['modal_volume']}")
    print(f"Next: {invocation('doctor', project)}")
    report(project, config, state)
    return 0


def validate_config(config, project):
    if config.get("format") != FORMAT or config.get("project") != str(project):
        raise ValueError("Project config is invalid or was moved. Initialize a new project at the intended location.")
    if not re.fullmatch(r"[a-z0-9]{1,20}", config.get("film", "")):
        raise ValueError("Film name must contain 1–20 lowercase letters or digits")
    agent = config["agent"]
    if agent.get("backend") not in ("claude", "codex", "opencode") or not agent.get("model", "").strip():
        raise ValueError("Choose a backend and a nonempty annotation model ID")
    if agent["backend"] == "opencode" and ("/" not in agent["model"] or not all(agent["model"].split("/", 1))):
        raise ValueError("OpenCode requires a provider/model ID")
    for key in ("quality_model", "judge_model"):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", config[key]):
            raise ValueError(f"Invalid {key}")
    for key in ("annotation_timeout_s", "quality_timeout_s"):
        if isinstance(config[key], bool) or not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")


def open_project(path, *, verify=False):
    project = path.expanduser().resolve(strict=True)
    if project.is_relative_to(REPO) or REPO.is_relative_to(project):
        raise ValueError("Private project must be outside the source checkout")
    config, state = load(owned(project, CONFIG)), load(owned(project, STATE))
    validate_config(config, project)
    if hashlib.sha256(canonical(config)).hexdigest() != state["config_sha256"]:
        raise ValueError("Project configuration changed. Use a new project so old caches and approvals are not reused.")
    if verify:
        if source_identity() != state["source_sha256"]:
            raise ValueError("Pipeline code/dependencies changed since init. Use a new project with this revision; existing outputs are preserved.")
        source_dir = project / "sources" / config["film"]
        if source_dir.is_symlink() or sorted(p.name for p in (project / "sources").iterdir()) != [config["film"]]:
            raise ValueError("This workflow requires its isolated single-film source directory")
        print("Checking movie and trailer fingerprints…", flush=True)
        for kind, filename in (("movie", "source"), ("trailer", "trailer")):
            pin = config["inputs"][kind]
            path = Path(pin["path"])
            link = source_dir / (filename + path.suffix.lower())
            if not link.is_file() or link.resolve() != path or sha(path) != pin["sha256"]:
                raise ValueError(f"{kind} bytes or source link changed. Use a new project; old caches cannot be reused safely.")
    return project, config, state


def environment(config):
    env = dict(os.environ)
    # Ambient profiles/legacy hooks must not silently change this saved workflow.
    for key in list(env):
        if key.startswith(("VIDEO_EDITING_", "S12_")):
            env.pop(key)
    env.update(VIDEO_EDITING_DATA_ROOT=config["project"], VIDEO_EDITING_MODAL_VOLUME=config["modal_volume"],
               VIDEO_EDITING_ANNOTATION_BACKEND=config["agent"]["backend"],
               VIDEO_EDITING_ANNOTATION_MODEL=config["agent"]["model"],
               VIDEO_EDITING_ANNOTATION_CLI=config["agent"]["cli"],
               VIDEO_EDITING_ANNOTATION_EFFORT=config["agent"]["effort"],
               VIDEO_EDITING_ANNOTATION_TIMEOUT_S=str(config["annotation_timeout_s"]),
               S12_GEMINI_MODEL=config["judge_model"], S12_JUDGE_PROVIDER="gemini",
               S12_JUDGE_TEMP="0", S12_JUDGE_SEED="42", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""))
    return env


def invocation(command, project):
    return shlex.join(["python", str(REPO / "video_editing.py"), command, "--project", str(project)])


@contextmanager
def locked(project):
    """Kernel lock is released on interruption; no unsafe stale-PID deletion."""
    import fcntl
    path = owned(project, "workflow/.lock")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another command is running in this project. Use status to inspect it.") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_command(argv, env, log):
    """Tee progress, keep logs private, and terminate descendants on Ctrl-C."""
    log.parent.mkdir(parents=True, exist_ok=True)
    secret = env.get("GOOGLE_AI_STUDIO_API_KEY", "")
    with log.open("w", encoding="utf-8") as stream:
        stream.write(f"Started {now()}\n")
        stream.flush()
        child = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", start_new_session=True, bufsize=1)
        try:
            for line in child.stdout:
                if secret:
                    line = line.replace(secret, "[REDACTED]")
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            return child.wait()
        except BaseException:
            try:
                # Let Python unwind annotation_process, whose CLI has its own
                # process group. SIGTERM would bypass that cleanup.
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
            raise
        finally:
            child.stdout.close()


@dataclass(frozen=True)
class Step:
    name: str
    phase: str
    title: str
    argv: list[str]
    required: tuple[str, ...] = ()


def steps(config):
    film = config["film"]
    exports, edit = f"exports/{film}", f"edit/task_{film}"
    def py(name, *args):
        return [sys.executable, str(REPO / "data_construction" / name), *args]
    film_args = ("--film", film)
    return [
        Step("probe", "probe", "Inspect source and trailer", py("pipeline.py", *film_args, "--only", "probe"), (f"{exports}/probe.json",)),
        Step("volume", "media", "Create this project's Modal volume", [sys.executable, "-c",
             "import modal,sys; modal.Volume.from_name(sys.argv[1], create_if_missing=True).hydrate()", config["modal_volume"]]),
        Step("media", "media", "Preprocess and align media (Modal jobs)", py("pipeline.py", *film_args),
             (f"{exports}/p2_refined.csv", f"{exports}/golden_transcript.json", f"{exports}/audio_cuts.json", f"{edit}/s3_units.json")),
        Step("script-inputs", "annotations", "Prepare annotation inputs", py("run_script_queue.py", *film_args, "--prep-only"), (f"{edit}/stages/script_inputs.json",)),
        Step("luminance", "annotations", "Measure shot brightness", py("shot_luma.py", "--all"), ("shot_luma.json",)),
        Step("match-check", "annotations", "Check source matches", py("verify_matches.py", "--all"), ("match_confidence.json",)),
        Step("card-text", "annotations", "Read title cards", py("card_text.py", *film_args), (f"{exports}/card_text.json",)),
        Step("script", "annotations", "Annotate shots and narrative (agent calls)", py("run_script_queue.py", *film_args), (f"{edit}/stages/script.json",)),
        Step("translation", "annotations", "Prepare English task descriptions (agent call)", py("translate_script.py", *film_args), (f"{edit}/stages/script.json",)),
        Step("cuts", "annotations", "Annotate edit boundaries (agent calls)", py("run_s4_queue.py", *film_args), (f"{edit}/stages/s4_cuts.json",)),
        Step("audio-extract", "dialogue", "Extract source-film audio", py("modal/modal_film_transcribe.py", "--extract", *film_args), (f"exports/_film_audio/{film}.opus",)),
        Step("audio-upload", "dialogue", "Upload source-film audio", py("modal/modal_film_transcribe.py", "--upload", *film_args)),
        Step("transcribe", "dialogue", "Transcribe source film (Modal job)", [sys.executable, "-m", "modal", "run", str(REPO / "data_construction/modal/modal_film_transcribe.py") + "::main", "--only", film]),
        Step("transcript-fetch", "dialogue", "Download source transcript", py("modal/modal_film_transcribe.py", "--fetch", *film_args), (f"{exports}/source_transcript.json",)),
        Step("locate-dialogue", "dialogue", "Locate dialogue in source film", py("locate_lines.py", *film_args), (f"{exports}/line_origins.json",)),
        Step("dialogue-takes", "dialogue", "Prepare dialogue candidates", py("cut_takes.py", *film_args), (f"{exports}/takes.json",)),
        Step("build-cs", "tasks", "Build Content Selection tasks", py("build_s1_tasks.py", *film_args)),
        Step("build-ns-rc", "tasks", "Build Narrative Sequencing and Rough Cut tasks", py("build_s23_tasks.py", *film_args)),
        Step("build-cuts", "tasks", "Build Fine Cut intermediate assets", py("build_s4_tasks.py", *film_args)),
        Step("build-fc", "tasks", "Build Fine Cut tasks", py("build_s4_editorial.py", *film_args)),
        Step("scoring-setup", "tasks", "Attach scoring rules", [sys.executable, str(REPO / "evaluation/task_pipeline.py"), "--tasks", str(Path(config["project"]) / "candidate_tasks.json"), "--prepare-only"]),
    ]


def discover_tasks(project, film, stages=STAGES):
    tasks = []
    for stage in stages:
        folder = owned(project, f"training_task/{film}/{stage}")
        found = sorted(folder.glob("c*/task.json"))
        if not found:
            raise ValueError(f"No {LABELS.get(stage, stage)} tasks were produced. Inspect builder logs and chunk eligibility before continuing.")
        for marker in found:
            task = marker.parent
            if task.is_symlink() or not task.resolve().is_relative_to(project):
                raise ValueError("Task path escapes the private project")
            for rel in ("task.json", "task_prompt.md", "initial_state/manifest.json", "verifier/answer.json"):
                path = owned(project, str((task / rel).relative_to(project)))
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Incomplete task: {task.name}/{rel}")
            tasks.append(str(task))
    return tasks


def validate_step(project, config, step):
    for relative in step.required:
        path = owned(project, relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Required artifact is missing or empty: {relative}. Restore it or initialize a new project.")
        if path.suffix == ".json":
            load(path)
    if step.name == "media":
        flags = load(project / f"exports/{config['film']}/state.json")
        expected = ("probe", "proxy", "upload", "shots", "fetch", "p2", "cards", "audio", "s3", "cutlist", "acuts", "report")
        if not all(flags.get(key, {}).get("ok") for key in expected):
            raise ValueError("Preprocessing did not finish every required stage; inspect its state and logs")
    if step.name == "translation":
        script = load(project / step.required[0])
        texts = [script.get("logline_en")] + [x.get(k) for group, keys in (("beats", ("name_en", "what_en")), ("chunks", ("what_en",))) for x in script.get(group, []) for k in keys]
        if not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("English task descriptions are incomplete")
    stage_map = {"build-cs": ("s_1",), "build-ns-rc": ("s_2", "s_3"), "build-fc": ("s_4e",), "scoring-setup": STAGES}
    if step.name in stage_map:
        tasks = discover_tasks(project, config["film"], stage_map[step.name])
        if step.name == "scoring-setup":
            for task in tasks:
                for name in ("rubric.json", "evaluation.json"):
                    load(Path(task) / "verifier" / name)


def checkpoint(project, config, state):
    state["updated_at"] = now()
    save(owned(project, STATE), state)
    report(project, config, state)


def execute_step(project, config, state, step, runner=run_command):
    previous = state["steps"].get(step.name, {})
    if previous.get("status") == "complete":
        validate_step(project, config, step)
        print(f"[done] {step.title}", flush=True)
        return
    if step.name == "scoring-setup":
        save(owned(project, "candidate_tasks.json"), discover_tasks(project, config["film"]))
    attempt = int(previous.get("attempt", 0)) + 1
    log = owned(project, f"workflow/logs/{step.name}-{attempt:03d}.log")
    row = {"status": "running", "attempt": attempt, "started_at": now(), "log": str(log)}
    state["steps"][step.name] = row
    checkpoint(project, config, state)
    print(f"\n[start] {step.title}\n        Log: {log}", flush=True)
    try:
        code = runner(step.argv, environment(config), log)
        row["exit_code"] = code
        if code:
            raise RuntimeError(f"{step.title} exited with status {code}")
        validate_step(project, config, step)
        row.update(status="complete", finished_at=now())
    except BaseException as exc:
        row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), finished_at=now())
        checkpoint(project, config, state)
        raise
    checkpoint(project, config, state)


def quality_module():
    # The audit helper uses sibling imports; no other construction stages are imported.
    directory = str(REPO / "data_construction")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("task_quality")


def verify_review(project, config, task, evidence):
    q = quality_module()
    evidence = owned(project, str(Path(evidence).relative_to(project)))
    terminal, bound, identity = (load(evidence / name) for name in ("terminal.json", "input.json", "identity.json"))
    if terminal.get("status") != "complete" or not terminal.get("source_integrity_verified"):
        raise ValueError("Review did not finish successfully")
    if terminal.get("task_path") != task or bound.get("task_path") != task:
        raise ValueError("Review belongs to another task")
    if identity.get("model") != config["quality_model"] or identity.get("source_sha256") != sha(q.__file__):
        raise ValueError("Review model or audit implementation changed")
    if terminal.get("input_sha256") != sha(evidence / "input.json") or terminal.get("response_sha256") != sha(evidence / "response.json"):
        raise ValueError("Review receipt bytes changed")
    q.verify_sources(bound)
    if q.canonical(q.prepare(task)) != q.canonical(bound):
        raise ValueError("Task, harness or scoring contract changed after review")
    result = load(evidence / "result.json")
    raw = load(evidence / "response.json")
    candidates = raw.get("candidates", [])
    if raw.get("modelVersion") != config["quality_model"] or len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Review response is incomplete or uses an unexpected model")
    parsed = q.load_json("\n".join(p.get("text", "") for p in candidates[0]["content"]["parts"] if not p.get("thought")))
    if q.canonical(parsed) != q.canonical(result) or result.get("action") != terminal.get("action"):
        raise ValueError("Review decision differs from its receipt")
    return terminal


def publish_reviews(project, config, state, tasks, *, validate=True):
    accepted, rows = [], []
    for task in tasks:
        record = state["reviews"].get(task, {})
        row = {"task_path": task, "status": record.get("status", "pending"), "action": record.get("action"), "evidence_path": record.get("evidence")}
        if record.get("status") == "complete":
            if validate:
                terminal = verify_review(project, config, task, record["evidence"])
                row.update(status="complete", action=terminal["action"])
                record["action"] = terminal["action"]
            if row["action"] == "keep_candidate":
                accepted.append(task)
        rows.append(row)
    save(owned(project, "accepted_tasks.json"), accepted)
    save(owned(project, "review_summary.json"), {"tasks": rows, "accepted": len(accepted), "total": len(tasks),
          "complete": all(r["status"] == "complete" for r in rows), "updated_at": now(),
          "scope": "Model task-quality assessments, not model outcome scores or historical author repairs"})
    return accepted


def review_tasks(project, config, state, *, limit=None, runner=run_command):
    if state["steps"].get("scoring-setup", {}).get("status") != "complete":
        raise ValueError("Tasks/scoring are not prepared. Run the workflow through --until tasks first.")
    tasks = discover_tasks(project, config["film"])
    # Validate older receipts before any paid work. During this invocation each
    # new receipt is checked separately; avoid rehashing every earlier task
    # after every request (quadratic media I/O). Recheck the full set at exit.
    publish_reviews(project, config, state, tasks)
    attempts = 0
    for task in tasks:
        row = state["reviews"].get(task, {})
        # The child may have published a complete receipt just before the
        # parent stopped. Recover it without issuing another paid request.
        if row.get("status") != "complete" and row.get("evidence"):
            evidence = owned(project, str(Path(row["evidence"]).relative_to(project)))
            terminal_path = evidence / "terminal.json"
            if terminal_path.is_file() and load(terminal_path).get("status") == "complete":
                terminal = verify_review(project, config, task, evidence)
                row.update(status="complete", action=terminal["action"], finished_at=now())
                checkpoint(project, config, state)
        if row.get("status") == "complete":
            verify_review(project, config, task, row["evidence"])
            print(f"[reviewed] {Path(task).parent.name}/{Path(task).name}: {row['action']}", flush=True)
            continue
        if limit is not None and attempts >= limit:
            break
        attempts += 1
        attempt = int(row.get("attempt", 0)) + 1
        key = hashlib.sha256(task.encode()).hexdigest()[:16]
        output = owned(project, f"workflow/reviews/{key}/attempt-{attempt:03d}")
        while output.exists():
            attempt += 1
            output = owned(project, f"workflow/reviews/{key}/attempt-{attempt:03d}")
        log = owned(project, f"workflow/logs/review-{key}-{attempt:03d}.log")
        row = {"status": "running", "attempt": attempt, "evidence": str(output / "task-00000"), "log": str(log), "started_at": now()}
        state["reviews"][task] = row
        checkpoint(project, config, state)
        print(f"\n[review {attempts}] {Path(task).parent.name}/{Path(task).name} — model request\nLog: {log}", flush=True)
        argv = [sys.executable, str(REPO / "data_construction/task_quality.py"), "--task", task,
                "--output", str(output), "--model", config["quality_model"], "--timeout", str(config["quality_timeout_s"])]
        try:
            code = runner(argv, environment(config), log)
            row["exit_code"] = code
            if code:
                raise RuntimeError(f"Quality review failed; inspect {log}. Re-running review retries this task only.")
            terminal = verify_review(project, config, task, row["evidence"])
            row.update(status="complete", action=terminal["action"], finished_at=now())
        except BaseException as exc:
            row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "held", error=str(exc), finished_at=now())
            publish_reviews(project, config, state, tasks)
            checkpoint(project, config, state)
            raise
        publish_reviews(project, config, state, tasks, validate=False)
        checkpoint(project, config, state)
    accepted = publish_reviews(project, config, state, tasks)
    print(f"Review progress: {sum(state['reviews'].get(t, {}).get('status') == 'complete' for t in tasks)}/{len(tasks)} complete; {len(accepted)} accepted.")
    print(f"Accepted tasks: {project / 'accepted_tasks.json'}\nDetails: {project / 'review_summary.json'}")
    return accepted


def run_workflow(project, config, state, until="review", review_limit=None, runner=run_command):
    print("Local media probe; no cloud or model calls." if until == "probe" else
          "This run may use paid Modal jobs, annotation CLI calls and Gemini quality reviews.", flush=True)
    for step in steps(config):
        if PHASES.index(step.phase) <= PHASES.index(until):
            execute_step(project, config, state, step, runner)
    if until == "review":
        review_tasks(project, config, state, limit=review_limit, runner=runner)
    print(f"\nWorkflow stopped at {until}. Summary: {project / 'README.md'}")


def score_tasks(project, config, state, execute=False, runner=run_command):
    tasks = discover_tasks(project, config["film"])
    accepted = publish_reviews(project, config, state, tasks)
    if not accepted:
        raise ValueError("No approved tasks. Run review and inspect review_summary.json first.")
    # Re-preparing scoring uses the same bound code; inspect descriptors again afterwards.
    stamp = uuid.uuid4().hex[:12]
    results = owned(project, f"workflow/scores/{stamp}.json")
    log = owned(project, f"workflow/logs/score-{stamp}.log")
    argv = [sys.executable, str(REPO / "evaluation/task_pipeline.py"), "--tasks", str(project / "accepted_tasks.json"), "--summary", str(results)]
    if not execute:
        argv.append("--preflight-only")
    print("Scoring submitted outputs (RC/FC call Gemini)." if execute else "Checking submission files only; no judge calls.", flush=True)
    row = {"status": "running", "executed": execute, "result": str(results), "log": str(log), "started_at": now()}
    state["last_score"] = row
    checkpoint(project, config, state)
    try:
        code = runner(argv, environment(config), log)
        for task in accepted:
            verify_review(project, config, task, state["reviews"][task]["evidence"])
        result = load(results)
        expected = {"scored", "invalid_submission"} if execute else {"preflight_passed"}
        rows = result.get("tasks", [])
        okay = (not code and len(rows) == len(accepted) and {r.get("task") for r in rows} == set(accepted)
                and all(r.get("status") in expected for r in rows))
        row.update(status="complete" if okay else "incomplete", exit_code=code, finished_at=now())
    except BaseException as exc:
        row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), finished_at=now())
        checkpoint(project, config, state)
        raise
    checkpoint(project, config, state)
    print(f"Results: {results}\nLog: {log}")
    if not okay:
        print("Some submissions are missing/invalid or scoring is incomplete. Read the result before retrying.")
    return 0 if okay else 1


def report(project, config, state):
    lines = [f"# Video-editing project: {config['film']}", "", "Original media are linked under `sources/`. Keep this entire project private.", "",
             "## Progress", "", "| Step | Status | Log |", "| --- | --- | --- |"]
    for step in steps(config):
        row = state["steps"].get(step.name, {})
        log = Path(row["log"]).relative_to(project).as_posix() if row.get("log") else ""
        lines.append(f"| {step.title} | {row.get('status', 'pending')} | {'[log](' + log + ')' if log else ''} |")
    rows = list(state["reviews"].values())
    lines += ["", f"Quality reviews completed: {sum(r.get('status') == 'complete' for r in rows)}. "
              f"Accepted: {sum(r.get('status') == 'complete' and r.get('action') == 'keep_candidate' for r in rows)}.", "",
              "`candidate_tasks.json` lists generated tasks; `accepted_tasks.json` lists reviewed candidates. "
              "`review_summary.json` separates keep, filter, repair, uncertainty and failed requests.", "",
              "Each task has a public `task_prompt.md` and initial media. Put the requested deliverables in its "
              "`output/` directory. Reference answers under `verifier/` are private grading evidence, not submission templates.", "",
              "## Continue", "", "```sh", invocation("run", project), invocation("status", project),
              invocation("score", project), invocation("score", project) + " --execute", "```", "",
              "`run` can incur Modal and model usage. `score` checks files; `score --execute` evaluates submitted work "
              "and calls Gemini for RC/FC. Generation does not run an editing policy or produce its performance scores.", ""]
    if state.get("last_score"):
        lines += [f"Latest score/check: `{state['last_score']['status']}` — {state['last_score']['result']}", ""]
    owned(project, "README.md").write_text("\n".join(lines), encoding="utf-8")


def show_status(project, config, state):
    print(f"Project: {project}\nAnnotation: {config['agent']['backend']} / {config['agent']['model']}")
    for step in steps(config):
        row = state["steps"].get(step.name, {})
        print(f"{row.get('status', 'pending'):12} {step.title}")
        if row.get("status") in ("failed", "running", "interrupted") and row.get("log"):
            print(f"             {row['log']}")
    counts = {}
    for row in state["reviews"].values():
        key = row.get("action") if row.get("status") == "complete" else row.get("status", "pending")
        counts[key] = counts.get(key, 0) + 1
    print(f"Reviews: {counts or 'not started'}\nSummary: {project / 'README.md'}")
    if state.get("last_score"):
        print(f"Last score/check: {state['last_score']['status']} — {state['last_score']['result']}")
    print("Status reads saved progress; it does not call services or validate media again.")


def doctor(config, scope):
    from .workflow_doctor import check
    rows = check(config, scope)
    for row in rows:
        print(f"[{'OK' if row['ok'] else 'MISSING'}] {row['name']}: {row['detail']}")
        if not row["ok"] and row.get("fix"):
            print(f"  {row['fix']}")
    print("Local checks do not verify provider login, remote quota, or model access.")
    return all(row["ok"] for row in rows)


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate, review and score private video-editing tasks from one movie and trailer.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("init", help="Create a new private project; no cloud/model calls")
    create.add_argument("--project", required=True, type=Path)
    create.add_argument("--movie", required=True, type=Path)
    create.add_argument("--trailer", required=True, type=Path)
    create.add_argument("--name", default="film", help="Short lowercase film ID (letters/digits, at most 20)")
    create.add_argument("--backend", choices=("claude", "codex", "opencode"), required=True)
    create.add_argument("--model", required=True)
    create.add_argument("--cli", help="Optional executable path/name")
    create.add_argument("--effort", help="Backend-native effort/variant")
    create.add_argument("--quality-model", default="gemini-3.8-flash")
    create.add_argument("--judge-model", default="gemini-3.1-pro-preview")
    create.add_argument("--annotation-timeout", type=positive, default=1800)
    create.add_argument("--quality-timeout", type=positive, default=600)
    for name, help_text in (("doctor", "Check local dependencies; no service calls"),
                            ("run", "Generate tasks and review them; can incur cloud/model usage"),
                            ("review", "Review remaining tasks; can incur Gemini usage"),
                            ("status", "Read progress without running work"),
                            ("score", "Check agent submissions; --execute runs paid judging")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--project", required=True, type=Path)
        if name == "doctor":
            sub.add_argument("--scope", choices=("probe", "build", "review", "preflight", "score", "all"), default="all")
        if name == "run":
            sub.add_argument("--until", choices=PHASES, default="review")
            sub.add_argument("--plan", action="store_true", help="Print steps without running commands or changing files")
        if name in ("run", "review"):
            sub.add_argument("--review-limit", type=int, help="Limit new task audits in this invocation; completed audits are reused")
        if name == "score":
            sub.add_argument("--execute", action="store_true", help="Score submissions, including paid Gemini RC/FC calls")
    args = parser.parse_args(argv)
    if getattr(args, "review_limit", None) is not None and args.review_limit < 1:
        parser.error("--review-limit must be positive")
    try:
        if args.command == "init":
            return init_project(args)
        read_only = args.command in ("status", "doctor") or getattr(args, "plan", False)
        project, config, state = open_project(args.project, verify=not read_only)
        if args.command == "status":
            show_status(project, config, state)
            return 0
        if getattr(args, "plan", False):
            print("Plan only — no files changed and no services called.")
            for step in steps(config):
                if PHASES.index(step.phase) <= PHASES.index(args.until):
                    print(f"{step.phase:12} {step.title}\n             {shlex.join(step.argv)}")
            if args.until == "review":
                print(f"review       Audit each candidate using {config['quality_model']}; reuse verified completed reviews.")
            return 0
        scope = (args.scope if args.command == "doctor" else ("score" if args.execute else "preflight") if args.command == "score" else "review" if args.command == "review"
                 else "probe" if args.until == "probe" else "all" if args.until == "review" else "build")
        if not doctor(config, scope):
            print("Fix the missing requirements above, then rerun this command.")
            return 1
        if args.command == "doctor":
            return 0
        with locked(project):
            # Refresh state after claiming the lock; other invocations may have just finished.
            state = load(project / STATE)
            if args.command == "run":
                run_workflow(project, config, state, args.until, args.review_limit)
            elif args.command == "review":
                review_tasks(project, config, state, limit=args.review_limit)
            else:
                return score_tasks(project, config, state, args.execute)
        return 0
    except KeyboardInterrupt:
        print("\nStopped. Completed work is saved. Use status, then rerun the same command to continue.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        message = str(exc)
        secret = os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
        print(f"Error: {message}", file=sys.stderr)
        if args.command not in ("init", "status"):
            print(f"Inspect: {invocation('status', args.project)}", file=sys.stderr)
        return 1
