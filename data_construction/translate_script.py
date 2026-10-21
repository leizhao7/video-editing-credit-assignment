"""Public construction draft: translate_script.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/translate_script.py [options]\n\n  --film\n  --all\n  -j, --workers\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (
        DATA_ROOT, require_target, validate_data_slug, export_slug, select_films,
        annotation_process
    )
    from .annotation_backend import build_annotation_command, AnnotationEvents
except ImportError:  # Direct script invocation.
    from release_config import (
        DATA_ROOT, require_target, validate_data_slug, export_slug, select_films,
        annotation_process
    )
    from annotation_backend import build_annotation_command, AnnotationEvents

import argparse, glob, json, os, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

ROOT = DATA_ROOT
EFFORT = "low"


def need(d):
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    return ([b for b in d.get("beats", []) if not text(b.get("name_en")) or not text(b.get("what_en"))],
            [c for c in d.get("chunks", []) if not text(c.get("what_en"))])


def source_fields(document):
    """Ignore only fields the translation step is authorized to add."""
    result = deepcopy(document)
    result.pop("logline_en", None)
    for beat in result.get("beats", []):
        beat.pop("name_en", None)
        beat.pop("what_en", None)
    for chunk in result.get("chunks", []):
        chunk.pop("what_en", None)
    return result


def translation_payload(document):
    """Select translatable fields and validate the identifiers used for merging."""
    if not isinstance(document, dict):
        raise ValueError("Source script must be a JSON object")
    payload = {"logline": document.get("logline")}
    for kind, fields in (("beats", ("id", "name", "what")), ("chunks", ("id", "what"))):
        entries = document.get(kind)
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError(f"Source {kind} must be an array of objects")
        identifiers = [entry.get("id") for entry in entries]
        if (any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise ValueError(f"Source {kind} IDs must be nonempty unique strings")
        if any(not isinstance(entry.get(field), str) for entry in entries for field in fields):
            raise ValueError(f"Source {kind} text fields must be strings")
        payload[kind] = [{field: entry[field] for field in fields} for entry in entries]
    return payload


def merge_translation_patch(document, patch):
    """Accept only the complete ordered English-field patch, without source edits."""
    payload = translation_payload(document)
    if not isinstance(patch, dict) or set(patch) != {"logline_en", "beats", "chunks"}:
        raise ValueError("Translation patch must contain exactly logline_en, beats, chunks")
    if not isinstance(patch["logline_en"], str) or not patch["logline_en"].strip():
        raise ValueError("Translation patch needs nonempty logline_en")
    merged = deepcopy(document)
    merged["logline_en"] = patch["logline_en"]
    for kind, fields in (("beats", ("name_en", "what_en")), ("chunks", ("what_en",))):
        entries = patch[kind]
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError(f"Translation patch {kind} must be an array of objects")
        if [entry.get("id") for entry in entries] != [entry["id"] for entry in payload[kind]]:
            raise ValueError(f"Translation patch {kind} IDs must match every source ID once, in order")
        for target, entry in zip(merged[kind], entries):
            if set(entry) != {"id", *fields}:
                raise ValueError(f"Translation patch {kind} entries allow only id and {', '.join(fields)}")
            for field in fields:
                if not isinstance(entry[field], str) or not entry[field].strip():
                    raise ValueError(f"Translation patch {entry['id']}.{field} must be nonempty text")
                target[field] = entry[field]
    return merged


def _atomic_write_bytes(path, content):
    """Replace a file atomically, also used to restore the original exact bytes."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=Path(path).parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _stream_status(line, count):
    """Summarize event metadata without copying model prose or reasoning."""
    try:
        event = json.loads(line)
    except (ValueError, TypeError):
        event = {}
    if not isinstance(event, dict):
        event = {}
    status = f"received {count} stream events"
    estimated = event.get("estimated_tokens")
    if event.get("subtype") == "thinking_tokens" and isinstance(estimated, (int, float)):
        status += f"; estimated reasoning tokens={estimated}"
    return status


def run(slug, name):
    validate_data_slug(ROOT, slug)
    p = f"{ROOT}/edit/task_{slug}/stages/script.json"
    original = Path(p).read_bytes()
    d = json.loads(original)
    try:
        payload = translation_payload(d)
    except ValueError as exc:
        print(f"  [{name}] ✗ Invalid translation source: {exc}", flush=True)
        return False
    nb, nc = need(d)
    if not nb and not nc and isinstance(d.get("logline_en"), str) and d["logline_en"].strip():
        print(f"  [{name}] English fields already complete; skipping", flush=True); return True
    task_dir = Path(ROOT) / "edit" / f"task_{slug}"
    work = task_dir / "work"
    work.mkdir(exist_ok=True)
    if not work.resolve().is_relative_to(task_dir.resolve()):
        raise ValueError("Translation working outputs must remain inside the task directory")
    working = Path(tempfile.mkdtemp(prefix="translation-", dir=work))
    patch_path = working / "fields.json"
    prompt = f"""Complete the English fields used in task briefs for the trailer annotation of {name}.
The source may already be in English. Copy natural English exactly into the corresponding `_en` fields. Do not modify any original fields.

```json
{json.dumps(payload, ensure_ascii=False, indent=1)}
```

Write only the additional English fields to {json.dumps(str(patch_path), ensure_ascii=False)}; do not rewrite the complete annotation.
The output JSON must contain exactly three top-level fields:
- `logline_en`: a nonempty English string.
- `beats`: every input beat in order, each containing only `id`, `name_en`, and `what_en`.
- `chunks`: every input chunk in order, each containing only `id` and `what_en`.
Every translated field must be a nonempty string. Preserve IDs and their order exactly; do not omit, duplicate, reorder, or add entries.
Do not output `shots`, original fields, or any extra fields. Do not modify `stages/script.json` or other inputs.
The caller validates this small patch and merges the English fields while preserving the original data programmatically.

Use clear, concise English suitable for an editing brief, describing concrete visible events rather than translating word for word.
Preserve proper names. End the task after saving the patch."""
    t0 = time.monotonic()
    command = build_annotation_command(prompt, task_dir, tools="Read,Write,Edit,Bash(python3:*)",
                                       effort=EFFORT)
    runs = Path(ROOT) / "edit/runs" / slug
    runs.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d_%H%M%S_translation-"), dir=runs))
    (log_dir / "PROMPT.md").write_text(prompt, encoding="utf-8")
    print(f"  [{name}] translation ({command.backend}) → {log_dir}/progress.log", flush=True)
    events = AnnotationEvents(command.backend)
    errors, ok, returncode = [], False, None
    try:
        with (log_dir / "session.jsonl").open("w", encoding="utf-8") as raw, \
             (log_dir / "progress.log").open("w", encoding="utf-8") as progress:
            def record(message):
                progress.write(f"[{time.monotonic()-t0:6.1f}s] {message}\n")
                progress.flush()

            record(f"translation started; backend={command.backend}; model={command.model}")
            count, last_status = 0, None
            try:
                with annotation_process(command.argv, cwd=task_dir,
                                        stderr_path=log_dir / "stderr.log") as process:
                    for line in process.stdout:
                        raw.write(line)
                        raw.flush()
                        count += 1
                        was_completed = events.completed
                        for tool in events.feed(line):
                            record(f"tool: {tool['name']}")
                        now = time.monotonic()
                        if last_status is None or now - last_status >= 15:
                            message = _stream_status(line, count)
                            record(message)
                            print(f"  [{name}] translation {message}", flush=True)
                            last_status = now
                        if events.completed and not was_completed:
                            record("CLI successful completion event received; output not yet validated")
                    returncode = process.wait()
            except (OSError, ValueError, TimeoutError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            errors.extend(events.failures())
            if returncode != 0:
                errors.append(f"Annotation process exited with status {returncode}; see stderr.log")
            if not Path(p).exists() or Path(p).read_bytes() != original:
                errors.append("Model changed or removed the original script; only a translation patch is permitted")
            if not errors:
                try:
                    patch = json.loads(patch_path.read_text(encoding="utf-8"))
                    d2 = merge_translation_patch(d, patch)
                    _atomic_write_bytes(p, (json.dumps(d2, ensure_ascii=False, indent=2,
                                                      allow_nan=False) + "\n").encode("utf-8"))
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    errors.append(f"Invalid translation output: {type(exc).__name__}: {exc}")
            ok = not errors
            record("validated translation" if ok else "translation rejected: " + "; ".join(errors))
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        ok = False
    finally:
        if not ok:
            _atomic_write_bytes(p, original)
    print(f"  [{name}] {time.monotonic()-t0:.0f}s  "
          + ("✓" if ok else f"✗ {errors[0][:240]}; original script restored; see {log_dir}/progress.log"),
          flush=True)
    return ok


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--all", action="store_true")
    P.add_argument("-j", "--workers", type=int, default=1)
    A = P.parse_args()
    require_target(P, A)
    films = {}
    for p in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p))
        slug = export_slug(pb, p)
        st = json.load(open(f"{ROOT}/exports/{slug}/state.json"))
        if st.get("script", {}).get("ok"):
            films[slug] = os.path.basename(pb["dir"])
    try:
        films = select_films(films, A)
    except ValueError as exc:
        P.error(str(exc))
    if not films:
        sys.exit("No films are available for translation")
    print(f"Preparing English fields for {len(films)} films\n", flush=True)
    if A.workers > 1:
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            n = sum(bool(x) for x in ex.map(lambda kv: run(*kv), films.items()))
    else:
        n = sum(bool(run(s, v)) for s, v in films.items())
    print(f"\nCompleted {n}/{len(films)} films")
    if n != len(films):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
