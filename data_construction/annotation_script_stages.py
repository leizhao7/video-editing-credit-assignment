"""Small, backend-independent helpers for staged script annotation.

Draft validation checks the data contract, not the accuracy of visual descriptions.
The caller must still run the complete script validator after merging structure.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

SHOT_BATCH_SIZE = 12


def _quoted_path(path):
    return json.dumps(str(path), ensure_ascii=False)


def build_shot_prompt(inputs_path, visuals):
    """Return a self-contained prompt that writes only the shot draft.

    ``visuals`` is the result of ``prepare_script_visuals`` and supplies
    ``manifest_path`` and ``draft_path``. This helper performs no file I/O.
    """
    return f"""# Inspect shots and save a shot-only draft

Read {_quoted_path(inputs_path)} and the contact-sheet manifest at
{_quoted_path(visuals['manifest_path'])}. The input provides the film slug, ordered
shot IDs, measured timing, kind, supplied dialogue and card-still names. Do not
modify these input files or re-derive their measurements. Do not delegate.

Write newly authored descriptions and explanations in English. Preserve visible
card text and proper names in their original form.

Your only output is the working draft at {_quoted_path(visuals['draft_path'])}.
Write a JSON object with `film` copied exactly from the input and a `shots` array.
For each input shot, in its original order, provide:
- `id`: the exact input ID.
- `desc`: a nonempty description grounded in inspected pictures, not the title.
- Do not output `line`. Supplied dialogue is measured input; the caller will
  insert its exact text/null mechanically. Do not rewrite or invent dialogue.
- `card_text`: transcribe visible card text, or use null when absent.
- `card_misflag`: optionally explain concrete visual evidence that contradicts
  an input card classification. An input card needs nonempty `card_text` or a
  nonempty evidence-based `card_misflag`; do not use this to avoid transcription.

Read the numbered contact sheets in order. Each tile has its own shot ID and
requested timestamp; the manifest records the actual extraction window. A still
does not establish motion or exact transition timing. Inspect listed individual
frames, `cards/` stills or nearby `reference/golden.mp4` frames when necessary.
Do not replace per-shot coverage with uniformly spaced video samples.
For EVERY shot with `dur_s < 0.75`, individually Read its high-resolution
`frame_path` from the manifest before saving that sheet's batch; the contact-sheet
thumbnail alone is insufficient. When an image is blurred or occluded, describe
only conservative visible facts. Do not infer off-screen people, actions or
identities, and do not invent accessories or details that are not clearly visible.

This is one independent batch of at most 12 shots. Annotate only the IDs in these
batch inputs, even if a sheet also shows other tiles. Do not inspect other batches
or edit any global draft or other annotation files. Start this batch output with
the input film slug and an empty shots array. Inspect its pictures and save those shot
entries BEFORE opening the next sheet. Add at most 12 new entries per save. Do
not prefill descriptions of unseen shots or postpone all writing until the end.
Preserve the draft if interrupted; it is a checkpoint, not a validated result.

Finish only when this batch includes every supplied shot once, in order, with all
required visual fields. Keep dialogue, timing, kind and continuation
offsets in the unchanged inputs; they need not be copied to this draft. Do not
generate logline, beats or chunks in this stage. Do not write stages/script.json.
After checking the complete shot draft, end with a brief completion message. If
the evidence is insufficient, report the limitation instead of inventing content.
"""


def _nonempty_text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_draft(inputs, draft):
    """Return contract errors for a shot draft; an empty list means it passes.

    Input ``line`` values are measured objects containing ``text`` or null.
    Output ``line`` values must be their exact text/null projection. Additional
    draft metadata is allowed, but never used by ``merge_structure``.
    """
    if not isinstance(inputs, dict) or not isinstance(draft, dict):
        return ["Inputs and draft must be JSON objects"]
    errors = []
    film = inputs.get("film")
    if not _nonempty_text(film) or draft.get("film") != film:
        errors.append("Draft film must exactly match the nonempty input film slug")
    source = inputs.get("shots")
    shots = draft.get("shots")
    if not isinstance(source, list) or not source:
        return errors + ["Inputs must contain a nonempty shots array"]
    if not isinstance(shots, list):
        return errors + ["Draft must contain a shots array"]

    expected_ids = []
    measured_lines = {}
    kinds = {}
    for index, shot in enumerate(source):
        if not isinstance(shot, dict) or not _nonempty_text(shot.get("id")):
            return errors + [f"Input shot {index + 1} needs a nonempty string ID"]
        sid = shot["id"]
        if sid in measured_lines:
            return errors + [f"Input contains duplicate shot ID {sid}"]
        expected_ids.append(sid)
        kinds[sid] = shot.get("kind")
        line = shot.get("line")
        if "line" not in shot or (line is not None and
                (not isinstance(line, dict) or not isinstance(line.get("text"), str))):
            return errors + [f"Input {sid}.line must be null or an object with string text"]
        measured_lines[sid] = None if line is None else line["text"]

    actual_ids = [shot.get("id") if isinstance(shot, dict) else None for shot in shots]
    if actual_ids != expected_ids:
        errors.append("Draft shot IDs must exactly match every input ID once, in order")
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            errors.append(f"Draft shot {index + 1} must be an object")
            continue
        sid = shot.get("id")
        if not isinstance(sid, str) or sid not in measured_lines:
            errors.append(f"Draft shot {index + 1} has an unknown or invalid ID")
            continue
        if not _nonempty_text(shot.get("desc")):
            errors.append(f"{sid} needs a nonempty visual description")
        if "line" not in shot or shot["line"] != measured_lines[sid]:
            errors.append(f"{sid}.line must preserve the exact supplied text/null")
        if kinds[sid] == "card" and not (
                _nonempty_text(shot.get("card_text")) or
                _nonempty_text(shot.get("card_misflag"))):
            errors.append(f"{sid} is a card: transcribe card_text or explain card_misflag")
    return errors


def validate_prefix(inputs, draft):
    """Validate a complete prefix without accepting gaps, duplicates or extras."""
    if not isinstance(inputs, dict) or not isinstance(draft, dict):
        return ["Inputs and draft must be JSON objects"]
    source, shots = inputs.get("shots"), draft.get("shots")
    if not isinstance(source, list) or not source or not isinstance(shots, list):
        return ["Inputs need nonempty shots and draft needs a shots array"]
    if len(shots) > len(source):
        return ["Draft prefix contains more shots than the input"]
    if not shots:
        return [] if _nonempty_text(inputs.get("film")) and draft.get("film") == inputs["film"] else [
            "Draft film must exactly match the input film slug"]
    return validate_draft({"film": inputs.get("film"), "shots": source[:len(shots)]}, draft)


def canonicalize_batch_lines(inputs, raw):
    """Insert measured dialogue after exact film/ordered-ID checks.

    Only ``line`` is replaced; model-authored visual fields are copied unchanged.
    This accepts exactly the supplied input subset, not an implicit prefix. It is
    not used to silently repair a previously saved global draft on resume.
    """
    if not isinstance(inputs, dict) or not isinstance(raw, dict):
        raise ValueError("Batch inputs and output must be JSON objects")
    source, shots = inputs.get("shots"), raw.get("shots")
    if (not _nonempty_text(inputs.get("film")) or raw.get("film") != inputs["film"]
            or not isinstance(source, list) or not source or not isinstance(shots, list)
            or any(not isinstance(s, dict) for s in source + shots)):
        raise ValueError("Batch film and shot arrays must match the supplied input")
    expected = [s.get("id") for s in source]
    if (any(not _nonempty_text(sid) for sid in expected) or len(set(expected)) != len(expected)
            or [s.get("id") for s in shots] != expected):
        raise ValueError("Batch shot IDs must exactly match every supplied ID once, in order")
    result = deepcopy(raw)
    for original, shot in zip(source, result["shots"]):
        line = original.get("line")
        if "line" not in original or (line is not None and
                (not isinstance(line, dict) or not isinstance(line.get("text"), str))):
            raise ValueError(f"Input {original['id']}.line must be null or an object with string text")
        shot["line"] = None if line is None else line["text"]
    return result


def make_shot_batch(inputs, manifest, start, stop, *, task_dir, inputs_path, output_path):
    """Create measured input and visual-manifest subsets for one bounded session.

    Media paths remain relative to the task root. Sheet pixels may include an
    already validated prefix, but only selected IDs are listed for annotation.
    """
    shots, records = inputs["shots"], manifest["shots"]
    if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int)
            or not isinstance(stop, int) or not 0 <= start < stop <= len(shots)
            or stop - start > SHOT_BATCH_SIZE):
        raise ValueError("Shot batch must contain 1 to 12 consecutive input shots")
    if [r["id"] for r in records] != [s["id"] for s in shots]:
        raise ValueError("Visual manifest shot IDs do not match current inputs")
    selected = deepcopy(shots[start:stop])
    wanted = {s["id"] for s in selected}
    batch_inputs = {"film": inputs["film"], "n_shots": len(selected),
                    "note": "Measured inputs for this batch only; do not modify or re-derive.",
                    "shots": selected}
    task = Path(task_dir).resolve()
    relative = lambda path: Path(path).resolve().relative_to(task).as_posix()
    batch_manifest = {
        "version": manifest.get("version"),
        "parent_visual_fingerprint": manifest.get("fingerprint"),
        "inputs_path": relative(inputs_path), "draft_path": relative(output_path),
        "video_path": manifest.get("video_path", "reference/golden.mp4"),
        "layout": deepcopy(manifest.get("layout", {})),
        "notes": deepcopy(manifest.get("notes", [])),
        "shots": deepcopy(records[start:stop]),
        "sheets": [{**deepcopy(sheet), "shot_ids": [sid for sid in sheet["shot_ids"] if sid in wanted]}
                   for sheet in manifest["sheets"] if wanted.intersection(sheet["shot_ids"])],
    }
    if {sid for sheet in batch_manifest["sheets"] for sid in sheet["shot_ids"]} != wanted:
        raise ValueError("Visual manifest sheets do not cover the selected batch")
    return batch_inputs, batch_manifest


def build_structure_prompt(frozen_draft_path, inputs_path, structure_path):
    """Return a fresh-session prompt that produces only narrative structure."""
    return f"""# Organize an already inspected shot log

Read the frozen shot draft at {_quoted_path(frozen_draft_path)} and measured inputs
at {_quoted_path(inputs_path)}. The draft already contains the complete ordered
shot descriptions and supplied dialogue. Treat both files as read-only. Do not
rewrite shots, repeat the full contact-sheet inspection, or regenerate dialogue.
Use the full shot log and supplied speech to reason about sequence continuity.
You may inspect an individual listed image or a nearby `reference/golden.mp4`
frame if a specific boundary remains ambiguous. Do not delegate.

Write all newly authored summaries, briefs, and explanations in English;
preserve IDs and proper names exactly.

Write one JSON object to {_quoted_path(structure_path)}, containing exactly these
three top-level fields (no `film`, `shots` or other fields):
- `logline`: a concise summary grounded in the supplied visible sequence.
- `beats`: ordered objects with `id`, `name`, `shots`, `what`, `function`, and
  `why_here`. Each `shots` is [first_shot_id, last_shot_id], a closed interval.
  `what` describes what occurs, `function` its narrative role, and `why_here`
  gives concrete evidence for the boundary. All text fields must be nonempty.
- `chunks`: ordered editing assignments with `id`, `beat`, `shots`, `what`, and
  `why_here`. `beat` is the containing beat ID; `shots` is a closed two-ID
  interval. `what` is a concrete local editing brief; `why_here` supports its
  boundary. All text fields must be nonempty.

Beats must cover every input shot exactly once in order, with no gaps or overlaps.
Chunks must do the same and stay entirely inside their named parent beat. Use at
least two beats and nonempty chunks. When a containing beat exceeds 15 shots,
each of its chunks must contain at most 15 shots. Preserve coherent boundaries;
these inherited construction heuristics do not prove that the sequence naturally
has a particular structure. If the evidence cannot support the contract, report
that limitation for human review instead of inventing content.

Review IDs, interval coverage and boundaries, then save the structure object and
end with a brief completion message. Do not write stages/script.json. The caller
will merge your structure with the unchanged frozen shots and validate the full
result before publishing it.
"""


def merge_structure(draft, structure):
    """Combine structure with a validated draft without sharing mutable objects.

    The caller must validate the draft first and validate complete coverage after
    this merge. Reject unexpected structure keys instead of letting model output
    replace the film identity or any shot annotation.
    """
    if not isinstance(draft, dict) or not _nonempty_text(draft.get("film")) or not isinstance(draft.get("shots"), list):
        raise ValueError("Draft must contain a nonempty film slug and a shots array")
    if not isinstance(structure, dict):
        raise ValueError("Structure must be a JSON object")
    allowed = {"logline", "beats", "chunks"}
    if set(structure) != allowed:
        missing = sorted(allowed - set(structure))
        extra = sorted(set(structure) - allowed)
        raise ValueError(f"Structure must contain only logline, beats, chunks; missing={missing}, extra={extra}")
    if not _nonempty_text(structure["logline"]):
        raise ValueError("Structure logline must be nonempty text")
    if not isinstance(structure["beats"], list) or not isinstance(structure["chunks"], list):
        raise ValueError("Structure beats and chunks must be arrays")
    return {"film": draft["film"], "shots": deepcopy(draft["shots"]),
            **deepcopy(structure)}


def atomic_write_json(path, document):
    """Write strict UTF-8 JSON beside the target and replace it atomically.

    Serialization or filesystem failure leaves an existing target untouched and
    removes this helper's temporary file. The caller owns path/scope validation.
    """
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         prefix=f".{target.name}-", suffix=".tmp",
                                         dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
