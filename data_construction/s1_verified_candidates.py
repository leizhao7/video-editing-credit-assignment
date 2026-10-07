"""Public construction draft: s1_verified_candidates.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""
try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

POLICY_VERSION = "s1-verified-candidate-v1"
_FIELDS = {"task_key", "candidate_id", "original_answer_sha256",
           "original_media_sha256", "evidence"}
_SLUG = re.compile(r"[a-z0-9_]+")
_TASK = re.compile(r"([a-z0-9_]+)/(c[1-9]\d*)")
_CID = re.compile(r"c\d{2,}")
_SID = re.compile(r"s\d{4,}")
_HASH = re.compile(r"[0-9a-fA-F]{64}")
_EPS = 1e-6  # float arithmetic only; never a frame/90-ms boundary tolerance


class VerifiedCandidateError(ValueError):
    """An explicit candidate reference failed identity or span validation."""


def _fail(message):
    raise VerifiedCandidateError(message)


def _number(value, name, *, positive=False):
    if isinstance(value, bool):
        _fail(name + " must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail(name + " must be finite")
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        _fail(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return number


def parse_verified_candidate(value, film):
    """Pure strict parsing of the explicit reviewer authorization object."""
    if not isinstance(film, str) or not _SLUG.fullmatch(film):
        _fail("Invalid film slug")
    if not isinstance(value, dict) or set(value) != _FIELDS:
        _fail("verified_candidate requires exactly: " + ", ".join(sorted(_FIELDS)))
    task = value["task_key"]
    match = _TASK.fullmatch(task) if isinstance(task, str) else None
    if not match or match.group(1) != film:
        _fail("verified_candidate must reference an old S1 task from the same film")
    cid = value["candidate_id"]
    if not isinstance(cid, str) or not _CID.fullmatch(cid):
        _fail("Invalid candidate_id")
    evidence = value["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        _fail("Explicit nonempty reviewer evidence is required")
    result = dict(value)
    result["evidence"] = evidence.strip()
    for field in ("original_answer_sha256", "original_media_sha256"):
        if not isinstance(value[field], str) or not _HASH.fullmatch(value[field]):
            _fail(field + " must be a full SHA-256 hex digest")
        result[field] = value[field].lower()
    return result


def validate_verified_candidate(value, *, film, candidate, source_manifest,
                                media_sources, actual_answer_sha256,
                                actual_media_sha256):
    """Pure validation; return exact ``src,t0,dur,crop,origin,src_shot``.

    ``candidate`` is the matching private answer.key row. ``media_sources`` maps
    film/trailer_only to {src, crop, duration}. Source indices are positional,
    matching the builder; the selected manifest row ID must also equal that index.
    Film spans must stay inside that declared shot. Partial/capped spans are valid,
    but are preserved verbatim, not expanded. Trailer spans must fit the trailer
    duration and declare src_shot=None. Inputs are never mutated.
    """
    ref = parse_verified_candidate(value, film)
    if actual_answer_sha256 != ref["original_answer_sha256"]:
        _fail("Original answer SHA-256 mismatch")
    if actual_media_sha256 != ref["original_media_sha256"]:
        _fail("Original candidate media SHA-256 mismatch")
    if not isinstance(candidate, dict) or candidate.get("id") != ref["candidate_id"]:
        _fail("Private candidate ID does not match explicit reference")
    if not isinstance(candidate.get("answer"), bool):
        _fail("Old candidate must have an explicit boolean label (true or false)")
    origin = candidate.get("origin")
    if origin not in {"film", "trailer_only"}:
        _fail("Only film or trailer_only candidates can be adopted")
    if not isinstance(media_sources, dict) or not isinstance(media_sources.get(origin), dict):
        _fail("Missing source metadata for candidate origin")
    media = media_sources[origin]
    src = media.get("src")
    if not isinstance(src, str) or not src or not Path(src).is_absolute():
        _fail("Media source must be an absolute local path")
    crop = media.get("crop", "")
    if not isinstance(crop, str) or "\x00" in crop or "\n" in crop or "\r" in crop:
        _fail("Invalid crop expression")
    duration = _number(media.get("duration"), "source video duration", positive=True)
    t0 = _number(candidate.get("src_t0"), "candidate src_t0")
    dur = _number(candidate.get("dur_s"), "candidate dur_s", positive=True)
    end = t0 + dur
    if not math.isfinite(end) or end > duration + _EPS:
        _fail("Candidate span exceeds source video duration")
    index = candidate.get("src_shot")
    if origin == "film":
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            _fail("Film candidate src_shot must be a nonnegative integer")
        if not isinstance(source_manifest, (list, tuple)) or index >= len(source_manifest):
            _fail("Film candidate src_shot is outside source_manifest")
        row = source_manifest[index]
        expected_id = f"s{index:04d}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            _fail("Declared source index disagrees with manifest row ID")
        if sum(isinstance(r, dict) and r.get("id") == expected_id for r in source_manifest) != 1:
            _fail("Duplicate declared source ID in source_manifest")
        if candidate.get("source_clip", expected_id) != expected_id:
            _fail("Candidate source_clip disagrees with its source index")
        start = _number(row.get("start_s"), "source shot start_s")
        stop = _number(row.get("end_s"), "source shot end_s", positive=True)
        if stop <= start or stop > duration + _EPS:
            _fail("Invalid declared source shot interval")
        if t0 < start - _EPS or t0 >= stop or end > stop + _EPS:
            _fail("Candidate span crosses the declared source shot boundary")
    elif index is not None:
        _fail("Trailer-only candidate must declare src_shot=None")
    return {"src": src, "t0": t0, "dur": dur, "crop": crop,
            "origin": origin, "src_shot": index}


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON object key: " + str(key))
        result[key] = value
    return result


def _local_file(path, root):
    """Refuse symlink indirection out of (or within) the original task tree."""
    path = Path(path)
    if ".." in path.parts:
        _fail("Parent traversal in original file path is not allowed")
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail("File is outside the declared project root")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail("Symlinked original provenance is not allowed: " + str(cursor))
    if not path.is_file() or path.stat().st_size <= 0:
        _fail("Missing or empty original file: " + str(path))
    return path


def _stamp(stat):
    return {"dev": stat.st_dev, "ino": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _read_stable(path, root, *, json_data=False):
    path = _local_file(path, root)
    digest, chunks = hashlib.sha256(), []
    before = _stamp(path.stat())
    with path.open("rb") as stream:
        if _stamp(os.fstat(stream.fileno())) != before:
            _fail("File changed while opening: " + str(path))
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if json_data:
                chunks.append(block)
        if _stamp(os.fstat(stream.fileno())) != before:
            _fail("File changed while reading: " + str(path))
    if _stamp(path.stat()) != before:
        _fail("File changed during verification: " + str(path))
    content = None
    if json_data:
        try:
            content = json.loads(b"".join(chunks).decode("utf8"),
                                 object_pairs_hook=_no_duplicate_object,
                                 parse_constant=lambda text: _fail("Nonfinite JSON value: " + text))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _fail("Invalid original JSON: " + str(path) + ": " + str(exc))
    return content, digest.hexdigest(), before


def _media_sources(root, slug, probe):
    if not isinstance(probe, dict) or probe.get("slug") != slug:
        _fail("Probe must explicitly identify the current film slug")
    directory = probe.get("dir")
    if not isinstance(directory, str) or not Path(directory).is_absolute():
        _fail("Probe dir must be absolute")
    directory = Path(directory)
    result, stamps = {}, {}
    for origin, name_field, info_field, crop_field in (
            ("film", "source", "src", "src_crop_expr"),
            ("trailer_only", "golden", "gold", "crop_expr")):
        name = probe.get(name_field)
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            _fail("Probe source filenames must be local basenames")
        path = _local_file(directory / name, root)
        info = probe.get(info_field)
        if not isinstance(info, dict):
            _fail("Probe has no duration metadata for " + origin)
        duration = _number(info.get("vdur", info.get("dur")), "probe video duration", positive=True)
        crop = probe.get(crop_field, "") or ""
        if origin == "trailer_only" and not crop and probe.get("crop"):
            ratio = _number(probe["crop"], "trailer crop ratio", positive=True)
            crop = f"crop=iw:iw/{ratio}:0:(ih-iw/{ratio})/2"
        result[origin] = {"src": str(path), "crop": crop, "duration": duration}
        stamps[origin] = {"path": str(path), **_stamp(path.stat())}
    return result, stamps


def resolve_verified_candidates(root, slug, source_manifest, shot_overrides,
                                probe=None, known_shot_ids=None):
    """Read-only resolver for private builder integration.

    ``shot_overrides`` is this film's {shot_id: override} mapping. Entries without
    verified_candidate are ignored. If known_shot_ids is omitted, the original
    script_inputs.json is read to reject nonexistent target shots. An optional probe
    must be the current film's trusted export metadata; otherwise it is read locally.

    Returns ``{spans, protected_source_indices, audit, policy_version}``. The caller
    must use these exact spans as positive material before p2/legacy routing, dedupe
    equivalent adopted spans, and union protected indices into both distractor
    selectors. This function does not change answer labels or publish anything.
    """
    root = Path(root).resolve()
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        _fail("Invalid film slug")
    if not isinstance(shot_overrides, dict):
        _fail("shot_overrides must be a per-film mapping")
    selected = []
    for sid, outer in shot_overrides.items():
        if not isinstance(sid, str) or not _SID.fullmatch(sid) or not isinstance(outer, dict):
            _fail("Invalid shot override entry")
        if "verified_candidate" not in outer:
            continue
        if outer.get("classification") in {"card", "featureless", "mixed_card_footage"}:
            _fail("A verified candidate conflicts with an excluded/unresolved classification")
        selected.append((sid, outer, parse_verified_candidate(outer["verified_candidate"], slug)))
    result = {"spans": {}, "protected_source_indices": [], "audit": [], "policy_version": POLICY_VERSION}
    if not selected:
        return result
    if known_shot_ids is None:
        inputs, _, _ = _read_stable(root / f"edit/task_{slug}/stages/script_inputs.json", root, json_data=True)
        if not isinstance(inputs, dict) or not isinstance(inputs.get("shots"), list):
            _fail("Invalid original script_inputs")
        known_shot_ids = [s.get("id") for s in inputs["shots"] if isinstance(s, dict)]
    if isinstance(known_shot_ids, str):
        _fail("known_shot_ids must be a collection, not a string")
    known = set(known_shot_ids)
    if any(sid not in known for sid, _, _ in selected):
        _fail("Unknown target shot for verified candidate")
    if probe is None:
        probe, _, _ = _read_stable(root / f"exports/{slug}/probe.json", root, json_data=True)
    media_sources, source_stamps = _media_sources(root, slug, probe)
    protected = set()
    for sid, outer, ref in selected:
        chunk = ref["task_key"].split("/")[1]
        tdir = root / "training_task" / slug / "s_1" / chunk
        task, _, _ = _read_stable(tdir / "task.json", root, json_data=True)
        if not isinstance(task, dict) or task.get("film") != slug or task.get("chunk") != chunk:
            _fail("Original task metadata does not identify the requested film/chunk")
        if task.get("stage") != "s1/clip_selection":
            _fail("Only original S1 clip-selection tasks may supply candidates")
        answer_path = tdir / "verifier/answer.json"
        answer, answer_hash, answer_stamp = _read_stable(answer_path, root, json_data=True)
        if answer_hash != ref["original_answer_sha256"]:
            _fail("Original answer SHA-256 mismatch")
        expected_task = f"{slug}_s1_{chunk}"
        if not isinstance(answer, dict) or answer.get("task") != expected_task:
            _fail("Original answer task identity mismatch")
        key = answer.get("key")
        if not isinstance(key, list) or any(not isinstance(k, dict) for k in key):
            _fail("Invalid private candidate key list")
        ids = [k.get("id") for k in key]
        if any(not isinstance(k, str) for k in ids) or len(ids) != len(set(ids)):
            _fail("Duplicate or invalid private candidate IDs")
        labels = [k.get("answer") for k in key]
        selected_ids = answer.get("answer")
        if (any(not isinstance(label, bool) for label in labels)
                or not isinstance(selected_ids, list)
                or any(not isinstance(cid, str) for cid in selected_ids)
                or len(selected_ids) != len(set(selected_ids))
                or set(selected_ids) != {k["id"] for k in key if k["answer"]}
                or answer.get("n_candidates") != len(key)
                or answer.get("n_answer") != len(selected_ids)):
            _fail("Original answer labels or counts are inconsistent")
        candidates = [k for k in key if k.get("id") == ref["candidate_id"]]
        if len(candidates) != 1:
            _fail("Candidate not present exactly once in original answer")
        candidate = candidates[0]
        manifest, manifest_hash, _ = _read_stable(tdir / "initial_state/manifest.json", root, json_data=True)
        if not isinstance(manifest, dict) or manifest.get("taskId") != expected_task:
            _fail("Original manifest task identity mismatch")
        public = manifest.get("candidates")
        if not isinstance(public, list) or any(not isinstance(k, dict) for k in public):
            _fail("Invalid public candidate list")
        public_ids = [k.get("id") for k in public]
        if any(not isinstance(k, str) for k in public_ids) or len(public_ids) != len(set(public_ids)) or set(public_ids) != set(ids):
            _fail("Public/private candidate IDs disagree or duplicate")
        if manifest.get("n_candidates") != len(public):
            _fail("Original manifest candidate count mismatch")
        row = next(k for k in public if k["id"] == ref["candidate_id"])
        if row.get("file") != f"clips/{ref['candidate_id']}.mp4":
            _fail("Candidate media path is not its canonical original clips path")
        if abs(_number(row.get("dur_s"), "public duration", positive=True) -
               _number(candidate.get("dur_s"), "private duration", positive=True)) > _EPS:
            _fail("Public/private candidate duration mismatch")
        media_path = tdir / "initial_state" / row["file"]
        _, media_hash, media_stamp = _read_stable(media_path, root)
        span = validate_verified_candidate(ref, film=slug, candidate=candidate,
            source_manifest=source_manifest, media_sources=media_sources,
            actual_answer_sha256=answer_hash, actual_media_sha256=media_hash)
        classification = outer.get("classification")
        if (classification == "source" and span["origin"] != "film") or (
                classification == "golden_only" and span["origin"] != "trailer_only"):
            _fail("Verified candidate origin conflicts with explicit classification")
        result["spans"][sid] = span
        if span["origin"] == "film":
            protected.add(span["src_shot"])
        result["audit"].append({"shot_id": sid, "reviewed_reference": copy.deepcopy(ref),
            "previous_candidate_label": candidate["answer"],
            "original_answer_path": str(answer_path), "original_answer_stat": answer_stamp,
            "original_media_path": str(media_path), "original_media_stat": media_stamp,
            "original_manifest_sha256": manifest_hash,
            "current_source_stat": source_stamps[span["origin"]],
            "adopted_span": dict(span), "private_only": True})
    result["protected_source_indices"] = sorted(protected)
    return result
