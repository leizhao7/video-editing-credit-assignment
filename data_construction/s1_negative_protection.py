"""Public construction draft: s1_negative_protection.py.

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
import csv
from functools import lru_cache
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

POLICY_VERSION = "s1-audit-negative-protection-v1"
AUDIT_DIR = Path("review_evidence/s1")
AUDIT_FILES = ("model_results.json", "blind_model_results.json")
_SLUG = re.compile(r"[a-z0-9_]+")
_TASK = re.compile(r"s_1_([a-z0-9_]+)_(c[1-9]\d*)")
_SID = re.compile(r"s\d{4,}")
_CID = re.compile(r"c(?:0[1-9]|[1-9]\d+)")
_HASH = re.compile(r"[0-9a-f]{64}")
_LIVE = {"live_action", "mixed"}
_TYPES = _LIVE | {"pure_card", "pure_black", "uncertain"}


class NegativeProtectionError(ValueError):
    """Audit provenance is present but cannot safely protect the negative pool."""


def _fail(message):
    raise NegativeProtectionError(message)


def _stamp(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _path(root, path):
    path = Path(path)
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        _fail("Evidence path is outside project root")
    if ".." in parts:
        _fail("Parent traversal in evidence path")
    cursor = root
    for part in parts:
        cursor /= part
        if cursor.is_symlink():
            _fail("Symlinked audit/original evidence: " + str(cursor))
    if not path.is_file() or path.stat().st_size == 0:
        _fail("Missing or empty audit/original evidence: " + str(path))
    return path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("Duplicate JSON key: " + key)
        result[key] = value
    return result


@lru_cache(maxsize=4096)
def _cached_file(path, stamp, kind):
    """Cache by stat identity, never just filename; parsed objects stay private."""
    h, blocks = hashlib.sha256(), []
    try:
        with open(path, "rb") as handle:
            if _stamp(os.fstat(handle.fileno())) != stamp:
                _fail("Evidence changed while opening: " + path)
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
                if kind != "hash":
                    blocks.append(block)
            if _stamp(os.fstat(handle.fileno())) != stamp:
                _fail("Evidence changed while reading: " + path)
        if _stamp(Path(path).stat()) != stamp:
            _fail("Evidence changed during verification: " + path)
        if kind == "hash":
            value = None
        elif kind == "text":
            value = b"".join(blocks).decode("utf8")
        else:
            value = json.loads(b"".join(blocks).decode("utf8"), object_pairs_hook=_pairs,
                               parse_constant=lambda v: _fail("Nonfinite JSON: " + v))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail("Cannot verify evidence " + path + ": " + str(exc))
    return value, h.hexdigest()


def _read(root, path, dependencies, kind="json"):
    path = _path(root, path)
    stamp = _stamp(path.stat())
    result = _cached_file(str(path), stamp, kind)
    if _stamp(path.stat()) != stamp:
        _fail("Evidence changed after cache lookup: " + str(path))
    dependencies[str(path)] = stamp
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _number(value, name, positive=False):
    if isinstance(value, bool):
        _fail(name + " must not be boolean")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        _fail(name + " must be numeric")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        _fail(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def _task(task):
    match = _TASK.fullmatch(task) if isinstance(task, str) else None
    if not match:
        _fail("Invalid S1 task identity: " + repr(task))
    return match.groups()


def _hash(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        _fail("Invalid SHA-256 digest")
    return value


@lru_cache(maxsize=8)
def _audit_index(path, stamp):
    document, digest = _cached_file(path, stamp, "json")
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        _fail("Audit must contain a results array")
    index = {}
    for category in ("results", "blocked", "failed"):
        rows = document.get(category, [])
        if not isinstance(rows, list):
            _fail("Invalid audit terminal category")
        for wrapper in rows:
            if not isinstance(wrapper, dict):
                _fail("Audit terminal entry must be an object")
            row = wrapper if category == "results" else wrapper.get("failure", wrapper)
            if not isinstance(row, dict):
                _fail("Invalid wrapped terminal audit entry")
            task = row.get("task")
            _task(task)
            if wrapper.get("task") != task or task in index:
                _fail("Duplicate/conflicting audit task: " + task)
            index[task] = (category, row)
    return index, digest


def _observations(entry, bundle, blind):
    category, row = entry
    task = bundle["task"]
    if row.get("task") != task or row.get("input_sha256") != bundle["task_bundle_sha256"]:
        _fail("Audit input SHA/task mismatch: " + task)
    if not isinstance(row.get("model"), str) or not row["model"].strip():
        _fail("Missing audit model attribution: " + task)
    if (blind and row.get("blind") is not True) or (not blind and row.get("blind") is True):
        _fail("Audit round/blind identity mismatch: " + task)
    if category != "results":
        if row.get("complete") is not False:
            _fail("Failure entry claims completion: " + task)
        return {}, {}, row
    audit = row.get("audit")
    if row.get("complete") is not True or not isinstance(audit, dict) or audit.get("task") != task:
        _fail("Incomplete or mismatched successful audit: " + task)
    cids = set(bundle["candidate_sha256"])
    sids = {r["id"] for r in bundle["reference_shots"]}
    maps = []
    for field, expected, edge_field, allowed in (
            ("reference_observations", sids, "matches", cids),
            ("candidate_observations", cids, "reference_matches", sids)):
        rows = audit.get(field)
        if (not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows)
                or len(rows) != len(expected) or {r.get("id") for r in rows} != expected):
            _fail("Audit ID coverage/range mismatch: " + task + "/" + field)
        for observation in rows:
            edges = observation.get(edge_field)
            if (not isinstance(edges, list) or any(not isinstance(e, str) or e not in allowed for e in edges)
                    or len(edges) != len(set(edges))):
                _fail("Invalid audit match ID/range: " + task + "/" + observation["id"])
            if (not isinstance(observation.get("observed"), str) or not observation["observed"].strip()
                    or observation.get("certainty") not in {"high", "medium", "low"}):
                _fail("Invalid audit observation evidence: " + task)
            if field == "reference_observations" and observation.get("visual_type") not in _TYPES:
                _fail("Unknown reference visual type: " + task)
        maps.append({r["id"]: r for r in rows})
    return maps[0], maps[1], row


def _bundle(root, path, slug, chunk, expected_hash, dependencies):
    bundle, file_sha = _read(root, path, dependencies)
    task = f"s_1_{slug}_{chunk}"
    if (not isinstance(bundle, dict) or bundle.get("task") != task
            or bundle.get("film") != slug or bundle.get("chunk") != chunk):
        _fail("Bundle film/chunk/task identity mismatch: " + task)
    bundle_sha = _hash(bundle.get("task_bundle_sha256"))
    if _digest({k: v for k, v in bundle.items() if k != "task_bundle_sha256"}) != bundle_sha:
        _fail("Original bundle self-hash mismatch: " + task)
    if bundle_sha != expected_hash:
        _fail("Bundle/inventory SHA mismatch: " + task)
    directory = root / "training_task" / slug / "s_1" / chunk
    metadata, _ = _read(root, directory / "task.json", dependencies)
    if metadata.get("film") != slug or metadata.get("chunk") != chunk or metadata.get("stage") != "s1/clip_selection":
        _fail("Original task metadata mismatch: " + task)
    answer, answer_sha = _read(root, directory / "verifier/answer.json", dependencies)
    manifest, _ = _read(root, directory / "initial_state/manifest.json", dependencies)
    prompt, _ = _read(root, directory / "task_prompt.md", dependencies, "text")
    if answer != bundle.get("existing_answer"):
        _fail("Original answer content differs from audit bundle: " + task)
    if manifest != bundle.get("manifest") or prompt != bundle.get("task_prompt"):
        _fail("Original manifest/prompt differs from audit bundle: " + task)
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else None
    key = answer.get("key") if isinstance(answer, dict) else None
    if (not isinstance(candidates, list) or not candidates or not isinstance(key, list)
            or any(not isinstance(c, dict) for c in candidates + key)):
        _fail("Invalid original candidate/key schema: " + task)
    # Candidate IDs may be noncontiguous after deletions.
    # Use the hash-bound manifest membership rather than renumbering original evidence.
    
    
    expected = {c.get("id") for c in candidates}
    if (len(expected) != len(candidates)
            or any(not isinstance(cid, str) or not _CID.fullmatch(cid) for cid in expected)):
        _fail("Original candidate ID coverage/range mismatch: " + task)
    for rows in (candidates, key):
        if len(rows) != len(expected) or {r.get("id") for r in rows} != expected:
            _fail("Original candidate ID coverage/range mismatch: " + task)
    if (isinstance(manifest.get("n_candidates"), bool) or manifest.get("n_candidates") != len(expected)
            or isinstance(answer.get("n_candidates"), bool) or answer.get("n_candidates") != len(expected)
            or not isinstance(bundle.get("candidate_sha256"), dict)
            or set(bundle["candidate_sha256"]) != expected):
        _fail("Original candidate count/hash coverage mismatch: " + task)
    for candidate in candidates:
        cid = candidate["id"]
        if candidate.get("file") != f"clips/{cid}.mp4":
            _fail("Unsafe original candidate file: " + cid)
        _, media_sha = _read(root, directory / "initial_state" / candidate["file"], dependencies, "hash")
        if media_sha != _hash(bundle["candidate_sha256"][cid]):
            _fail("Original candidate SHA mismatch: " + task + "/" + cid)
    references = bundle.get("reference_shots")
    if (not isinstance(references, list) or not references
            or any(not isinstance(r, dict) or not isinstance(r.get("id"), str)
                   or not _SID.fullmatch(r["id"]) for r in references)
            or len({r["id"] for r in references}) != len(references)):
        _fail("Invalid reference ID coverage: " + task)
    for reference in references:
        timing = reference.get("timing")
        if not isinstance(timing, dict) or timing.get("id") != reference["id"]:
            _fail("Reference timing identity mismatch: " + task)
        _number(timing.get("start_s"), "reference start")
        _number(timing.get("dur_s"), "reference duration", True)
    if bundle.get("reference_file") != f"references/{slug}.mp4":
        _fail("Invalid canonical reference path: " + task)
    _, reference_sha = _read(root, root / AUDIT_DIR / "inputs" / bundle["reference_file"], dependencies, "hash")
    if reference_sha != _hash(bundle.get("reference_sha256")):
        _fail("Canonical reference SHA mismatch: " + task)
    return bundle, {c["id"]: c for c in key}, {"bundle_file": str(path),
        "bundle_file_sha256": file_sha, "task_bundle_sha256": bundle_sha,
        "original_answer_path": str(directory / "verifier/answer.json"),
        "original_answer_sha256": answer_sha,
        "original_answer_binding": "raw SHA recorded after parsed content equality with audit snapshot",
        "canonical_reference_sha256": reference_sha}


def load_audit_negative_protection(root, slug):
    """Return ``{chunk: {film_source_indices, trailer_intervals, evidence}}``.

    Empty/non-audited synthetic roots return {}. Once evidence exists for this
    film, missing/inconsistent input or either audit fails closed by exception.
    A declared blocked/failed round may contribute no observations; the successful
    round still protects its reported matches. This does not approve that task.
    All returned objects are caller-owned copies. No selection labels are returned.
    """
    root = Path(root).resolve()
    if not isinstance(slug, str) or not _SLUG.fullmatch(slug):
        _fail("Invalid film slug")
    base, dependencies = root / AUDIT_DIR, {}
    audit_paths = [base / name for name in AUDIT_FILES]
    bundle_paths = sorted((base / "inputs/tasks").glob(f"s_1_{slug}_c*.json"))
    inventory_path = base / "inputs/inventory.json"
    if (not bundle_paths and not any(p.exists() or p.is_symlink() for p in audit_paths)
            and not inventory_path.exists() and not inventory_path.is_symlink()):
        return {}
    audits = []
    for path in audit_paths:
        path = _path(root, path)
        stamp = _stamp(path.stat())
        index, digest = _audit_index(str(path), stamp)
        if _stamp(path.stat()) != stamp:
            _fail("Audit changed after index lookup: " + str(path))
        dependencies[str(path)] = stamp
        audits.append((index, {"path": str(path), "sha256": digest}))
    audited_tasks = {task for index, _ in audits for task in index if _task(task)[0] == slug}
    if not audited_tasks and not bundle_paths:
        return {}
    inventory, inventory_sha = _read(root, base / "inputs/inventory.json", dependencies)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("tasks"), list):
        _fail("Invalid audit input inventory")
    entries = {}
    for row in inventory["tasks"]:
        if not isinstance(row, dict):
            _fail("Invalid inventory entry")
        task = row.get("task")
        _task(task)
        if task in entries or row.get("file") != f"tasks/{task}.json":
            _fail("Duplicate/unsafe inventory identity")
        entries[task] = _hash(row.get("sha256"))
    targets = {task for task in entries if _task(task)[0] == slug}
    original_tasks = {f"s_1_{slug}_{p.parent.name}" for p in
                      (root / "training_task" / slug / "s_1").glob("c*/task.json")}
    if targets != audited_tasks or targets != {p.stem for p in bundle_paths} or targets != original_tasks:
        _fail("Original/bundle/audit task coverage mismatch for " + slug)
    source_text, source_manifest_sha = _read(root, root / "exports" / slug / "source_manifest.csv", dependencies, "text")
    source_rows = list(csv.DictReader(io.StringIO(source_text)))
    if not source_rows or any(row.get("id") != f"s{i:04d}" for i, row in enumerate(source_rows)):
        _fail("Source manifest index/ID mismatch")
    result = {}
    for task in sorted(targets):
        _, chunk = _task(task)
        bundle, candidates, identity = _bundle(root, base / "inputs/tasks" / (task + ".json"),
                                             slug, chunk, entries[task], dependencies)
        rounds = []
        for blind, (index, source) in enumerate(audits):
            if task not in index:
                _fail("Missing audit terminal record: " + task)
            refs, observations, record = _observations(index[task], bundle, bool(blind))
            rounds.append((refs, observations, record, dict(source, round="blind" if blind else "answer_aware")))
        narrative = {sid for refs, _, _, _ in rounds for sid, ref in refs.items() if ref["visual_type"] in _LIVE}
        protected, intervals, evidence = set(), set(), []
        unavailable = [src["round"] for _, _, rec, src in rounds if rec.get("complete") is not True]
        for _, _, rec, src in rounds:
            if rec.get("complete") is not True:
                evidence.append({"kind": "audit_round_unavailable", "task": task,
                    "policy_version": POLICY_VERSION, "audit_source": src,
                    "model": rec["model"], "audit_input_sha256": rec["input_sha256"],
                    "status": rec.get("status", "failed"), "failure_class": rec.get("failure_class"),
                    "available_evidence_only": True,
                    "blind_blocked": src["round"] == "blind" and rec.get("status") == "blocked",
                    "task_quality_approved": False,
                    "explanation": "Use reported matches from available rounds only; a provider block is not a bad-task judgment."})
        for refs, observations, record, source in rounds:
            for cid, observation in observations.items():
                matches = sorted(set(observation["reference_matches"]) & narrative)
                if not matches:
                    continue
                candidate = candidates[cid]
                if not isinstance(candidate.get("answer"), bool):
                    _fail("Original candidate label must be explicit boolean: " + task + "/" + cid)
                start = _number(candidate.get("src_t0"), "candidate source start")
                duration = _number(candidate.get("dur_s"), "candidate source duration", True)
                if not math.isfinite(start + duration):
                    _fail("Candidate interval end is not finite: " + task + "/" + cid)
                origin, index = candidate.get("origin"), candidate.get("src_shot")
                if origin == "film":
                    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(source_rows):
                        _fail("Protected candidate source index outside manifest: " + task + "/" + cid)
                    protected.add(index)
                elif origin == "trailer_only":
                    if index is not None:
                        _fail("Trailer-only candidate must have src_shot=None")
                    intervals.add((start, duration))
                else:
                    _fail("Unknown protected candidate origin: " + task + "/" + cid)
                evidence.append({"kind": "candidate_negative_protection", "policy_version": POLICY_VERSION,
                    "task": task, "candidate_id": cid, "available_evidence_only": bool(unavailable),
                    "unavailable_rounds": unavailable,
                    "protection_only": True, "positive_adoption_authorized": False,
                    "attribution": "model_reported_possible_match_not_human_verified_truth",
                    "model": record["model"], "audit_source": source,
                    "audit_input_sha256": record["input_sha256"],
                    "audit_recipe": record.get("audit_recipe"),
                    "candidate_observation": copy.deepcopy(observation),
                    "protected_reference_matches": matches,
                    "reference_evidence": [{"reference_id": sid, "round": src["round"],
                        "audit_source_sha256": src["sha256"], "model": rec["model"],
                        "observation": copy.deepcopy(rr[sid])}
                        for sid in matches for rr, _, rec, src in rounds if sid in rr],
                    "original_candidate": copy.deepcopy(candidate),
                    "original_candidate_sha256": bundle["candidate_sha256"][cid],
                    "input_identity": dict(identity, inventory_sha256=inventory_sha,
                                           source_manifest_sha256=source_manifest_sha)})
        result[chunk] = {"film_source_indices": sorted(protected),
                         "trailer_intervals": [{"start_s": s, "duration_s": d} for s, d in sorted(intervals)],
                         "evidence": evidence}
    for path, stamp in dependencies.items():
        if _stamp(_path(root, Path(path)).stat()) != stamp:
            _fail("Evidence changed before return: " + path)
    return copy.deepcopy(result)
