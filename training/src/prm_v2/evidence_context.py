"""Deterministic provenance cards for the isolated PRM judge experiment.

This module does not score actions, execute commands, inspect files, infer
semantic novelty, or decide whether a claimed artifact is correct. It operates
only on the canonical packet and transport's verified image receipts.
"""
from __future__ import annotations

import hashlib
import json
import re

try:
    from . import judge
except ImportError:  # unittest discovery / direct local import
    import judge


VERSION = "evidence_context_v1"
MAX_TURN = 10000
MAX_PAIRS_PER_CARD = 2048
TEXT_OBSERVATION_KINDS = frozenset({"tool_text", "tool_result", "observation"})
IMAGE_KINDS = frozenset({"image", "media_image"})
TRUNCATED_TEXT = "ARCHIVE TEXT TRUNCATED; unshown content is unknown."
OMITTED_MEDIA = "MEDIA NOT ATTACHED. Judge cannot inspect this evidence; omission="
ABBREVIATED_ACTION = "[Non-tool prose abbreviated; tool arguments preserved.]"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text_digest(value):
    # No whitespace collapse, case folding, JSON interpretation or fuzzy match.
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _action_text(action):
    return action if isinstance(action, str) else _canonical(action).decode("utf-8")


def _ledger(action):
    return isinstance(action, dict) and action.get("record_type") == "incomplete_history_ledger"


def _omitted(text):
    # transport may prepend the truncation marker before the omission marker.
    return text.startswith(OMITTED_MEDIA) or text.startswith(TRUNCATED_TEXT + "\n" + OMITTED_MEDIA)


def _receipts(value, known):
    """Validate externally verified receipts; do not claim to verify media here."""
    if value is None:
        return None
    if type(value) is not list:
        raise judge.ProtocolError("image_receipts must be a list or None")
    result = {}
    for receipt in value:
        if type(receipt) is not dict:
            raise judge.ProtocolError("image receipt must be an object")
        ident = receipt.get("id")
        if not isinstance(ident, str) or ident not in known or ident in result:
            raise judge.ProtocolError("image receipt ID is unknown or repeated")
        ev = known[ident]
        if ev["kind"] not in IMAGE_KINDS:
            raise judge.ProtocolError("image receipt must identify image evidence")
        if type(receipt.get("turn")) is not int or receipt["turn"] != ev["turn"]:
            raise judge.ProtocolError("image receipt turn does not match its evidence")
        sha = receipt.get("sha256")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise judge.ProtocolError("image receipt sha256 must be lowercase SHA-256")
        if "bytes" in receipt and (type(receipt["bytes"]) is not int or receipt["bytes"] <= 0):
            raise judge.ProtocolError("image receipt bytes must be a positive integer")
        if "mime_type" in receipt and receipt["mime_type"] not in {
                "image/jpeg", "image/png", "image/webp"}:
            raise judge.ProtocolError("unsupported image receipt MIME")
        if _omitted(ev.get("text", "")):
            raise judge.ProtocolError("attached receipt contradicts explicit omitted media")
        result[ident] = dict(receipt)
    return result


def _append_record(packet, known, record):
    ident = record["id"]
    if ident in known:
        if known[ident] != record:
            raise judge.ProtocolError("reserved evidence ID collides with different content: " + ident)
        return
    packet["evidence"].append(record)
    known[ident] = record


def prepare(packet, image_receipts=None):
    """Return {packet: augmented COPY, context: deterministic provenance sidecar}.

    Original canonical fields/evidence are preserved. Added ``task.original``
    and ``action.tNNN`` are requirements/actions, NOT environment observations.
    An opaque middle-history ledger is never promoted to a complete action.
    ``image_receipts=None`` means attachment status is unknown; ``[]`` means
    the transport attests that no images were attached. Receipts are not media
    bytes and this module relies on the caller's verification of their hashes.
    """
    augmented = judge.validate_packet(packet)
    last_turn = augmented["targets"][-1]["turn"]
    if last_turn > MAX_TURN:
        raise judge.ProtocolError("turn index exceeds bounded context-card limit")
    original_hash = _digest(augmented)
    known = {item["id"]: item for item in augmented["evidence"]}
    receipts = _receipts(image_receipts, known)
    _append_record(augmented, known, {
        "id": "task.original", "turn": 0,
        "kind": "task_requirement", "text": augmented["task"],
    })
    actions, incomplete_turns, recorded_turns = [], [], []
    for group in ("history", "targets"):
        for row in augmented[group]:
            turn, action = row["turn"], row["action"]
            recorded_turns.append(turn)
            if _ledger(action):
                incomplete_turns.append(turn)
                actions.append({"turn": turn, "source": group, "evidence_id": None,
                                "record_type": "incomplete_history_ledger",
                                "complete_provided_action": False,
                                "proves_execution": False})
                continue
            text = _action_text(action)
            ident = f"action.t{turn:03d}"
            _append_record(augmented, known, {
                "id": ident, "turn": turn, "kind": "action_record", "text": text,
            })
            actions.append({"turn": turn, "source": group, "evidence_id": ident,
                            "record_type": "action_record",
                            "complete_provided_action": ABBREVIATED_ACTION not in text,
                            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                            "proves_execution": False})

    index, fingerprints = {}, {}
    for ident, ev in sorted(known.items(), key=lambda kv: (kv[1]["turn"], kv[0])):
        kind, text = ev["kind"], ev.get("text", "")
        truncated = text.startswith(TRUNCATED_TEXT)
        item = {"turn": ev["turn"], "kind": kind, "category": "unknown",
                "availability": "unknown", "truncated": truncated}
        if kind in TEXT_OBSERVATION_KINDS:
            item.update(category="observation", availability="available" if "text" in ev else "unavailable")
            if "text" in ev and not truncated:
                item["normalized_text_sha256"] = _text_digest(text)
                fingerprints[ident] = ("line_ending_normalized_text", item["normalized_text_sha256"])
        elif kind in IMAGE_KINDS:
            item["category"] = "observation"
            if receipts is not None and ident in receipts:
                item.update(availability="available", sha256=receipts[ident]["sha256"])
                fingerprints[ident] = ("verified_image_sha256", item["sha256"])
            elif _omitted(text) or receipts is not None:
                item["availability"] = "unavailable"
            item["omission_marker"] = _omitted(text)
        elif kind == "action_record":
            item.update(category="action", availability="available",
                        truncated=ABBREVIATED_ACTION in text)
        elif kind == "task_requirement":
            item.update(category="requirement", availability="available")
        elif kind == "archive_limitation":
            item.update(category="limitation", availability="available")
        elif "text" in ev:
            # Unknown evidence kinds remain unknown; text presence alone does
            # not establish that this is an environment observation.
            item["availability"] = "available"
        index[ident] = item

    available_obs = [ident for ident, ev in index.items()
                     if ev["category"] == "observation" and ev["availability"] == "available"]
    action_ids = [a["evidence_id"] for a in actions if a["evidence_id"] is not None]
    cards = []
    for target in augmented["targets"]:
        turn = target["turn"]
        prior = [i for i in available_obs if index[i]["turn"] < turn]
        current = [i for i in available_obs if index[i]["turn"] == turn]
        pairs, pair_count = [], 0
        for current_id in current:
            fingerprint = fingerprints.get(current_id)
            if fingerprint is None:
                continue
            for prior_id in prior:
                if fingerprints.get(prior_id) == fingerprint:
                    pair_count += 1
                    if len(pairs) < MAX_PAIRS_PER_CARD:
                        pairs.append({"prior_id": prior_id, "current_id": current_id,
                                      "basis": fingerprint[0], "sha256": fingerprint[1]})
        prior_actions = [i for i in action_ids if index[i]["turn"] < turn]
        current_actions = [i for i in action_ids if index[i]["turn"] == turn]
        in_scope = [i for i, ev in index.items() if ev["turn"] <= turn]
        missing = [n for n in range(1, turn) if n not in recorded_turns]
        ledger = [n for n in incomplete_turns if n < turn]
        truncated_ids = [i for i in in_scope if index[i]["truncated"]]
        unavailable = [i for i in in_scope if index[i]["availability"] == "unavailable"]
        unknown = [i for i in in_scope if index[i]["availability"] == "unknown"]
        cards.append({
            "turn": turn, "task_requirement_id": "task.original",
            "current_action_id": current_actions[0] if current_actions else None,
            "prior_action_ids": prior_actions,
            "prior_evidence_ids": prior, "current_evidence_ids": current,
            "prior_citable_ids": ["task.original", *prior_actions, *prior],
            "current_citable_ids": [*current_actions, *current],
            "unavailable_evidence_ids": unavailable,
            "unknown_availability_evidence_ids": unknown,
            "truncated_evidence_ids": truncated_ids,
            "missing_prior_turns": missing, "incomplete_history_action_turns": ledger,
            "history_transcript_incomplete": bool(missing or ledger or [
                i for i in truncated_ids if index[i]["turn"] < turn]),
            "history_evidence_incomplete": bool(missing or ledger or [
                i for i in [*truncated_ids, *unavailable, *unknown] if index[i]["turn"] < turn]),
            "identical_evidence_pairs": pairs,
            "identical_evidence_pair_count": pair_count,
            "identical_evidence_pairs_omitted": pair_count - len(pairs),
        })

    # This check also guarantees added records remain compatible with v2.
    judge.validate_packet(augmented)
    context = {
        "version": VERSION, "packet_id": augmented["packet_id"],
        "original_packet_sha256": original_hash,
        "augmented_packet_sha256": _digest(augmented),
        "evidence_index": index, "actions": actions, "cards": cards,
        "task_requirement_id": "task.original",
        "available_observation_ids": available_obs,
        "attached_image_ids": sorted(receipts) if receipts is not None else [],
        "image_receipts_supplied": receipts is not None,
        "limitations": {
            "missing_transcript_turns": [n for n in range(1, last_turn + 1) if n not in recorded_turns],
            "incomplete_history_action_turns": incomplete_turns,
            "archive_limitation_ids": [i for i, ev in index.items() if ev["category"] == "limitation"],
            "truncated_evidence_ids": [i for i, ev in index.items() if ev["truncated"]],
            "unavailable_evidence_ids": [i for i, ev in index.items() if ev["availability"] == "unavailable"],
            "unknown_availability_evidence_ids": [i for i, ev in index.items() if ev["availability"] == "unknown"],
        },
        "interpretation": {
            "action_records": "Actions record what was requested/said, not execution or success.",
            "requirements": "task.original is a requested constraint, not an observed environment fact.",
            "identity": "Exact shown tool text (line endings only) or receipt SHA matches; not semantic equivalence, state identity, redundancy, correctness, or a reward label.",
            "truncation": "Truncated text can expose its shown prefix but is excluded from content-identity matches.",
            "history": "Missing/ledger-only turns and unattached media cannot establish what an agent or judge previously learned.",
            "scope": "Each card is time-causal. The global evidence index also contains later targets; never use those to score an earlier target.",
            "media": "Attachment and SHA information comes only from caller-provided verified image receipts; this module does not read media.",
        },
    }
    return {"packet": augmented, "context": context}


def available_observation_ids(context):
    """Return a COPY of global observable IDs; use cards for per-turn scope."""
    return list(context["available_observation_ids"])
