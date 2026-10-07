"""Deterministic evidence capabilities; no scoring, execution, files or network.

The input is the canonical augmented packet returned by
``transport_v3.prepare_context``. Its context is a trusted transport sidecar,
not actor-supplied text. Optional execution_receipts must likewise originate
from a trusted adapter's structured result, never parsed from stdout/LLM prose.
These cards constrain provenance; they cannot establish semantic entailment.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from . import judge

VERSION = "evidence-capabilities-v5-1"
TEXT_KINDS = frozenset({"tool_text", "tool_result", "observation"})
IMAGE_KINDS = frozenset({"image", "media_image"})
ACTION_KINDS = frozenset({"action_record", "action_ledger"})
REQUIREMENT_KINDS = frozenset({"task_requirement", "task_rule"})
TRUNCATED = "ARCHIVE TEXT TRUNCATED; unshown content is unknown."
OMITTED = "MEDIA NOT ATTACHED. Judge cannot inspect this evidence; omission="
EMPTY_RETURNS = frozenset({"", "(no output)", "(empty output)", "no output"})
CAN_SUPPORT = frozenset({
    "task_requirement", "actor_action_or_claim", "archive_limitation", "visible_content",
    "reported_text", "media_reference", "no_text_return", "reported_rejection",
    "command_execution_status", "command_exit_zero", "command_exit_nonzero", "command_not_executed",
})
# Exact archived adapter wrappers only. Never substring-search arbitrary stdout.
# Without an out-of-band structured receipt their origin remains unverified.
REJECTION_WRAPPERS = frozenset({
    "(拒绝：命令试图访问工作目录之外的路径)",
})
MEDIA_LABEL = re.compile(r"\([^\r\n()]{1,512}\.(?:jpe?g|png|webp|gif|wav|mp3|m4a|mp4|webm)\)", re.I)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _context(packet, context, known):
    if context is None:
        return {}, set()
    if type(context) is not dict:
        raise judge.ProtocolError("provenance context must be an object")
    context = judge._plain_json(context)
    if context.get("packet_id") != packet["packet_id"]:
        raise judge.ProtocolError("provenance context belongs to a different packet")
    if context.get("augmented_packet_sha256") != _digest(packet):
        raise judge.ProtocolError("provenance context does not bind this packet")
    attached = context.get("attached_image_ids", [])
    if (type(attached) is not list or any(not isinstance(x, str) for x in attached)
            or len(set(attached)) != len(attached)):
        raise judge.ProtocolError("invalid attached-image index")
    for ident in attached:
        if not isinstance(ident, str) or ident not in known or known[ident]["kind"] not in IMAGE_KINDS:
            raise judge.ProtocolError("attachment does not identify image evidence")
        text = known[ident].get("text", "")
        if text.startswith(OMITTED) or text.startswith(TRUNCATED + "\n" + OMITTED):
            raise judge.ProtocolError("attached image contradicts explicit omission")
    index = context.get("evidence_index", {})
    if type(index) is not dict:
        raise judge.ProtocolError("invalid context evidence index")
    for ident in attached:
        record = index.get(ident, {})
        if (record.get("availability") != "available" or
                record.get("kind") != known[ident]["kind"] or
                record.get("turn") != known[ident]["turn"] or
                re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is None):
            raise judge.ProtocolError("image attachment lacks a bound verified receipt")
    return context, set(attached)


def _execution_receipts(value, known):
    """Accept explicit adapter records, never a JSON-looking stdout string.

    Scope is one whole command invocation, not necessarily every tool in a turn.
    Exit0 establishes that invocation's reported process exit, not its effects.
    """
    if value is None:
        return {}
    if type(value) is not list:
        raise judge.ProtocolError("execution_receipts must be a list")
    result = {}
    for original in value:
        if type(original) is not dict:
            raise judge.ProtocolError("execution receipt must be structured")
        record = judge._plain_json(original)
        required = {"evidence_id", "turn", "scope"}
        if not required <= set(record) or set(record) - required - {"exit_code", "executed"}:
            raise judge.ProtocolError("unknown execution receipt fields")
        ident = record["evidence_id"]
        if (not isinstance(ident, str) or ident not in known or ident in result or
                known[ident]["kind"] not in TEXT_KINDS):
            raise judge.ProtocolError("execution receipt must identify unique tool-result evidence")
        if type(record["turn"]) is not int or record["turn"] != known[ident]["turn"]:
            raise judge.ProtocolError("execution receipt turn mismatch")
        if record["scope"] != "whole_command":
            raise judge.ProtocolError("execution receipt must cover the whole command invocation")
        if "exit_code" in record:
            if type(record["exit_code"]) is not int:
                raise judge.ProtocolError("exit_code must be an integer, not bool/string")
            if "executed" in record and record["executed"] is not True:
                raise judge.ProtocolError("exit code contradicts not-executed receipt")
            event = {"status": "exit_code_observed", "exit_code": record["exit_code"],
                     "basis": "trusted_structured_exit_code"}
        elif record.get("executed") is False:
            event = {"status": "not_executed", "exit_code": None,
                     "basis": "trusted_whole_command_not_executed"}
        else:
            raise judge.ProtocolError("receipt needs explicit exit_code or executed=false")
        event.update(evidence_id=ident, turn=record["turn"], scope="whole_command",
                     establishes_artifact_effect=False)
        result[ident] = event
    return result


def prepare(packet, context=None, *, execution_receipts=None):
    """Return a new provenance sidecar, preserving every input evidence ID.

    No command parsing/execution, semantic classification, automatic reward,
    filesystem lookups or guessed audio attachments. Missing data stays missing.
    ``context`` and execution_receipts are trusted caller inputs: this module
    checks consistency, but cannot authenticate an arbitrary caller.
    """
    packet = judge.validate_packet(packet)
    known = {e["id"]: e for e in packet["evidence"]}
    context, attached = _context(packet, context, known)
    receipts = _execution_receipts(execution_receipts, known)
    index = {}
    for ident, ev in sorted(known.items(), key=lambda x: (x[1]["turn"], x[0])):
        kind, text = ev["kind"], ev.get("text", "")
        truncated = text.startswith(TRUNCATED)
        omission = text.startswith(OMITTED) or text.startswith(TRUNCATED + "\n" + OMITTED)
        item = {"turn": ev["turn"], "source_kind": kind, "role": "unknown",
                "modality": "unknown", "availability": "unknown", "truncated": truncated,
                "can_support": [], "limitations": [], "execution": None}
        if kind in REQUIREMENT_KINDS:
            item.update(role="task_requirement", modality="text", availability="available",
                        can_support=["task_requirement"])
            item["limitations"] = ["Not observation history, file presence/absence, or task completion."]
        elif kind in ACTION_KINDS:
            item.update(role="action_intent", modality="text", availability="available",
                        can_support=["actor_action_or_claim"])
            item["limitations"] = ["Proves recorded intent/text, not execution or effect."]
        elif kind == "archive_limitation":
            item.update(role="archive_limitation", modality="text", availability="available",
                        can_support=["archive_limitation"])
        elif kind in IMAGE_KINDS:
            availability = "available" if ident in attached else (
                "unavailable" if omission or context.get("image_receipts_supplied") is True else "unknown")
            item.update(role="image_observation", modality="image", availability=availability,
                        can_support=["visible_content"] if availability == "available" else [])
            item["limitations"] = ["Only attached pixels; no unheard words/audio, complete video timeline, or unshown frames."]
            if availability != "available":
                item["limitations"].append("Referenced media is not established as attached; actor may have seen it.")
        elif kind in {"audio", "video", "media_audio", "media_video"}:
            item.update(role="unverified_media_reference", modality="audio" if "audio" in kind else "video",
                        availability="unavailable" if omission else "unknown")
            item["limitations"] = ["No trusted audio/video attachment adapter is implemented; kind/text alone do not attest bytes."]
        elif kind in TEXT_KINDS:
            item.update(role="tool_result_text", modality="text",
                        availability="available" if "text" in ev else "unavailable",
                        can_support=["reported_text"] if "text" in ev else [])
            item["limitations"] = ["Literal tool-return text; not by itself trusted execution status, semantic validation, or audio/video content."]
            if not truncated and MEDIA_LABEL.fullmatch(text.strip()):
                item.update(role="media_label", can_support=["media_reference"])
                item["limitations"] = ["Filename/caption reference only; not the referenced media content."]
            elif text.strip() in EMPTY_RETURNS:
                item.update(role="empty_tool_return", can_support=["no_text_return"] if "text" in ev else [])
                item["limitations"] = ["No text returned; no explicit exit status or confirmed artifact effect."]
            elif not truncated and text.strip() in REJECTION_WRAPPERS:
                item.update(role="reported_command_rejection", can_support=["reported_rejection"])
                item["execution"] = {"status": "reported_not_executed", "exit_code": None,
                    "basis": "exact_archived_rejection_wrapper_unverified_origin", "evidence_id": ident,
                    "turn": ev["turn"], "scope": "whole_command_report", "establishes_artifact_effect": False}
                item["limitations"] = ["Exact archived wrapper reports rejection; text origin is not independently authenticated. Do not infer intended side effects or endorse its path diagnosis."]
        else:
            item["availability"] = "available" if "text" in ev else "unknown"
            item["limitations"] = ["Unknown evidence kind is not promoted to an authoritative observation."]
        if truncated:
            item["limitations"].append("Only the retained prefix is available; omitted text cannot support facts.")
        if ident in receipts:
            if item["execution"] and receipts[ident]["status"] == "exit_code_observed":
                raise judge.ProtocolError("structured exit code conflicts with whole-command rejection report")
            item["execution"] = deepcopy(receipts[ident])
            item["can_support"].append("command_execution_status")
            if receipts[ident]["status"] == "not_executed":
                item["can_support"].append("command_not_executed")
            else:
                item["can_support"].append("command_exit_zero" if receipts[ident]["exit_code"] == 0
                                           else "command_exit_nonzero")
        index[ident] = item

    turns = []
    for group in ("history", "targets"):
        for row in packet[group]:
            turn = row["turn"]
            current = [ident for ident, ev in index.items() if ev["turn"] == turn]
            events = [deepcopy(index[ident]["execution"]) for ident in current if index[ident]["execution"]]
            # Even one successful command receipt cannot prove an entire turn
            # containing other invocations completed or changed artifacts.
            states = {event["status"] for event in events}
            status = next(iter(states)) if len(states) == 1 else ("mixed_reports" if states else "unknown")
            turns.append({"turn": turn, "source": group, "execution_status": status,
                "execution_scope": "reported_command_invocations_only_not_whole_turn",
                "execution_events": events, "current_evidence_ids": current,
                "prior_evidence_ids": [ident for ident, ev in index.items() if ev["turn"] < turn],
                "artifact_effect_status": "unknown",
                "limitations": ["No inference that all commands ran or that a file was created/deleted; inspect a properly scoped state observation."]})
    return {"version": VERSION, "packet_id": packet["packet_id"], "packet_sha256": _digest(packet),
            "evidence_index": index, "turns": turns,
            "scope": {"scoring": "none", "semantics": "not_adjudicated",
                "audio_video_adapter": "not_implemented", "execution_receipts": "trusted_caller_structured_only",
                "text_rejection_wrappers": "exact_allowlist_reports_not_authenticated_execution",
                "no_output": "unknown_execution", "identity": "references_and_capabilities_not_entailment"}}
