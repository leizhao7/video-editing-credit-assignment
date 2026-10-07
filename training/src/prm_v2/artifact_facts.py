"""Bounded JSON artifact facts for archived evidence; never execute or score.

Recognized schemas are exactly {cuts:[{clip,in_s,out_s}, ...]} and one of
{selected:[ID,...]}, {selection:[ID,...]}, {order:[ID,...]}.  Decimal arithmetic
reports the sum of supplied intervals, NOT a measured rendered-video duration.
Action literals and actual tool-returned JSON remain separate evidence bases.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import re
import shlex


VERSION = "artifact_json_facts_v1"
LIMITS = {"rows": 256, "evidence": 4096, "action_characters": 100000,
          "json_characters": 65536, "json_depth": 4, "items": 256,
          "records": 512, "facts": 4096, "matches": 1024, "number_characters": 80,
          "number_digits": 40, "number_exponent": 20}
OBSERVATION_KINDS = {"tool_text", "tool_result", "observation"}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
SCOPE = ("Deterministic facts from finite recognized JSON only. action_only is intended content, "
         "not executed code or a successful file write. tool_observation is JSON returned in the "
         "archive, not proof of creative correctness. Interval sums are arithmetic, not measured "
         "render durations. Equality preserves list order and compares exact Decimal values; "
         "it establishes neither redundancy, novelty, reward nor unchanged filesystem state. "
         "Paths are literal strings; no cwd inference, file loading, shell/Python evaluation or network.")


class FactError(ValueError):
    pass


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(raw):
    if len(raw) > LIMITS["number_characters"]:
        raise FactError("number_length_limit")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise FactError("invalid_decimal") from exc
    if not value.is_finite():
        raise FactError("nonfinite_number")
    parts = value.as_tuple()
    if len(parts.digits) > LIMITS["number_digits"] or abs(parts.exponent) > LIMITS["number_exponent"]:
        raise FactError("number_precision_or_exponent_limit")
    if value.copy_abs() > Decimal("1000000000"):
        raise FactError("numeric_magnitude_limit")
    return value


def _decimal_text(value):
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FactError("duplicate_json_key")
        result[key] = value
    return result


def _check_depth(text):
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > LIMITS["json_depth"]:
                raise FactError("json_depth_limit")
        elif char in "]}":
            depth -= 1


def _parse_json(text):
    if not isinstance(text, str) or len(text) > LIMITS["json_characters"]:
        raise FactError("json_size_or_type_limit")
    raw = text.strip()
    _check_depth(raw)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
            parse_float=_number, parse_int=_number,
            parse_constant=lambda _: (_ for _ in ()).throw(FactError("nonfinite_number")))
    except (ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, FactError):
            raise
        raise FactError("not_one_strict_json_object") from exc
    if type(value) is not dict or len(value) != 1:
        raise FactError("unknown_or_ambiguous_top_level_schema")
    kind = next(iter(value))
    if kind not in {"cuts", "selected", "selection", "order"}:
        raise FactError("unknown_artifact_schema")
    items = value[kind]
    if type(items) is not list or len(items) > LIMITS["items"]:
        raise FactError("artifact_items_type_or_count_limit")
    if kind != "cuts":
        if any(not isinstance(item, str) or not ID_PATTERN.fullmatch(item) for item in items):
            raise FactError("expected_simple_id_list")
        return kind, {kind: list(items)}, text, []
    normalized, intervals = [], []
    # 40 coefficient digits + at most 20 decimal exponent places + at most
    # three carry digits from 256 intervals fits exactly in this local context.
    with localcontext() as context:
        context.prec = 128
        for index, cut in enumerate(items):
            if type(cut) is not dict or set(cut) != {"clip", "in_s", "out_s"}:
                raise FactError("unknown_cut_layout_or_units")
            if not isinstance(cut["clip"], str) or not ID_PATTERN.fullmatch(cut["clip"]):
                raise FactError("invalid_clip_id")
            start, end = cut["in_s"], cut["out_s"]
            if type(start) is not Decimal or type(end) is not Decimal:
                raise FactError("cut_seconds_must_be_json_numbers")
            normalized.append({"clip": cut["clip"], "in_s": _decimal_text(start), "out_s": _decimal_text(end)})
            intervals.append({"index": index, "clip": cut["clip"], "in_s": _decimal_text(start),
                "out_s": _decimal_text(end), "duration_s": _decimal_text(end - start),
                "nonnegative_boundaries": start >= 0 and end >= 0, "in_before_out": start < end})
    return kind, {kind: normalized}, text, intervals


def _literal_path(raw):
    try:
        parts = shlex.split(raw, posix=True)
    except ValueError:
        return None
    if len(parts) != 1 or not parts[0] or len(parts[0]) > 512:
        return None
    path = parts[0]
    if path.startswith("-") or any(c in path for c in "$`*?[];|&<>\n\r\x00") or "://" in path:
        return None
    return path


def _action_source(action):
    """Only one full legacy bash call; all text is inert transcript data."""
    if not isinstance(action, str):
        return None, None, None, "full_action_unavailable"
    if len(action) > LIMITS["action_characters"]:
        return None, None, None, "action_size_limit"
    matches = re.findall(r"<function=bash>\s*<parameter=(?:cmd|command)>(.*?)</parameter>\s*</function>", action, re.S)
    if len(matches) != 1 or action.count("<function=") != 1:
        return None, None, None, "unsupported_or_multiple_action_calls"
    command = matches[0].strip()
    lines = command.splitlines()
    if len(lines) >= 3:
        # A QUOTED delimiter makes the body literal in shell syntax. We still
        # never execute it and never infer that this proposed write succeeded.
        header = re.fullmatch(r"cat\s*>\s*(.*?)\s*<<\s*(['\"])([A-Za-z_][A-Za-z0-9_]{0,31})\2\s*", lines[0])
        if header and lines[-1] == header[3] and lines[1:-1].count(header[3]) == 0:
            path = _literal_path(header[1])
            if path:
                return path, "literal_heredoc_write_intent", "\n".join(lines[1:-1]), None
        return None, None, None, "unsupported_multiline_or_unquoted_action"
    if len(lines) == 1:
        matched = re.fullmatch(r"cat\s+(?:--\s+)?(.+)", command)
        if matched:
            path = _literal_path(matched[1])
            if path:
                return path, "direct_cat_read", None, None
    return None, None, None, "unsupported_action_not_executed"


def prepare(packet):
    """Return sidecar facts only; never mutate packet or external state.

    Pass evidence_context's augmented packet to obtain citable action.tNNN
    sources. With a raw packet, an action fact has source_evidence_id=None and
    citable=False rather than inventing an evidence ID.
    """
    if not isinstance(packet, dict):
        raise FactError("packet_must_be_object")
    history, targets, evidence = (packet.get(key, []) for key in ("history", "targets", "evidence"))
    if any(type(items) is not list for items in (history, targets, evidence)):
        raise FactError("packet_collections_must_be_lists")
    rows = history + targets
    if len(rows) > LIMITS["rows"] or len(evidence) > LIMITS["evidence"]:
        raise FactError("packet_size_limit")
    if any(type(row) is not dict or type(row.get("turn")) is not int or row["turn"] < 1 for row in rows):
        raise FactError("invalid_source_turn")
    if len({row["turn"] for row in rows}) != len(rows):
        raise FactError("duplicate_source_turn")
    known = {}
    for item in evidence:
        if type(item) is not dict or not isinstance(item.get("id"), str) or item["id"] in known:
            raise FactError("invalid_or_duplicate_evidence_id")
        known[item["id"]] = item
    cards, records, matches, skipped = [], [], [], []
    rows = sorted(rows, key=lambda row: row["turn"])

    def add_record(turn, source_id, basis, text, path, binding):
        try:
            kind, normalized, raw, intervals = _parse_json(text)
        except FactError as exc:
            skipped.append({"turn": turn, "source_evidence_id": source_id, "basis": basis, "reason": str(exc)})
            return
        if len(records) >= LIMITS["records"]:
            raise FactError("record_count_limit")
        if len(cards) + (len(intervals) + 1 if kind == "cuts" else 1) > LIMITS["facts"]:
            raise FactError("fact_count_limit")
        value_text = _canonical(normalized)
        value_sha = _sha(value_text)
        rid = "artifact.t%03d.%s.%s" % (turn, basis, _sha((source_id or "provided_action") + "\n" + raw)[:20])
        common = {"turn": turn, "source_evidence_id": source_id, "basis": basis,
                  "citable": source_id is not None, "artifact_path": path, "path_binding": binding,
                  "epistemic_status": "intended_content_only" if basis == "action_only" else "observed_tool_json"}
        record = {"id": rid, **common, "schema": kind, "normalized_json": normalized,
                  "json_value_sha256": value_sha, "source_json_text_sha256": _sha(raw)}
        record["scope"] = ("Literal intended JSON; does not prove execution or a successful write." if basis == "action_only" else
                           "JSON returned by a tool; only direct_cat_read binds it to a literal path. No rendered-media or semantic correctness claim.")
        for prior in records:
            if prior["turn"] >= turn or prior["normalized_json"] != normalized:
                continue
            if len(matches) >= LIMITS["matches"]:
                raise FactError("exact_match_count_limit")
            matches.append({"prior_record_id": prior["id"], "current_record_id": rid,
                "prior_turn": prior["turn"], "turn": turn,
                "prior_source_evidence_id": prior["source_evidence_id"], "source_evidence_id": source_id,
                "prior_basis": prior["basis"], "basis": basis,
                "comparison_basis": "exact_known_structure_and_decimal_values",
                "same_json_text_bytes": prior["source_json_text_sha256"] == record["source_json_text_sha256"],
                "text_comparison_scope": "Supplied observation/extracted literal JSON text, not complete artifact file bytes.",
                "same_literal_path": path is not None and path == prior["artifact_path"],
                "scope": "Content equality only; not proof of redundant work, no state change, or execution of a prior action."})
        records.append(record)
        if kind != "cuts":
            cards.append({"id": rid + ".ids", **common, "kind": kind + "_ids", "value": list(normalized[kind]),
                          "id_count": len(normalized[kind]), "record_id": rid, "scope": record["scope"]})
            return
        with localcontext() as context:
            context.prec = 128
            total = sum((Decimal(item["duration_s"]) for item in intervals), Decimal(0))
        for interval in intervals:
            cards.append({"id": rid + ".segment_%03d.duration_seconds" % interval["index"], **common,
                "kind": "cut_interval_seconds", "record_id": rid, **interval,
                "derived_decimal": interval["duration_s"], "unit": "seconds",
                "scope": record["scope"] + " Exact out_s minus in_s arithmetic only."})
        cards.append({"id": rid + ".total_interval_seconds", **common, "record_id": rid,
            "kind": "cuts_total_interval_seconds", "derived_decimal": _decimal_text(total), "unit": "seconds",
            "segment_count": len(intervals), "all_nonnegative_boundaries": all(item["nonnegative_boundaries"] for item in intervals),
            "all_in_before_out": all(item["in_before_out"] for item in intervals),
            "scope": record["scope"] + " Sum of supplied (out_s-in_s); not measured render duration, transition-aware duration, or task compliance."})

    for row in rows:
        turn, action = row["turn"], row.get("action")
        path, binding, literal, error = _action_source(action)
        if literal is not None:
            action_ids = [item["id"] for item in evidence if item.get("turn") == turn and item.get("kind") == "action_record" and item.get("text") == action]
            source_id = action_ids[0] if len(action_ids) == 1 else None
            add_record(turn, source_id, "action_only", literal, path, binding)
        elif error:
            skipped.append({"turn": turn, "source_evidence_id": None, "basis": "action_only", "reason": error})
        refs = []
        observations = row.get("observations", [])
        if isinstance(observations, list):
            refs.extend(obs["evidence_id"] for obs in observations if isinstance(obs, dict) and isinstance(obs.get("evidence_id"), str))
        if isinstance(row.get("evidence_ids"), list):
            refs.extend(ref for ref in row["evidence_ids"] if isinstance(ref, str))
        for ref in dict.fromkeys(refs):
            item = known.get(ref)
            if not item or item.get("turn") != turn or item.get("kind") not in OBSERVATION_KINDS:
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.lstrip().startswith(("{", "[")):
                continue
            if item.get("truncated") or "ARCHIVE TEXT TRUNCATED; unshown content is unknown." in text:
                skipped.append({"turn": turn, "source_evidence_id": ref, "basis": "tool_observation", "reason": "truncated_observation"})
                continue
            add_record(turn, ref, "tool_observation", text, path if binding == "direct_cat_read" else None,
                       "direct_cat_read" if binding == "direct_cat_read" else "unbound_tool_output")
    index = {card["id"]: card for card in cards}
    if len(index) != len(cards):
        raise FactError("fact_id_collision")
    return {"version": VERSION, "cards": cards, "fact_index": index, "records": records,
            "exact_content_matches": matches, "skipped": skipped, "scope": SCOPE}
