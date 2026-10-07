"""Inert ffprobe measurement cards; not novelty judgements or rewards.

Only visible, full source actions are used.  No command is executed and no path
is loaded.  Absence here means unobserved by this narrow parser, not unknown to
the agent: manifests, images, omitted turns and other tools are not parsed.
"""
from __future__ import annotations

import json
import re
import shlex
from decimal import Decimal, InvalidOperation


VERSION = "ffprobe_numeric_facts_v1"
LIMITS = {"rows": 256, "observation_characters": 100000, "facts_per_observation": 64}


def _decimal(value):
    if isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not result.is_finite() or result < 0 or len(str(value)) > 80 or abs(result.adjusted()) > 1000:
        return None
    # normalize() would apply the ambient Decimal precision and round long
    # values.  Plain decimal formatting and zero stripping remain exact.
    rendered = format(result, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _source(action):
    if not isinstance(action, str):
        return None, "full_source_action_unavailable"
    calls = re.findall(r"<function=bash>\s*<parameter=(?:cmd|command)>(.*?)</parameter>\s*</function>", action, re.S)
    if len(calls) != 1:
        return None, "no_single_full_bash_action"
    command = calls[0].strip()
    # These constructs require shell evaluation or hide executable commands.
    if any(marker in command for marker in ("$", "`", "<<", "\n")):
        return None, "unsupported_shell_structure"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None, "malformed_shell_tokens"
    probe_tokens = [i for i, value in enumerate(tokens) if value in ("ffprobe", "/usr/bin/ffprobe", "/usr/local/bin/ffprobe")]
    starts = [i for i in probe_tokens if i == 0 or tokens[i - 1] in ("&&", ";")]
    if len(starts) != 1 or len(probe_tokens) != 1:
        return None, "no_single_direct_ffprobe"
    start = starts[0]
    end = next((i for i in range(start + 1, len(tokens)) if tokens[i] in ("|", "&&", ";", "||", "&")), len(tokens))
    args = tokens[start + 1:end]
    values = {"-v", "-loglevel", "-show_entries", "-of", "-print_format"}
    flags = {"-show_format", "-show_streams", "-hide_banner"}
    options, assets = {}, []
    i = 0
    while i < len(args):
        token = args[i]
        if token in values and i + 1 < len(args):
            options[token] = args[i + 1]
            i += 2
        elif token in flags:
            options[token] = True
            i += 1
        elif token == "2" and args[i:i + 3] == ["2", ">&", "1"]:
            i += 3
        elif token.startswith("-") or token in ("<", ">", ">&"):
            return None, "unsupported_ffprobe_option_or_redirection"
        else:
            assets.append(token)
            i += 1
    if len(assets) != 1 or not assets[0] or any(c in assets[0] for c in ("*", "?", "[", "]")) or "://" in assets[0]:
        return None, "explicit_single_media_path_unavailable"
    # Do not resolve relative paths against an invented cwd.
    return {"asset": assets[0], "options": options}, None


def _measurements(text, source):
    facts, notes = [], []

    def add(field, value):
        number = _decimal(value)
        if number is None:
            notes.append("invalid_or_nonfinite_duration")
        else:
            facts.append({"field": field, "value": number, "unit": "seconds"})

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except (ValueError, TypeError):
            return [], ["malformed_ffprobe_json"]
        if not isinstance(obj, dict):
            return [], ["malformed_ffprobe_json"]
        fmt = obj.get("format")
        if isinstance(fmt, dict) and "duration" in fmt:
            add("format.duration", fmt["duration"])
        counts = {}
        for stream in obj.get("streams", []) if isinstance(obj.get("streams", []), list) else []:
            if not isinstance(stream, dict):
                continue
            kind = stream.get("codec_type")
            if kind not in ("video", "audio", "data", "subtitle", "attachment"):
                continue
            ordinal = counts.get(kind, 0)
            counts[kind] = ordinal + 1
            if "duration" in stream:
                add("streams.%s[%d].duration" % (kind, ordinal), stream["duration"])
            if "duration_ts" in stream:
                notes.append("duration_ts_skipped_units_unspecified")
        return facts, sorted(set(notes))

    options = source["options"]
    entries = options.get("-show_entries", "")
    format_allowed = bool(options.get("-show_format")) or "format=" in entries
    streams_allowed = bool(options.get("-show_streams")) or "stream=" in entries
    counts, stream_field = {}, None
    stream_has_duration, section = False, None
    duration_rows = [(index, line.split("=", 1)[1]) for index, line in enumerate(stripped.splitlines()) if line.startswith("duration=")]
    last_duration_index = duration_rows[-1][0] if duration_rows else None
    seen_format = False
    for index, raw in enumerate(stripped.splitlines()):
        line = raw.strip()
        if line == "[FORMAT]":
            section, stream_field = "format", None
        elif line == "[STREAM]":
            section, stream_field, stream_has_duration = "stream", None, False
        elif line in ("[/STREAM]", "[/FORMAT]"):
            section, stream_field = None, None
        elif line.startswith("codec_type=") and streams_allowed:
            kind = line.split("=", 1)[1]
            if kind not in ("video", "audio", "data", "subtitle", "attachment"):
                stream_field = None
                notes.append("unknown_codec_type")
                continue
            ordinal = counts.get(kind, 0)
            counts[kind] = ordinal + 1
            stream_field = "streams.%s[%d].duration" % (kind, ordinal)
            stream_has_duration = False
        elif line.startswith("duration_ts="):
            notes.append("duration_ts_skipped_units_unspecified")
        elif line.startswith("duration="):
            value = line.split("=", 1)[1]
            if section == "format" or (format_allowed and not streams_allowed):
                add("format.duration", value)
                seen_format = True
            elif stream_field and not stream_has_duration:
                add(stream_field, value)
                stream_has_duration = True
            elif stream_field and section is None and format_allowed and not seen_format and index == last_duration_index:
                # grep may remove [FORMAT]; a final second duration cannot
                # overwrite the preceding stream's already assigned duration.
                add("format.duration", value)
                seen_format = True
                notes.append("final_format_duration_scoped_from_show_streams_show_format_order")
            else:
                notes.append("unscoped_duration_skipped")
    return facts, sorted(set(notes))


def prepare(packet):
    """Return bounded typed measurements, preserving packet and external state."""
    evidence = {e.get("id"): e for e in packet.get("evidence", []) if isinstance(e, dict)}
    history = packet.get("history", [])
    targets = packet.get("targets", [])
    rows = sorted([r for r in history + targets if isinstance(r, dict) and isinstance(r.get("turn"), int)], key=lambda r: r["turn"])
    target_turns = {row["turn"] for row in targets}
    known, cards, skipped = {}, [], []
    if len(rows) > LIMITS["rows"]:
        return {"version": VERSION, "cards": [], "skipped": [{"reason": "row_limit_exceeded"}], "scope": "visible full ffprobe actions only; no semantic progress or first-ever knowledge claims"}
    for row in rows:
        turn = row["turn"]
        source, error = _source(row.get("action"))
        current, notes = [], []
        if error:
            skipped.append({"turn": turn, "reason": error})
        else:
            refs = []
            for obs in row.get("observations", []) if isinstance(row.get("observations", []), list) else []:
                if isinstance(obs, dict) and obs.get("evidence_id"):
                    refs.append(obs["evidence_id"])
            refs.extend(row.get("evidence_ids", []))
            for ref in dict.fromkeys(refs):
                item = evidence.get(ref)
                if not item or item.get("turn") != turn or item.get("kind") not in ("tool_text", "text"):
                    continue
                text = item.get("text", "")
                if not isinstance(text, str) or item.get("truncated") or len(text) > LIMITS["observation_characters"]:
                    notes.append("incomplete_or_oversized_observation_skipped")
                    continue
                facts, fact_notes = _measurements(text, source)
                notes.extend(fact_notes)
                if len(facts) > LIMITS["facts_per_observation"]:
                    notes.append("fact_limit_exceeded_observation_skipped")
                    continue
                current.extend(dict(fact, asset=source["asset"], source_evidence_id=ref, source_action_turn=turn) for fact in facts)
        assets = {fact["asset"] for fact in current}
        prior = [dict(fact) for key, fact in sorted(known.items()) if key[0] in assets]
        new, changed, unchanged = [], [], []
        for fact in current:
            key = (fact["asset"], fact["field"])
            previous = known.get(key)
            if previous is None:
                new.append(dict(fact))
            else:
                pair = {"asset": key[0], "field": key[1], "unit": "seconds", "prior": dict(previous), "current": dict(fact)}
                (unchanged if Decimal(previous["value"]) == Decimal(fact["value"]) else changed).append(pair)
        if turn in target_turns:
            cards.append({"turn": turn, "prior_known_values": prior, "current_values": current,
                          "newly_observed_in_visible_history_fields": new, "changed": changed,
                          "unchanged": unchanged, "notes": sorted(set(notes + ([error] if error else [])))})
        # Updates happen after the entire turn: one observation is not prior
        # history for another observation of the same action.
        for fact in current:
            known[(fact["asset"], fact["field"])] = dict(fact)
    return {"version": VERSION, "cards": cards, "skipped": skipped,
            "scope": "Only measurements from visible, full, recognized ffprobe actions; new fields are not semantic progress, first-ever knowledge, or proof that other tools/history lacked the fact. Paths are literal strings; no cwd inference. Values are observations at their source turns, not assertions that an asset stayed unchanged."}
