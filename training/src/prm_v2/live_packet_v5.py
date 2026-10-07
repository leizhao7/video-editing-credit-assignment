"""Pure, bounded live-turn adapter for the offline v5 judge protocol.

No paths, network, commands, stdout status parsing, or training-token mutation.
Only a completed prefix is inspected. Images are the exact admitted inline
bytes, never fetched or regenerated. Missing/invalid/limited media remain
explicitly unavailable evidence. This adapter is not an execution attestor.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import re

from . import context_v5, dataset, evidence_context, judge

VERSION = "prm-live-packet-v5-3"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_DATA = re.compile(r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]*={0,2})\Z")
_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def _native_call(call):
    """Lossless argument spelling; XML only when delimiters cannot be injected.

    The existing deterministic facts recognize simple legacy bash XML. Native
    malformed calls and XML-like arguments stay a complete JSON tool_call and
    therefore do not acquire unsupported typed arithmetic/path facts.
    """
    if type(call) is not dict:
        raise judge.ProtocolError("native tool call must be an object")
    function = call.get("function", call)
    name = function.get("name") if type(function) is dict else None
    args = function.get("arguments") if type(function) is dict else None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            args = None
    if (isinstance(name, str) and _NAME.fullmatch(name) and type(args) is dict
            and all(isinstance(k, str) and _NAME.fullmatch(k) for k in args)):
        values = {k: v if isinstance(v, str) else _json(v) for k, v in args.items()}
        if all(not re.search(r"</?(?:tool_call|function|parameter)(?:[\s=>]|$)", v)
               for v in values.values()):
            params = "\n".join(f"<parameter={k}>{v}</parameter>" for k, v in values.items())
            return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>"
    return "<tool_call>\n" + _json(call) + "\n</tool_call>"


def action_text(action):
    """Accept raw public text, native assistant message, or append parser view.

    append backend stores the ORIGINAL native message as JSON in action.text;
    prefer it over dispatch-only calls (which may rename invalid arguments).
    Separate reasoning_content/thinking fields are intentionally not exposed.
    """
    if isinstance(action, str):
        return dataset.visible_action(action)
    if type(action) is not dict:
        raise judge.ProtocolError("action must be public text or a native action object")
    message = action
    if "text" in action and "calls" in action:
        raw = action["text"]
        if not isinstance(raw, str) or type(action["calls"]) is not list:
            raise judge.ProtocolError("invalid append action view")
        try:
            candidate = json.loads(raw)
        except (ValueError, TypeError):
            candidate = None
        if type(candidate) is dict and ("content" in candidate or "tool_calls" in candidate):
            message = candidate
        else:
            prose = dataset.visible_action(raw)
            # A raw XML response already contains its calls; do not duplicate
            # dispatch records, or replace an originally malformed model call.
            if "<tool_call>" in prose:
                return prose
            return "\n".join([prose, *[_native_call(c) for c in action["calls"]]]).strip()
    content = message.get("content") or ""
    if isinstance(content, list):
        if any(type(p) is not dict or p.get("type") != "text" for p in content):
            raise judge.ProtocolError("unsupported nontext assistant content")
        content = "\n".join(p.get("text", "") for p in content)
    if not isinstance(content, str):
        raise judge.ProtocolError("assistant content must be text")
    calls = message.get("tool_calls") or []
    if type(calls) is not list:
        raise judge.ProtocolError("native tool_calls must be a list")
    public = dataset.visible_action(content)
    return "\n".join([public, *[_native_call(c) for c in calls]]).strip()


def observation_messages(observation):
    """Normalize actual ToolSession results, or a tool_result transport frame.

    Mirrors SuffixRenderer's output+attachments mapping, before token rendering.
    runtime_exit.returncode is deliberately NOT a per-command success receipt.
    """
    if type(observation) is not dict:
        raise judge.ProtocolError("tool observation must be an object")
    if observation.get("kind") == "tool_result":
        observation = observation.get("result")
    if type(observation) is not dict or set(observation) != {"results"} or type(observation["results"]) is not list:
        raise judge.ProtocolError("expected ToolSession results, not a transcript/runtime status")
    messages = []
    for result in observation["results"]:
        if (type(result) is not dict or not isinstance(result.get("output"), str)
                or type(result.get("attachments")) is not list):
            raise judge.ProtocolError("invalid native tool result")
        content = [{"type": "text", "text": result["output"]}]
        for part in result["attachments"]:
            if type(part) is not dict:
                raise judge.ProtocolError("invalid native attachment")
            # Preserve mime/hash hints only for checking, never as authority.
            content.append({"type": "image_url", "image_url": {"url": part.get("url")},
                            **{k: part[k] for k in ("mime", "sha256", "bytes") if k in part}})
        messages.append({"role": "tool", "tool_call_id": result.get("id"),
                         "name": result.get("name"), "content": content})
    return messages


def _parts(messages):
    if type(messages) is dict:
        messages = observation_messages(messages)
    if type(messages) is not list or len(messages) > 128:
        raise judge.ProtocolError("messages must be a bounded list of actual observations")
    for mi, msg in enumerate(messages, 1):
        if type(msg) is not dict or msg.get("role") not in {"tool", "user", "function"}:
            raise judge.ProtocolError("only actual tool/user observations belong after an action")
        content = msg.get("content", "")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if type(parts) is not list or len(parts) > 4096:
            raise judge.ProtocolError("invalid or oversized observation content")
        for pi, part in enumerate(parts, 1):
            if type(part) is not dict:
                raise judge.ProtocolError("observation content part must be an object")
            yield mi, pi, part


def native_execution_receipts(observation, *, turn):
    """Trusted ToolSession boundary ONLY; never pass actor/transcript JSON here.

    Preserve the actual bash invocation's structured exit, independently of
    public output rendering. is_error=False alone conveys no execution success.
    Other tools, missing exits, aborted host calls and stdout are not attested.
    """
    if type(turn) is not int or turn < 1:
        raise judge.ProtocolError("invalid native receipt turn")
    observation_messages(observation)  # Same strict native result envelope.
    native = observation['result'] if observation.get('kind') == 'tool_result' else observation
    receipts, seen = [], set()
    for index, result in enumerate(native['results'], 1):
        metadata = result.get('metadata')
        if result.get('name') != 'bash' or result.get('is_error') is not False:
            continue
        if type(metadata) is not dict or metadata.get('exit') is None:
            continue
        identity = result.get('id')
        if (type(metadata['exit']) is not int or not isinstance(identity, str)
                or not identity or identity in seen):
            raise judge.ProtocolError("invalid or duplicated structured bash exit identity")
        seen.add(identity)
        receipts.append({'evidence_id': f't{turn:03d}.o{index:02d}.p01',
            'turn': turn, 'scope': 'whole_command', 'exit_code': metadata['exit'],
            'tool_call_id': identity, 'tool_name': 'bash', 'output_sha256': _sha(result['output'])})
    return receipts


def _checked_execution_receipts(value, bindings):
    """Require exact observation/call/turn/hash binding of trusted sidecars."""
    if value is None:
        return [], []
    fields = {'evidence_id', 'turn', 'scope', 'exit_code', 'tool_call_id', 'tool_name', 'output_sha256'}
    if type(value) is not list or len(value) > 4096:
        raise judge.ProtocolError("bounded trusted execution receipt list required")
    receipts, provenance, seen = [], [], set()
    for record in value:
        if type(record) is not dict or set(record) != fields:
            raise judge.ProtocolError("unexpected trusted execution receipt fields")
        ident = record['evidence_id']
        if not isinstance(ident, str) or ident not in bindings or ident in seen:
            raise judge.ProtocolError("execution receipt lacks unique original observation")
        binding = bindings[ident]
        if (type(record['turn']) is not int or type(record['exit_code']) is not int
                or record['scope'] != 'whole_command' or record['tool_name'] != 'bash'
                or any(record[key] != binding[key]
                       for key in ('turn', 'tool_call_id', 'tool_name', 'output_sha256'))):
            raise judge.ProtocolError("execution receipt call/turn/output binding mismatch")
        seen.add(ident)
        receipts.append({key: record[key] for key in ('evidence_id', 'turn', 'scope', 'exit_code')})
        provenance.append(deepcopy(record))
    return receipts, provenance


def _inline_image(part, max_bytes):
    value = part.get("image_url", part.get("url"))
    url = value.get("url") if type(value) is dict else value
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError("not_inline_image_original_bytes")
    if len(url) > 128 + 4 * ((max_bytes + 2) // 3):
        raise ValueError("per_image_byte_limit")
    match = _DATA.fullmatch(url)
    if match is None:
        raise ValueError("invalid_or_unsupported_image_data_uri")
    mime, encoded = match.groups()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid_image_base64") from exc
    if not raw or len(raw) > max_bytes:
        raise ValueError("per_image_byte_limit")
    # Never re-encode pixels. The decoder only verifies the original payload;
    # missing Pillow causes explicit unavailable evidence, not MIME guessing.
    try:
        from PIL import Image
        with Image.open(BytesIO(raw)) as image:
            if _FORMATS.get(image.format) != mime:
                raise ValueError("image_mime_signature_mismatch")
            if not 0 < image.width * image.height <= 40_000_000:
                raise ValueError("image_pixel_limit")
            image.verify()
    except ImportError as exc:
        raise ValueError("image_verifier_unavailable") from exc
    except (OSError, SyntaxError) as exc:
        raise ValueError("image_decoder_rejected_bytes") from exc
    sha = _sha(raw)
    for meta in (part, value if type(value) is dict else {}):
        declared_mime = meta.get("mime", meta.get("mime_type"))
        if declared_mime is not None and declared_mime != mime:
            raise ValueError("image_declared_mime_mismatch")
        if "sha256" in meta and meta["sha256"] != sha:
            raise ValueError("image_declared_sha256_mismatch")
        if "bytes" in meta and (type(meta["bytes"]) is not int or meta["bytes"] != len(raw)):
            raise ValueError("image_declared_byte_count_mismatch")
    return raw, mime, sha


def _uniform(items, count):
    if len(items) <= count:
        return items
    if count == 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [items[i * (len(items) - 1) // (count - 1)] for i in range(count)]


def _turn_balanced(items, count):
    """Cover visual turns before filling the remaining image budget uniformly.

    A large multi-image return must not crowd another target turn out entirely.
    Selection stays deterministic, byte-preserving and bounded by the same cap.
    """
    if len(items) <= count:
        return items
    groups = {}
    for item in items:
        groups.setdefault(item[0], []).append(item)
    representatives = [_uniform(group, 1)[0] for group in groups.values()]
    chosen = _uniform(representatives, min(count, len(representatives)))
    identities = {item[1] for item in chosen}
    remaining = [item for item in items if item[1] not in identities]
    chosen += _uniform(remaining, count-len(chosen))
    identities = {item[1] for item in chosen}
    return [item for item in items if item[1] in identities]


def prepare(*, task_text, attempt_id, stage, turns, start_turn, end_turn,
            task_id=None, max_images=16, max_image_bytes=8_000_000,
            max_total_image_bytes=24_000_000, max_history_images=4,
            execution_receipts=None, harness_notices=()):
    """Freeze <=10 completed 1-based turns and return v5 packet/context/media.

    turns must contain the original contiguous prefix 1..end_turn. Any future
    entries are untouched and have no effect on IDs/hashes/evidence. Each row
    has turn/action/messages; messages can also be a ToolSession results object.
    Call on complete tool returns; a terminal no-tool action uses messages=[].
    execution_receipts is a separate trusted-caller sidecar produced at the
    native ToolSession boundary, not metadata parsed from arbitrary messages.

    Images: reserve up to 4 (and at most one quarter of the total) for detailed
    history, cover target visual turns first and fill uniformly, then use the
    remainder for history. With the default budget, every target turn with an
    eligible image gets at least one selection before byte/validity checks.
    Unused history reserve goes to targets. Limits are judge-only, never actor
    context limits. Default 16 means up to 12 target + 4 history images.
    """
    if (not isinstance(task_text, str) or not task_text.strip()
            or not isinstance(attempt_id, str) or not attempt_id.strip()):
        raise judge.ProtocolError("original task and attempt identity are required")
    if type(turns) is not list or type(start_turn) is not int or type(end_turn) is not int:
        raise judge.ProtocolError("turns/bounds have invalid types")
    if not 1 <= start_turn <= end_turn <= min(len(turns), 256) or end_turn - start_turn >= 10:
        raise judge.ProtocolError("expected <=10 completed turns within the bounded original prefix")
    for key, value, minimum in (("max_images", max_images, 0), ("max_history_images", max_history_images, 0),
                                ("max_image_bytes", max_image_bytes, 1),
                                ("max_total_image_bytes", max_total_image_bytes, 1)):
        if type(value) is not int or value < minimum:
            raise judge.ProtocolError(f"invalid {key}")
    detailed_history = set(range(1, min(3, start_turn))) | set(range(max(1, start_turn - 4), start_turn))
    detailed = detailed_history | set(range(start_turn, end_turn + 1))
    evidence, rows, source_records, omissions, candidates = [], [], [], [], []
    execution_bindings = {}
    for expected, turn in enumerate(turns[:end_turn], 1):
        if type(turn) is not dict or type(turn.get("turn")) is not int or turn["turn"] != expected:
            raise judge.ProtocolError("turns must be original contiguous 1-based assistant boundaries")
        number = expected
        public = action_text(turn.get("action"))
        source = {"turn": number, "raw_action_sha256": _sha(_json(turn.get("action"))), "observations": []}
        ids, observations = [], []
        messages = turn.get("messages", [])
        if type(messages) is dict:
            messages = observation_messages(messages)
        for mi, pi, part in _parts(messages):
            if len(evidence) >= 4096 - end_turn - 2:
                raise judge.ProtocolError("bounded evidence inventory exceeded")
            ident = f"t{number:03d}.o{mi:02d}.p{pi:02d}"
            ids.append(ident)
            kind = part.get("type")
            if kind == "text":
                raw_text = part.get("text")
                if not isinstance(raw_text, str):
                    raise judge.ProtocolError("observation text must be a string")
                limit = 16000 if number in detailed else 240
                truncated = len(raw_text) > limit
                text = (evidence_context.TRUNCATED_TEXT + "\n" if truncated else "") + raw_text[:limit]
                ev = {"id": ident, "turn": number, "kind": "tool_text", "text": text}
                source["observations"].append({"id": ident, "kind": "tool_text", "sha256": _sha(raw_text),
                                               "original_chars": len(raw_text), "truncated": truncated})
                message = messages[mi-1]
                if (pi == 1 and message.get('role') == 'tool' and message.get('name') == 'bash'
                        and isinstance(message.get('tool_call_id'), str) and message['tool_call_id']):
                    execution_bindings[ident] = {'turn': number, 'tool_call_id': message['tool_call_id'],
                        'tool_name': 'bash', 'output_sha256': _sha(raw_text)}
            elif kind in {"image_url", "image", "input_image"}:
                ev = {"id": ident, "turn": number, "kind": "image"}
                reason = "middle_history_ledger" if number not in detailed else None
                if reason is None:
                    candidates.append((number, ident, deepcopy(part), ev))
                else:
                    ev["text"] = evidence_context.OMITTED_MEDIA + reason
                    omissions.append({"id": ident, "turn": number, "reason": reason})
                source["observations"].append({"id": ident, "kind": "image", "source_part_sha256": _sha(_json(part))})
            else:
                ev = {"id": ident, "turn": number, "kind": "unsupported_observation",
                      "text": "Unsupported observation part; unavailable to judge; type=" + str(kind)[:80]}
                source["observations"].append({"id": ident, "kind": "unsupported_observation", "sha256": _sha(_json(part))})
                omissions.append({"id": ident, "turn": number, "reason": "unsupported_observation_kind"})
            evidence.append(ev)
            observations.append({"evidence_id": ident, "kind": ev["kind"]})
        action = public if number in detailed else {
            "record_type": "incomplete_history_ledger", "turn": number,
            "tool_names": re.findall(r"<function=([^>]+)>", public),
            "action_sha256": _sha(public), "evidence_ids": ids,
            "note": "Earlier middle-history ledger; action and media omitted; unshown history is unknown."}
        row = {"turn": number, "action": action, "observations": observations}
        if number >= start_turn:
            row["evidence_ids"] = ids
        rows.append(row)
        source_records.append(source)

    history_candidates = [c for c in candidates if c[0] < start_turn]
    reserve = min(max_history_images, len(history_candidates), max_images // 4)
    chosen_targets = _turn_balanced([c for c in candidates if c[0] >= start_turn], max_images - reserve)
    chosen_history = _uniform(history_candidates, min(max_history_images, max_images - len(chosen_targets)))
    chosen = {c[1] for c in chosen_targets + chosen_history}
    image_parts, receipts, used_bytes = [], [], 0
    # Allocation order prioritizes target evidence, but labels preserve turns.
    ordered = sorted(candidates, key=lambda c: (c[0] < start_turn, c[0], c[1]))
    for number, ident, part, ev in ordered:
        reason = "judge_image_count_limit" if ident not in chosen else None
        if reason is None:
            try:
                raw, mime, sha = _inline_image(part, max_image_bytes)
                if used_bytes + len(raw) > max_total_image_bytes:
                    raise ValueError("judge_total_image_byte_limit")
                used_bytes += len(raw)
                receipts.append({"id": ident, "turn": number, "sha256": sha, "bytes": len(raw), "mime_type": mime})
                image_parts.extend([
                    {"text": f"IMMUTABLE IMAGE EVIDENCE id={ident} observed_at_turn={number} sha256={sha}"},
                    {"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode("ascii")}}])
            except (ValueError, binascii.Error) as exc:
                reason = str(exc)
        if reason is not None:
            ev["text"] = evidence_context.OMITTED_MEDIA + reason
            omissions.append({"id": ident, "turn": number, "reason": reason})
    execution, execution_sources = _checked_execution_receipts(execution_receipts, execution_bindings)
    metadata = {"version": VERSION, "attempt_id": attempt_id, "start_turn": start_turn, "end_turn": end_turn,
                "task_sha256": _sha(task_text), "prefix_source_sha256": _sha(_json(source_records)),
                "detailed_history_turns": sorted(detailed_history),
                "history_ledger_turns": [n for n in range(1, start_turn) if n not in detailed_history],
                "image_policy": "cover target visual turns, then uniform fill; uniform history reserve <=4 and <=one-quarter; exact original bytes",
                "max_images": max_images, "max_history_images": max_history_images, "max_image_bytes": max_image_bytes,
                "max_total_image_bytes": max_total_image_bytes, "images_sent": len(receipts),
                "image_bytes_sent": used_bytes, "omissions": omissions,
                "source_records": source_records,
                "execution_source_bindings": execution_sources,
                "execution_scope": "Only supplied, call/turn/output-bound native bash exits; no inferred effects or other tool execution."}
    visible_notices = []
    if harness_notices:
        from append_agent_rl.budget_notice import checked_notices
        visible_notices = checked_notices(harness_notices, end_turn=end_turn)
    if visible_notices:
        metadata["harness_budget_notices"] = visible_notices
    packet = {"packet_id": "prm5live_" + _sha(_json(metadata))[:24],
              "task_id": task_id if task_id is not None else "task_" + _sha(task_text)[:20],
              "stage": stage, "task": task_text,
              "history": [r for r in rows if r["turn"] < start_turn],
              "targets": [r for r in rows if r["turn"] >= start_turn], "evidence": evidence}
    if omissions:
        packet["evidence"].append({"id": "live_missing_evidence", "turn": 0,
                                    "kind": "archive_limitation", "text": _json(omissions)})
    item = context_v5.prepare(packet, receipts, execution_receipts=execution)
    if visible_notices:
        item["context"]["harness_budget_notices_v1"] = deepcopy(visible_notices)
    metadata["packet_sha256"] = item["context"]["augmented_packet_sha256"]
    return {**item, "image_parts": image_parts, "image_receipts": receipts, "metadata": metadata}
