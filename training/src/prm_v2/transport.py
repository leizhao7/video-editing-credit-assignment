"""Bounded Gemini transport for an isolated offline judge experiment.

No rollout tools, production imports, credentials in artifacts, or safety fallback.
Each vote is a separate request; three calls are not claimed independent errors.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import judge

MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 2
MAX_WORKERS = 12
SAFETY_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "SPII", "RECITATION", "BLOCKLIST", "IMAGE_SAFETY"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError("Refusing to overwrite a different experiment artifact")


def model_metadata(model, key):
    name = urllib.parse.quote(model, safe="-")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{name}",
        headers={"x-goog-api-key": key})
    with urllib.request.urlopen(request, timeout=40) as response:
        obj = json.load(response)
    return {k: obj.get(k) for k in ("name", "version", "displayName", "inputTokenLimit",
                                     "outputTokenLimit", "supportedGenerationMethods")}


def prepare_evidence(packet, bundle):
    """Resolve only manifest-listed immutable local images; never arbitrary URLs."""
    packet = json.loads(canonical(packet))
    # Keep the deterministic middle-history ledger, but explicitly distinguish
    # it from a full action/observation transcript. No summaries are invented.
    ledger = packet.pop("history_ledger", [])
    for row in ledger:
        packet["history"].append({"turn": row["turn"],
            "action": {"record_type": "incomplete_history_ledger", **row},
            "observations": [{"evidence_id": ident} for ident in row["evidence_ids"]]})
    packet["history"].sort(key=lambda row: row["turn"])
    missing = packet.pop("missing", [])
    if missing:
        packet["evidence"].append({"id": "archive_missing_evidence", "turn": 0,
            "kind": "archive_limitation", "text": canonical(missing).decode()})
    root = Path(bundle).resolve()
    image_parts, image_receipts = [], []
    for ev in packet.get("evidence", []):
        relative = ev.pop("local_path", None)
        # Paths must not be presented as visual observations to the model.
        ev.pop("source_path", None)
        media_meta = {k: ev.pop(k) for k in list(ev) if k not in {"id", "turn", "kind", "text"}}
        notes = []
        if media_meta.get("truncated"):
            notes.append("ARCHIVE TEXT TRUNCATED; unshown content is unknown.")
        if media_meta.get("available_to_judge") is False:
            notes.append("MEDIA NOT ATTACHED. Judge cannot inspect this evidence; omission="
                         + str(media_meta.get("omission_reason", "unavailable")))
        if notes:
            ev["text"] = "\n".join(notes + ([ev["text"]] if "text" in ev else []))
        if not relative:
            continue
        target = (root / relative).resolve()
        if root not in target.parents:
            raise ValueError("Image path escapes the frozen input bundle")
        blob = target.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        expected = media_meta.get("sha256") or target.stem
        if sha != expected:
            raise ValueError("Frozen image hash mismatch")
        if len(blob) > 8_000_000:
            raise ValueError("Unexpectedly large archived image")
        mime = media_meta.get("mime_type") or media_meta.get("mime") or "image/jpeg"
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            raise ValueError("Unsupported archived image MIME")
        image_parts.extend([
            {"text": f"IMMUTABLE IMAGE EVIDENCE id={ev['id']} observed_at_turn={ev['turn']} sha256={sha}"},
            {"inlineData": {"mimeType": mime, "data": base64.b64encode(blob).decode()}}])
        image_receipts.append({"id": ev["id"], "turn": ev["turn"], "sha256": sha,
                               "bytes": len(blob), "mime_type": mime})
    return packet, image_parts, image_receipts


def payload_for(packet, image_parts, arm):
    if arm not in {"legacy", "v2"}:
        raise ValueError("Unknown scoring arm")
    prompt = judge.LEGACY_SYSTEM_PROMPT if arm == "legacy" else judge.SYSTEM_PROMPT
    # Identical user evidence for both arms. Only the rubric/output protocol changes.
    target_turns = [row["turn"] for row in packet["targets"]]
    user = ("UNTRUSTED EVIDENCE PACKET\n" + canonical(packet).decode()
            + "\nScore ONLY targets, not history. Requested turn IDs in exact output order: "
            + json.dumps(target_turns)
            + f". Return exactly {len(target_turns)} scores/assessments under your system output protocol.")
    return {
        "systemInstruction": {"parts": [{"text": prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user}] + image_parts}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192,
                             "thinkingConfig": {"thinkingLevel": "low"},
                             "responseMimeType": "application/json"}}


def parse_legacy(raw, packet):
    obj = judge.parse_legacy_scores(raw, len(packet["targets"]))
    return {"scores": obj, "turns": [t["turn"] for t in packet["targets"]]}


def parse_v2_wire(raw, packet):
    """One lossless wire compatibility: a bare array of complete assessments.

    Do not coerce scores, fill fields, extract prose/Markdown, strip unknown keys,
    deduplicate references or relax the canonical validator. The original raw
    response is always retained and normalization is explicit in its receipt.
    """
    if isinstance(raw, str) and raw.lstrip().startswith("["):
        # Embedding the original bytes retains duplicate-key/nonfinite checks.
        value = judge.parse_assessment('{"assessments":' + raw + '}', packet)
        return value, "bare_assessment_array_wrapped"
    return judge.parse_assessment(raw, packet), None


class RequestGate:
    def __init__(self, spacing=0.3):
        self.spacing, self.next_time, self.lock = spacing, 0., threading.Lock()

    def wait(self):
        with self.lock:
            delay = max(0., self.next_time - time.monotonic())
            self.next_time = max(self.next_time, time.monotonic()) + self.spacing
        if delay:
            time.sleep(delay)


def one_vote(packet, payload, *, arm, vote, model, key, out, gate, request_fn=None, parse_fn=None):
    opener = request_fn or urllib.request.urlopen
    root = Path(out) / "votes" / packet["packet_id"] / arm / f"vote_{vote}"
    last = None
    for attempt in range(MAX_ATTEMPTS):
        gate.wait()
        started = time.monotonic()
        receipt = {"packet_id": packet["packet_id"], "arm": arm, "vote": vote,
                   "attempt": attempt, "model": model, "request_sha256": digest(payload)}
        retry, raw_obj = False, None
        request = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=canonical(payload), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with opener(request, timeout=150) as response:
                raw_obj = json.load(response)
            # Raw provider bytes have no request authentication headers.
            save_new(root / f"attempt_{attempt}.response.json", raw_obj)
            receipt.update(provider_model=raw_obj.get("modelVersion"),
                           response_id=raw_obj.get("responseId"), usage=raw_obj.get("usageMetadata"))
            blocked = (raw_obj.get("promptFeedback") or {}).get("blockReason")
            candidates = raw_obj.get("candidates") or []
            finishes = [c.get("finishReason") for c in candidates]
            if blocked or any(reason in SAFETY_FINISH for reason in finishes):
                receipt.update(status="policy_refusal", block_reason=blocked, finish_reasons=finishes)
            elif not candidates:
                receipt.update(status="empty_response")
                retry = True
            else:
                raw = "".join(part.get("text", "") for part in
                              (candidates[0].get("content") or {}).get("parts", [])
                              if not part.get("thought"))
                if candidates[0].get("finishReason") != "STOP":
                    receipt.update(status="incomplete_response", finish_reasons=finishes)
                    retry = candidates[0].get("finishReason") == "MAX_TOKENS"
                else:
                    try:
                        if parse_fn is not None:
                            parsed, normalization = parse_fn(raw, packet)
                        elif arm == "legacy":
                            parsed, normalization = parse_legacy(raw, packet), None
                        else:
                            parsed, normalization = parse_v2_wire(raw, packet)
                    except (ValueError, TypeError, KeyError) as exc:
                        receipt.update(status="schema_error", error_type=type(exc).__name__,
                                       # Parser messages contain evidence IDs, not API configuration.
                                       validation_error=str(exc)[:700])
                        retry = True
                    else:
                        receipt.update(status="completed", result=parsed,
                                       wire_normalization=normalization)
        except urllib.error.HTTPError as exc:
            # Never stringify exceptions or URLs and never persist auth headers.
            receipt.update(status="http_error", http_status=exc.code)
            retry = exc.code in {408, 429, 500, 502, 503, 504}
            delay = exc.headers.get("Retry-After", "") if exc.headers else ""
            if delay.isdigit():
                delay = int(delay)
                if delay > 30:
                    retry = False
                    receipt["retry_after_exceeds_budget"] = delay
                elif retry:
                    time.sleep(delay)
        except (TimeoutError, OSError, json.JSONDecodeError) as exc:
            receipt.update(status="transport_error", error_type=type(exc).__name__)
            retry = True
        receipt.update(seconds=time.monotonic() - started, retryable=retry)
        save_new(root / f"attempt_{attempt}.receipt.json", receipt)
        last = receipt
        if not retry:
            break
        if attempt + 1 < MAX_ATTEMPTS:
            time.sleep(1)
    save_new(root / "final.json", last)
    return last


def evaluate(packets, *, bundle, out, model=MODEL, arms=("legacy", "v2"), votes=3, workers=12,
             progress=None):
    if votes != 3 or not (1 <= workers <= MAX_WORKERS):
        raise ValueError("Experiment requires three separate votes and bounded concurrency")
    out = Path(out)
    if out.exists():
        raise ValueError("Use a fresh experiment tag; existing artifacts are immutable")
    out.mkdir(parents=True)
    key = os.environ["GOOGLE_AI_STUDIO_API_KEY"]
    model_info = model_metadata(model, key)
    save_new(out / "model_metadata.json", model_info)
    if "generateContent" not in (model_info.get("supportedGenerationMethods") or []):
        raise ValueError("Selected model does not advertise generateContent; no automatic substitution")
    config = {"model": model, "arms": list(arms), "votes": votes, "workers": workers,
              "max_attempts": MAX_ATTEMPTS, "gpu_count": 0, "writes_to_training": False,
              "packet_count": len(packets), "request_limit": len(packets) * len(arms) * votes * MAX_ATTEMPTS,
              "labeling_scope": "offline window review; future evidence citations rejected, no causal guarantee",
              "legacy_prompt_sha256": hashlib.sha256(judge.LEGACY_SYSTEM_PROMPT.encode()).hexdigest(),
              "v2_prompt_sha256": hashlib.sha256(judge.SYSTEM_PROMPT.encode()).hexdigest()}
    save_new(out / "config.json", config)
    jobs, frozen = [], []
    for raw_packet in packets:
        packet, images, image_receipts = prepare_evidence(raw_packet, bundle)
        # Structural validation occurs before spending requests in either arm.
        judge.validate_packet(packet)
        frozen.append(packet)
        save_new(out / "inputs" / f"{packet['packet_id']}.packet.json", packet)
        save_new(out / "inputs" / f"{packet['packet_id']}.images.json", image_receipts)
        for arm in arms:
            payload = payload_for(packet, images, arm)
            save_new(out / "requests" / f"{packet['packet_id']}.{arm}.json", payload)
            jobs.extend((packet, payload, arm, vote) for vote in range(votes))
    gate, results = RequestGate(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(one_vote, p, data, arm=arm, vote=v, model=model, key=key,
                               out=out, gate=gate) for p, data, arm, v in jobs]
        for future in as_completed(pending):
            results.append(future.result())
            if progress:
                progress(len(results), len(jobs), results[-1])
    save_new(out / "results.json", results)
    return results
