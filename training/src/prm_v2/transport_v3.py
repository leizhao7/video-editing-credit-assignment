"""Offline v2-control/v3 experiment with identical augmented evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re

from . import judge, judge_v3, transport, evidence_context, numeric_facts

ARMS = ("v2_control", "v3")


def prepare_context(packet, receipts):
    item = evidence_context.prepare(packet, image_receipts=receipts)
    numeric = numeric_facts.prepare(packet)
    # Avoid padding unrelated selection/image-review windows with a long list
    # of unsupported numeric-parser cases. Never assign a score from these facts.
    if any(card["current_values"] for card in numeric["cards"]):
        item["context"]["typed_numeric_observations"] = numeric
    return item


def payload_for(packet, context, images, arm):
    if arm not in ARMS:
        raise ValueError("Unsupported offline experiment arm")
    targets = [row["turn"] for row in packet["targets"]]
    user = ("UNTRUSTED EVIDENCE PACKET\n" + transport.canonical(packet).decode()
            + "\nDETERMINISTIC PROVENANCE INDEX (not gold labels or semantic judgments)\n"
            + transport.canonical(context).decode()
            + "\nScore ONLY target turns, not history. Exact output order: " + json.dumps(targets)
            + f". Return exactly {len(targets)} assessments under your system output protocol.")
    return {"systemInstruction": {"parts": [{"text": judge_v3.SYSTEM_PROMPT if arm == "v3" else judge.SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user}] + images}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192,
            "thinkingConfig": {"thinkingLevel": "low"}, "responseMimeType": "application/json",
            "responseJsonSchema": judge_v3.response_schema(len(targets), comparisons=arm == "v3")}}


def evaluate(packets, *, bundle, out, arms=ARMS, model=transport.MODEL, workers=12, progress=None):
    if not set(arms) <= set(ARMS) or not arms or len(arms) != len(set(arms)):
        raise ValueError("Invalid arms")
    if not (1 <= workers <= 12):
        raise ValueError("Invalid concurrency")
    if not packets:
        raise ValueError("Empty offline experiment is not a valid test")
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh output tag required")
    prepared, ids = [], set()
    # Validate all evidence before any provider call; never silently drop packets.
    for raw in packets:
        packet, images, receipts = transport.prepare_evidence(raw, bundle)
        pid = packet["packet_id"]
        if not re.fullmatch(r"[A-Za-z0-9:_-]+", pid) or pid in ids:
            raise ValueError("Invalid or duplicate packet identity")
        ids.add(pid)
        original_sha = transport.digest(packet)
        item = prepare_context(packet, receipts)
        judge.validate_packet(item["packet"])
        prepared.append((item["packet"], item["context"], images, receipts, original_sha))
    key = os.environ["GOOGLE_AI_STUDIO_API_KEY"]
    info = transport.model_metadata(model, key)
    if "generateContent" not in (info.get("supportedGenerationMethods") or []):
        raise ValueError("Selected model unavailable; no fallback")
    out.mkdir(parents=True)
    transport.save_new(out / "model_metadata.json", info)
    transport.save_new(out / "config.json", {"model": model, "arms": list(arms), "votes": 3,
        "workers": workers, "max_attempts": transport.MAX_ATTEMPTS, "packet_count": len(packets),
        "request_limit": len(packets) * len(arms) * 3 * transport.MAX_ATTEMPTS,
        "writes_to_training": False, "gpu_count": 0, "protocol": judge_v3.PROTOCOL_VERSION,
        "same_augmented_evidence": True, "different_output_schemas": True,
        "numeric_card_version": numeric_facts.VERSION,
        "effect_scope": "rubric plus output comparison protocol; not pure prompt-only or training effect",
        "v2_prompt_sha256": hashlib.sha256(judge.SYSTEM_PROMPT.encode()).hexdigest(),
        "v3_prompt_sha256": hashlib.sha256(judge_v3.SYSTEM_PROMPT.encode()).hexdigest()})
    jobs = []
    for packet, context, images, receipts, original_sha in prepared:
        pid = packet["packet_id"]
        transport.save_new(out / "inputs" / f"{pid}.packet.json", packet)
        transport.save_new(out / "inputs" / f"{pid}.context.json", context)
        transport.save_new(out / "inputs" / f"{pid}.images.json", receipts)
        transport.save_new(out / "inputs" / f"{pid}.provenance.json", {
            "before_augmentation_sha256": original_sha, "augmented_sha256": transport.digest(packet)})
        for arm in arms:
            payload = payload_for(packet, context, images, arm)
            transport.save_new(out / "requests" / f"{pid}.{arm}.json", payload)
            parser = partial(judge_v3.parse_wire, context=context) if arm == "v3" else transport.parse_v2_wire
            jobs.extend((packet, payload, arm, vote, parser) for vote in range(3))
    gate, results = transport.RequestGate(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transport.one_vote, packet, payload, arm=arm, vote=vote,
            model=model, key=key, out=out, gate=gate, parse_fn=parser)
            for packet, payload, arm, vote, parser in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress:
                progress(len(results), len(jobs), result)
    transport.save_new(out / "results.json", results)
    return results
