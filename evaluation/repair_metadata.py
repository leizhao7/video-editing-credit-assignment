"""Create auditable S4E dialogue metadata candidates from the actual take audio.

Never modifies the input corpus. ASR proposals are not declared human-verified;
uncertain/empty transcripts are retained as review-required, not silently fixed.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from grading_contract import media_info, source_file

FIELDS = ("speech_spans_s", "text", "words_start_at_s", "words_dur_s", "take_dur_s")


def repair_take(row, path, transcribe):
    info = media_info(path)
    stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    duration = float(stream.get("duration", info["format"]["duration"]))
    segments = list(transcribe(str(path)))
    spans, words, rejected = [], [], []
    for seg in segments:
        selected = []
        for word in seg.words or []:
            start, end = float(word.start), float(word.end)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
                rejected.append({"s": start, "e": end, "text": word.word})
                continue
            selected.append({"s": round(start, 4), "e": round(end, 4),
                             "text": word.word, "probability": float(word.probability)})
        if selected:
            spans.append({"s": selected[0]["s"], "e": selected[-1]["e"],
                          "text": "".join(w["text"] for w in selected).strip()})
            words.extend(selected)
    proposed = copy.deepcopy(row)
    proposed["take_dur_s"] = duration
    proposed["speech_spans_s"] = spans
    proposed["text"] = " ".join(x["text"] for x in spans)
    proposed["words_start_at_s"] = words[0]["s"] if words else None
    proposed["words_dur_s"] = round(words[-1]["e"]-words[0]["s"], 4) if words else 0.
    uncertain = not words or bool(rejected) or any(w["probability"] < .5 for w in words)
    receipt = {"take": row["id"], "media_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               "actual_duration_s": duration, "before": {k: row.get(k) for k in FIELDS},
               "after": {k: proposed.get(k) for k in FIELDS}, "words": words,
               "discarded_out_of_range_words": rejected,
               "status": "review_required" if uncertain else "asr_candidate",
               "verified_semantics": False,
               "note": "Fresh file-relative ASR; no time-only clamp retaining stale text. Full-take transcript replaces the old located-snippet hint."}
    return proposed, receipt


def repair_task(source, output, transcribe, expected_manifest_sha256, only_invalid=True):
    source, output = Path(source), Path(output)
    raw = (source/"initial_state/manifest.json").read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_manifest_sha256:
        raise RuntimeError("source manifest changed since selection; refusing candidate repair")
    manifest = json.loads(raw)
    receipts, candidates = [], []
    for row in manifest.get("dialogue", []):
        media = source_file(source, row["file"])
        info = media_info(media)
        stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
        duration = float(stream.get("duration", info["format"]["duration"]))
        bad = any(not (0 <= float(s["s"]) < float(s["e"]) <= duration + .1)
                  for s in row.get("speech_spans_s", []))
        if only_invalid and not bad:
            candidates.append(copy.deepcopy(row))
            continue
        new, receipt = repair_take(row, media, transcribe)
        candidates.append(new); receipts.append(receipt)
    manifest["dialogue"] = candidates
    # Candidate is deliberately outside the data volume and not auto-applied.
    output.mkdir(parents=True, exist_ok=False)
    (output/"manifest.candidate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    receipt = {"source": str(source), "manifest_before_sha256": actual,
               "candidate_sha256": hashlib.sha256((output/"manifest.candidate.json").read_bytes()).hexdigest(),
               "model": "faster-whisper small.en int8", "takes": receipts,
               "applied": False, "semantically_verified": False}
    (output/"repair_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return {"task": source.name, "takes": len(receipts),
            "review_required": sum(x["status"] == "review_required" for x in receipts),
            "output": str(output), "applied": False}
