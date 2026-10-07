"""Build a frozen, label-blind PRM calibration set from existing rollout windows.

This is an offline data adapter, not a training or judge implementation. Archive
labels and outcomes are retained in metadata.json only. Media are original bytes,
addressed by their verified SHA256. No media fetching or API calls occur here.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


STAGES = ("s_1", "s_2", "s_3", "s_4e")
ERROR_MARKER = re.compile(
    r"Traceback \(most recent call last\)|(?:^|\n)(?:Error|ERROR|PermissionError|FileNotFoundError|SyntaxError):"
    r"|(?:exit(?:ed)? (?:code|status)[=: ]+)[1-9]\d*|No such file or directory|Invalid argument",
    re.I,
)
TOOL_CALL = re.compile(r"<tool_call>.*?</tool_call>", re.S)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def text_content(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def visible_action(text: str, *, prose_limit: int = 2400) -> str:
    """Strip both paired thinking and the Qwen closing-only thinking format.

    Tool calls/arguments are never truncated. Long non-tool prose is explicitly
    abbreviated; raw content remains recoverable from the hashed source archive.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*$", "", text, flags=re.S)
    text = text.replace("<|im_end|>", "").strip()
    calls = TOOL_CALL.findall(text)
    prose = TOOL_CALL.sub("", text).strip()
    if len(prose) > prose_limit:
        prose = prose[:prose_limit] + "\n[Non-tool prose abbreviated; tool arguments preserved.]"
    return "\n".join([prose, *calls]).strip()


def split_turns(row: dict) -> tuple[str, list[dict]]:
    """Use original assistant boundaries, without inventing extra tool turns."""
    task_parts, turns = [], []
    for message in row.get("conversation", []):
        role = message.get("role")
        if role == "assistant":
            raw = text_content(message)
            action = visible_action(raw)
            if message.get("tool_calls"):
                action += "\n" + json.dumps({"tool_calls": message["tool_calls"]}, ensure_ascii=False)
            turns.append({"turn": len(turns) + 1, "action": action, "messages": []})
        elif not turns and role == "user":
            task_parts.append(text_content(message))
        elif turns and role in ("tool", "user", "function"):
            turns[-1]["messages"].append(message)
    return "\n\n".join(task_parts), turns


def iter_rows(sources: Iterable[Path]):
    for source in sources:
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    yield source, line_number, json.loads(line)


def index_candidates(sources: list[Path], source_hashes: dict[str, str]) -> tuple[list[dict], list[dict]]:
    candidates, rejected = [], []
    for source, line_number, row in iter_rows(sources):
        task, turns = split_turns(row)
        uid, stage, task_id = row.get("trajectory_uid"), row.get("stage"), row.get("task")
        if not uid or not task_id or stage not in STAGES:
            rejected.append({"source": str(source), "line": line_number, "reason": "missing_identity"})
            continue
        for window_index, window in enumerate(row.get("process_reward", {}).get("windows", [])):
            start, end = window.get("start_turn"), window.get("end_turn")
            if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start <= end <= len(turns) or end - start >= 10:
                rejected.append({"source": str(source), "line": line_number, "window": window_index, "reason": "invalid_window_bounds"})
                continue
            selected = turns[start - 1:end]
            observation_text = "\n".join(text_content(m) for t in selected for m in t["messages"])
            scores = window.get("mean_scores", [])
            candidates.append({
                "source": str(source), "source_sha256": source_hashes[str(source)], "line": line_number,
                "trajectory_uid": uid, "stage": stage, "task_id": task_id, "window_index": window_index,
                "start_turn": start, "end_turn": end, "target_count": end - start + 1,
                "trajectory_turns": len(turns), "terminal_window": end == len(turns),
                "has_error_marker": bool(ERROR_MARKER.search(observation_text)),
                "old_all_plus_one": bool(scores) and all(score == 1 for score in scores),
                "task_missing": not bool(task.strip()),
            })
    return candidates, rejected


def stable_order(seed: int, candidate: dict) -> str:
    identity = [seed, candidate["source_sha256"], candidate["trajectory_uid"], candidate["start_turn"], candidate["end_turn"]]
    return sha256_bytes(canonical(identity))


def select_candidates(candidates: list[dict], *, seed: int = 42, per_stage: int = 6,
                      challenges: dict[str, str] | None = None) -> list[dict]:
    """Deduplicate tasks before selecting pre-existing windows, never by new score.

    Optional caller-selected challenges are labeled separately from the seeded
    strata. A shortage is reported instead of silently duplicating tasks.
    """
    challenges = challenges or {}
    if any(stage not in STAGES or not isinstance(task, str) or not task.strip()
           for stage, task in challenges.items()):
        raise ValueError("Challenges must map supported stages to non-empty task IDs")
    selected = []
    for stage in STAGES:
        available = [c for c in candidates if c["stage"] == stage and not c["task_missing"]]
        selected_tasks = set()
        roles = ["challenge_saturation", "early", "middle", "terminal", "error_marker", "late"]
        for position in range(min(per_stage, len({c["task_id"] for c in available}))):
            pool = [c for c in available if c["task_id"] not in selected_tasks]
            role = roles[position % len(roles)]
            challenge = challenges.get(stage)
            if role == "challenge_saturation" and challenge and any(c["task_id"] == challenge for c in pool):
                pool = [c for c in pool if c["task_id"] == challenge]
                selection_kind = "challenge_saturation"
            else:
                if role == "challenge_saturation":
                    role = "early"
                selection_kind = "seeded_stratum"
            def rank(c):
                n = c["target_count"]
                # Retain useful late windows without allowing one-turn tails
                # to dominate this small calibration set.
                substantial = n >= 7
                if role == "challenge_saturation":
                    preferred = c["old_all_plus_one"]
                    distance = 0
                elif role == "early":
                    preferred, distance = c["start_turn"] == 1, c["start_turn"]
                elif role == "middle":
                    preferred = c["start_turn"] > 1 and not c["terminal_window"]
                    distance = abs((c["start_turn"] + c["end_turn"]) / 2 - c["trajectory_turns"] / 2)
                elif role == "terminal":
                    preferred, distance = c["terminal_window"], -n
                elif role == "error_marker":
                    preferred, distance = c["has_error_marker"], 0
                else:
                    preferred = c["start_turn"] > 1
                    distance = c["trajectory_turns"] - c["end_turn"]
                return (not substantial, not preferred, -n, distance, stable_order(seed, c))
            winner = min(pool, key=rank)
            selected.append({**winner, "selection_kind": selection_kind, "requested_stratum": role})
            selected_tasks.add(winner["task_id"])
    return selected


def _image_asset(part: dict) -> tuple[str | None, str | None]:
    if part.get("type") == "image" and isinstance(part.get("asset"), dict):
        return part["asset"].get("sha256"), part["asset"].get("name")
    return None, None


def make_packet(row: dict, choice: dict, *, media_dir: Path) -> tuple[dict, dict]:
    task, turns = split_turns(row)
    start, end = choice["start_turn"], choice["end_turn"]
    if not task.strip():
        raise ValueError("Task prompt is missing; cannot construct judge input")
    targets = list(range(start, end + 1))
    if not 1 <= start <= end <= len(turns):
        raise ValueError("Selected target window no longer matches the archive")
    history = sorted(set(range(1, min(3, start))) | set(range(max(1, start - 4), start)))
    detailed = set(history + targets)
    ledger_turns = [t for t in range(1, start) if t not in detailed]
    packet_id = "prm2_" + sha256_bytes(canonical([choice["source_sha256"], choice["trajectory_uid"], start, end]))[:20]
    evidence, built_turns, ledger, missing = [], {}, [], []
    media_dir.mkdir(parents=True, exist_ok=True)
    assets = row.get("image_assets", {})
    for turn in turns[:end]:
        number, observations, ids = turn["turn"], [], []
        detail = number in detailed
        for message_index, message in enumerate(turn["messages"], 1):
            content = message.get("content", "")
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
            if not isinstance(parts, list):
                parts = [{"type": "text", "text": json.dumps(content, ensure_ascii=False)}]
            for part_index, part in enumerate(parts, 1):
                if not isinstance(part, dict):
                    part = {"type": "text", "text": str(part)}
                eid = f"t{number:03d}.o{message_index:02d}.p{part_index:02d}"
                ids.append(eid)
                if part.get("type") == "text":
                    text = str(part.get("text", ""))
                    # Full tool results for targets / selected history; the
                    # middle history keeps an explicit, deterministic ledger.
                    limit = 16000 if detail else 240
                    entry = {"id": eid, "turn": number, "kind": "tool_text", "text": text[:limit],
                             "original_chars": len(text), "text_sha256": sha256_bytes(text.encode()),
                             "truncated": len(text) > limit, "context_detail": "full" if detail else "ledger"}
                    evidence.append(entry)
                    observations.append({"evidence_id": eid, "kind": "tool_text"})
                    continue
                asset_hash, name = _image_asset(part)
                entry = {"id": eid, "turn": number, "kind": "image", "sha256": asset_hash,
                         "source_path": f"frames/{name}" if name and "/" not in name else name,
                         "archive_asset_name": name, "context_detail": "full" if detail else "ledger"}
                asset = assets.get(asset_hash, {}) if asset_hash else {}
                if not detail:
                    entry.update(available_to_judge=False, omission_reason="middle_history_ledger")
                elif not isinstance(asset, dict) or not asset.get("base64"):
                    entry.update(available_to_judge=False, missing=True, missing_reason="original_image_bytes_absent")
                    missing.append({"evidence_id": eid, "reason": "original_image_bytes_absent"})
                else:
                    try:
                        raw = base64.b64decode(asset["base64"], validate=True)
                        if sha256_bytes(raw) != asset_hash:
                            raise ValueError("image_sha256_mismatch")
                        if asset.get("bytes") is not None and len(raw) != asset["bytes"]:
                            raise ValueError("image_size_mismatch")
                        mime = asset.get("mime_type", "application/octet-stream")
                        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(mime, ".bin")
                        path = media_dir / f"{asset_hash}{suffix}"
                        if path.exists() and path.read_bytes() != raw:
                            raise ValueError("existing_media_conflicts_with_source")
                        if not path.exists():
                            path.write_bytes(raw)
                        entry.update(local_path=f"media/{path.name}", mime_type=mime, bytes=len(raw), available_to_judge=True)
                    except (ValueError, TypeError) as exc:
                        entry.update(available_to_judge=False, missing=True, missing_reason=str(exc))
                        missing.append({"evidence_id": eid, "reason": str(exc)})
                evidence.append(entry)
                observations.append({"evidence_id": eid, "kind": "image"})
        data = {"turn": number, "action": turn["action"], "observations": observations}
        if number in targets:
            data["evidence_ids"] = ids
        if detail:
            built_turns[number] = data
        else:
            tools = re.findall(r"<function=([^>]+)>", turn["action"])
            ledger.append({"turn": number, "tool_names": tools, "action_sha256": sha256_bytes(turn["action"].encode()),
                           "evidence_ids": ids, "note": "Earlier middle-history ledger; no reasoning transcript."})
    packet = {"packet_id": packet_id, "task_id": choice["task_id"], "stage": choice["stage"], "task": task,
              "history": [built_turns[t] for t in history], "history_ledger": ledger,
              "targets": [built_turns[t] for t in targets], "evidence": evidence,
              "missing": missing}
    window = row["process_reward"]["windows"][choice["window_index"]]
    metadata = {"packet_id": packet_id, **choice, "archive_global_step": row.get("global_step"),
                "outcome_reward": row.get("reward"), "old_window": window,
                "old_turn_scores": row["process_reward"].get("turn_scores", [])[start - 1:end],
                "old_turn_valid": row["process_reward"].get("turn_score_valid", [])[start - 1:end],
                "history_turns": history, "history_ledger_turns": ledger_turns,
                "task_sha256": sha256_bytes(task.encode()), "packet_sha256": sha256_bytes(canonical(packet)),
                "missing": missing}
    return packet, metadata


def build_dataset(sources: list[Path], output: Path, *, seed: int = 42, per_stage: int = 6,
                  challenges: dict[str, str] | None = None) -> dict:
    if not sources:
        raise ValueError("At least one caller-supplied source archive is required")
    sources = [Path(source).resolve() for source in sources]
    source_hashes = {str(source): file_sha256(source) for source in sources}
    candidates, rejected = index_candidates(sources, source_hashes)
    choices = select_candidates(candidates, seed=seed, per_stage=per_stage, challenges=challenges)
    by_source_line = defaultdict(list)
    for choice in choices:
        by_source_line[(choice["source"], choice["line"])].append(choice)
    output.mkdir(parents=True, exist_ok=True)
    packets, metadata = {}, {}
    for source, line_number, row in iter_rows(sources):
        for choice in by_source_line.get((str(source), line_number), []):
            packet, meta = make_packet(row, choice, media_dir=output / "media")
            key = (choice["source"], choice["line"], choice["window_index"])
            packets[key], metadata[key] = packet, meta
    keys = [(c["source"], c["line"], c["window_index"]) for c in choices]
    packet_list, metadata_list = [packets[k] for k in keys], [metadata[k] for k in keys]
    available = {stage: len({c["task_id"] for c in candidates if c["stage"] == stage}) for stage in STAGES}
    media = {e["local_path"]: {"sha256": e["sha256"], "bytes": e["bytes"], "mime_type": e["mime_type"]}
             for p in packet_list for e in p["evidence"] if e.get("local_path")}
    manifest = {"format": "prm-calibration-original-windows-v2", "seed": seed,
                "source_sha256": source_hashes, "source_rows": sum(1 for _ in iter_rows(sources)),
                "candidate_windows": len(candidates), "rejected": rejected,
                "available_unique_tasks_by_stage": available,
                "selected_packets": len(packet_list), "unique_selected_tasks": len({p["task_id"] for p in packet_list}),
                "selected_by_stage": dict(Counter(p["stage"] for p in packet_list)),
                "target_turns": sum(len(p["targets"]) for p in packet_list),
                "target_turns_by_stage": {s: sum(len(p["targets"]) for p in packet_list if p["stage"] == s) for s in STAGES},
                "selection_counts": dict(Counter(m["selection_kind"] for m in metadata_list)),
                "terminal_windows": sum(m["terminal_window"] for m in metadata_list),
                "error_marker_windows": sum(m["has_error_marker"] for m in metadata_list),
                "missing_evidence_count": sum(len(p["missing"]) for p in packet_list),
                "media": media, "media_files": len(media), "media_bytes": sum(m["bytes"] for m in media.values()),
                "shortfall_policy": "Task uniqueness takes priority when the supplied archives contain fewer tasks than requested.",
                "label_separation": "packets.json contains no outcomes, old judge scores, votes or selection labels; these exist only in metadata.json.",
                "provenance": "Local saved rollout subset, not actual optimizer batch; selection is fixed before new judging.",
                "packets_sha256": sha256_bytes(canonical(packet_list)), "metadata_sha256": sha256_bytes(canonical(metadata_list))}
    for name, data in (("packets.json", packet_list), ("metadata.json", metadata_list), ("manifest.json", manifest)):
        path = output / name
        new = canonical(data)
        if path.exists() and path.read_bytes() != new:
            raise ValueError(f"Refusing to replace changed frozen calibration file: {path}")
        if not path.exists():
            path.write_bytes(new)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True,
                        help="Caller-supplied rollout archive; repeat for multiple archives")
    parser.add_argument("--output", type=Path, required=True,
                        help="Private output directory outside the source repository")
    parser.add_argument("--challenge", action="append", default=[], metavar="STAGE=TASK_ID",
                        help="Optional caller-selected task; repeat once per stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-stage", type=int, default=6)
    args = parser.parse_args()
    challenges = {}
    for spec in args.challenge:
        stage, separator, task = spec.partition("=")
        if not separator or stage not in STAGES or not task.strip() or stage in challenges:
            parser.error("--challenge must be STAGE=TASK_ID with one entry per supported stage")
        challenges[stage] = task
    manifest = build_dataset(args.source, args.output, seed=args.seed, per_stage=args.per_stage,
                             challenges=challenges)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("media", "rejected")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
