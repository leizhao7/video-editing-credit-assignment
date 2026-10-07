"""Private, hash-bound training answers. Never import this module in a sandbox.

S4E reference constraints are intentionally NOT fabricated executable edit.json.
Only fields actually present in the released answers are called ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

FORMAT = "answer-privileged-opsd-v1"
STAGES = ("s_1", "s_2", "s_3", "s_4e")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid reference time")
    return value


def fields(row, names):
    return {name: row[name] for name in names if name in row}


def answer_card(stage, answer, manifest):
    require(stage in STAGES, "Unsupported stage")
    require(answer.get("task") == manifest.get("taskId"), "Answer/manifest task identity differs")
    if stage == "s_1":
        ids = {x["id"] for x in manifest["candidates"]}
        selected = answer["answer"]
        require(bool(selected) and len(selected) == len(set(selected)) and set(selected) <= ids,
                "Invalid S1 selected IDs")
        require(len(selected) == answer["n_answer"], "S1 answer count differs")
        return {"selected_clip_ids": selected, "selection_count": len(selected)}
    if stage == "s_2":
        ids = {x["id"] for x in manifest["clips"]}
        orders = [answer["order"], *answer.get("alternative_orders", [])]
        for order in orders:
            require(len(order) == len(ids) and set(order) == ids, "Invalid S2 order")
        return {"reference_order": orders[0], "equally_accepted_alternative_orders": orders[1:]}
    if stage == "s_3":
        clips = {x["id"]: x for x in manifest["clips"]}
        cuts = answer["cuts"]
        require(len(cuts) == len(clips) and {x["clip"] for x in cuts} == set(clips), "Invalid S3 clip coverage")
        for cut in cuts:
            lo, hi = finite(cut["in_s"]), finite(cut["out_s"])
            # The released metadata rounds media durations; allow one source frame.
            tolerance = 1 / finite(answer["fps"]) + 0.002
            require(0 <= lo < hi <= clips[cut["clip"]]["clip_dur_s"] + tolerance,
                    f"S3 reference escaped supplied clip: {cut['clip']}")
        return {"reference_is_not_unique": True, "time_base": "seconds relative to each supplied clip",
                **fields(answer, ("fps", "target_duration_s", "dur_tol_s")),
                "reference_cuts": [fields(x, ("clip", "in_s", "out_s", "in_tol_s")) for x in cuts]}
    shots = manifest["shots"]
    units = {x["unit"]: x["id"] for x in shots}
    require(len(units) == len(shots), "Ambiguous S4E private-to-public shot mapping")
    public_pairs = {(a["id"], b["id"]) for a, b in zip(shots, shots[1:])}
    public_joins = {x["join"]: (x["leaving"], x["arriving"]) for x in manifest["joins"]}
    joins = []
    for join in answer["golden_joins"]:
        require(join["cut"] in public_joins, "Unknown S4E join number")
        pair = public_joins[join["cut"]]
        if "after" in join or "before" in join:
            require(join.get("after") in units and join.get("before") in units, "Unknown S4E shot unit")
            require(pair == (units[join["after"]], units[join["before"]]), "S4E join mapping disagrees")
        require(pair in public_pairs, "Nonadjacent S4E reference join")
        joins.append({"after_clip": pair[0], "before_clip": pair[1],
                      **fields(join, ("cut", "picture", "audio", "audio_offset_s", "carried_by_sound",
                                      "fade_out_s", "black_hold_s", "fade_in_s"))})
    require(len(joins) == answer["n_joins"] == len(public_pairs), "S4E join coverage differs")
    takes = {x["id"] for x in manifest["dialogue"]}
    dialogue = []
    for line in answer["golden_dialogue"]:
        require(line["id"] in takes, "S4E reference dialogue unavailable")
        dialogue.append({"take_id": line["id"], "text": line["text"],
                         "preview_onset_s": finite(line["trailer_s"])})
    return {"reference_is_not_unique": True,
            "reference_kind": "partial editing plan, NOT a ready-to-submit edit.json",
            "target_duration_s": finite(manifest["stretch_dur_s"]),
            "reference_joins": joins, "reference_dialogue": dialogue,
            "distractor_take_ids": answer.get("distractors", []),
            "music_source_start_s": finite(answer["music_align_s"]),
            "time_semantics": "Dialogue onsets use preview t=0, NOT the original trailer clock. "
                "A negative dialogue onset denotes pre-roll: crop the preceding portion at preview t=0. "
                "Music start is an offset within supplied music.wav. J leads the picture cut; L trails it.",
            "unknown_not_ground_truth": ["dialogue take trim endpoints", "gains and mixing envelopes",
                "unprovided transition durations", "render commands and implementation"],
            "guidance": "Use the tools and actual supplied media to decide missing details. "
                "Do not treat coarse ASR word timestamps as verified trims. Other coherent edits can be valid."}


def build_catalog(release_path, corpus_root=None):
    raw = Path(release_path).read_bytes()
    release = json.loads(raw)
    result = {}
    for row in release["tasks"]:
        if row["probe"]["split"] != "train":
            continue
        task = row["task"]
        require(task not in result and "/" not in task and task not in (".", ".."), "Duplicate/unsafe task")
        root = Path(corpus_root) / "_tt_train" / task if corpus_root else Path(row["task_path"])
        contents, hashes = {}, {}
        for name in ("verifier/answer.json", "initial_state/manifest.json"):
            data = (root / name).read_bytes()
            require(sha(data) == row["files"][name]["sha256"], f"Released {name} changed: {task}")
            contents[name], hashes[name] = json.loads(data), sha(data)
        card = answer_card(row["stage"], contents["verifier/answer.json"], contents["initial_state/manifest.json"])
        result[task] = {"stage": row["stage"], "source_hashes": hashes, "answer": card,
                        "answer_sha256": sha(canonical(card))}
    require(len(result) == release["train"], "Training answer coverage differs from release")
    return {"format": FORMAT, "split": "train", "release_sha256": sha(raw), "tasks": result,
            "counts": dict(Counter(x["stage"] for x in result.values()))}


class PrivateAnswers:
    def __init__(self, path, expected_sha256, release_path, prepared_rows=None):
        raw = Path(path).read_bytes()
        require(sha(raw) == expected_sha256, "Private catalog digest differs from launch plan")
        data = json.loads(raw)
        require(data["format"] == FORMAT and data["split"] == "train", "Not a training-only answer catalog")
        release_raw = Path(release_path).read_bytes()
        require(data["release_sha256"] == sha(release_raw), "Private answers use another release")
        selected = {x["task"]: x for x in json.loads(release_raw)["tasks"] if x["probe"]["split"] == "train"}
        require(set(selected) == set(data["tasks"]), "Missing/extra training answers (or validation leakage)")
        for task, entry in data["tasks"].items():
            require(entry["stage"] == selected[task]["stage"], "Answer stage differs")
            require(entry["answer_sha256"] == sha(canonical(entry["answer"])), "Answer card digest differs")
            require(set(entry["source_hashes"]) == {"verifier/answer.json", "initial_state/manifest.json"},
                    "Incomplete private answer source binding")
            for name, digest in entry["source_hashes"].items():
                require(digest == selected[task]["files"][name]["sha256"], "Answer not bound to released source")
                if prepared_rows is not None:
                    require(prepared_rows[task]["source_hashes"][name] == digest,
                            "Teacher answer differs from actual prepared student task")
        self.tasks, self.sha256, self.release_sha256 = data["tasks"], expected_sha256, data["release_sha256"]

    def get(self, task):
        require(task in self.tasks, "No privileged answer for this training task; validation is forbidden")
        return self.tasks[task]

    def hint(self, task):
        # JSON-escape '<' so untrusted answer text cannot inject chat/control tokens.
        answer = canonical(self.get(task)["answer"]).decode().replace("<", "\\u003c")
        return "\n\n[PRIVATE TEACHER REFERENCE — not visible to the student]\n" \
            "Use this reference to favor useful next actions from the current student history, " \
            "including verification and recovery from mistakes. Do not claim actions or observations " \
            "that did not occur. Continue obeying the public task and tool protocol; do not mention " \
            "or quote this private note. A reference marked non-unique is guidance, not an exact-match rule.\n" \
            + answer + "\n[END PRIVATE TEACHER REFERENCE]\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-root")
    args = parser.parse_args()
    value = build_catalog(args.release, args.corpus_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # This is a generated data artifact, not a source edit. Never overwrite it.
    with output.open("xb") as stream:
        stream.write(canonical(value))
    output.chmod(0o600)
    print(json.dumps({"path": str(output.resolve()), "sha256": sha(canonical(value)), "counts": value["counts"]}))


if __name__ == "__main__":
    main()
