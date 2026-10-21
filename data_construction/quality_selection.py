"""Film-disjoint, deterministic size sampling after semantic quality review.

The largest-remainder and SHA256 ranking rules are recovered from the historical
selection helpers. This portable CLI never makes a task-quality decision. It
ships no task rows, media, film list, or historical private release manifest.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path

PAPER_SEED = "quality797-20260906-v3"
PAPER_PARENT_SHA256 = "5381ad35843c3a90441710a6cba11964220394b0550162703feec15e3b805d5d"
PAPER_TARGETS = {"_tt_train": {"s_1": 122, "s_2": 132, "s_3": 231, "s_4e": 231},
                 "_tt_val": {"s_1": 14, "s_2": 15, "s_3": 26, "s_4e": 26}}
HISTORICAL_HELPER_SHA256 = "2fe9c6f51cf460505dc0c230baf84542465d6ea3e871aeed472de5ba174061d6"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def apportion(counts, total, cover_all=False):
    """Recovered largest remainder allocation, with deterministic film-name ties."""
    if (not isinstance(counts, dict) or any(not isinstance(k, str) or not k or type(v) is not int or v < 1
                                           for k, v in counts.items())
            or type(total) is not int or not 0 <= total <= sum(counts.values())):
        raise ValueError("Invalid counts or target size")
    base = {k: int(cover_all and total >= len(counts) and n > 0) for k, n in counts.items()}
    remain = total - sum(base.values())
    capacity = {k: counts[k] - base[k] for k in counts}
    denominator = sum(capacity.values())
    if not denominator:
        return base
    result = {k: base[k] + remain * n // denominator for k, n in capacity.items()}
    ordered = sorted(counts, key=lambda k: (-(remain * capacity[k] % denominator), k))
    for k in ordered[:total - sum(result.values())]:
        result[k] += 1
    return result


def film_of(row):
    film = row.get("film")
    if film is None:
        # Historical indexes encode film/stage/chunk as the last three folders.
        parts = Path(row.get("task_path", "")).parts
        if len(parts) < 3 or parts[-2] != row.get("stage"):
            raise ValueError("Supply film explicitly or a film/stage/chunk task_path")
        film = parts[-3]
    if not isinstance(film, str) or not film:
        raise ValueError("Invalid film identity")
    return film


def select_group(rows, total, seed):
    by_film = defaultdict(list)
    for row in rows:
        by_film[film_of(row)].append(row)
    quota = apportion({k: len(v) for k, v in by_film.items()}, total, cover_all=True)
    chosen = set()
    for film, members in sorted(by_film.items()):
        ranked = sorted(members, key=lambda r: hashlib.sha256(
            (seed + "\0" + r["original_split"] + "\0" + r["stage"] + "\0" + film + "\0" + r["task"]).encode()).hexdigest())
        chosen.update(r["task"] for r in ranked[:quota[film]])
    return chosen, {film: {"before": len(members), "after": quota[film]} for film, members in sorted(by_film.items())}


def select(rows, targets, seed):
    if not isinstance(rows, list) or not rows or not isinstance(seed, str) or not seed:
        raise ValueError("Nonempty candidate list and explicit seed required")
    seen, film_splits, groups = set(), {}, defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                                             for k in ("task", "original_split", "stage", "task_path")):
            raise ValueError("Each candidate needs task, original_split, stage and task_path")
        if row["task"] in seen:
            raise ValueError("Duplicate task identity")
        seen.add(row["task"])
        film = film_of(row)
        if film_splits.setdefault(film, row["original_split"]) != row["original_split"]:
            raise ValueError("A film crosses splits")
        groups[(row["original_split"], row["stage"])].append(row)
    if not isinstance(targets, dict) or any(not isinstance(v, dict) for v in targets.values()):
        raise ValueError("Targets must map split to stage counts")
    requested = {(split, stage): n for split, stages in targets.items() for stage, n in stages.items()}
    if set(requested) != set(groups):
        raise ValueError("Targets must explicitly cover every split/stage group")
    selected_ids, allocations = set(), {}
    for (split, stage), group in sorted(groups.items()):
        chosen, allocation = select_group(group, requested[(split, stage)], seed)
        selected_ids.update(chosen)
        allocations.setdefault(split, {})[stage] = allocation
    selected = [r for r in rows if r["task"] in selected_ids]
    return {"schema": "video-editing-quality-sampling-v1", "seed": seed, "tasks": selected,
            "counts": targets, "count": len(selected), "allocations": allocations,
            "sampled_out": [r for r in rows if r["task"] not in selected_ids],
            "sampled_out_tasks_are_not_quality_rejections": True,
            "original_split_unchanged": True, "policy_reward_scores_used": False,
            "historical_selection_helper_sha256": HISTORICAL_HELPER_SHA256}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path, help="JSON index containing reviewed tasks")
    parser.add_argument("--output", required=True, type=Path, help="New directory outside the source repository")
    parser.add_argument("--targets", type=Path, help="JSON split -> stage -> target count")
    parser.add_argument("--seed", help="Explicit deterministic ranking seed")
    parser.add_argument("--paper-797", action="store_true", help="Require the exact hash-bound private 925-task parent")
    args = parser.parse_args(argv)
    raw = args.candidates.read_bytes()
    parent_sha = hashlib.sha256(raw).hexdigest()
    if args.paper_797:
        if args.targets or args.seed or parent_sha != PAPER_PARENT_SHA256:
            parser.error("--paper-797 requires the exact historical parent and no target/seed override")
        targets, seed = PAPER_TARGETS, PAPER_SEED
    else:
        if not args.targets or not args.seed:
            parser.error("Use --targets and --seed, or --paper-797")
        targets, seed = json.loads(args.targets.read_text()), args.seed
    index = json.loads(raw)
    result = select(index["tasks"], targets, seed)
    result.update(parent_sha256=parent_sha, portable_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  historical_parent_verified=args.paper_797)
    output = args.output.expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Private selection outputs must be outside the source repository")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "selection.json").write_bytes(canonical(result))
    (output / "accepted_tasks.json").write_bytes(canonical([r["task_path"] for r in result["tasks"]]))
    print(json.dumps({"selected": result["count"], "sampled_out": len(result["sampled_out"]), "output": str(output)}))


if __name__ == "__main__":
    main()
