"""Public construction draft: build_s23_tasks.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/build_s23_tasks.py [options]\n\n  --film\n  --all\n  -j, --workers\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import argparse, csv, glob, hashlib, json, math, os, random, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_s1_tasks import (cut, source_shots, shot_index, bound, cover,
                            mark_cards, load_matches, build_id, OUT, ROOT)

FPS_DEFAULT = 24000 / 1001
IN_TOL_MIN = 6             # Existing minimum, in source-video frames.
DUR_TOL = 4                # 出点/时长容差（帧）
MIN_SLACK = 0.6            # 互裁后每个 clip 至少要留的余量（秒）；见 separate()


def chunk_members(ch, inputs):
    ids = [s["id"] for s in inputs["shots"]]
    a, z = ids.index(ch["shots"][0]), ids.index(ch["shots"][1])
    return ids[a:z + 1]


def collect(ch, inputs, p2, shots, pb, srcpath, gold, scrop, gcrop):
    """the chunk's clips, in trailer order, with everything grading needs"""
    by_id = {s["id"]: s for s in inputs["shots"]}
    seq, seen, cards = [], {}, []
    for sid in chunk_members(ch, inputs):
        sh_ = by_id[sid]
        if sh_["kind"] == "card":
            cards.append(sh_["dur_s"]); continue
        r = p2.get(sid, {})
        if sh_["kind"] == "source" and r.get("refined_src_s"):
            i = shot_index(shots, float(r["refined_src_s"]))
            if i in seen:
                seen[i]["extra_uses"].append(sid); continue
            a, b = shots[i]
            fd = pb["src"].get("vdur") or pb["src"]["dur"]
            t0, dur = cover(a, b, float(r["refined_src_s"]), sh_["dur_s"], fd)
            if t0 is None:
                # cannot give this shot a clip that contains its own answer
                seq.append({"shot": sid, "src": gold, "clip_t0": sh_["start_s"],
                            "clip_dur": sh_["dur_s"], "crop": gcrop,
                            "origin": "trailer_only", "src_shot": None, "in_s": 0.0,
                            "use_dur": sh_["dur_s"], "plateau": 0, "extra_uses": []})
                continue
            e = {"shot": sid, "src": srcpath, "clip_t0": t0, "clip_dur": dur,
                 "crop": scrop, "origin": "film", "src_shot": i,
                 "in_s": float(r["refined_src_s"]), "use_dur": sh_["dur_s"],
                 "plateau": int(r.get("plateau_frames") or 0), "extra_uses": [],
                 # Pipeline refinement samples at reference FPS, even when the
                 # source video has a different native FPS. Old CSVs lack this
                 # field; retain an explicit legacy metric label in that case.
                 "alignment_fps": float(r.get("alignment_fps") or pb["gold"]["fps"]),
                 "alignment_score_kind": r.get("alignment_score_kind") or "legacy-all-pairs-window-mean",
                 "plateau_kind": r.get("plateau_kind") or "legacy-near-peak-count",
                 "peak_band": float(r.get("peak_band") or 0.02),
                 "near_peak_count": int(r.get("near_peak_count") or r.get("plateau_frames") or 0),
                 "disjoint_near_peaks": int(r.get("disjoint_near_peaks") or 0),
                 "alignment_review_reason": r.get("alignment_review_reason") or ""}
            seen[i] = e; seq.append(e)
        else:
            seq.append({"shot": sid, "src": gold, "clip_t0": sh_["start_s"],
                        "clip_dur": sh_["dur_s"], "crop": gcrop, "origin": "trailer_only",
                        "src_shot": None, "in_s": 0.0, "use_dur": sh_["dur_s"],
                        "plateau": 0, "extra_uses": []})
    return separate(seq), cards


def separate(seq):
    """Separate overlapping candidate windows without removing either reference-used interval.
    Trim slack jointly across the chunk, leaving a gap where possible.
    Windows whose required intervals overlap cannot be separated this way."""
    fam = sorted([e for e in seq if e["origin"] == "film"], key=lambda e: e["clip_t0"])
    orig = {id(e): (e["clip_t0"], e["clip_dur"]) for e in fam}
    for a, b in zip(fam, fam[1:]):
        if a["clip_t0"] + a["clip_dur"] <= b["clip_t0"] + 1e-6:
            continue
        a_keep = a["in_s"] + a["use_dur"]      # a 必须留到这里
        b_reach = b["in_s"]                    # b 必须在这之前开始
        if a_keep > b_reach - 1e-6:
            continue                           # 用到的画面本身重叠，裁不开
        # A gap, not a shared boundary. Cutting both at one point leaves the clips
        # exactly abutting, and the last frame of one is still the frame before the
        # first of the other — continuity a solver can read even with no shared
        # footage. The gap comes out of slack neither clip needs; it is capped by how
        # much slack there is, so the parts the trailer used are never touched.
        m = (a_keep + b_reach) / 2.0
        g = min(0.5, (b_reach - a_keep) * 0.5)
        b_end = b["clip_t0"] + b["clip_dur"]
        a["clip_dur"] = (m - g / 2) - a["clip_t0"]
        b["clip_t0"] = m + g / 2
        b["clip_dur"] = b_end - (m + g / 2)

    # A clip in the middle of a chain may be trimmed from both sides.
    # Restore its original cover() window if the remaining slack is insufficient.


    for e in fam:
        lead = e["in_s"] - e["clip_t0"]
        tail = (e["clip_t0"] + e["clip_dur"]) - (e["in_s"] + e["use_dur"])
        if max(lead, tail) < MIN_SLACK:
            e["clip_t0"], e["clip_dur"] = orig[id(e)]
    return seq


def write_clips(seq, cdir, prefix, seed):
    """cut every clip and hand out ids from a shuffled order"""
    rng = random.Random(int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16))
    order = list(range(len(seq)))
    rng.shuffle(order)
    ids = [None] * len(seq)
    made = []
    for n, k in enumerate(order, 1):
        cid = f"{prefix}{n:02d}"
        dst = f"{cdir}/{cid}.mp4"
        c = seq[k]
        if not cut(c["src"], c["clip_t0"], c["clip_dur"], c["crop"], dst):
            # Include the failed clip identity to distinguish unreadable media from an invalid source span.


            return None, (f"{os.path.basename(c['src'])} @ {c['clip_t0']:.3f}s "
                          f"+{c['clip_dur']:.3f}s")
        ids[k] = cid
        made.append({"id": cid, "file": f"clips/{cid}.mp4",
                     "dur_s": round(c["clip_dur"], 3)})
    made.sort(key=lambda m: m["id"])
    return ids, made


def build_s2(slug, name, ch, seq, pb, tdir):
    cdir = f"{tdir}/initial_state/clips"
    for d in (cdir, f"{tdir}/output", f"{tdir}/verifier"):
        os.makedirs(d, exist_ok=True)
    ids, made = write_clips(seq, cdir, "k", f"s2/{slug}/{ch['id']}")
    if not ids:
        return None, f"{ch['id']}: 切片失败 ({made})"
    tid = f"{slug}_s2_{ch['id']}"
    json.dump({"id": tid, "seed": "clip_ordering", "stage": "s2/ordering",
               "film": slug, "chunk": ch["id"], "beat": ch.get("beat"),
               "deliverables": ["output/order.json"]},
              open(f"{tdir}/task.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"taskId": tid, "n_clips": len(made),
               "note": "Ids were assigned after a shuffle and say nothing about order; "
                       "clips are silent and identically transcoded.",
               "clips": made},
              open(f"{tdir}/initial_state/manifest.json", "w"), ensure_ascii=False, indent=1)
    open(f"{tdir}/initial_state/SETUP.md", "w").write(
        f"# Material\n\n`clips/` holds {len(made)} clips. All of them belong to this stretch —\n"
        "none missing, none extra. They are in random order; the ids carry no sequence.\n")
    open(f"{tdir}/task_prompt.md", "w").write(f"""# Task — put this stretch in order

You are the assistant editor on the *{name}* trailer. The **{len(made)} shots** this
stretch uses have already been chosen and sit in `initial_state/clips/` — but they are
**shuffled**.

This stretch covers:

> {ch.get('what_en') or ch['what']}

It sits inside the beat **{ch.get('beat_name_en') or ch.get('beat_name') or ch.get('beat')}**.

## What to do

Put the {len(made)} shots into the order the trailer used.

Use all of them, each exactly once. The ids were assigned after a shuffle and **carry
no ordering information**.

Judge from the picture: how actions run on from one shot to the next, where eyelines
and movement point, how the shot sizes step, where the location changes, and the order
things happen in the description above.

## Deliverable

Write `output/order.json`:

```json
{{"order": ["k03", "k01", "k05", "k02", "k04"]}}
```

Every id, first to last.
""")
    open(f"{tdir}/output/.gitkeep", "w").write("")
    json.dump({"task": tid, "order": ids,
               "clips": [{"id": i, "trailer_shot": c["shot"], "origin": c["origin"]}
                         for i, c in zip(ids, seq)]},
              open(f"{tdir}/verifier/answer.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"id": tid, "deliverable_kind": "json_sequence",
               "verifier_class": "mechanical",
               "golden_semantics": "answer_key — the trailer's actual order; golden self-score 1.0",
               "reward": "(1 − ND) × LIS × ADJ (arXiv 2605.27705 Eq. 3); pass at 0.5",
               "source_evidence": f"{slug} chunk {ch['id']}: member shots, in golden shot-list order",
               "leak_controls": "ids shuffled under an independent seed; prefix k differs from S1's c, so nothing maps across stages"},
              open(f"{tdir}/verifier/task_meta.json", "w"), ensure_ascii=False, indent=1)
    n = len(ids)
    # Rubric assembly is separate from task construction; do not embed an obsolete scoring rule here.
    
    
    shutil.copy(f"{CODE_ROOT}/grade_s2.py", f"{tdir}/verifier/grade.py")
    # LAST. This file is what --skip-existing trusts, so it must not exist until
    # everything it vouches for does — a build killed midway would otherwise leave a
    # half-written task stamped current and get skipped forever after.
    run_legacy_rubric_hook("s2")
    json.dump({"build": build_id()}, open(f"{tdir}/verifier/build.json", "w"))
    return {"chunk": ch["id"], "n": len(made)}, None


def inpoint_tolerance(clip, source_fps, reference_fps):
    """Return the existing heuristic tolerance with explicit measurement units.

    The six-source-frame minimum is unchanged. The measured peak width is in
    alignment samples, which need not have the source video's native FPS.
    Legacy pipeline CSVs used reference FPS but counted all near-peak offsets;
    keep that provenance rather than relabeling them as paired contiguous widths.
    """
    measured_fps = clip.get("alignment_fps")
    alignment_fps = float(reference_fps if measured_fps in (None, "") else measured_fps)
    if any(not math.isfinite(x) or x <= 0 for x in (source_fps, alignment_fps)):
        raise ValueError("in-point tolerance requires positive finite source and alignment FPS")
    width = int(clip.get("plateau") or 0)
    if width < 0:
        raise ValueError("alignment peak width must be nonnegative")
    seconds = max(IN_TOL_MIN / source_fps, width / (2.0 * alignment_fps))
    return {"in_tol_s": round(seconds, 4),
            "in_tol_frames": round(seconds * source_fps, 1),
            "plateau_frames": width, "alignment_fps": alignment_fps,
            "alignment_score_kind": clip.get("alignment_score_kind") or "legacy-all-pairs-window-mean",
            "plateau_kind": clip.get("plateau_kind") or "legacy-near-peak-count"}


def build_s3(slug, name, ch, seq, cards, pb, tdir):
    fps = pb["gold"]["fps"]        # 预告片：镜头长度量在这个帧率下
    src_fps = pb["src"]["fps"]     # Native source FPS sets only the minimum tolerance.
    cdir = f"{tdir}/initial_state/clips"
    for d in (cdir, f"{tdir}/output", f"{tdir}/verifier"):
        os.makedirs(d, exist_ok=True)
    # ids follow trailer order here — the order is given, this stage is about lengths
    made, ids = [], []
    for n, c in enumerate(seq, 1):
        cid = f"p{n:02d}"
        if not cut(c["src"], c["clip_t0"], c["clip_dur"], c["crop"], f"{cdir}/{cid}.mp4"):
            return None, (f"{ch['id']}: 切片失败 "
                          f"({os.path.basename(c['src'])} @ {c['clip_t0']:.3f}s "
                          f"+{c['clip_dur']:.3f}s)")
        ids.append(cid)
        made.append({"id": cid, "file": f"clips/{cid}.mp4", "position": n,
                     "clip_dur_s": round(c["clip_dur"], 3)})

    target = sum(c["use_dur"] for c in seq)          # 已天然不含字幕卡
    tid = f"{slug}_s3_{ch['id']}"
    json.dump({"id": tid, "seed": "rough_cut", "stage": "s3/rough_cut",
               "film": slug, "chunk": ch["id"], "beat": ch.get("beat"),
               "deliverables": ["output/cuts.json"]},
              open(f"{tdir}/task.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"taskId": tid, "n_clips": len(made),
               "target_duration_s": round(target, 3),
               # Repeated uses of one source shot share one clip, so the sum of available clip durations
               # need not equal the full reference stretch.


               "target_note": ("the total the " + str(len(seq)) + " clips below add up to "
                               "in the finished trailer"
                               + (f"; the {len(cards)} non-footage insert(s) in this "
                                  f"stretch ({sum(cards):.2f}s) — title cards, idents, "
                                  f"black — are not cut from the film and are not counted" if cards else "")
                               + ("; where the trailer returns to a shot it already used, "
                                  "that shot is one clip here and counted once"
                                  if any(c["extra_uses"] for c in seq) else "")),
               # Trailer-only clips may already have their final duration; do not promise trimming slack for every clip.


               "note": "Clips are numbered p01, p02… in the trailer's order. "
                       "You decide which part of each to use.",
               "clips": made},
              open(f"{tdir}/initial_state/manifest.json", "w"), ensure_ascii=False, indent=1)
    open(f"{tdir}/initial_state/SETUP.md", "w").write(
        f"# Material\n\n`clips/` holds {len(made)} shots, **already in the trailer's order** "
        "(`p01` is first).\nYou decide which part of each one to use.\n"
        "`target_duration_s` in `manifest.json` is how long this "
        "stretch runs in the finished trailer.\n")
    open(f"{tdir}/task_prompt.md", "w").write(f"""# Task — rough-cut this stretch

You are the assistant editor on the *{name}* trailer. The shots for this stretch are
chosen and ordered already, in `initial_state/clips/` — **{len(made)} of them**, `p01`
first.

Your job is to **set an in point and an out point on each** — where to start using
that shot and where to leave it. Most run longer than the trailer's use of them.

This stretch covers:

> {ch.get('what_en') or ch['what']}

**Target total: {target:.2f} seconds** across the {len(made)} shots.
{"This stretch also carries " + str(len(cards)) + " non-footage insert(s) totalling " + f"{sum(cards):.2f}" + "s — title cards, studio idents, black — which are not cut from the film and are NOT part of your total." if cards else ""}

## What to do

For each shot, decide which part of it the trailer would use:

- where the action reads strongest — where to come in, where to leave
- whether to cut away mid-gesture or let it finish
- the lengths should add up to close to the target

## Deliverable

Write `output/cuts.json`:

```json
{{"cuts": [
  {{"clip": "p01", "in_s": 12.34, "out_s": 13.90}},
  {{"clip": "p02", "in_s": 4.05,  "out_s": 5.20}}
]}}
```

`in_s` / `out_s` are seconds **within that clip** (the clip starts at 0).
Give all {len(made)}, in order.
""")
    open(f"{tdir}/output/.gitkeep", "w").write("")

    key = []
    for cid, c in zip(ids, seq):
        rel_in = max(0.0, c["in_s"] - c["clip_t0"]) if c["origin"] == "film" else 0.0
        tolerance = inpoint_tolerance(c, src_fps, fps)
        key.append({"clip": cid, "trailer_shot": c["shot"], "origin": c["origin"],
                    "in_s": round(rel_in, 4), "out_s": round(rel_in + c["use_dur"], 4),
                    "dur_s": round(c["use_dur"], 4),
                    **tolerance,   # in_tol_frames is expressed in native source frames.
                    "peak_band": c.get("peak_band"),
                    "near_peak_count": c.get("near_peak_count"),
                    "disjoint_near_peaks": c.get("disjoint_near_peaks"),
                    "alignment_review_reason": c.get("alignment_review_reason", ""),
                    "scored": c["origin"] == "film"})
    json.dump({"task": tid, "fps": fps, "src_fps": round(src_fps, 3),
               "target_duration_s": round(target, 3),
               "dur_tol_s": round(DUR_TOL / fps, 4), "dur_tol_frames": DUR_TOL,
               "cards_excluded_s": round(sum(cards), 3),
               "cuts": key},
              open(f"{tdir}/verifier/answer.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"id": tid, "deliverable_kind": "json_cutlist",
               "verifier_class": "mechanical_component_with_rubric_judges",
               "golden_semantics": "answer_key — the trailer's actual in-point and length; golden 1.0",
               "reward": "The deterministic component is the fraction of scored shots whose in-point and duration both fall within tolerance. Run evaluation/task_pipeline.py --prepare-only to attach the released RC scoring contract: 0.4 deterministic, 0.4 reference-aware judge, 0.2 blind judge.",
               "source_evidence": f"{slug} chunk {ch['id']}: in-point is refined_src_s minus the film shot's start; length is dur_s from the golden shot list",
               "tolerance_note": "Heuristic in-point tolerance: half the contiguous paired-score peak-band width divided by alignment_fps, with a floor of 6/source_fps seconds. in_tol_frames is the equivalent native-source frame count. This is not a calibrated confidence interval; disjoint near peaks remain flagged for review. Legacy rows are labelled separately and use the pipeline's reference FPS when alignment_fps is absent.",
               "golden_reference": "verifier/golden/ — the answer as footage, not numbers. One file per clip: the film "
                   "cut at the in-point and length this key specifies, so it is what a correct "
                   "deliverable produces for that clip. Shots with no film counterpart are "
                   "already at final length and are copied through unchanged. _cut.mp4 is all "
                   "of them joined in trailer order — the whole stretch as the finished trailer "
                   "plays it, which is the thing an agent's rough cut is competing with. "
                   "Written by golden_s3.py after the task is built; never shown to the agent.",
               "leak_controls": "target length has cards, idents and black frames removed; trailer_only shots are carried but not scored — there is no in-point to choose"},
              open(f"{tdir}/verifier/task_meta.json", "w"), ensure_ascii=False, indent=1)
    ns = sum(1 for k in key if k["scored"])
    # The current S3 rubric pipeline is separate from this task builder.
    # Do not reinstate the legacy exact-match grader during construction.


    run_legacy_rubric_hook("s3")
    json.dump({"build": build_id()}, open(f"{tdir}/verifier/build.json", "w"))
    return {"chunk": ch["id"], "n": len(made), "target": round(target, 2),
            "scored": ns}, None


def build_film(slug, name):
    T, E = f"{ROOT}/edit/task_{slug}", f"{ROOT}/exports/{slug}"
    script = json.load(open(f"{T}/stages/script.json"))
    inputs = json.load(open(f"{T}/stages/script_inputs.json"))
    mark_cards(script, inputs, slug)
    p2 = load_matches(slug)
    pb = json.load(open(f"{E}/probe.json"))
    srcpath = f"{pb['dir']}/{pb['source']}"
    gold = f"{pb['dir']}/{pb['golden']}"
    gcrop = pb.get("crop_expr") or (f"crop=iw:iw/{pb['crop']}:0:(ih-iw/{pb['crop']})/2"
                                    if pb["crop"] else "")
    scrop = pb.get("src_crop_expr", "")
    shots = source_shots(slug)
    beat_name = {b["id"]: b["name"] for b in script["beats"]}
    beat_en = {b["id"]: b.get("name_en") for b in script["beats"]}

    n2 = n3 = 0; errs = []
    for chunk_number, ch in enumerate(script["chunks"], 1):
        print(f"  [NS/RC] {slug} processing chunk {chunk_number}/{len(script['chunks'])}: {ch['id']}",
              flush=True)
        ch = {**ch, "beat_name": beat_name.get(ch.get("beat")),
              "beat_name_en": beat_en.get(ch.get("beat"))}
        seq, cards = collect(ch, inputs, p2, shots, pb, srcpath, gold, scrop, gcrop)
        anchored = sum(1 for c in seq if c["origin"] == "film")
        if len(seq) < 3 or anchored < 2:
            errs.append(f"{ch['id']}: {len(seq)} 镜 / {anchored} 个锚在正片上，跳过")
            continue
        for stage, fn in (("s_2", build_s2), ("s_3", build_s3)):
            d = f"{OUT}/{slug}/{stage}/{ch['id']}"
            args = (slug, name, ch, seq, pb, d) if stage == "s_2" else \
                   (slug, name, ch, seq, cards, pb, d)
            r, err = fn(*args)
            if err:
                errs.append(err); shutil.rmtree(d, ignore_errors=True)
            elif stage == "s_2":
                n2 += 1
            else:
                n3 += 1
    print(f"  [{name}] S2 {n2} 个 / S3 {n3} 个"
          + (f"  跳过 {len(errs)}: {errs[0][:52]}" if errs else ""), flush=True)
    return n2 + n3


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--all", action="store_true")
    P.add_argument("-j", "--workers", type=int, default=1)
    A = P.parse_args()
    require_target(P, A)
    films = {}
    for p in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p))
        st = json.load(open(f"{ROOT}/exports/{pb['slug']}/state.json"))
        if st.get("script", {}).get("ok"):
            films[pb["slug"]] = os.path.basename(pb["dir"])
    if A.film:
        films = {k: v for k, v in films.items() if k == A.film}
        if not films:
            sys.exit(f"{A.film} 的场记表还没做好")
    if not films:
        sys.exit("还没有片子完成场记表")
    print(f"建 S2/S3 任务: {len(films)} 部\n", flush=True)
    if A.workers > 1:
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            n = sum(ex.map(lambda kv: build_film(*kv), films.items()))
    else:
        n = sum(build_film(s, v) for s, v in films.items())
    print(f"\n共 {n} 个任务 -> {OUT}/")


if __name__ == "__main__":
    main()
