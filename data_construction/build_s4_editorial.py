"""Public construction draft: build_s4_editorial.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print("Usage: python data_construction/build_s4_editorial.py [options]\n\n  --film\n  --all\n  --music-pad  Seconds of music padding on each side; a wider window preserves the alignment decision\n  --distractors  Dialogue distractor count = used dialogue count multiplied by this factor\n  --force  Rebuild existing tasks while preserving submissions in output/\n  --keep-transitions  Keep black/fade transition units; removed by default to avoid embedding answers in input media.\n  --explore  Exploratory variant: omit take transcripts so the agent must listen; outputs go to s_4x/\n  --stage  Output stage directory; defaults according to --explore\n  -h, --help  Show this static help without loading optional dependencies.\n")
        raise SystemExit(0)


def _complete_task(directory, task_id, explore=False):
    import json
    from pathlib import Path
    try:
        from .build_s4_tasks import _task_media_ready
    except ImportError:
        from build_s4_tasks import _task_media_ready
    root = Path(directory)
    try:
        task = json.loads((root / "task.json").read_text())
        manifest = json.loads((root / "initial_state/manifest.json").read_text())
        answer = json.loads((root / "verifier/answer.json").read_text())
        seed = json.loads((root / "verifier/_rubric_seed.json").read_text())
        metadata = json.loads((root / "verifier/task_meta.json").read_text())
        if explore:
            json.loads((root / "verifier/takes_content.json").read_text())
        return (task["id"] == manifest["taskId"] == answer["task"] == seed["task"] == task_id
                and task["stage"] == metadata["stage"]
                and task["deliverables"] == ["output/edit.json", "output/preview.mp4"]
                and manifest["n_joins"] == len(manifest["joins"]) == answer["n_joins"]
                and bool((root / "task_prompt.md").read_text().strip())
                and (root / "output").is_dir() and _task_media_ready(root, manifest))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
        from .build_s4_tasks import (_media_duration, _require_media, _write_media,
                                     _invalidate_task, _task_media_ready)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
        from build_s4_tasks import (_media_duration, _require_media, _write_media,
                                    _invalidate_task, _task_media_ready)

    import argparse, glob, json, os, random, re, shutil, subprocess, sys, zlib

    ROOT = DATA_ROOT
    TT = f"{ROOT}/training_task"


    P = argparse.ArgumentParser()
    P.add_argument("--film")
    P.add_argument("--all", action="store_true")
    P.add_argument("--music-pad", type=float, default=20.0,
                   help="Seconds of music padding on each side; a wider window preserves the alignment decision")
    P.add_argument("--distractors", type=float, default=1.5,
                   help="Dialogue distractor count = used dialogue count multiplied by this factor")
    P.add_argument("--force", action="store_true",
                   help="Rebuild existing tasks while preserving submissions in output/")
    P.add_argument("--keep-transitions", action="store_true",
                   help="Keep black/fade transition units. By default they are removed to avoid embedding transition answers in input media.")
    P.add_argument("--explore", action="store_true",
                   help="Exploratory variant: omit take transcripts so the agent must listen; outputs go to s_4x/")
    P.add_argument("--stage", default=None, help="Output stage directory; defaults according to --explore")
    A = P.parse_args()
    require_target(P, A)


    def sh(c):
        return subprocess.run(c, capture_output=True, text=True)


    def remap_joins(all_shots, kept, gold, dropped):
        """Map reference join annotations to the sequence after transition units are removed.

        A|T|B, with transition unit T, originally has two joins; removing T leaves one join A|B.
        The removed transition duration becomes the black hold the agent must choose.
        """
        kept_ids = [s["id"] for s in kept]
        idx = {s["id"]: i for i, s in enumerate(all_shots)}
        out = []
        for i in range(len(kept_ids) - 1):
            a, b = kept_ids[i], kept_ids[i + 1]
            # Units between a and b in the original sequence
            between = [s["id"] for s in all_shots[idx[a] + 1: idx[b]]]
            hold = sum(dropped.get(x, 0.0) for x in between)
            if hold > 0.01:
                out.append({"cut": i + 1, "picture": "dip_to_black",
                            "black_hold_s": round(hold, 3),
                            "audio": (gold[idx[a]]["audio"] if idx[a] < len(gold) else "hard"),
                            "note": f"Removed transition media at this join: {'+'.join(between)}"})
            else:
                g = gold[idx[a]] if idx[a] < len(gold) else {"picture": "cut", "audio": "hard"}
                out.append({**{k: v for k, v in g.items() if k != "cut"}, "cut": i + 1})
        return out


    def is_transition(path):
        """Detect an embedded black/fade transition from low frame luminance.
        This heuristic does not remove every type of visual transition."""
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=32:18",
                            "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True)
        if r.returncode:
            raise RuntimeError(f"Cannot inspect transition media: {path}")
        import numpy as np
        a = np.frombuffer(r.stdout, dtype=np.uint8)
        if a.size < 18 * 32:
            raise RuntimeError(f"No decodable frames for transition inspection: {path}")
        fr = a[: (a.size // (18 * 32)) * 18 * 32].reshape(-1, 18 * 32).mean(axis=1)
        return bool(fr.min() < 8)


    def build(slug, chunk):
        src4 = f"{TT}/{slug}/s_4/{chunk}"
        dst = f"{TT}/{slug}/{A.stage or ('s_4x' if A.explore else 's_4e')}/{chunk}"
        tid = f"{slug}_s4e_{chunk}"
        if not os.path.isdir(src4):
            return None
        lo_p = f"{ROOT}/exports/{slug}/takes.json"
        if not os.path.isfile(lo_p):
            _invalidate_task(dst)
            raise RuntimeError("Missing takes.json: run source transcription, locate_lines.py, "
                               "then cut_takes.py before FC editorial assembly")
        try:
            with open(lo_p) as stream:
                takes = json.load(stream)
            tk = takes["takes"]
            if (not isinstance(tk, list) or type(takes.get("n_takes")) is not int
                    or takes["n_takes"] != len(tk) or takes.get("film", slug) != slug):
                raise ValueError("takes.json has inconsistent film/count metadata")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _invalidate_task(dst)
            raise RuntimeError(f"Invalid take preparation metadata: {lo_p}") from exc
        if not tk:
            print(f"  Warning: {slug} takes.json contains zero takes; this task has no dialogue pool.", flush=True)
        if not A.force and _complete_task(dst, tid, A.explore):
            return "skip"
        _invalidate_task(dst)
        # Preserve existing submitted output while rebuilding the surrounding task files.
        
        keep = {}
        for f in glob.glob(f"{dst}/output/*"):
            if os.path.basename(f) != ".gitkeep":
                try:
                    keep[os.path.basename(f)] = open(f, "rb").read()
                except OSError:
                    pass

        man4 = json.load(open(f"{src4}/initial_state/manifest.json"))
        key4 = json.load(open(f"{src4}/verifier/answer.json"))
        units = json.load(open(f"{ROOT}/edit/task_{slug}/s3_units.json"))["units"]
        by = {u["id"]: u for u in units}
        pos, t = {}, 0.0
        for u in units:
            pos[u["id"]] = t
            t += u["dur"]
        order = [s["unit"] for s in man4["shots"]]
        t0 = pos[order[0]]
        t1 = pos[order[-1]] + by[order[-1]]["dur"]

        # Remove embedded black/fade transition units so transition construction remains an editing decision.
        # Keep their duration for the remapped reference black_hold field.


        dropped = {}
        if not A.keep_transitions:
            for s_ in man4["shots"]:
                if s_.get("origin") != "trailer":
                    continue
                p_ = f"{src4}/initial_state/shots/{s_['id']}.mp4"
                if os.path.exists(p_) and is_transition(p_):
                    dropped[s_["id"]] = s_["dur_s"]
        shots_all = [s_ for s_ in man4["shots"] if s_["id"] not in dropped]
        if len(shots_all) < 2:
            return None

        for d in ("initial_state/shots", "initial_state/handles", "initial_state/takes",
                  "verifier", "output"):
            os.makedirs(f"{dst}/{d}", exist_ok=True)
        open(f"{dst}/output/.gitkeep", "w").close()
        mus = f"{ROOT}/exports/{slug}/stems/music.wav"
        music_duration = _require_media(mus, "audio")

        # Reuse picture and production sound from S4, but mute trailer-only media.
        # Its finished mix could otherwise reveal music alignment by direct correlation.


        for s in shots_all:
            p = f"{src4}/initial_state/shots/{s['id']}.mp4"
            if s.get("origin") != "to_build":
                q = f"{dst}/initial_state/shots/{s['id']}.mp4"
                if _media_duration(q, "video") is None:
                    _require_media(p, "video")
                    if os.path.lexists(q):
                        os.remove(q)
                    if s.get("origin") == "trailer":
                        _write_media(["ffmpeg", "-v", "error", "-i", p, "-c:v", "copy", "-an",
                                      "-y", q], q, "video")
                    else:
                        os.link(p, q) if os.stat(p).st_dev == os.stat(f"{dst}/initial_state/shots").st_dev \
                            else shutil.copy(p, q)
            h = (s.get("handle") or {}).get("file")
            if h:
                p = f"{src4}/initial_state/{h}"
                q = f"{dst}/initial_state/{h}"
                if _media_duration(q, "audio") is None:
                    _require_media(p, "audio")
                    if os.path.lexists(q):
                        os.remove(q)
                    os.link(p, q) if os.stat(p).st_dev == os.stat(f"{dst}/initial_state/handles").st_dev \
                        else shutil.copy(p, q)

        # Widen the music window so choosing its entry point remains an editing decision.
        
        # Randomize leading padding per task and omit it from the manifest; a fixed value reveals the alignment
        random.seed(zlib.crc32(f"music/{slug}/{chunk}".encode()))
        # Record the actual clamped head padding used by ffmpeg, not merely the requested padding.
        
        
        want_head = random.uniform(A.music_pad * 0.5, A.music_pad * 1.5)
        cut_from = max(0.0, t0 - want_head)
        head_pad = t0 - cut_from
        tail_pad = min(A.music_pad, max(0.0, music_duration - t1))
        mo = f"{dst}/initial_state/music.wav"
        if _media_duration(mo, "audio") is None:
            _write_media(["ffmpeg", "-v", "error", "-ss", f"{cut_from:.3f}",
                          "-t", f"{head_pad + (t1 - t0) + tail_pad:.3f}", "-i", mus, "-y", mo], mo, "audio")

        # ---- Dialogue pool: used takes plus distractors
        used = [x for x in tk if x["trailer_e"] > t0 and x["trailer_s"] < t1]
        rest = [x for x in tk if x not in used]
        random.seed(zlib.crc32(f"{slug}/{chunk}".encode()))
        # Keep at least three dialogue distractors even when no dialogue is used in the reference,
        # so choosing to add no dialogue remains a meaningful option.
        n_d = max(3, int(len(used) * A.distractors + 0.5))
        pool = used + random.sample(rest, min(n_d, len(rest)))
        # Reshuffle if all correct dialogue choices appear first; list position must not reveal them.
        gold_ids = {id(x) for x in used}
        for _ in range(20):
            random.shuffle(pool)
            gi = [i for i, x in enumerate(pool) if id(x) in gold_ids]
            if not gi or max(gi) >= len(used):
                break

        lines = []
        for i, x in enumerate(pool, 1):
            did = f"d{i:02d}"
            p = f"{ROOT}/exports/{slug}/{x['file']}"
            q = f"{dst}/initial_state/takes/{did}.m4a"
            if _media_duration(q, "audio") is None:
                _require_media(p, "audio")
                shutil.copy(p, q)
            lines.append({"id": did, "file": f"takes/{did}.m4a",
                          # Include every detected speech span in the file; the matched window alone does not reveal
                          # other dialogue in the handles and could cause valid choices to be rejected.
                          "speech_spans_s": x.get("speech_spans_s") or [],
                          "words_start_at_s": x["words_start_at_s"],
                          "words_dur_s": x["words_dur_s"],
                          "take_dur_s": x["take_dur_s"],
                          # Provide audible dialogue content to avoid unnecessary transcription work
                          "text": x["film_text"],
                          # Do not reveal whether the reference uses this take
                          "_used_in_golden": x in used,
                          "_trailer_s": x["trailer_s"]})

        # ---- Join table
        fps = json.load(open(f"{ROOT}/exports/{slug}/probe.json")).get("fps") or key4["fps"]
        joins, t = [], 0.0
        for i in range(len(shots_all) - 1):
            a, b = shots_all[i], shots_all[i + 1]
            t += a["dur_s"]
            joins.append({"join": i + 1, "t_s": round(t, 3), "frame": round(t * fps, 1),
                          "leaving": a["id"], "arriving": b["id"],
                          "leaving_origin": a.get("origin"), "arriving_origin": b.get("origin")})

        if not _task_media_ready(dst, {"shots": shots_all, "dialogue": lines,
                                       "music": {"file": "music.wav"}}):
            raise RuntimeError(f"Incomplete editorial media: {dst}")

        json.dump({"taskId": f"{slug}_s4e_{chunk}", "fps": round(fps, 6),
                   "n_shots": len(shots_all), "n_joins": len(joins),
                   "stretch_dur_s": round(t1 - t0, 3),
                   # The S4 manifest retained audio for trailer-only footage, whereas this variant mutes it.
                   # Update muted metadata to describe the actual media supplied in this task.
                   "shots": [{**{k: v for k, v in s.items() if not k.startswith("_")},
                              "has_sound": s.get("origin") == "film"}
                             for s in ({k: v for k, v in x.items() if k != "muted"}
                                       for x in shots_all)],
                   "joins": joins,
                   # The exploratory variant omits content descriptions so the agent must inspect the audio.
                   # Keep the validation copy privately under verifier/.
                   "dialogue": [{k: v for k, v in l.items()
                                 if not k.startswith("_")
                                 and (not A.explore or k in ("id", "file", "take_dur_s"))}
                                for l in lines],
                   # Omit padding duration because head_pad equals the reference alignment position
                   "music": {"file": "music.wav",
                             "window_is_wider_than_stretch": True,
                             "note": "The trailer's music. The window runs wider than "
                                     "this stretch and how much spills past each end is "
                                     "not stated. It is not aligned to 0 — where it comes "
                                     "in is your call."}},
                  open(f"{dst}/initial_state/manifest.json", "w"), ensure_ascii=False, indent=1)

        # ---- Reference answers: the finished trailer provides one valid edit, not a unique solution.
        json.dump({"task": f"{slug}_s4e_{chunk}", "fps": round(fps, 6),
                   "grading": "rubric — the finished trailer is one valid solution, not the only solution. "
                              "Assess editing quality with the rubric rather than requiring an exact reference match.",
                   "n_joins": len(joins),
                   # Renumber joins after transition units are removed; combine their duration into the reference hold.
                   
                   "golden_joins": remap_joins(man4["shots"], shots_all, key4["cuts"], dropped),
                   "dropped_transitions": {k: round(v, 3) for k, v in dropped.items()},
                   "golden_dialogue": [{"id": l["id"], "text": l["text"],
                                        "trailer_s": round(l["_trailer_s"] - t0, 3)}
                                       for l in lines if l["_used_in_golden"]],
                   "distractors": [l["id"] for l in lines if not l["_used_in_golden"]],
                   "golden_window_s": [round(t0, 3), round(t1, 3)],
                   "music_align_s": round(head_pad, 3)},
                  open(f"{dst}/verifier/answer.json", "w"), ensure_ascii=False, indent=1)

        # Record the creative-task criteria without treating the reference edit as the only acceptable result.
        
        gold_ids = [l["id"] for l in lines if l["_used_in_golden"]]
        # Use remapped joins after removing transition units so embedded effects do not inflate reference counts.
        
        gj = remap_joins(man4["shots"], shots_all, key4["cuts"], dropped)
        n_zero = sum(1 for c in gj if c["picture"] != "cut")
        # Keep hard checks in the rubric seed; final rubric assembly remains a separate step.
        
        
        json.dump({"task": tid, "hard_checks": {
            "joins_complete": f"Provide all {len(joins)} joins, numbered 1..{len(joins)}",
            "no_invented_takes": "Every dialogue take must come from the supplied pool",
            "no_repicture": "Do not change shot order or duration; these are fixed by RC",
            "distractors_are_wrong":
                f"The reference does not use the matched main lines of these takes: "
                f"{[l['id'] for l in lines if not l['_used_in_golden']]}. "
                f"Selecting their matched main-line interval (starting at manifest words_start_at_s) is incorrect; "
                f"other speech in the same file handles is not automatically incorrect because the reference may use it."},
            "reference": {
                "note": "The finished trailer is one valid solution, not the only one. Do not grade by exact match.",
                "golden_dialogue_ids": gold_ids,
                "distractor_ids": [l["id"] for l in lines if not l["_used_in_golden"]],
                "golden_music_align_s": round(head_pad, 3),
                "golden_nonplain_joins": n_zero}},
            open(f"{dst}/verifier/_rubric_seed.json", "w"), ensure_ascii=False, indent=1)

        json.dump({"stage": "s4e/editorial_fine_cut",
                   "golden_semantics": "reference — the original finished edit, not a unique solution",
                   "source_evidence": "Dialogue takes are located using the source-film transcript; music comes from separated trailer stems; "
                                      "join positions come from RC and follow the reference frame boundaries",
                   "leak_controls": "Do not supply the finished mix, picture, or vocal stem; music padding is randomized and undisclosed; "
                                    "trailer-only footage is muted because its music could reveal alignment; "
                                    "the dialogue pool includes distractors in deterministic order without putting answers first"},
                  open(f"{dst}/verifier/task_meta.json", "w"), ensure_ascii=False, indent=1)


        # Recover the beat context from the S3 prompt for dialogue and card instructions.
        
        beat = None
        pp = f"{src4}/task_prompt.md"
        if os.path.exists(pp):
            m = re.search(r"covers:\s*\n+>\s*(.+?)\n", open(pp).read())
            if m:
                beat = m.group(1).strip()
        # Mask the text of cards that this task asks the agent to create;
        # copying that text into the beat description would reveal the answer.
        
        
        if beat:
            ct = f"{ROOT}/exports/{slug}/card_text.json"
            if os.path.exists(ct):
                known = json.load(open(ct))["cards"]
                for sh_ in shots_all:
                    if sh_.get("origin") != "to_build":
                        continue
                    for st in (known.get(sh_["unit"]) or {}).get("states", []):
                        txt = (st.get("text") or "").strip()
                        for frag in re.findall(r"[A-Za-z][A-Za-z' ]{7,}", txt):
                            f = frag.strip()
                            if len(f) >= 8:
                                beat = re.sub(re.escape(f), "[a title card]", beat,
                                              flags=re.I)
        if A.explore:
            # The exploratory manifest omits speech_spans_s / text / words_* (see the
            # A.explore filtering above), requiring the agent to listen. Validation still needs these fields,
            # so retain an unchanged private copy in verifier/, which is hidden from the task-solving agent.
            json.dump({"note": "These fields are omitted from the exploratory manifest but retained for validation",
                       "takes": [{k: v for k, v in l.items() if not k.startswith("_")}
                                 for l in lines]},
                      open(f"{dst}/verifier/takes_content.json", "w"),
                      ensure_ascii=False, indent=1)

        write_prompt(dst, slug, chunk, shots_all, joins, lines, head_pad, tail_pad,
                     key4, beat=beat, dropped=dropped)

        for fn, blob in keep.items():
            q = f"{dst}/output/{fn}"
            if not os.path.exists(q):
                open(q, "wb").write(blob)
        return len(joins)


    def write_prompt(dst, slug, chunk, shots_all, joins, lines, head_pad, tail_pad,
                     key4, beat=None, dropped=None):
        man4 = {"shots": shots_all}
        n_j = len(joins)
        n_d = len(lines)
        L = [f"# Fine cut — {slug} / {chunk}\n\n",
             f"You have a rough cut: {len(man4['shots'])} shots in order, each already trimmed "
             f"to the frame, joined with {n_j} straight cuts. The pictures and their timing are "
             f"settled and **are not yours to change**.\n\n",
             "What is not settled is everything you can hear, and what happens at the joins.\n\n"]
        if beat:
            L.append(f"This stretch covers:\n\n> {beat}\n\n")
        L += [
             "## What you have\n\n",
             "| | |\n|---|---|\n",
             f"| `shots/pNN.mp4` | the rough cut's shots. They carry the **film's own production "
             f"sound** — whatever the microphone picked up on set. |\n",
             f"| `handles/pNN.m4a` | the same sound, but running 4 seconds before each shot's "
             f"first frame and 4 seconds past its last. `manifest.json` gives "
             f"`handle.picture_in_s` — where the picture starts inside that file. |\n"]
        if lines:
            if A.explore:
                L.append(f"| `takes/dNN.m4a` | **{n_d} pieces of dialogue, lifted from the film "
                         f"uncut.** What is said in them, and where in each file the words sit, "
                         f"is **not written down anywhere** — listen. Each runs longer than the "
                         f"line it was cut for, so there is usually more than one thing in a "
                         f"file. Not all of these belong in this stretch. |\n")
            else:
                L.append(f"| `takes/dNN.m4a` | **{n_d} pieces of dialogue, lifted from the film "
                         f"uncut.** `manifest.json` lists every stretch of speech in each file "
                         f"(`speech_spans_s`) — the line that was looked for, and whatever else "
                         f"happens to sit either side of it. All of it is yours to use. Not all "
                         f"of these pieces belong in this stretch. |\n")
        L.append(f"| `music.wav` | the trailer's music. The window runs wider than this stretch "
                 f"at both ends — how much wider is not stated — and it is **not lined up**. "
                 f"Where the music comes in is one of your decisions. |\n")
        L.append(f"| `manifest.json` | shots, joins with their exact times and frames, dialogue "
                 f"index, music window. |\n\n")

        L.append("## The work\n\n")
        L.append("**Sound.** The rough cut plays each shot's own production sound, cut off hard "
                 "at both edges. That is not a trailer. Decide what is heard:\n\n")
        if lines:
            L.append(f"- Which of the {n_d} lines belong in this stretch, and which do not. "
                     f"Some are here to be left out.\n")
            L.append("- Where each line you keep starts — under which shot, and how much of it "
                     "runs before or after the picture changes.\n")
            L.append("- How much of a take to use. They run long on purpose; the trailer's "
                     "version of a line is usually shorter than the film's.\n")
        L.append("- Where the production sound is worth keeping, and where it should give way.\n")
        L.append("- Where the music enters, and how it sits against the pictures.\n\n")

        tr = [s for s in man4["shots"] if s.get("origin") == "trailer"]
        if tr:
            one = len(tr) == 1
            L.append(f"**Silent shots.** {', '.join(x['id'] for x in tr)} "
                     f"{'was' if one else 'were'} made for the trailer rather than shot for the "
                     f"film, so {'it has' if one else 'they have'} no production sound at all — "
                     f"the picture arrives silent. What plays under "
                     f"{'it' if one else 'them'} is entirely your call.\n\n")
        cards = [s for s in man4["shots"] if s.get("origin") == "to_build"]
        if cards:
            one = len(cards) == 1
            L.append(f"**Cards.** {'One shot' if one else str(len(cards)) + ' shots'} in this "
                     f"stretch {'is' if one else 'are'} a title card — "
                     f"{', '.join(c['id'] for c in cards)}. There is no file for "
                     f"{'it' if one else 'them'}; making {'it' if one else 'them'} is part of "
                     f"the job. The slot is already the right length, so what you decide is what "
                     f"it says, how it is set, and how it enters and leaves.\n\n")
        L.append("**Picture.** Most joins want a straight cut and nothing else — this is the "
                 "common case, and adding something everywhere is the most reliable way to make "
                 "a trailer worse. Where a join wants more, the moves are: a **dip to black** "
                 "(how long it holds, and whether either side ramps, is part of the call), a "
                 "**dissolve**, or something faster — a whip pan, a flash frame.\n\n")
        if dropped:
            n = len(dropped)
            L.append(f"The rough cut you have is **all picture, edge to edge** — where the "
                     f"finished trailer breathes, this does not. {'A join' if n == 1 else 'Some joins'} "
                     f"here may want the picture to sink away and come back before the next shot "
                     f"starts: a beat of black, held long enough to feel like a breath and not an "
                     f"accident. That beat is time you add — the finished stretch runs longer than "
                     f"the rough cut when you use it. How long to hold, and whether either side "
                     f"ramps into it, is yours.\n\n")
        L.append("None of this is a checklist. Which of these a join wants — or whether it "
                 "wants any — is the question being asked.\n\n")

        L.append("## Deliverable\n\n")
        L.append("Write `output/edit.json`:\n\n```json\n{\n")
        L.append('  "music": {"starts_at_s": 3.20, "note": "file time that lands on 0.0 of the cut"},\n')
        if lines:
            L.append('  "dialogue": [\n')
            L.append('    {"take": "d03", "in_s": 1.40, "out_s": 3.10, "at_s": 0.00, "gain_db": 14,\n')
            L.append('     "why": "carries into the second shot before the picture turns"}\n')
            L.append('  ],\n')
        L.append('  "joins": [\n')
        L.append('    {"join": 1, "picture": "cut", "sound": "hard"},\n')
        L.append('    {"join": 2, "picture": "dip_to_black", "black_hold_s": 0.30, '
                 '"fade_in_s": 0.35, "sound": "under"},\n')
        L.append('    {"join": 3, "picture": "cut", "sound": "carries_over", "note": "line d03 runs on"},\n')
        L.append('    {"join": 4, "picture": "cut", "sound": "under", "note": "music holds it"}\n')
        L.append("  ],\n")
        L.append('  "production_sound": [{"shot": "p04", "keep": true, "gain_db": -6}],\n')
        L.append('  "cards": [{"shot": "p04", "text": "FROM THE DIRECTOR OF", '
                 '"look": "white type, centred, on black", "in": "cut", "out": "dip"}]\n')
        L.append("}\n```\n\n")
        L.append("`take` / `in_s` / `out_s` are which line and which part of it; `at_s` is where "
                 "it lands on this stretch's timeline, where 0.0 is the first frame. "
                 "`gain_db` is how far you lift or drop it.\n\n")
        # Separated music retains level automation from the reference mix.
        # Dialogue added in new positions needs its own level balancing.
        
        
        L.append("**On levels.** The music retains level changes from the reference mix. "
                 "Listen to the dialogue and music together, and adjust `gain_db` on the line, "
                 "the production sound, or both so the intended speech remains clear.\n\n")
        # Define a consistent sound vocabulary so outputs can be interpreted and checked.
        
        
        L.append("On each join:\n\n")
        L.append("- `picture` — one of `cut` / `dip_to_black` / `dissolve` / `other`\n")
        # Define carries_over / prelap using the established L-cut / J-cut terminology.
        # Explaining a technique gives vocabulary rather than assigning an answer to a join.
        # Do not reveal how many joins use each technique or which joins use it.
        L.append("- `sound` — one of these four, and give it on **every** join:\n"
                 "  - `hard` — the sound breaks where the picture breaks\n"
                 "  - `carries_over` — something belonging to the leaving shot runs past it. "
                 "An **L-cut**.\n"
                 "  - `prelap` — something belonging to the arriving shot starts early. "
                 "A **J-cut**.\n"
                 "  - `under` — neither side's own sound matters here; what carries the join is "
                 "laid on top. A **sound bridge** — usually the music bed, sometimes a line "
                 "from elsewhere running across both shots.\n\n")
        L.append(f"There are **{n_j} joins**, numbered 1 to {n_j} in order. Give all of them.\n\n")
        # Specify the render artifact consistently so the result can be reviewed, including audible level balance.
        
        L.append("**Also render it.** Build the result as `output/preview.mp4` — this stretch's "
                 "picture with your mix under it. It runs the length of the rough cut plus "
                 "whatever black you chose to add. `ffmpeg` is "
                 "here. The JSON says what you decided; the render is where anyone can hear "
                 "whether it worked. A line you placed but cannot hear in the render is a line "
                 "you did not place.\n\n")
        L.append("## How this is judged\n\n")
        L.append("Not against a single correct answer. The trailer that was actually cut from "
                 "this material is one solution; a different set of choices can be as good or "
                 "better. What is looked at is whether the choices hold together: does a line "
                 "land where the picture supports it, does the music arrive somewhere that means "
                 "something, is the restraint at the plain joins deliberate rather than absent.\n")
        open(f"{dst}/task_prompt.md", "w").write("".join(L))

        json.dump({"id": f"{slug}_s4e_{chunk}", "seed": "editorial_fine_cut",
                   "stage": "s4e/editorial_fine_cut", "film": slug, "chunk": chunk,
                   "beat": beat, "deliverables": ["output/edit.json", "output/preview.mp4"]},
                  open(f"{dst}/task.json", "w"), ensure_ascii=False, indent=1)


    films = sorted(os.listdir(TT)) if A.all else [A.film]
    tot = n_task = 0
    for slug in films:
        base = f"{TT}/{slug}/s_4"
        if not os.path.isdir(base):
            continue
        got = []
        chunks = sorted(os.listdir(base), key=lambda x: (len(x), x))
        for chunk_number, c in enumerate(chunks, 1):
            print(f"  [FC] {slug} processing chunk {chunk_number}/{len(chunks)}: {c}",
                  flush=True)
            r = build(slug, c)
            if r == "skip" or r is None:
                continue
            got.append(f"{c}:{r}")
            tot += r
            n_task += 1
        if got:
            print(f"  {slug:<24} {len(got):2d} chunks  " + " ".join(got[:12]), flush=True)
    print(f"\n{n_task} creative editing tasks, {tot} joins total")


if __name__ == "__main__":
    main()
