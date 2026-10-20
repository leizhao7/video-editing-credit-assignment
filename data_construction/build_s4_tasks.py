"""Public construction draft: build_s4_tasks.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/build_s4_tasks.py [options]\n\n  --film\n  --chunk\n  --all\n  --handle\n  --pad  音乐段两端各多留几秒\n  --keep-plain  保留一刀都不用动的段落——创作题里那是克制力测试\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)


def _media_duration(path, kind):
    """Return duration only for nonempty, probe-readable media of the expected kind."""
    import json, math, subprocess
    from pathlib import Path
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration:stream=codec_type", "-of", "json", str(path)],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            return None
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        if math.isfinite(duration) and duration > 0 and any(
                stream.get("codec_type") == kind for stream in info["streams"]):
            return duration
    except (OSError, ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired):
        pass
    return None


def _require_media(path, kind):
    duration = _media_duration(path, kind)
    if duration is None:
        raise RuntimeError(f"Missing or unreadable {kind} media: {path}")
    return duration


def _write_media(command, output, kind):
    """Do not leave a failed conversion as a reusable cached media file."""
    import subprocess
    from pathlib import Path
    output = Path(output)
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Media conversion failed ({result.returncode}): {output}")
        _require_media(output, kind)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _invalidate_task(directory):
    """A rebuilding task is incomplete until both publication files are rewritten."""
    from pathlib import Path
    for relative in ("task.json", "initial_state/manifest.json"):
        (Path(directory) / relative).unlink(missing_ok=True)


def _task_media_ready(directory, manifest):
    from pathlib import Path
    base = (Path(directory) / "initial_state").resolve()
    try:
        def valid(relative, kind):
            path = (base / relative).resolve()
            return base in path.parents and _media_duration(path, kind) is not None
        shots = manifest["shots"]
        if not isinstance(shots, list) or not shots:
            return False
        for shot in shots:
            if shot.get("origin") != "to_build" and not valid(f"shots/{shot['id']}.mp4", "video"):
                return False
            if shot.get("handle") and not valid(shot["handle"]["file"], "audio"):
                return False
        if not valid(manifest["music"]["file"], "audio"):
            return False
        return all(valid(line["file"], "audio") for line in manifest.get("dialogue", []))
    except (OSError, KeyError, TypeError, AttributeError):
        return False


def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys
    from build_s1_tasks import load_matches, cut, build_id

    ROOT = DATA_ROOT
    TT = f"{ROOT}/training_task"


    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--chunk"); P.add_argument("--all", action="store_true")
    # Audio handle length in seconds.
    P.add_argument("--handle", type=float, default=4.0)
    P.add_argument("--pad", type=float, default=2.0, help="音乐段两端各多留几秒")
    P.add_argument("--keep-plain", action="store_true",
                   help="保留一刀都不用动的段落——创作题里那是克制力测试")
    A = P.parse_args()
    require_target(P, A)

    # 画面转场归并到 4 类：agent 要选的是"做什么"，不是复述我们的词表
    PIC = {"cut": "cut", "no_change": "cut",
           "dip_to_black": "dip_to_black", "mid_dip": "dip_to_black",
           "dissolve": "dissolve"}
    TAGS = {"dip_to_black": "经过黑", "dissolve": "叠化", "other": "其他画面效果"}


    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True)


    def chunk_cuts(slug, chunk):
        """这一段内部的切点，以及每一刀的答案"""
        T = f"{ROOT}/edit/task_{slug}"
        key = json.load(open(f"{TT}/{slug}/s_3/{chunk}/verifier/answer.json"))
        order = [c["trailer_shot"] for c in key["cuts"]]
        s4 = {c["idx"]: c for c in json.load(open(f"{T}/stages/s4_cuts.json"))["cuts"]}
        au = {c["idx"]: c for c in json.load(open(f"{T}/stages/audio_measured.json"))["cuts"]}
        cl = json.load(open(f"{T}/stages/cut_list.json"))["cuts"]
        ids = [u["id"] for u in json.load(open(f"{T}/s3_units.json"))["units"]]
        pos = {u: i for i, u in enumerate(ids)}
        lo, hi = pos[order[0]], pos[order[-1]]

        # S3 自己标了 origin 的镜头：非 film 的，素材只能从预告片切，带成品混音
        s3_film = {c["trailer_shot"]: c.get("origin") == "film" for c in key["cuts"]}

        out = []
        for c in cl:
            if not (pos[c["after"]] >= lo and pos[c["before"]] <= hi):
                continue
            i = c["idx"]
            tr = s4[i].get("transition") or {}
            raw = tr.get("type", "cut")
            pic = PIC.get(raw, "other")
            sp = (au[i].get("speech") or {}).get("type", "hard")
            mus = set(au[i].get("music") or [])
            row = {"cut": len(out) + 1, "after": c["after"], "before": c["before"],
                   "picture": pic, "audio": sp,
                   "_raw_picture": raw,
                   "_kinds": [c["after_kind"], c["before_kind"]]
                                  + ["golden_only"] * (not s3_film.get(c["after"], True)
                                                       or not s3_film.get(c["before"], True))}
            if sp in ("J", "L"):
                row["audio_offset_s"] = round((au[i].get("speech") or {}).get("offset") or 0, 3)
            if pic == "dip_to_black":
                for k in ("fade_out", "black_hold", "fade_in"):
                    if isinstance(tr.get(k), (int, float)):
                        row[k + "_s"] = round(float(tr[k]), 3)
            if pic == "dissolve" and isinstance(tr.get("dur"), (int, float)):
                row["dur_s"] = round(float(tr["dur"]), 3)
            row["_bridge"] = bool("bridge" in mus and sp == "hard" and pic == "cut")
            out.append(row)
        return key, order, out


    def tags_for(rows):
        t = []
        if any(r["audio"] == "J" for r in rows): t.append("J-cut")
        if any(r["audio"] == "L" for r in rows): t.append("L-cut")
        if any(r["_bridge"] for r in rows): t.append("声音桥")
        if any(r["picture"] == "dip_to_black" for r in rows): t.append("经过黑")
        if any(r["picture"] == "dissolve" for r in rows): t.append("叠化")
        if any(r["picture"] == "other" for r in rows): t.append("其他画面效果")
        if any("card" in r["_kinds"] for r in rows): t.append("字幕卡进出")
        return t


    DESC_EN = {
        "J-cut": "**J-cut** — the arriving shot's sound starts before its picture does",
        "L-cut": "**L-cut** — the leaving shot's sound runs on after its picture is gone",
        "声音桥": "**a sound bridge** — picture and dialogue both break, and something "
                  "continuous underneath carries the join",
        "经过黑": "**a dip through black** — the picture sinks to black and comes back; "
                  "how long it holds, and whether either side ramps, is part of the call",
        "叠化": "**a dissolve** — the two pictures overlap and blend",
        "其他画面效果": "**something else on the picture** — a whip, a flash, a flare",
        "字幕卡进出": "**cutting to and from a card** — a title card sits in this stretch",
    }

    DESC = {
        "J-cut": "**J-cut** —— 下一镜的声音先进来，画面还没换",
        "L-cut": "**L-cut** —— 上一镜的声音拖过切点，画面已经换了",
        "声音桥": "**声音桥** —— 画面硬切、台词也断，靠一条跨过切点不变的声音把两镜连住",
        "经过黑": "**经过黑** —— 画面沉到黑再起来；黑压多久、两侧斜坡多长要给出来",
        "叠化": "**叠化** —— 两个画面重叠渐变",
        "其他画面效果": "**其他画面效果** —— 甩镜、闪白、闪光一类",
        "字幕卡进出": "**字幕卡进出** —— 这一段有切进或切出字幕卡的地方",
    }


    def _build(slug, chunk, film_name):
        src3 = f"{TT}/{slug}/s_3/{chunk}"
        dst = f"{TT}/{slug}/s_4/{chunk}"
        if not os.path.isdir(src3):
            return None
        key3, order, rows = chunk_cuts(slug, chunk)
        if len(rows) < 2:
            return None
        # --keep-plain retains segments requiring no added transition, allowing restraint as an editing choice.
        
        
        if not A.keep_plain and not any(
                "golden_only" not in r["_kinds"]
                and not (r["picture"] == "cut" and r["audio"] == "hard") for r in rows):
            return None
        for d in ("initial_state", "verifier", "output"):
            os.makedirs(f"{dst}/{d}", exist_ok=True)
        open(f"{dst}/output/.gitkeep", "w").close()
        mus = f"{ROOT}/exports/{slug}/stems/music.wav"
        mus_dur = _require_media(mus, "audio")

        pb = json.load(open(f"{ROOT}/exports/{slug}/probe.json"))
        GOLD = f"{pb['dir']}/{pb['golden']}"
        CROP = (pb.get("crop_expr", "") + ",") if pb.get("crop_expr") else (
            f"crop=iw:iw/{pb['crop']}:0:(ih-iw/{pb['crop']})/2," if pb["crop"] else "")
        units = json.load(open(f"{ROOT}/edit/task_{slug}/s3_units.json"))["units"]
        units_by_id = {u["id"]: u for u in units}
        order_units = [u["id"] for u in units]
        t, pos = 0.0, {}
        for u in units:
            pos[u["id"]] = t; t += u["dur"]
        lo_i, hi_i = order_units.index(order[0]), order_units.index(order[-1])
        t0 = pos[order[0]]
        t1 = pos[order[-1]] + units_by_id[order[-1]]["dur"]

        # Include every timeline unit, including cards and trailer-only media.
        # Audio handling differs by origin; copied finished audio must not be scored as a new sound-editing decision.


        os.makedirs(f"{dst}/initial_state/shots", exist_ok=True)
        bysrc = {c["trailer_shot"]: c for c in key3["cuts"]}
        # J/L edits need production sound before or after the visible interval.
        # Provide longer handles from the source film; the visual edit points are already supplied in S4.
        
        
        p2 = load_matches(slug)
        SRC = f"{pb['dir']}/{pb['source']}"
        src_dur = float(pb["src"].get("vdur") or pb["src"]["dur"])
        os.makedirs(f"{dst}/initial_state/handles", exist_ok=True)

        def handle_for(uid, pid, dur):
            """入点前后各切 --handle 秒的正片声音；片头片尾不够就有多少给多少"""
            r = p2.get(uid) or {}
            if not r.get("refined_src_s"):
                return None
            t = float(r["refined_src_s"])
            head = min(A.handle, t)
            tail = min(A.handle, max(0.0, src_dur - (t + dur)))
            if head + tail < 0.25:
                return None
            out = f"{dst}/initial_state/handles/{pid}.m4a"
            if _media_duration(out, "audio") is None:
                _write_media(["ffmpeg", "-v", "error", "-ss", f"{t - head:.4f}",
                              "-t", f"{head + dur + tail:.4f}", "-i", SRC,
                              "-vn", "-ac", "2", "-ar", "48000",
                              "-c:a", "aac", "-b:a", "128k", "-y", out], out, "audio")
            return {"file": f"handles/{pid}.m4a", "head_s": round(head, 3),
                    "tail_s": round(tail, 3), "picture_in_s": round(head, 3)}
        span = order_units[lo_i:hi_i + 1]
        shots, cards = [], []
        for n, uid in enumerate(span, 1):
            u = units_by_id[uid]
            pid = f"p{n:02d}"
            out = f"{dst}/initial_state/shots/{pid}.mp4"
            rec = {"id": pid, "unit": uid, "dur_s": round(u["dur"], 3), "kind": u["kind"]}
            if u["kind"] == "card":
                # 字幕卡不给素材 —— 任务的一部分就是把它做出来。它在时间线上的位置、
                # 说什么、什么底色、几个文字状态，都写进提示词。
                rec["origin"] = "to_build"; rec["build_it"] = True
                cards.append((n, uid, u))
                shots.append(rec)
                continue
            if (uid in bysrc and u["kind"] == "source"
                    and bysrc[uid].get("origin") == "film"):
                c = bysrc[uid]
                rec["origin"] = "film"
                h = handle_for(uid, pid, c["dur_s"])
                if h:
                    rec["handle"] = h
                if _media_duration(out, "video") is None:
                    if os.path.lexists(out):
                        os.remove(out)
                    # cut(), not a hand-rolled ffmpeg line. Both paths here have to produce
                    # media that is identical in every property but the picture, and the
                    # only place that is guaranteed is the function the other stages use.
                    if not cut(f"{src3}/initial_state/clips/{c['clip']}.mp4",
                               c["in_s"], c["dur_s"], "", out):
                        if os.path.exists(out):
                            os.remove(out)
                        raise RuntimeError(f"S4 shot conversion failed: {out}")
            else:
                # Cards use a uniform silent track. Trailer-only footage has no matched production track,
                # so its finished audio is retained and adjacent audio decisions are marked unscored.


                rec["origin"] = "trailer"
                mute = u["kind"] == "card"
                rec["muted"] = mute
                # Use the shared cut() normalization so encoding details do not identify the source route.


                if _media_duration(out, "video") is None:
                    if os.path.lexists(out):
                        os.remove(out)
                    # CROP carries a trailing comma and falls back to a computed expression
                    # when crop_expr is absent; cut() wants the bare filter. Passing only
                    # crop_expr would silently drop the crop on every film that has none.
                    if not cut(GOLD, pos[uid], u["dur"], CROP.rstrip(","), out, mute=mute):
                        if os.path.exists(out):
                            os.remove(out)
                        raise RuntimeError(f"S4 shot conversion failed: {out}")
            shots.append(rec)

        # Pad the music segment without exposing its reference start.
        # Record the actual padding after clamping to the start of the media.
        
        
        mout = f"{dst}/initial_state/music.wav"
        head_pad = min(A.pad, t0)
        tail_pad = min(A.pad, max(0.0, mus_dur - t1)) if mus_dur else A.pad
        if _media_duration(mout, "audio") is None:
            _write_media(["ffmpeg", "-v", "error", "-ss", f"{max(0, t0 - A.pad):.3f}",
                          "-t", f"{(t1 - t0) + 2 * A.pad:.3f}", "-i", mus, "-y", mout], mout, "audio")
        if not _task_media_ready(dst, {"shots": shots, "music": {"file": "music.wav"}}):
            raise RuntimeError(f"Incomplete S4 media: {dst}")

        tags = tags_for(rows)
        n = len(rows)

        # 这一段的散文描述，从 S3 的提示词里取 —— 同一段材料，同一句话，别再造一版
        beat = ""
        p3 = f"{src3}/task_prompt.md"
        if os.path.exists(p3):
            m = re.search(r"covers:\s*\n+>\s*(.+?)\n", open(p3).read(), re.S)
            if m:
                beat = " ".join(m.group(1).split())

        # 要 agent 自己做的字幕卡
        ct = f"{ROOT}/exports/{slug}/card_text.json"
        ctext = json.load(open(ct))["cards"] if os.path.exists(ct) else {}
        card_specs = []
        for pos_n, uid, u in cards:
            info = ctext.get(uid, {})
            where = ("at the very start of this stretch" if pos_n == 1 else
                     "at the very end of this stretch" if pos_n == len(span) else
                     f"about {round(100 * (pos_n - 1) / max(len(span) - 1, 1))}% of the way through")
            lines = []
            for st in info.get("states", []):
                if st.get("text"):
                    lines.append(st["text"])
            bg = u.get("bg") or info.get("bg") or [0, 0, 0]
            card_specs.append({"unit": uid, "position": pos_n, "where": where,
                               "dur_s": round(u["dur"], 3), "n_states": u.get("n_states", 1),
                               "bg_rgb": bg, "reads": lines})

        n_film = sum(1 for x in shots if x.get("origin") == "film")
        L = ["# Task — finish this stretch\n\n",
             f"You are cutting the *{film_name}* trailer. The rough cut for this stretch is "
             f"done — the shots are chosen, ordered and trimmed, in `initial_state/shots/`, "
             f"**{len(shots)} of them**, `p01` first.\n\n"]
        if beat:
            L.append(f"This stretch covers:\n\n> {beat}\n\n")
        L.append("Each shot from the film comes with a **handle** — `initial_state/handles/"
                 "<id>.wav`, the film's own sound reaching past the picture at both ends. "
                 "`handle.head_s` in the manifest says how far into that file the picture "
                 "starts. This is the material a J or an L is made out of: a line that runs "
                 "on past a shot's out-point is in its tail, a line already under way before "
                 "a shot's in-point is in its head. Where the handle is quiet, there is "
                 "nothing to carry across and the join wants a straight cut.\n\n")
        L.append(f"The shots carry the **film's own audio, uncut**. `initial_state/music.wav` "
                 f"is the trailer's music for this stretch, with {head_pad:.1f}s of slack before "
                 f"it and {tail_pad:.1f}s after — you line it up yourself.\n\n")
        L.append(f"Your job is the fine cut: decide how each of the **{n} joins** is handled.\n")

        L.append("\n## What you can do at a join\n\n")
        L.append("Most joins are a straight cut and want nothing done to them. Where more is "
                 "wanted, these are the moves:\n\n")
        # 声音桥不在这张表里：它不是接点上的动作，是音乐床铺对之后自然成立的性质，
        # 交付格式里也写不出来（audio 只有 hard/J/L）。当成背景讲，不当成手法讲。
        for tg in tags:
            if tg == "声音桥":
                continue
            L.append(f"- {DESC_EN[tg]}\n")
        if "声音桥" in tags:
            L.append("\nMusic runs underneath much of this stretch and carries a good many "
                     "joins on its own — where it does, the join wants nothing else.\n")
        L.append("\nA join can take more than one at once — the picture going through black "
                 "while a line carries across it, say.\n")

        if card_specs:
            L.append("\n## Cards to build\n\n")
            one = len(card_specs) == 1
            L.append(f"This stretch runs {'a card' if one else str(len(card_specs)) + ' cards'} "
                     f"you have to make — there is no clip for {'it' if one else 'them'} in "
                     f"`shots/`, only the slot where {'it goes' if one else 'they go'}.\n\n")
            for k, cs in enumerate(card_specs, 1):
                r, g, b = cs["bg_rgb"]
                tone = ("black" if max(r, g, b) < 40 else "white" if min(r, g, b) > 210
                        else f"rgb({r},{g},{b})")
                extra = (f", the type arriving in **{cs['n_states']} steps** rather than all "
                         f"at once" if cs["n_states"] > 1 else "")
                L.append(f"**Card {k}** — {cs['where']}, running **{cs['dur_s']:.2f}s** on a "
                         f"{tone} field{extra}.\n")
                if cs["reads"]:
                    for ln in cs["reads"]:
                        L.append(f"  - reads: *{ln}*\n")
                else:
                    L.append("  - the text did not survive extraction — write something that "
                             "fits the beat above\n")
            L.append("\nBuild each at the same size and frame rate as the shots, and say "
                     "which join it sits on.\n")

        L.append(f"""
    ## How to work

    - Listen across each join. Is a line running through it? If so, does it belong to the
      shot that is leaving (L) or the one arriving (J), and by how much?
    - Listen to the music. Where it runs on through a join, the two shots may need nothing
      else holding them together.
    - Watch the picture. Does the stretch want a breath — a dip through black — anywhere?

    ## Deliverable

    Write `output/treatments.json`:

    ```json
    {{"cuts": [
      {{"cut": 1, "picture": "cut", "audio": "L", "audio_offset_s": 1.20}},
      {{"cut": 2, "picture": "dip_to_black", "black_hold_s": 0.30, "fade_in_s": 0.35, "audio": "hard"}},
      {{"cut": 3, "picture": "cut", "audio": "hard"}}
    ]}}
    ```

    `picture` is one of `cut` / `dip_to_black` / `dissolve` / `other`.
    `audio` is one of `hard` / `J` / `L`; for J or L give `audio_offset_s` — how far the sound
    runs ahead of or behind the picture, in seconds.
    `dip_to_black` takes `black_hold_s`, plus `fade_out_s` / `fade_in_s` where there is a ramp
    (0 where there is not). `dissolve` takes `dur_s`.

    Give all **{n}** joins, numbered 1 to {n} in order.
    """)
        if card_specs:
            L.append("""
    Put the cards you built under `output/cards/` and list them in the same file:

    ```json
    {"cards": [{"file": "cards/card1.mp4", "after_cut": 4}]}
    ```

    `after_cut` is the join it follows.
    """)
        open(f"{dst}/task_prompt.md", "w").write("".join(L))

        open(f"{dst}/initial_state/SETUP.md", "w").write(f"""# Material

    `shots/` is the rough cut for this stretch — already in the trailer's order and already
    trimmed to length (`p01` first). The shots carry the film's own audio, uncut.

    `music.wav` is the trailer's music for this stretch, with {head_pad:.1f}s of slack before it
    and {tail_pad:.1f}s after; lining it up against the picture is yours to do.

    You are not re-cutting the shots. You are deciding how the {n} joins between them are
    handled.
    """)
        fps = key3["fps"]
        ans = {"task": f"{slug}_s4_{chunk}", "fps": fps, "n_cuts": n,
               "tol": {"audio_offset_s": 0.35, "picture_param_s": 0.20},
               "tags_given": tags,
               "cuts": [{k: v for k, v in r.items() if not k.startswith("_")} |
                        {"carried_by_sound": r["_bridge"],
                         # 一侧是 golden_only 的接点，素材带的是成片原声，J/L 听得出来
                         "scored": "golden_only" not in r["_kinds"]} for r in rows]}
        json.dump(ans, open(f"{dst}/verifier/answer.json", "w"), ensure_ascii=False, indent=1)

        # This historical S4 intermediate has no bundled conform grader.
        # The editorial task stage is produced by build_s4_editorial.py.
        
        
        json.dump({"task": f"{slug}_s4_{chunk}",
                   "type": "mechanical — per-cut treatment against the trailer",
                   "formula": "read output/treatments.json; a cut is a hit when picture type "
                              "and audio type both match, and every numeric parameter the "
                              "answer carries is inside tolerance; score = hits/n; pass 0.7",
                   "n_cuts": n, "golden_score": 1.0,
                   "tolerances": {"audio_offset_s": 0.35, "picture_param_s": 0.20},
                   "scoring_status": "PROVISIONAL — 画面和声音两类判定权重相同，参数只在类型判对时才查。"
                                     "是否该给部分分（类型对、参数差）未定。",
                   "controls": [{"name": "全部答 cut + hard", "expect": f"约 {sum(1 for r in rows if r['picture']=='cut' and r['audio']=='hard')}/{n}"},
                                {"name": "每刀都加转场", "expect": "低"}]},
                  open(f"{dst}/verifier/rubric.json", "w"), ensure_ascii=False, indent=1)
        json.dump({"deliverable_kind": "json_treatments", "verifier_class": "mechanical",
                   "golden_semantics": "answer_key — 成片在这一刀实际做的处理；golden 1.0",
                   "source_evidence": "picture 来自 s4_cuts.json（agent 判读+实测校验过）；"
                                      "audio 来自 audio_measured.json（转录词级时间戳+分轨实测）",
                   "leak_controls": "音乐轨两端各留 2 秒且不给对齐位置；正片原声未经剪辑，"
                                    "J/L 的编辑点不在素材里；提示词只列这一段用到的手法，不指明哪一刀"},
                  open(f"{dst}/verifier/task_meta.json", "w"), ensure_ascii=False, indent=1)
        # Keep build provenance and answer content hashes separate: the build identifies construction inputs,
        # while answer_hash identifies this task reference.


        json.dump({"build": build_id(),
                   "answer_hash": hashlib.sha1(
                       json.dumps(ans, sort_keys=True).encode()).hexdigest()[:16]},
                  open(f"{dst}/verifier/build.json", "w"))

        # Publish only after all media, prompts, and verifier artifacts succeeded.
        with open(f"{dst}/initial_state/manifest.json", "w") as stream:
            json.dump({"taskId": f"{slug}_s4_{chunk}", "n_shots": len(shots), "n_cuts": n,
                       "shots": shots,
                       "music": {"file": "music.wav",
                                 "head_pad_s": round(head_pad, 3),
                                 "tail_pad_s": round(tail_pad, 3),
                                 "note": "Head and tail padding as given above (a stretch "
                                         "near the head of the trailer gets less head room). "
                                         "Not aligned to 0."}},
                      stream, ensure_ascii=False, indent=1)
        with open(f"{dst}/task.json", "w") as stream:
            json.dump({"id": f"{slug}_s4_{chunk}", "seed": "fine_cut", "stage": "s4/fine_cut",
                       "film": slug, "chunk": chunk, "beat": key3.get("beat"),
                       "deliverables": ["output/treatments.json"]},
                      stream, ensure_ascii=False, indent=1)
        return n


    def build(slug, chunk, film_name):
        dst = f"{TT}/{slug}/s_4/{chunk}"
        _invalidate_task(dst)
        completed = False
        try:
            result = _build(slug, chunk, film_name)
            completed = True
            return result
        finally:
            if not completed:
                # This also removes either marker after a partial publication failure.
                _invalidate_task(dst)


    films = sorted(os.listdir(TT)) if A.all else ([A.film] if A.film else [])
    names = {}
    for p in sorted(__import__("glob").glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p)); names[pb["slug"]] = os.path.basename(pb["dir"])
    tot = 0
    for slug in films:
        base = f"{TT}/{slug}/s_3"
        if not os.path.isdir(base):
            continue
        chunks = [A.chunk] if A.chunk else sorted(
            [c for c in os.listdir(base) if os.path.isdir(f"{base}/{c}")],
            key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        made = []
        for c in chunks:
            n = build(slug, c, names.get(slug, slug))
            if n:
                made.append(f"{c}:{n}"); tot += n
        print(f"  {slug:<24} {len(made):2d} 段  {' '.join(made)}", flush=True)
    print(f"\n  共 {tot} 个接点")


if __name__ == "__main__":
    main()
