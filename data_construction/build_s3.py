"""Public construction draft: build_s3.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/build_s3.py [options]\n\n  --film\n  --export-resolve  Also export a Resolve project using your configured bridge\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, csv, json, os, shutil, subprocess, sys, time
    import numpy as np


    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    ROOT = DATA_ROOT
    RES = os.environ.get("VIDEO_EDITING_RESOLVE_MCP_ROOT", "")

    _P = argparse.ArgumentParser(); _P.add_argument("--film", required=True)
    _P.add_argument("--export-resolve", action="store_true", help="Also export a Resolve project using your configured bridge")
    _ARGS = _P.parse_args()
    FILM = _ARGS.film
    E = f"{ROOT}/exports/{FILM}"
    _pb = json.load(open(f"{E}/probe.json"))
    G = f"{_pb['dir']}/{_pb['golden']}"
    PROXY = f"{_pb['dir']}/_proxy.mp4"
    CROP = (_pb.get("crop_expr", "") + ",") if _pb.get("crop_expr") else (
        f"crop=iw:iw/{_pb['crop']}:0:(ih-iw/{_pb['crop']})/2," if _pb["crop"] else "")
    FPS = _pb["gold"]["fps"]
    OUT = f"{ROOT}/edit/task_{FILM}"
    CARD_W = 1920
    CARD_H = int(round(1920 / (_pb["src_ratio"] or 2.39) / 2)) * 2


    def rv(code):
        p = subprocess.run([f"{RES}/.venv/bin/python", f"{RES}/cli.py", "eval_python",
                            "--code", code], capture_output=True, text=True, cwd=RES)
        try: return json.loads(p.stdout).get("result")
        except Exception: return {"err": (p.stdout or p.stderr)[-400:]}


    gm = {r["id"]: r for r in csv.DictReader(open(f"{E}/golden_clips/manifest.csv"))}
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_s1_tasks import mark_cards, load_matches
    p2 = load_matches(FILM)
    _sc = f"{ROOT}/edit/task_{FILM}/stages"
    if os.path.exists(f"{_sc}/script.json"):
        mark_cards(json.load(open(f"{_sc}/script.json")),
                   json.load(open(f"{_sc}/script_inputs.json")), FILM)
    cls = json.load(open(f"{E}/card_class.json"))

    for d in ["cards", "cards_mp4", "golden_only"]:
        shutil.rmtree(f"{OUT}/{d}", ignore_errors=True); os.makedirs(f"{OUT}/{d}", exist_ok=True)

    units, ncard, nlive, nmatch = [], 0, 0, 0
    for gid in sorted(gm):
        g = gm[gid]; t0, t1 = float(g["start_s"]), float(g["end_s"]); dur = t1 - t0
        v = p2[gid]["verdict"]
        if v.startswith("MATCH"):
            nmatch += 1
            units.append({"id": gid, "kind": "source", "dur": round(dur, 4),
                          "media": PROXY, "src_in": round(float(p2[gid]["refined_src_s"]), 4),
                          # How determined that in-point is. A static shot has a
                          # self-similarity plateau hundreds of frames wide, so argmax
                          # returns noise — the number is not wrong, it is undefined.
                          # Carrying it forward keeps downstream honest about which
                          # in-points are pinned and which are only approximately placed.
                          "in_plateau_frames": int(p2[gid].get("plateau_frames") or 0),
                          "clip": p2[gid]["source_clip"]})
            continue
        c = cls.get(gid, {"kind": "live", "states": 1})
        if c["kind"] == "card":
            # one still per text state: the lines animate in, so a single grab loses steps
            n = max(1, min(c.get("states", 1), 6))
            shots = []
            for k in range(n):
                t = t0 + dur*(k+0.75)/n
                o = f"{OUT}/cards/{gid}_s{chr(97+k)}.png"
                subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.4f}", "-i", G,
                                "-frames:v", "1", "-vf", CROP + "scale=1920:-2", "-y", o], check=True)
                shots.append(os.path.basename(o))
            nf = int(round(dur*FPS)); per = max(1, nf//n)
            movs = []
            for k, st_ in enumerate(shots):
                fr = nf - per*(n-1) if k == n-1 else per
                mo = f"{OUT}/cards_mp4/{gid}_s{chr(97+k)}.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-loop", "1", "-i", f"{OUT}/cards/{st_}",
                                "-vf", f"fps={FPS},scale={CARD_W}:{CARD_H},setsar=1", "-frames:v", str(fr),
                                "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                                "-pix_fmt", "yuv420p", "-y", mo], check=True)
                movs.append(os.path.basename(mo))
            ncard += 1
            units.append({"id": gid, "kind": "card", "dur": round(dur, 4),
                          "stills": shots, "movs": movs, "n_states": n, "bg": c.get("bg")})
        else:
            o = f"{OUT}/golden_only/{gid}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-t", f"{dur:.4f}",
                            "-i", G, "-vf", CROP + "scale=1920:-2", "-c:v", "libx264",
                            "-crf", "18", "-preset", "veryfast", "-an", "-y", o], check=True)
            nlive += 1
            units.append({"id": gid, "kind": "golden_only", "dur": round(dur, 4),
                          "media": os.path.basename(o)})

    json.dump({"stage": "S3", "film": FILM, "fps": FPS,
               "counts": {"source": nmatch, "card": ncard, "golden_only": nlive},
               "units": units}, open(f"{OUT}/s3_units.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(units)} units: source {nmatch} · card {ncard} · golden_only {nlive}")
    print(f"Card stills: {len(os.listdir(f'{OUT}/cards'))}")

    # ---------------- optional Resolve project export ----------------
    if not _ARGS.export_resolve:
        return
    if not RES:
        raise RuntimeError("Set VIDEO_EDITING_RESOLVE_MCP_ROOT for --export-resolve")
    media = [PROXY] + [f"{OUT}/golden_only/{u['media']}" for u in units if u["kind"] == "golden_only"] \
            + [f"{OUT}/cards_mp4/{m}" for u in units if u["kind"] == "card" for m in u["movs"]]
    print(f"\nImporting {len(media)} media files...")
    PROJ = f"{FILM.upper()}_S3"
    TL_H = int(round(1920 / (_pb["crop"] or (_pb["gold"]["w"] / _pb["gold"]["h"])) / 2)) * 2

    # Start from a fresh project to avoid appending clips to an existing timeline.
    
    
    _wipe = ('_cur = pm.GetCurrentProject()\n'
             'if _cur and _cur.GetName() == "%s": pm.CloseProject(_cur)\n'
             'result = {"deleted": bool(pm.DeleteProject("%s"))}') % (PROJ, PROJ)
    print(rv(_wipe))
    print(rv(f'result = bridge.project_create(name="{PROJ}")'))
    for k_, v_ in [("timelineFrameRate", f"{FPS:.3f}"), ("timelineResolutionWidth", "1920"),
                   ("timelineResolutionHeight", str(TL_H)),
                   ("timelineOutputResolutionWidth", "1920"),
                   ("timelineOutputResolutionHeight", str(TL_H))]:
        rv(f'result = bridge.project_set_setting(name="{k_}", value="{v_}")')
    print(f"Project settings: {PROJ}  {FPS:.3f}fps 1920x{TL_H}")
    print(rv(f'result = bridge.media_import(paths={json.dumps(media)})'))
    print(rv('result = bridge.timeline_create(name="S3_ROUGH")'))

    clips = []
    for u in units:
        n = int(round(u["dur"]*FPS))
        if u["kind"] == "source":
            s = int(round(u["src_in"]*FPS))
            clips.append({"clip": os.path.basename(PROXY), "start_frame": s, "end_frame": s+n})
        elif u["kind"] == "golden_only":
            clips.append({"clip": u["media"], "start_frame": 0, "end_frame": n})
        else:
            for m in u["movs"]:
                clips.append({"clip": m})          # Already rendered to the exact frame count; use the whole clip
    print(f"Arranging {len(clips)} clips...")
    for i in range(0, len(clips), 25):
        r = rv(f'result = bridge.timeline_append(clips={json.dumps(clips[i:i+25])})')
        if isinstance(r, dict) and r.get("err"): print("  !", r["err"][:200]); break

    info = rv('result = bridge.timeline_info()')
    print(info)
    n_items = ((info or {}).get("tracks", {}).get("video") or [{}])[0].get("items", 0)
    if n_items != len(clips):
        sys.exit(f"Timeline has {n_items} clips; expected {len(clips)}")

    # Save the project before exporting; ExportProject serializes persisted database contents.


    print(rv('result = bridge.project_save()'))

    drp = f"{OUT}/{FILM}_s3.drp"
    if os.path.exists(drp): os.remove(drp)
    print(rv(f'result = {{"exported": pm.ExportProject("{PROJ}", {json.dumps(drp)})}}'))

    # Verify the file, not the API's return value: count the clips actually written.
    import zipfile as _z, re as _re
    _zz = _z.ZipFile(drp)
    _seq = [n for n in _zz.namelist() if n.startswith("SeqContainer/")]
    _got = sum(len(_re.findall(r"<Sm2TiVideoClip\b", _zz.read(n).decode("utf8", "replace")))
               for n in _seq)
    if _got != len(clips):
        sys.exit(f"Exported .drp has {_got} clips; expected {len(clips)} ({len(_seq)} timelines)")
    print(f"\n-> {drp}   {_got} clips  {len(_seq)} timelines")


if __name__ == "__main__":
    main()
