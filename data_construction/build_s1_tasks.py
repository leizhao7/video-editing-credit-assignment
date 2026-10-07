"""Public construction draft: build_s1_tasks.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/build_s1_tasks.py [options]\n\n  --film\n  --all\n  -j, --workers\n  --text-only  只按当前 script.json 重写提示词，不重切素材\n  --skip-existing  已经建好且素材完整的作业段直接跳过\n  --repair-cards  仅 S1：应用有证据的实拍/字幕卡纠错并记录私有分类审计；需要完整重建\n  --card-overrides  私有 JSON：{film_slug: {shot_id: {classification, evidence}}}\n  --gc  清掉无人引用的缓存片段后退出\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import argparse, ast, copy, csv, glob, hashlib, json, os, random, re, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = DATA_ROOT
OUT = f"{ROOT}/training_task"
N_DISTRACT = 20            # 每个作业段的干扰项目标数


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def pipe(prod, cons):
    """Run a producer/consumer pipeline and reap the producer on every exit path.
    Use a temporary stderr file to avoid blocking on an unread pipe."""
    with tempfile.TemporaryFile() as errf:
        p1 = subprocess.Popen(prod, stdout=subprocess.PIPE, stderr=errf)
        try:
            r = subprocess.run(cons, stdin=p1.stdout, capture_output=True, text=True)
        finally:
            p1.stdout.close()
            try:
                p1.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p1.kill()
                p1.wait()
        if p1.returncode:
            errf.seek(0)
            r.stderr = (errf.read().decode("utf8", "replace") + "\n" + (r.stderr or ""))
            if r.returncode == 0:
                r.returncode = p1.returncode      # a broken decode is not a success
    return r


# Keep the hardlink cache outside task directories so copying tasks does not duplicate cached media.


CACHE = f"{ROOT}/.clipcache"
# bump when the ffmpeg invocation changes in any way that alters the output
RECIPE = "v15-mute-keeps-track"


def frame_vf(crop, w=854, h=480):
    """Return the shared crop, scale, padding, and aspect-ratio filter chain.
    Construction and verification must use the same normalization."""
    return (((crop + ",") if crop else "")
            + f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
              f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")


def _valid_cached_cut(path, nframes):
    """A cache entry is valid only if it fully decodes with enough frames."""
    if not os.path.isfile(path) or os.path.getsize(path) <= 500:
        return False
    try:
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False
    if decoded.returncode != 0 or (decoded.stderr or "").strip():
        return False
    try:
        counted = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False
    try:
        got = int(counted.stdout.strip().strip(","))
    except (AttributeError, ValueError):
        return False
    return counted.returncode == 0 and got >= nframes - 1


def cut(src, t0, dur, crop, dst, w=854, h=480, mute=False, independent_silence=False):
    """Cut one candidate with explicit video/audio normalization.
    Uniform dimensions, streams, metadata, and audio properties avoid exposing the source route.
    Cache the result and hardlink repeated uses of the same recipe."""
    # Include every output-affecting parameter in the cache key, including the recipe and mute policy.


    if independent_silence and not mute:
        raise ValueError("Independent silence is restricted to explicitly silent repair cuts")
    # Keep the exact historical key and command for every default/shared-stage call.
    # Strict S1 repair must not reuse a cut whose silent track depended on source audio.
    key_text = f"{src}|{t0:.4f}|{dur:.4f}|{crop}|{w}x{h}|{int(mute)}|{RECIPE}"
    if independent_silence:
        key_text += (f"|independent-silence-v2-source-timebase24|trim-end={dur:.8f}"
                     f"|frames={max(1, int(round(dur * 24)))}")
    key = hashlib.sha1(key_text.encode()).hexdigest()[:20]
    cached = f"{CACHE}/{key}.mp4"
    os.makedirs(CACHE, exist_ok=True)
    nframes = max(1, int(round(dur * 24)))
    if os.path.exists(cached) and _valid_cached_cut(cached, nframes):
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(cached, dst)
        except OSError:
            shutil.copy(cached, dst)
        return True
    if os.path.exists(cached):
        # Unlink only this cache name.  Existing task hardlinks remain available for
        # diagnosis until their own task is rebuilt; the replacement below gets a new
        # inode and can never mutate those old paths.
        try:
            os.remove(cached)
        except FileNotFoundError:
            # Another build worker rejected the same bad inode first.
            pass
    # Critical hardlink boundary: opening an existing dst with ffmpeg -y truncates the
    # inode in place.  If dst is linked into the cache (the normal steady state), that
    # corrupts the cache and every task sharing it.  Always unlink before a cache-miss
    # encode so the encoder creates a private inode.
    if os.path.exists(dst):
        os.remove(dst)
    # Normalize sample aspect ratio so encoding metadata cannot identify the source route.


    vf = frame_vf(crop, w, h)
    if independent_silence:
        # Rawvideo has no timestamps. Taking dur*24 NATIVE frames before assigning
        # 24 fps loses the last 20% of a 30 fps source interval. Resample source PTS
        # first, with a half-open source-duration trim so a following shot cannot
        # enter the last output frame. The output duration is still quantized to
        # round(dur*24)/24; fps may hold the last in-range frame to round that edge.
        # Anchor to the first actually decoded in-range frame: a fractional seek
        # otherwise makes fps pad a duplicate first frame even on native 24 fps.
        # This retains the original native-24 sequence for off-grid seeks too.
        vf += (f",trim=start=0:end={dur:.8f},setpts=PTS-STARTPTS,"
               "fps=fps=24:start_time=0:round=near:eof_action=pass")
    # Use an exact output frame count independent of the source frame rate.
    # Decode to raw frames before encoding so stream-level HDR metadata cannot leak through.


    prod = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.4f}",
            "-i", src, "-map", "0:v:0", "-vf", vf,
            "-frames:v", str(nframes),      # output option — before -i it is silently wrong
            "-an", "-dn", "-sn", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"]
    # Normalize audio channels, rate, codec, and level across source routes.
    # Repair-only mode supplies an independent silent clock bounded by the video frame count;
    # it does not open or decode original audio, which may end before the video.


    audio_input = (["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"] if independent_silence else
                   ["-ss", f"{max(0.0, t0):.4f}", "-t", f"{nframes / 24.0 + 0.5:.4f}", "-i", src])
    cons = ["ffmpeg", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
            *audio_input,
            "-map", "0:v:0", "-map", "1:a:0" if independent_silence else "1:a:0?", "-shortest",
            # Keep only the two explicitly mapped streams; the second input may contain unrelated tracks.


            "-dn", "-sn", "-map_metadata", "-1", "-map_chapters", "-1",
            # Silencing retains a uniform audio stream, avoiding source-identifying stream counts.


            "-af", (("volume=0," if mute else "")
                    + "aformat=channel_layouts=stereo,"
                    + ("" if mute else "loudnorm=I=-18:TP=-1.5:LRA=11,")
                    + "aresample=48000:first_pts=0"),
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
            "-metadata:s:a:0", "handler_name=", "-metadata:s:a:0", "language=und",
            # Stated, not inherited — there is now nothing to inherit from, but these
            # also have to be the same for every clip in the corpus, not merely the same
            # within a film.
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv",
            "-chroma_sample_location", "left",
            "-x264-params", "keyint=48:min-keyint=48:scenecut=0",
            "-video_track_timescale", "12288",
            "-metadata:s:v:0", "handler_name=", "-metadata:s:v:0", "language=und",
            "-metadata", "encoder=",
            "-brand", "mp42", "-movflags", "+faststart",
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-y", dst]
    r = pipe(prod, cons)
    # Digital silence can make loudnorm produce a nonfinite level rejected by AAC.
    # Retry without loudnorm only on this silent-input failure path.


    if r.returncode and "NaN" in (r.stderr or ""):
        cons_q = [(a.replace("loudnorm=I=-18:TP=-1.5:LRA=11,", "") if "loudnorm" in a else a)
                  for a in cons]
        r = pipe(prod, cons_q)
    # Validate by decoding, not by file size or container metadata alone.
    # A failed faststart rewrite can leave a readable index with corrupt frame data.


    ok = r.returncode == 0 and _valid_cached_cut(dst, nframes)
    if ok:
        try:
            os.link(dst, cached)
        except OSError:
            pass
    if not ok and os.path.exists(dst):
        # ffmpeg errors and still writes what it managed to encode. Left alone that
        # partial file sits in clips/ unlisted — the agent sees a candidate the manifest
        # does not have, and had it been an answer clip the key would name a missing id.
        os.remove(dst)
    return ok


_CARD_MARK = (r"字幕卡|片名卡|档期卡|标题卡|演职员|片头卡|厂标|信息卡|引语卡|logo|LOGO"
              r"|纯色底.{0,6}字|黑底.{0,4}字|全屏.{0,4}字|黑场字幕")
_CARD_POS = re.compile(r"(?:" + _CARD_MARK + r")(?:之后|之前|后的|后接|前的)")
_CARD_OVER = re.compile(r"叠|压在|覆盖|上方叠")
_LUMA = None

# Use the same luminance sampling policy for construction and verification.
# Selecting the brightest sample prevents a fade-up from being classified by its dark midpoint alone.


LUMA_SAMPLES = (0.2, 0.5, 0.8)      # 取样位置（占镜头长度的比例）
LUMA_FLAT_STD = 8.0                 # 标准差低于此值 = 画面里没有可判断的内容


def featureless(luma):
    """Return whether sampled frames lack sufficient contrast for visual judgment.
    Use the same brightest-of-LUMA_SAMPLES measurement as shot_luma.py so fade endpoints
    are not classified from a single dark middle frame."""
    return bool(luma) and luma[1] < LUMA_FLAT_STD


too_dark = featureless          # 旧名，仍被引用


def mark_cards(script, inputs, slug):
    """Reclassify logos, title cards, and other non-footage units from shot-log evidence.
    Distinguish a card itself from a temporal reference to a card or text over live action."""
    global _LUMA
    if _LUMA is None:
        q = f"{ROOT}/shot_luma.json"
        _LUMA = json.load(open(q)) if os.path.exists(q) else {}
        if not _LUMA:
            print("  ★★ 找不到 shot_luma.json —— 黑场不会被剔除，"
                  "本次构建与已交付的语料不一致。先跑: python3 shot_luma.py", flush=True)
    desc = {s["id"]: (s.get("desc") or "") for s in script.get("shots", [])}
    n = 0
    for s in inputs["shots"]:
        if s.get("kind") == "card":
            continue
        d = desc.get(s["id"], "")
        if re.search(_CARD_MARK, d) and not _CARD_POS.search(d) and not _CARD_OVER.search(d):
            s["kind"] = "card"
            n += 1
            continue
        # Route featureless frames like cards because their content cannot support clip selection.
        # Use measured contrast rather than treating every matcher BLACK verdict as reliable.


        if featureless(_LUMA.get(f"{slug}/{s['id']}")):
            s["kind"] = "card"
            n += 1
    return n


# Apply separate agreement thresholds to geometric matches and colour fallbacks.
# Rejecting a valid match uses the trailer route; accepting an invalid one can create a wrong target.


MATCH_MIN = {"MATCH": 0.75, "MATCH(colour)": 0.85}
MATCH_MIN_DEFAULT = 0.85
_CONF = None


def drop_weak_matches(slug, p2, thr=None):
    """Reject source matches with insufficient measured visual agreement.
    Clear their refined source timestamps so downstream construction uses the reference-shot route.
    The geometric and colour-fallback routes use separate thresholds."""
    global _CONF
    if _CONF is None:
        p = f"{ROOT}/match_confidence.json"
        _CONF = json.load(open(p)) if os.path.exists(p) else {}
        if not _CONF:
            # Warn when this measured build input is absent; weak-match filtering then cannot be applied.
            
            
            print("  ★★ 找不到 match_confidence.json —— 弱匹配不会被剔除，"
                  "本次构建与已交付的语料不一致。", flush=True)
            print("     先跑: python3 verify_matches.py", flush=True)
    if not _CONF:
        return 0
    n = 0
    for sid, r in p2.items():
        c = _CONF.get(f"{slug}/{sid}")
        if c is None or r.get("refined_src_s") in (None, "", "None"):
            continue
        lim = thr if thr is not None else MATCH_MIN.get(r.get("verdict"), MATCH_MIN_DEFAULT)
        if c < lim:
            # Update both the timestamp and verdict so downstream consumers see a consistent record.


            r["refined_src_s"] = ""
            r["verdict"] = "NOMATCH(weak picture agreement)"
            n += 1
    return n


_BUILD_ID = None


def build_id():
    """Fingerprint builder code, encoding recipe, and measured filtering inputs.
    Use this to distinguish current tasks from artifacts produced by different construction inputs."""
    global _BUILD_ID
    if _BUILD_ID is None:
        h = hashlib.sha1()
        for f in ("build_s1_tasks.py", "build_s23_tasks.py", "build_s4_tasks.py",
                  "grade_s1.py", "grade_s2.py", "grade_s3.py",
                  "match_confidence.json", "shot_luma.json"):
            p_ = f"{CODE_ROOT if f.endswith('.py') else ROOT}/{f}"
            if not os.path.exists(p_):
                h.update(b"<missing>")
                continue
            raw = open(p_, "rb").read()
            # Hash the Python parse tree so ordinary comments do not invalidate cached tasks.
            # String literals remain part of the fingerprint, including generated prompt text.


            if f.endswith(".py"):
                try:
                    h.update(ast.dump(ast.parse(raw)).encode())
                    continue
                except SyntaxError:
                    pass                  # 解析不了就退回原文，宁可多重建
            h.update(raw)
        h.update(RECIPE.encode())
        # The graders are copied into every task, so editing one and then running
        # --skip-existing would leave the old copy in place and call the task current.
        # A separate check catches that after the fact; the fingerprint is what stops it
        # happening. ffmpeg decides what the clips actually contain, and swapping it out
        # under an unchanged recipe is the same split this whole mechanism exists for,
        # one dimension over.
        h.update(sh(["ffmpeg", "-version"]).stdout.splitlines()[0].encode()
                 if sh(["ffmpeg", "-version"]).returncode == 0 else b"<no ffmpeg>")
        _BUILD_ID = h.hexdigest()[:16]
    return _BUILD_ID


def s1_build_id(repair_cards=False, card_decisions=None):
    """S1 opt-in provenance without changing other stages' classification policy."""
    if not repair_cards:
        return build_id()
    from s1_card_classification import POLICY_VERSION
    with open(f"{CODE_ROOT}/s1_card_classification.py", "rb") as source:
        tree = ast.dump(ast.parse(source.read()))
    evidence = json.dumps(card_decisions or [], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1((build_id() + POLICY_VERSION + tree + evidence).encode()).hexdigest()[:16]


def load_matches(slug):
    """Load source matches with shared card and visual-agreement corrections.
    Consumers use this function; matching and measurement tools read the raw file
    so they can produce independent updated measurements."""
    p2 = {r["golden_id"]: r for r in
          csv.DictReader(open(f"{ROOT}/exports/{slug}/p2_refined.csv"))}
    drop_weak_matches(slug, p2)
    return p2


def source_shots(slug):
    rows = list(csv.DictReader(open(f"{ROOT}/exports/{slug}/source_manifest.csv")))
    return [(float(r["start_s"]), float(r["end_s"])) for r in rows]


MAX_CLIP_S = 45.0          # 单个候选最长多少秒


def cover(a, b, use_from, use_dur, film_dur, cap=MAX_CLIP_S):
    """Return a padded window containing the entire reference-used source stretch.
    Padding can create overlap between neighbouring windows; callers may separate them jointly.
    The cap is soft because the narrowing branch can add a lead-out after applying it."""
    need_a, need_b = use_from, use_from + use_dur
    lo = min(a, need_a)
    hi = max(b, need_b)
    pad = min(1.5, max(0.0, (cap - (hi - lo)) / 2))
    lo = max(0.0, lo - pad)
    hi = min(film_dur, hi + pad)
    if hi - lo > cap:                      # 还是太长：围绕用到的那段收窄
        lo = max(0.0, min(need_a - 1.0, hi - cap))
        hi = min(film_dur, max(need_b + 1.0, lo + cap))
    # The use can run past the end of the film — the matched in-point carries a frame
    # or two of error and a shot near the credits has nowhere to extend into. Report
    # that instead of returning a window that silently fails to contain the answer.
    if lo > need_a + 1e-6 or hi < need_b - 1e-6:
        return None, None
    return lo, hi - lo


def bound(a, b, around=None, cap=MAX_CLIP_S):
    """Bound a long source shot around the interval used in the reference.
    The target interval must remain inside the returned candidate window."""
    if b - a <= cap:
        return a, b - a
    if around is None:
        return a, cap
    t0 = min(max(a, around - cap / 3), b - cap)
    return max(a, t0), cap


def shot_index(shots, t):
    """which source shot contains time t"""
    import bisect
    starts = [a for a, _ in shots]
    return max(0, min(len(shots) - 1, bisect.bisect_right(starts, t) - 1))


def pick_distractors(home, shots, need, exclude=()):
    """the film shots either side of the true ones, closest first

    Walking outward one ring at a time keeps the set spread across the chunk instead of
    piling twenty distractors around whichever shot happens to be first. Each comes out
    at its own boundaries — same kind of object as a true candidate.
    """
    # Exclude source shots already selected by match_short to avoid duplicate distractors.


    taken = set(home) | set(exclude)
    out, ring = [], 1
    while len(out) < need and ring < 60:
        for i in home:
            for j in (i - ring, i + ring):
                if len(out) >= need or not (0 <= j < len(shots)) or j in taken:
                    continue
                a, b = shots[j]
                if b - a < 0.5:          # 太短的镜头看不出内容，当不了干扰项
                    continue
                taken.add(j)
                out.append(j)
        ring += 1
    return out


def match_short(true, ids, a, z, by_id, gold, gcrop, shots, home, srcpath,
                scrop, rng, gap=4, excluded_trailer_ids=()):
    """Choose duration-matched distractors from reference shots outside the current chunk.
    Use a gap around chunk boundaries to avoid ambiguous membership.
    Different reference shots can reuse the same source take; callers must handle known aliases."""
    # Reject an invalid chunk index range rather than accidentally widening the distractor pool.
    
    
    if not (isinstance(a, int) and isinstance(z, int) and 0 <= a <= z < len(ids)):
        raise ValueError(f"作业段索引区间不合法: a={a!r} z={z!r} 共 {len(ids)} 个镜头")
    # Match distractor durations to every true candidate, including both source routes,
    # so unusually short or long clips do not identify the target set by duration alone.


    want = [(t, 2 if t["origin"] == "trailer_only" else 1) for t in true]
    if not want:
        return []
    # Use both trailer and source-film pools to cover the target duration range.
    # The source pool can supply clips shorter than the trailer minimum.


    pool = [{"src": gold, "t0": by_id[s]["start_s"], "dur": by_id[s]["dur_s"],
             "crop": gcrop, "origin": "trailer_only"}
            for k, s in enumerate(ids)
            if (k < a - gap or k > z + gap) and by_id[s]["kind"] != "card"
            and s not in excluded_trailer_ids
            and by_id[s]["dur_s"] > 0.05]
    lo = max(0.05, min(t["dur"] for t, _ in want) * 0.4)
    hi = max(t["dur"] for t, _ in want) * 1.6
    pool += [{"src": srcpath, "t0": bound(x, y)[0], "dur": bound(x, y)[1],
              "crop": scrop, "origin": "film", "src_shot": j}
             for j, (x, y) in enumerate(shots)
             if j not in home and lo <= y - x <= hi]
    out, used = [], set()
    for t, per in want:
        # Prefer a distractor at or below the target duration before ranking by closeness.
        
        
        near = sorted(range(len(pool)),
                      key=lambda i: (pool[i]["dur"] > t["dur"],
                                     abs(pool[i]["dur"] - t["dur"]), rng.random()))
        took = 0
        for i in near:
            if took >= per:
                break
            if i in used:
                continue
            used.add(i)
            took += 1
            out.append({"shot": None, "src_shot": None, **pool[i], "answer": False})
    return out


def _cut_all(cands, cdir, mute=False, strict=False):
    made, answer, keymap = [], [], []
    for i, c in enumerate(cands, 1):
        cid = f"c{i:02d}"
        dst = f"{cdir}/{cid}.mp4"
        if c.get("reviewed_media_reuse"):
            if not mute or not strict:
                raise ValueError("Audited video reuse is restricted to strict silent S1 repair")
            from s1_media_reuse import remux_silent
            ok = remux_silent(c["reviewed_media_reuse"], dst, c["dur"],
                              require_timebase_certification=True)
        else:
            ok = cut(c["src"], c["t0"], c["dur"], c["crop"], dst, mute=mute,
                     independent_silence=bool(mute and strict))
        if not ok:
            if strict:
                raise RuntimeError(f"S1 candidate cut failed: {cid}, source={c['src']}, "
                                   f"start={c['t0']}, duration={c['dur']}")
            # Fail explicitly when a candidate cannot be cut; silently dropping it can desynchronize task stages.


            print(f"    ★ 切片失败，候选被丢弃: {os.path.basename(c['src'])} "
                  f"@{c['t0']:.2f}+{c['dur']:.2f}s "
                  f"({'正确项' if c['answer'] else '干扰项'})", flush=True)
            continue
        made.append({"id": cid, "file": f"clips/{cid}.mp4", "dur_s": round(c["dur"], 3)})
        if c["answer"]:
            answer.append(cid)
        keymap.append({"id": cid, "answer": c["answer"], "trailer_shots": c["shot"],
                       "origin": c["origin"], "src_shot": c.get("src_shot"),
                       "src_t0": round(c["t0"], 3), "dur_s": round(c["dur"], 3)})
    return made, answer, keymap


def build_chunk(slug, film_name, ch, script, inputs, p2, units, shots, pb, srcpath,
                text_only=False, skip_existing=False, repair_cards=False,
                card_decisions=None, plan_only=False, verified_spans=None, reuse_media=None,
                negative_protection=None):
    verified_spans = verified_spans or {}
    if verified_spans and not repair_cards:
        raise ValueError("Verified candidate spans are restricted to isolated S1 repair")
    if repair_cards and text_only:
        raise ValueError("S1 card correction changes candidates/answers; text-only is unsafe")
    if repair_cards and card_decisions is None:
        raise ValueError("S1 repair requires classified inputs and card_decisions provenance")
    if plan_only and not repair_cards:
        raise ValueError("plan_only is supported only for the isolated S1 repair path")
    ids = [s["id"] for s in inputs["shots"]]
    a, z = ids.index(ch["shots"][0]), ids.index(ch["shots"][1])
    members = ids[a:z + 1]
    by_id = {s["id"]: s for s in inputs["shots"]}
    # Expose the chunk description rather than a per-shot description index that would reveal the answer.


    tdir = f"{OUT}/{slug}/s_1/{ch['id']}"
    cdir = f"{tdir}/initial_state/clips"
    if not plan_only:
        for d in (cdir, f"{tdir}/output", f"{tdir}/verifier"):
            os.makedirs(d, exist_ok=True)

    gold = f"{pb['dir']}/{pb['golden']}"
    gcrop = pb.get("crop_expr") or (f"crop=iw:iw/{pb['crop']}:0:(ih-iw/{pb['crop']})/2"
                                    if pb["crop"] else "")
    scrop = pb.get("src_crop_expr", "")

    # ---- true candidates: the FILM's own shot containing each trailer moment ----
    true, skipped, home, seen = [], [], [], {}
    for sid in members:
        sh_ = by_id[sid]
        if sh_["kind"] == "card":
            skipped.append((sid, "card")); continue
        if sid in verified_spans:
            span = verified_spans[sid]
            identity = (span["src"], span["t0"], span["dur"], span["crop"])
            if identity in seen:
                seen[identity]["shot"].append(sid)
                continue
            e = {"shot": [sid], **span}
            seen[identity] = e
            true.append(e)
            if span.get("src_shot") is not None:
                home.append(span["src_shot"])
            continue
        r = p2.get(sid, {})
        if sh_["kind"] == "source" and r.get("refined_src_s"):
            i = shot_index(shots, float(r["refined_src_s"]))
            # Keep the chunk index range separate from source timestamps; match_short needs the original indices.


            sa, sb = shots[i]
            t0, dur = bound(sa, sb, around=float(r["refined_src_s"]))
            identity = (srcpath, t0, dur, scrop) if repair_cards else i
            if identity in seen:
                # Same reviewed interval, irrespective of whether it arrived via
                # a verified old candidate or the legacy film route.
                seen[identity]["shot"].append(sid); continue
            e = {"shot": [sid], "src": srcpath, "t0": t0, "dur": dur,
                 "crop": scrop, "origin": "film", "src_shot": i}
            seen[identity] = e; true.append(e); home.append(i)
        else:
            # no counterpart in the film — the trailer's own shot goes in
            true.append({"shot": [sid], "src": gold, "t0": sh_["start_s"],
                         "dur": sh_["dur_s"], "crop": gcrop, "origin": "trailer_only",
                         "src_shot": None})

    if len(true) < (1 if repair_cards else 3):
        return None, f"{ch['id']}: 只有 {len(true)} 个可用镜头，太少"
    # A chunk with no source-film anchor cannot supply neighbouring distractors.
    # Skip it instead of creating a select-all task.


    if len(home) < 2 and not repair_cards:
        return None, (f"{ch['id']}: 只有 {len(home)} 个镜头锚在正片上"
                      f"（{len(true)-len(home)} 个是片头片尾图形），不构成选素材任务")

    # ---- distractors: neighbouring film shots, own boundaries ------------
    drng = random.Random(int(hashlib.sha1(f"short/{slug}/{ch['id']}".encode())
                             .hexdigest()[:8], 16))
    protected = set(home)
    negative_protection = negative_protection or {}
    if negative_protection and not repair_cards:
        raise ValueError("Audit negative protection only applies to isolated S1 repair")
    extra_indices = negative_protection.get("film_source_indices", [])
    if any(type(index) is not int or not 0 <= index < len(shots) for index in extra_indices):
        raise ValueError("Audit protection names an invalid source index")
    protected.update(extra_indices)
    if repair_cards:
        # A boundary-rejected film match is now represented by its trailer shot.
        # Do not immediately draw that declared film shot back as a distractor.
        # IDs are accepted as indices only after checking their declared timestamp
        # against the actual source manifest interval, never by name alone.
        for sid in members:
            matched = p2.get(sid, {})
            source_id = matched.get("source_clip", "")
            if not isinstance(source_id, str) or not re.fullmatch(r"s\d{4,}", source_id):
                continue
            source_index = int(source_id[1:])
            try:
                declared_start = float(matched.get("clip_start_s", ""))
            except (TypeError, ValueError):
                continue
            if (0 <= source_index < len(shots)
                    and abs(shots[source_index][0] - declared_start) <= 0.001):
                protected.add(source_index)
    excluded_trailer = set()
    if repair_cards:
        # Trailer moments outside this chunk can still show the same source take.
        # Exclude every known alias of protected footage, including weak p2 routes:
        # weak alignment is not enough for a positive, but enough to avoid a risky
        # distractor. Also cover exact trailer interval overlaps.
        for other in ids:
            span = verified_spans.get(other)
            matched = p2.get(other, {})
            declared = matched.get("source_clip", "")
            index = int(declared[1:]) if re.fullmatch(r"s\d{4,}", str(declared)) else None
            if (span and span.get("src_shot") in protected) or index in protected:
                excluded_trailer.add(other)
            sh_ = by_id[other]
            if any(min(sh_["start_s"] + sh_["dur_s"], r["start_s"] + r["duration_s"]) >
                   max(sh_["start_s"], r["start_s"])
                   for r in negative_protection.get("trailer_intervals", [])):
                excluded_trailer.add(other)
            if any(t["origin"] == "trailer_only" and
                   min(sh_["start_s"] + sh_["dur_s"], t["t0"] + t["dur"]) >
                   max(sh_["start_s"], t["t0"]) for t in true):
                excluded_trailer.add(other)
    short = match_short(true, ids, a, z, by_id, gold, gcrop, shots, protected,
                        srcpath, scrop, drng, excluded_trailer_ids=excluded_trailer)
    dis = pick_distractors(home, shots, N_DISTRACT - len(short),
                           exclude=protected | {s["src_shot"] for s in short
                                                if s.get("src_shot") is not None})
    if repair_cards and len(short) + len(dis) < max(3, len(true)):
        return None, f"{ch['id']}: 只有 {len(short) + len(dis)} 个有效干扰项，不足以构成选片任务"
    cands = [{**t, "answer": True} for t in true] + short + \
            [{"src": srcpath, "t0": bound(*shots[j])[0], "dur": bound(*shots[j])[1],
              "crop": scrop, "origin": "film", "shot": None, "src_shot": j,
              "answer": False} for j in dis]

    rng = random.Random(int(hashlib.sha1(f"{slug}/{ch['id']}".encode()).hexdigest()[:8], 16))
    rng.shuffle(cands)
    if reuse_media:
        if not repair_cards:
            raise ValueError("Audited media reuse only applies to S1 repair")
        from s1_media_reuse import identity, timebase_reuse_allowed
        for candidate in cands:
            ref = reuse_media.get(identity(candidate))
            if ref and timebase_reuse_allowed(ref):
                candidate["reviewed_media_reuse"] = ref
            elif ref:
                candidate["media_reuse_skipped_reason"] = "uncertified_or_non_native_24fps_source_timebase"
    if plan_only:
        return {"chunk": ch["id"], "n": len(cands), "answer": len(true),
                "cards_dropped": len(skipped), "plan": {
                    "candidates": cands, "members": members, "skipped_cards": skipped,
                    "short_distractors": len(short), "neighbour_distractors": len(dis),
                    "protected_source_indices": sorted(protected),
                    "audit_negative_protection": negative_protection,
                    "excluded_alias_trailer_ids": sorted(excluded_trailer)}}, None

    if skip_existing and not text_only:
        # Skip an unchanged, complete task after validating its artifacts; seeded shuffling is deterministic.


        mf = f"{tdir}/initial_state/manifest.json"
        av = f"{tdir}/verifier/answer.json"
        bid = f"{tdir}/verifier/build.json"
        stale = (not os.path.exists(bid)
                 or json.load(open(bid)).get("build") != s1_build_id(repair_cards, card_decisions))
        if stale:
            pass                       # 用别的代码建的，重建
        elif os.path.exists(mf) and os.path.exists(av):
            existing = json.load(open(mf))["candidates"]
            listed = {m["id"] for m in existing}
            disk = {os.path.basename(f)[:-4] for f in glob.glob(f"{cdir}/*.mp4")}
            # Path and count consistency do not establish media integrity.
            # Validate cached candidates fully before trusting --skip-existing.


            media_ok = all(
                _valid_cached_cut(
                    os.path.join(tdir, "initial_state", item["file"]),
                    max(1, int(round(float(item["dur_s"]) * 24))),
                )
                for item in existing
            )
            # Also check the count against what this chunk SHOULD hold. Comparing the
            # manifest to the clips folder only proves the task is internally
            # consistent — a candidate that failed to cut is absent from both, so the
            # pair agrees and the defective task survives every rebuild.
            want = len(cands) if repair_cards else len(true) + len(dis)
            count_ok = (len(listed) == len(existing) == want if repair_cards else len(listed) >= want)
            answer_ok = (set(json.load(open(av))["answer"]) ==
                         {f"c{i:02d}" for i, cand in enumerate(cands, 1) if cand["answer"]}
                         if repair_cards else True)
            if listed and listed == disk and count_ok and answer_ok and media_ok:
                return {"chunk": ch["id"], "n": len(listed),
                        "answer": len(json.load(open(av))["answer"]),
                        "cards_dropped": 0, "skipped": True}, None

    if text_only:
        # Rewrite task text through the shared builder path without re-cutting media.
        
        
        if not os.path.exists(f"{tdir}/initial_state/manifest.json"):
            return None, f"{ch['id']}: 还没建过，不能只重写文本"
        mf = json.load(open(f"{tdir}/initial_state/manifest.json"))
        made = mf["candidates"]
        answer = json.load(open(f"{tdir}/verifier/answer.json"))["answer"]
        keymap = json.load(open(f"{tdir}/verifier/answer.json"))["key"]
    else:
        made, answer, keymap = _cut_all(cands, cdir, mute=repair_cards, strict=repair_cards)
    if not text_only and len(answer) < (1 if repair_cards else 3):
        return None, f"{ch['id']}: 正确项只切出 {len(answer)} 个"
    if not text_only:
        # what the agent can see must be exactly what the manifest lists
        on_disk = {os.path.basename(f)[:-4] for f in glob.glob(f"{cdir}/*.mp4")}
        listed = {m["id"] for m in made}
        if on_disk != listed:
            return None, (f"{ch['id']}: clips 目录与清单不一致，"
                          f"多出 {sorted(on_disk-listed)[:3]} 缺 {sorted(listed-on_disk)[:3]}")

    # ---- task files ------------------------------------------------------
    tid = f"{slug}_s1_{ch['id']}"
    json.dump({"id": tid, "seed": "clip_selection", "stage": "s1/clip_selection",
               "film": slug, "chunk": ch["id"], "beat": ch.get("beat"),
               "deliverables": ["output/selection.json"]},
              open(f"{tdir}/task.json", "w"), ensure_ascii=False, indent=1)

    json.dump({"taskId": tid, "n_candidates": len(made),
               # Describe only properties guaranteed by the current candidate construction.


               "note": ("Every supplied candidate can be inspected in full; all are "
                       "silent and identically normalized. Spans may include transitions."
                       if repair_cards else "Every candidate is a complete shot at its source's own shot "
                       "boundaries: silent, identically transcoded, numbered after a "
                       "shuffle."),
               "candidates": made},
              open(f"{tdir}/initial_state/manifest.json", "w"), ensure_ascii=False, indent=1)

    open(f"{tdir}/initial_state/SETUP.md", "w").write(
        f"# Material\n\n`clips/` holds {len(made)} candidate clips; `manifest.json` lists their\n"
        "durations. They are silent — picture only.\n\n"
        "Inspect them however you like: extract frames, build contact sheets, compare\n"
        "them frame by frame.\n")

    count_rule = (f"Select exactly **{len(answer)} clips**. Over-picking and under-picking are both penalized."
                  if repair_cards else "The count is not given. Decide it yourself; over-picking costs as much as\n  under-picking.")
    clip_rule = ("Every supplied candidate can be inspected **in full**, including its beginning and end.\n"
                 "  They are silent and identically normalized. **Judge from the picture.**\n"
                 "- Select narrative footage, including animation, fades and footage with text overlays.\n"
                 "  Standalone logos, title/date/credit cards and wholly blank clips are not selections."
                 if repair_cards else "Every candidate is a **complete shot at its own boundaries** — silent, identically\n  transcoded, numbered after a shuffle. **Judge from the picture.**")
    open(f"{tdir}/task_prompt.md", "w").write(f"""# Task — pick the clips for this stretch

You are the assistant editor on the *{film_name}* trailer. There are **{len(made)}
candidate clips** under `initial_state/clips/`.

This stretch of the trailer covers:

> {ch.get('what_en') or ch['what']}

It sits inside the beat **{ch.get('beat_name_en') or ch.get('beat_name') or ch.get('beat')}**.

## What to do

Screen every candidate and pick the ones this stretch actually uses — the shots that
cut together into the description above. The rest are real footage from the same film
that the trailer did not use here; most are neighbouring shots from the same scenes.

- Order does not matter here. This stage is only about which clips.
- {count_rule}
- {clip_rule}

## Deliverable

Write `output/selection.json`:

```json
{{"selected": ["c03", "c07", "c19"]}}
```

Just the ids you picked, in any order.
""")
    open(f"{tdir}/output/.gitkeep", "w").write("")

    # ---- verifier （agent 看不到） ----------------------------------------
    if not text_only:
        json.dump({"task": tid, "answer": sorted(answer),
                   "n_candidates": len(made), "n_answer": len(answer),
                   "skipped_cards": [s for s, _ in skipped],
                   "key": keymap},
                  open(f"{tdir}/verifier/answer.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"id": tid, "deliverable_kind": "json_set",
               "verifier_class": "mechanical",
               "golden_semantics": "answer_key（机械导出，golden 自评 1.0）",
               "reward": "F1(选中集合, 正确集合)；pass 门槛 0.8",
               "source_evidence": ("S1 isolated repair: see s1_card_classification.json and "
                                  "s1_shadow_provenance.json for reviewed source spans, "
                                  "reference fallback and old/new candidate mapping. "
                                  "Mechanical key consistency alone is not semantic approval."
                                  if repair_cards else f"{slug} 作业段 {ch['id']}：正确项是 p2_refined 把预告镜头"
                                  f"匹配回正片后、包含该时刻的那条正片镜头（按 source_manifest "
                                  f"的边界切）；干扰项是正片里相邻的镜头，同样按自身边界切"),
               "leak_controls": ("Uniform silent encoding and shuffled IDs; known same-take "
                                 "aliases excluded from distractors. Duration balancing is "
                                 "heuristic, not a guarantee of zero statistical leakage. "
                                 "Final public-task-only quality review is required."
                                 if repair_cards else
                                 "全部候选都是正片自然镜头，时长无信号；统一转码/去声/打乱编号；"
                                 "字幕卡整体剔除")},
              open(f"{tdir}/verifier/task_meta.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"task": tid, "type": "mechanical — 集合比对",
               "formula": "读 output/selection.json 的 selected；F1 = 2PR/(P+R)，"
                          "P=|选中∩正确|/|选中|，R=|选中∩正确|/|正确|；pass 0.8",
               "n_answer": len(answer), "n_candidates": len(made),
               "golden_score": 1.0,
               "controls": [
                   {"name": "全选", "expect_f1": round(2 * len(answer) / (len(made) + len(answer)), 3)},
                   {"name": "空集", "expect_f1": 0.0}]},
              open(f"{tdir}/verifier/rubric.json", "w"), ensure_ascii=False, indent=1)
    shutil.copy(f"{CODE_ROOT}/grade_s1.py", f"{tdir}/verifier/grade.py")
    # LAST. This file is what --skip-existing trusts, so it must not exist until
    # everything it vouches for does — a build killed midway would otherwise leave a
    # half-written task stamped current and get skipped forever after.
    if repair_cards:
        # Private construction evidence must never enter the model's initial_state.
        json.dump({"decisions": [r for r in card_decisions or [] if r["shot_id"] in members]},
                  open(f"{tdir}/verifier/s1_card_classification.json", "w"),
                  ensure_ascii=False, indent=1)
    json.dump({"build": s1_build_id(repair_cards, card_decisions), "s1_repair_cards": repair_cards},
              open(f"{tdir}/verifier/build.json", "w"))
    return {"chunk": ch["id"], "n": len(made), "answer": len(answer),
            "cards_dropped": len(skipped)}, None


def build_film(slug, name, text_only=False, skip_existing=False, repair_cards=False,
               card_overrides=None, card_audit_dir=None):
    if repair_cards and text_only:
        raise ValueError("S1 card correction requires a full media/answer rebuild")
    if card_overrides and not repair_cards:
        raise ValueError("S1 card overrides require repair_cards=True")
    T, E = f"{ROOT}/edit/task_{slug}", f"{ROOT}/exports/{slug}"
    script = json.load(open(f"{T}/stages/script.json"))
    inputs = json.load(open(f"{T}/stages/script_inputs.json"))
    raw_inputs = copy.deepcopy(inputs) if repair_cards else None
    mark_cards(script, inputs, slug)
    p2 = load_matches(slug)
    card_decisions = None
    if repair_cards:
        from s1_card_classification import correct_s1_classification, POLICY_VERSION
        inputs, p2, card_decisions = correct_s1_classification(
            script, raw_inputs, inputs, p2, slug, luma=_LUMA,
            match_confidence=_CONF, flat_std=LUMA_FLAT_STD,
            match_minimum=MATCH_MIN["MATCH"], overrides=card_overrides,
            source_manifest=list(csv.DictReader(open(f"{E}/source_manifest.csv"))))
        audit_dir = card_audit_dir or f"{ROOT}/review_evidence/s1/classification"
        os.makedirs(audit_dir, exist_ok=True)
        json.dump({"policy": POLICY_VERSION, "slug": slug, "decisions": card_decisions},
                  open(f"{audit_dir}/{slug}.json", "w"), ensure_ascii=False, indent=1)
        pending = sum(row["review_required"] for row in card_decisions)
        if pending:
            print(f"  [{name}] S1 分类有 {pending} 个镜头需复核，见 {audit_dir}/{slug}.json",
                  flush=True)
    units = {u["id"]: u for u in json.load(open(f"{T}/s3_units.json"))["units"]}
    pb = json.load(open(f"{E}/probe.json"))
    srcpath = f"{pb['dir']}/{pb['source']}"
    shots = source_shots(slug)
    beat_name = {b["id"]: b["name"] for b in script["beats"]}
    beat_en = {b["id"]: b.get("name_en") for b in script["beats"]}

    rows, errs = [], []
    for ch in script["chunks"]:
        ch = {**ch, "beat_name": beat_name.get(ch.get("beat")),
              "beat_name_en": beat_en.get(ch.get("beat"))}
        r, err = build_chunk(slug, name, ch, script, inputs, p2, units, shots, pb,
                             srcpath, text_only, skip_existing, repair_cards, card_decisions)
        (rows.append(r) if r else errs.append(err))
        if err and os.path.isdir(f"{OUT}/{slug}/s_1/{ch['id']}"):
            shutil.rmtree(f"{OUT}/{slug}/s_1/{ch['id']}")   # 别留半个任务
    nsk = sum(1 for r in rows if r.get("skipped"))
    print(f"  [{name}] {len(rows)}/{len(script['chunks'])} 个任务"
          + (f"（跳过已建 {nsk}）" if nsk else "") + "  "
          f"候选中位 {sorted(r['n'] for r in rows)[len(rows)//2] if rows else 0}  "
          f"正确项中位 {sorted(r['answer'] for r in rows)[len(rows)//2] if rows else 0}"
          + (f"  跳过 {len(errs)}: {errs[0][:60]}" if errs else ""), flush=True)
    return len(rows)


def gc_cache():
    """drop cache entries nothing links to any more

    Every rebuild with different bounds leaves its old cuts behind — the cache is keyed
    on the parameters, so changing them orphans the previous entries rather than
    replacing them. They are invisible: hardlinked files show their full size in the
    cache directory while costing nothing extra, until the last task pointing at them
    is rebuilt and they start costing everything.
    """
    # Not while something is building. A run in progress has torn down the tasks that
    # referenced half the cache and not yet relinked the rest, so a large share of it is
    # momentarily unreferenced — and deleting those means the build re-encodes them, or
    # worse, opens a path another thread is writing.
    if subprocess.run(["pgrep", "-f", "build_s1_tasks|build_s23_tasks"],
                      capture_output=True).stdout.decode().split():
        sys.exit("拒绝清理：正在构建。构建过程中大量缓存会暂时无人引用，"
                 "此时清理等于删掉马上要用的片段。")
    # Track every media hardlink under the task tree, including verifier references,
    # so garbage collection does not discard cache entries still used outside candidate folders.


    def _ino(f):
        try:
            return os.stat(f).st_ino
        except OSError:
            return None          # 构建正在写，枚举与 stat 之间文件可能消失

    live = {i for i in (_ino(f) for f in
                        glob.glob(f"{OUT}/*/s_*/*/**/*.mp4", recursive=True)) if i}
    cached = glob.glob(f"{CACHE}/*.mp4")
    # An empty live set may indicate a moved or incomplete task tree.
    # Refuse destructive cleanup when that cannot be distinguished from genuine orphaning.


    if cached and not live:
        sys.exit(f"拒绝清理：{CACHE} 里有 {len(cached)} 个片段，但 {OUT} 下一个在用的都没找到。"
                 f"\n  任务树是空的，还是路径不对？确认后再跑。")
    doomed = [c for c in cached if os.stat(c).st_ino not in live]
    if cached and len(doomed) > 0.9 * len(cached):
        sys.exit(f"拒绝清理：{len(doomed)}/{len(cached)} 个片段被判为孤儿（超过九成）。"
                 f"\n  重建到一半时这是正常的，但也可能是判断存活的方式坏了。"
                 f"\n  确实要清就先补齐任务树，或手动删 {CACHE}。")
    freed = 0
    for c in doomed:
        freed += os.path.getsize(c)
        os.remove(c)
    print(f"  清理缓存 {len(doomed)} 个孤儿片段，释放 {freed/2**20:.0f} MB")


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--all", action="store_true")
    P.add_argument("-j", "--workers", type=int, default=1)
    P.add_argument("--text-only", action="store_true",
                   help="只按当前 script.json 重写提示词，不重切素材")
    P.add_argument("--skip-existing", action="store_true",
                   help="已经建好且素材完整的作业段直接跳过")
    P.add_argument("--repair-cards", action="store_true",
                   help="仅 S1：应用有证据的实拍/字幕卡纠错并记录私有分类审计；需要完整重建")
    P.add_argument("--card-overrides", help="私有 JSON：{film_slug: {shot_id: {classification, evidence}}}")
    P.add_argument("--gc", action="store_true", help="清掉无人引用的缓存片段后退出")
    A = P.parse_args()
    require_target(P, A)
    if A.repair_cards and A.text_only:
        P.error("--repair-cards 不能与 --text-only 合用：素材和答案必须同步重建")
    if A.card_overrides and not A.repair_cards:
        P.error("--card-overrides 需要 --repair-cards")
    overrides = json.load(open(A.card_overrides)) if A.card_overrides else {}
    if A.gc:
        gc_cache(); return
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
    print(f"建 S1 任务: {len(films)} 部\n", flush=True)
    if A.workers > 1:
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            n = sum(ex.map(lambda kv: build_film(*kv, A.text_only, A.skip_existing,
                                                A.repair_cards, overrides.get(kv[0])), films.items()))
    else:
        n = sum(build_film(s, v, A.text_only, A.skip_existing, A.repair_cards,
                           overrides.get(s)) for s, v in films.items())
    print(f"\n共 {n} 个任务 -> {OUT}/")


if __name__ == "__main__":
    main()
