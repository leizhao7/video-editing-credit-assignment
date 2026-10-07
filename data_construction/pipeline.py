"""Public construction draft: pipeline.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/pipeline.py [options]\n\n  --film\n  --all\n  --films  文本文件，每行一个片名/slug/片段；只跑这些\n  --from\n  --only\n  --modal-dir  Optional override for bundled modal/ adapters\n  -j, --workers\n  --repair  按盘上的产物反推 state.json，不跑任何阶段\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (
        CODE_ROOT, DATA_ROOT, require_target, cloud_volume, env_text, env_positive_int, validate_data_slug
    )
except ImportError:  # Direct script invocation.
    from release_config import (
        CODE_ROOT, DATA_ROOT, require_target, cloud_volume, env_text, env_positive_int, validate_data_slug
    )

import argparse, contextlib, json, math, os, re, shutil, subprocess, sys, threading, time
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Limit stage concurrency according to local encoder, memory, and editor resource requirements.


LOCK_RESOLVE = threading.Lock()
# PROXY_LANES controls concurrent local proxy encodes.


SEM_PROXY = threading.Semaphore(env_positive_int("PROXY_LANES", 2))
SEM_LOCAL = threading.Semaphore(3)      # Limit concurrent local matching jobs.

ROOT = DATA_ROOT
SRC = f"{ROOT}/sources"
SCRATCH = str(Path(DATA_ROOT) / ".scratch")
MODAL = None          # resolved at runtime: the dir holding modal_*.py
STAGES = ["probe", "proxy", "upload", "shots", "fetch", "p2", "cards",
          "audio", "s3", "cutlist", "acuts", "report"]


def slugify(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "", s.lower())
    return s[:20] or "film"


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                          text=True, **kw)


def sh_checked(cmd):
    result = sh(cmd)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {(result.stderr or result.stdout or '')[-500:]}")
    return result


class Film:
    def __init__(self, d):
        if not d or d in (".", "..") or Path(d).name != d or "\\" in d:
            raise ValueError("Film directory must be one source-directory name")
        validate_data_slug(ROOT, slugify(d))
        self.dir = f"{SRC}/{d}"
        self.name = d
        self.slug = slugify(d)
        self.E = f"{ROOT}/exports/{self.slug}"
        os.makedirs(self.E, exist_ok=True)
        self.sf = f"{self.E}/state.json"
        self.state = json.load(open(self.sf)) if os.path.exists(self.sf) else {}

    def mark(self, k, **kw):
        self.state[k] = {"ok": True, "at": time.strftime("%F %T"), **kw}
        json.dump(self.state, open(self.sf, "w"), ensure_ascii=False, indent=1)

    def done(self, k): return self.state.get(k, {}).get("ok")

    # ---------------------------------------------------------------- stages
    def probe(self):
        r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/probe.py",
                "--dir", self.dir, "--slug", self.slug])
        if r.returncode: raise RuntimeError(r.stderr[-400:])
        p = json.load(open(f"{self.E}/probe.json"))
        self.mark("probe", crop=p["crop"], wm=p["wm"], fps_match=p["fps_match"])
        return f"crop={p['crop'] or '无'} wm={'有' if p['wm'] else '无'} " \
               f"{'同帧率' if p['fps_match'] else '帧率不同'}"

    def proxy(self):
        with SEM_PROXY:
            return self._proxy()

    def _proxy(self):
        p = json.load(open(f"{self.E}/probe.json"))
        out = f"{self.dir}/_proxy.mp4"
        # Write a temporary proxy and rename it only after duration validation,
        # so the final path does not expose a partial encode.


        if not _proxy_complete(self):
            tmp = f"{self.dir}/_proxy.part.mp4"
            vf = ((p.get("src_crop_expr", "") + ",") if p.get("src_crop_expr") else "") \
                 + "scale=960:-2"
            r = sh(["ffmpeg", "-y", "-v", "error", "-hwaccel", "videotoolbox",
                    "-i", f"{self.dir}/{p['source']}", "-map", "0:v:0",
                    "-vf", vf, "-c:v", "h264_videotoolbox",
                    "-b:v", "1800k", "-an", tmp])
            if r.returncode:
                if os.path.exists(tmp): os.remove(tmp)
                raise RuntimeError(r.stderr[-400:])
            want = p["src"].get("vdur") or p["src"]["dur"]
            if abs(_dur(tmp) - want) > max(2.0, want * 0.002):
                d = _dur(tmp); os.remove(tmp)
                raise RuntimeError(f"proxy 只有 {d:.0f}s，应为 {want:.0f}s")
            os.replace(tmp, out)
        self.mark("proxy", gb=round(os.path.getsize(out) / 2**30, 2),
                  cropped=bool(p.get("src_crop_expr")))
        return f"{os.path.getsize(out)/2**30:.2f} GB" + (" (已裁黑边)" if p.get("src_crop_expr") else "")

    def upload(self):
        p = json.load(open(f"{self.E}/probe.json"))
        for local, remote in [(f"{self.dir}/_proxy.mp4", "source.mp4"),
                              (f"{self.dir}/{p['golden']}", "golden.mp4")]:
            r = sh(["modal", "volume", "put", cloud_volume(), local,
                    f"/{self.slug}/{remote}"])
            if r.returncode and "already exists" not in (r.stderr or ""):
                raise RuntimeError(r.stderr[-300:])
        self.mark("upload")
        return "ok"

    def shots(self):
        p = json.load(open(f"{self.E}/probe.json"))
        for name, th, crop in [("source", 0.30, 0.0), ("golden", 0.20, p["crop"])]:
            cmd = ["modal", "run", f"{MODAL}/modal_tn_generic.py::detect",
                   "--film", self.slug, "--name", name, "--th", str(th),
                   "--crop", str(crop)]
            if name == "golden" and p.get("crop_expr"):
                cmd += ["--crop-expr", p["crop_expr"]]
            if name == "source": cmd += ["--scale", "854:-2", "--crf", "26"]
            r = sh(cmd)
            if r.returncode: raise RuntimeError(r.stdout[-500:] + r.stderr[-500:])
        self.mark("shots")
        return "ok"

    def _drop_proxy(self):
        """Remove the local upload proxy after confirming the remote source exists.
        The remote copy remains available if preprocessing needs to be rerun."""
        p = f"{self.dir}/_proxy.mp4"
        if not os.path.exists(p):
            return
        r = sh(["modal", "volume", "ls", cloud_volume(), f"/{self.slug}"])
        if r.returncode or "source.mp4" not in (r.stdout or ""):
            return                       # 卷上没有就留着，不冒险
        n = os.path.getsize(p)
        os.remove(p)
        return f"清代理 {n/1e9:.1f}GB"

    def fetch(self):
        """golden clips + the source shot manifest; source clips stay on the volume"""
        sh_checked(["modal", "volume", "get", cloud_volume(), f"/{self.slug}/golden_clips",
            f"{self.E}/", "--force"])
        if os.path.isdir(f"{self.E}/golden_clips/golden_clips"):
            for f in os.listdir(f"{self.E}/golden_clips/golden_clips"):
                shutil.move(f"{self.E}/golden_clips/golden_clips/{f}", f"{self.E}/golden_clips/{f}")
            os.rmdir(f"{self.E}/golden_clips/golden_clips")
        sh_checked(["modal", "volume", "get", cloud_volume(),
            f"/{self.slug}/source_clips/manifest.csv",
            f"{self.E}/source_manifest.csv", "--force"])
        n = len([f for f in os.listdir(f"{self.E}/golden_clips") if f.endswith(".mp4")])
        self.mark("fetch", golden_shots=n)
        return f"{n} 个预告镜头"

    def _afps(self):
        """sampling rate from this film's own shot lengths: enough frames inside the
        shortest shots for temporal consistency to mean anything"""
        import csv as _c
        import numpy as _np
        d = _np.array([float(r["dur_s"]) for r in
                       _c.DictReader(open(f"{self.E}/golden_clips/manifest.csv"))])
        d = d[d > 0.15]
        need = 3.0 / max(_np.percentile(d, 15), 0.2)     # 15th pct shot gets 3 samples
        import math as _m
        return float(min(12.0, max(4.0, _m.ceil(need))))

    def p2(self):
        with SEM_LOCAL:
            return self._p2()

    def _p2(self):
        p = json.load(open(f"{self.E}/probe.json"))
        wm = ",".join(str(x) for x in p["wm"]) if p["wm"] else ""
        base = ["--film", self.slug, "--source", f"{self.dir}/_proxy.mp4",
                "--golden", f"{self.dir}/{p['golden']}", "--crop", str(p["crop"])]
        if p.get("crop_expr"): base += ["--crop-expr", p["crop_expr"]]
        af = self._afps()
        base += ["--afps", str(af)]
        if wm: base += ["--wm", wm]
        r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/p2_match.py", *base,
                "--out", f"{self.E}/p2_final.csv"])
        if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
        r2 = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/p2_refine.py", *base,
                 "--fps", str(p["gold"]["fps"])])
        if r2.returncode: raise RuntimeError(r2.stderr[-400:])
        import csv as _c
        rows = list(_c.DictReader(open(f"{self.E}/p2_refined.csv")))
        m = sum(1 for x in rows if x["verdict"].startswith("MATCH"))
        inl = sorted(int(x["orb_inliers"]) for x in rows
                     if (x.get("orb_inliers") or "").strip().isdigit()
                     and (x.get("refined_src_s") or "").strip())
        med = inl[len(inl) // 2] if inl else 0
        lo = sum(1 for x in inl if x < 30) / len(inl) if inl else 1.0
        self.mark("p2", matched=m, total=len(rows), afps=af,
                  inliers_med=med, low_inlier_frac=round(lo, 3))
        warn = ""
        if inl and (med < 30 or lo > 0.6):
            warn = (f"   ⚠ 匹配质量存疑：inliers 中位 {med}，{lo:.0%} 的命中低于 30。"
                    f"先把几对匹配的画面拿出来"
                    f"看一眼，确认正片和预告是同一部电影、同一个版本。")
        return f"匹配 {m}/{len(rows)}  (采样 {af:.0f}fps  inliers中位 {med}){warn}"

    def cards(self):
        r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/classify_cards.py",
                "--film", self.slug])
        if r.returncode: raise RuntimeError(r.stderr[-400:])
        d = json.load(open(f"{self.E}/card_class.json"))
        nc = sum(1 for v in d.values() if v["kind"] == "card")
        self.mark("cards", cards=nc, live=len(d) - nc)
        return f"卡 {nc} / 实拍 {len(d)-nc}"

    def audio(self):
        for fn in ["modal_transcribe.py::run", "modal_stems.py::split"]:
            r = sh(["modal", "run", f"{MODAL}/{fn}", "--film", self.slug, "--name", "golden"])
            if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
        sh_checked(["modal", "volume", "get", cloud_volume(),
            f"/{self.slug}/golden_transcript.json",
            f"{self.E}/golden_transcript.json", "--force"])
        sh_checked(["modal", "volume", "get", cloud_volume(), f"/{self.slug}/golden_stems",
            f"{self.E}/", "--force"])
        if os.path.isdir(f"{self.E}/golden_stems"):
            os.makedirs(f"{self.E}/stems", exist_ok=True)
            for f in os.listdir(f"{self.E}/golden_stems"):
                shutil.move(f"{self.E}/golden_stems/{f}", f"{self.E}/stems/{f}")
            os.rmdir(f"{self.E}/golden_stems")
        self.mark("audio")
        return "转录+分轨 ok"

    def cutlist(self):
        r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/build_cutlist.py",
                "--film", self.slug])
        if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
        n = json.load(open(f"{ROOT}/edit/task_{self.slug}/stages/cut_list.json"))["n_cuts"]
        self.mark("cutlist", n=n)
        return f"{n} 个切点"

    def acuts(self):
        """Measure speech J/L boundaries and music events after units and the cut list exist."""
        r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/audio_cuts.py", "--film", self.slug])
        if r.returncode: raise RuntimeError(r.stdout[-400:] + r.stderr[-400:])
        d = json.load(open(f"{self.E}/audio_cuts.json"))
        from collections import Counter
        sp = Counter(c["speech"]["type"] for c in d["cuts"])
        self.mark("acuts", **{k: v for k, v in sp.items()})
        return f"语音 J{sp['J']}/L{sp['L']}/hard{sp['hard']}"

    def s3(self):
        with LOCK_RESOLVE, _resolve_lock():   # one Resolve, one writer — across processes too
            r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/build_s3.py",
                    "--film", self.slug])
        if r.returncode: raise RuntimeError(r.stdout[-500:] + r.stderr[-500:])
        self.mark("s3")
        return "unit metadata and media ok"

    def report(self):
        r = sh([sys.executable, "-W", "ignore", f"{CODE_ROOT}/build_report.py",
                "--film", self.slug])
        if r.returncode: raise RuntimeError(r.stdout[-500:] + r.stderr[-500:])
        self.mark("report")
        return "report ok"


# state.json is a convenience, the artifacts are the truth. When the two disagree —
# a stage marker cleared by mistake, a crash between writing output and writing state —
# rebuild the markers from what is actually on disk instead of redoing hours of work.
ARTIFACT = {
    "probe":   lambda F: os.path.exists(f"{F.E}/probe.json"),
    # Check proxy completeness rather than treating file existence as completion.

    "proxy":   lambda F: _proxy_complete(F),
    "upload":  lambda F: os.path.exists(f"{F.E}/source_manifest.csv"),
    "shots":   lambda F: os.path.exists(f"{F.E}/source_manifest.csv"),
    "fetch":   lambda F: os.path.isdir(f"{F.E}/golden_clips")
                         and any(x.endswith(".mp4") for x in os.listdir(f"{F.E}/golden_clips")),
    "p2":      lambda F: os.path.exists(f"{F.E}/p2_refined.csv"),
    "cards":   lambda F: os.path.exists(f"{F.E}/card_class.json"),
    "audio":   lambda F: os.path.exists(f"{F.E}/golden_transcript.json")
                         and os.path.isdir(f"{F.E}/stems"),
    "s3":      lambda F: os.path.exists(f"{ROOT}/edit/task_{F.slug}/s3_units.json"),
    "cutlist": lambda F: os.path.exists(f"{ROOT}/edit/task_{F.slug}/stages/cut_list.json"),
    "acuts":   lambda F: os.path.exists(f"{F.E}/audio_cuts.json"),
    "report":  lambda F: os.path.exists(f"{F.E}/cuts_report.html"),
}


def _dur(p):
    r = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", p])
    if r.returncode:
        raise RuntimeError(f"ffprobe failed for {p}: {(r.stderr or '')[-300:]}")
    try:
        duration = float(r.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned no valid duration for {p}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"ffprobe returned an invalid duration for {p}")
    return duration


def _proxy_complete(F):
    p = f"{F.dir}/_proxy.mp4"
    if not os.path.exists(p): return False
    src = json.load(open(f"{F.E}/probe.json"))["src"]
    want = src.get("vdur") or src["dur"]        # picture length, not container length
    return abs(_dur(p) - want) < max(2.0, want * 0.002)


def repair(f):
    got = []
    for k, fn in ARTIFACT.items():
        if f.done(k): continue
        try:
            if fn(f): f.mark(k, at="repair"); got.append(k)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"[{f.slug}] 无法检查 {k}: {exc}", file=sys.stderr, flush=True)
    if got: print(f"[{f.slug}] 恢复 {len(got)}: {' '.join(got)}", flush=True)
    return len(got)


def _resolve(q, dirs):
    """Resolve --film as a directory name, a slug, or a fragment of either."""
    q = q.strip()
    if not q:
        raise ValueError("Film selector must not be blank")
    ql = q.lower()
    for pick in (lambda d: d == q,
                 lambda d: slugify(d) == ql,
                 lambda d: ql in d.lower() or ql in slugify(d)):
        hit = [d for d in dirs if pick(d)]
        if len(hit) == 1: return hit
        if len(hit) > 1:
            sys.exit(f"--film {q!r} 匹配到多部: {hit}")
    sys.exit(f"--film {q!r} 匹配不到任何片。可选:\n  " + "\n  ".join(
        f"{d}   (slug: {slugify(d)})" for d in dirs))


@contextlib.contextmanager
def _resolve_lock(timeout=1800):
    """Serialize editor access across processes using a filesystem lock."""
    p = f"{ROOT}/.resolve.lock"
    t0 = time.time()
    while True:
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try: pid = int(open(p).read().strip() or 0)
            except Exception: pid = 0
            if not pid or not _alive(pid):
                try: os.unlink(p)
                except OSError: pass
                continue
            if time.time() - t0 > timeout:
                raise RuntimeError("等 Resolve 锁超时")
            time.sleep(3)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        yield
    finally:
        try: os.unlink(p)
        except OSError: pass


def _alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def _claim(f):
    """Use a filesystem lock to prevent concurrent pipelines from modifying one film."""
    p = f"{f.E}/.lock"
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try: pid = int(open(p).read().strip() or 0)
        except Exception: pid = 0
        if pid and _alive(pid):
            return None
        os.unlink(p)                                  # 锁主已死，接管
        try: fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError: return None
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return p


def run(f, stages):
    lock = _claim(f)
    if lock is None:
        print(f"[{f.slug}] 跳过: 另一个进程正在处理", flush=True)
        return False
    try:
        return _run(f, stages)
    finally:
        try: os.unlink(lock)
        except OSError: pass


def _run(f, stages):
    print(f"[{f.slug}] 开始", flush=True)
    for s in stages:
        if f.done(s):
            continue
        t0 = time.time()
        try:
            msg = getattr(f, s)()
            print(f"[{f.slug}] {s:<8} {time.time()-t0:6.0f}s  {msg}", flush=True)
        except Exception as e:
            print(f"[{f.slug}] {s:<8} 失败: {str(e)[:300]}", flush=True)
            f.state[s] = {"ok": False, "err": str(e)[:500], "at": time.strftime("%F %T")}
            json.dump(f.state, open(f.sf, "w"), ensure_ascii=False, indent=1)
            return False
    return True


def main():
    global MODAL
    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--all", action="store_true")
    P.add_argument("--films", help="文本文件，每行一个片名/slug/片段；只跑这些")
    P.add_argument("--from", dest="start", choices=STAGES, default="probe")
    P.add_argument("--only", default=None)
    P.add_argument("--modal-dir", default="")
    P.add_argument("-j", "--workers", type=int, default=1)
    P.add_argument("--repair", action="store_true",
                   help="按盘上的产物反推 state.json，不跑任何阶段")
    A = P.parse_args()
    require_target(P, A)

    # Bundled generic adapters are the default; explicit paths can override them.
    MODAL = str(Path(A.modal_dir.strip() or env_text("VIDEO_EDITING_MODAL_DIR")
                     or str(Path(CODE_ROOT) / "modal")).expanduser().resolve())
    stages = [value.strip() for value in A.only.split(",")] if A.only is not None else STAGES[STAGES.index(A.start):]
    invalid = [value for value in stages if value not in STAGES]
    if invalid:
        P.error(f"Unknown or empty stages: {invalid}")
    if len(set(stages)) != len(stages):
        P.error("Do not repeat stages in --only")
    needs_cloud = not A.repair and bool(set(stages) & {"upload", "shots", "fetch", "audio"})
    if needs_cloud:
        cloud_volume()
    missing = [m for m in ("modal_tn_generic.py", "modal_transcribe.py", "modal_stems.py")
               if needs_cloud and not os.path.exists(f"{MODAL}/{m}")]
    if missing: sys.exit(f"{MODAL} 缺少: {missing}")

    all_films = sorted(d for d in os.listdir(SRC)
                       if os.path.isdir(f"{SRC}/{d}"))
    by_slug = {}
    for directory in all_films:
        by_slug.setdefault(slugify(directory), []).append(directory)
    collisions = [names for names in by_slug.values() if len(names) > 1]
    if collisions:
        P.error(f"Source directory names produce colliding export slugs: {collisions}")
    if A.films:
        entries = Path(A.films).expanduser().read_text(encoding="utf-8").splitlines()
        films = list(dict.fromkeys(_resolve(x.strip(), all_films)[0] for x in entries if x.strip()))
        if not films:
            P.error("Film list is empty")
    elif A.film:
        films = _resolve(A.film, all_films)
    else:
        films = all_films
    if A.repair:
        n = sum(repair(Film(d)) for d in films)
        print(f"共恢复 {n} 个阶段标记")
        return
    print(f"{len(films)} 部片 × {len(stages)} 阶段   modal={MODAL}")
    t0 = time.time()
    if A.workers > 1:
        print(f"并行 {A.workers} 路   proxy≤2 · 本地≤3 · Resolve 串行", flush=True)
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            ok = sum(ex.map(lambda d: run(Film(d), stages), films))
    else:
        ok = sum(run(Film(d), stages) for d in films)
    print(f"\n完成 {ok}/{len(films)}   墙钟 {(time.time()-t0)/60:.0f} 分")
    if ok != len(films):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
