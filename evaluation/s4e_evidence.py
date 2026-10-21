"""Build trusted render-clock audio, continuous video and navigational measurements.
Source observations/declarations never substitute for the actual delivered render.
"""
import argparse, atexit, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile, warnings
from grading_contract import validate_task
from grading_video import s4_video, VIDEO_CONTRACT
from grading_media import render_audio, map_time, unexplained_black, verified_bridges
_RENDER_AUDIO = {}

if __name__ != "__main__":
    raise ImportError(f"{__file__} is a command-line script, not an importable module (it writes files and starts subprocesses)")

P = argparse.ArgumentParser()
P.add_argument("--task")
P.add_argument("--tasks")
P.add_argument("--asr", default="small.en")
P.add_argument("--force", action="store_true")
A = P.parse_args()

warnings.filterwarnings("ignore")
_ASR = []
ASR_CONTRACT = "s12-asr-v5-render-clock"
_CANNED_HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "please subscribe",
    "like and subscribe", "subtitles by the amara org community",
}
_HEALTH = {
    "format": "s12-evidence-health-v3",
    "asr": {
        "status": "ok",
        "model": A.asr,
        "contract": ASR_CONTRACT,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "cache_hits": 0,
        "shared_cache_hits": 0,
        "shared_cache_writes": 0,
        "runtime_transcriptions": 0,
        "empty_results": 0,
        "filtered_hallucination_segments": 0,
        "legacy_cache_bypasses": 0,
        "cache_errors": [],
    },
    "audio_analysis": {
        "status": "ok",
        "librosa_calls": 0,
        "empty_windows": 0,
    },
}
_TMP_DIRS = []
# Require the confirmation threshold before treating soundtrack correlation
# as evidence that the declared cue appears in the final mix.
MUSIC_RENDER_CONFIRM_CORR = 0.15


def _tempdir(prefix):
    path = tempfile.mkdtemp(prefix=prefix)
    _TMP_DIRS.append(path)
    return path


@atexit.register
def _cleanup_tempdirs():
    for path in _TMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


def _bind_evidence_slot():
    """Keep one evidence process and all of its FFmpeg children CPU-bounded."""
    raw = os.environ.get("S12_EVIDENCE_SLOT")
    if raw is None or not hasattr(os, "sched_getaffinity"):
        return
    try:
        allowed = sorted(os.sched_getaffinity(0))
        per_slot = max(1, int(os.environ.get("S12_EVIDENCE_CPUS_PER_SLOT", "2")))
        start = (int(raw) * per_slot) % len(allowed)
        selected = {allowed[(start + i) % len(allowed)] for i in range(per_slot)}
        os.sched_setaffinity(0, selected)
    except (OSError, TypeError, ValueError, ZeroDivisionError):
        return


_bind_evidence_slot()


def _is_canned_hallucination(segment) -> bool:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(getattr(segment, "text", "")).lower())
    text = " ".join(text.split())
    try:
        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
    except (TypeError, ValueError):
        no_speech = 0.0
    return text in _CANNED_HALLUCINATIONS and no_speech >= 0.20


def asr(path, words=False):
    """Prefer a precomputed <media>.asr.json cache with full text and word timings.

    ASR depends on the media, not the policy output; caching avoids repeated work
    during rollouts. A cache miss must run ASR. Dependency, initialization and
    transcription failures propagate; they must not imply dialogue is absent.
    An empty transcript is valid only after successful decoding/transcription."""
    cache = os.path.splitext(path)[0] + ".asr.json"
    if os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as fh:
                d = json.load(fh)
            if not isinstance(d, dict) or d.get("contract") != ASR_CONTRACT:
                raise ValueError("legacy/unversioned ASR cache")
            result = d.get("words", []) if words else d.get("text", "")
            if words and not isinstance(result, list):
                raise TypeError("ASR cache words must be a list")
            if not words and not isinstance(result, str):
                raise TypeError("ASR cache text must be a string")
            _HEALTH["asr"]["cache_hits"] += 1
            if not result:
                _HEALTH["asr"]["empty_results"] += 1
            return result
        except Exception as exc:  # fall back to live ASR, but expose corruption
            if str(exc) == "legacy/unversioned ASR cache":
                _HEALTH["asr"]["legacy_cache_bypasses"] += 1
            else:
                _HEALTH["asr"]["cache_errors"].append(
                    f"{os.path.basename(cache)}: {type(exc).__name__}: {exc}"
                )
    shared_cache = None
    cache_root = os.environ.get("S12_ASR_CACHE_DIR", "").strip()
    real = os.path.realpath(path)
    # See the S3 counterpart: cache only trusted initial_state media.  The
    # submitted output/preview must always be transcribed per trajectory.
    source_media = "/initial_state/" in path.replace(os.sep, "/")
    if cache_root and source_media:
        stat = os.stat(real)
        mode = "words" if words else "text"
        key = hashlib.sha256(
            f"s4e-{mode}\0{ASR_CONTRACT}\0{A.asr}\0{real}\0"
            f"{stat.st_size}\0{stat.st_mtime_ns}".encode()
        ).hexdigest()
        shared_cache = os.path.join(cache_root, "s4e", key[:2], key + ".json")
        if os.path.isfile(shared_cache):
            try:
                with open(shared_cache, encoding="utf-8") as handle:
                    wrapper = json.load(handle)
                if (not isinstance(wrapper, dict)
                        or wrapper.get("contract") != ASR_CONTRACT
                        or wrapper.get("mode") != mode):
                    raise TypeError("shared ASR cache contract mismatch")
                result = wrapper.get("result")
                expected = list if words else str
                if not isinstance(result, expected):
                    raise TypeError(
                        f"shared ASR cache expected {expected.__name__}, "
                        f"got {type(result).__name__}"
                    )
                _HEALTH["asr"]["shared_cache_hits"] += 1
                if not result:
                    _HEALTH["asr"]["empty_results"] += 1
                return result
            except Exception as exc:
                _HEALTH["asr"]["cache_errors"].append(
                    f"shared/{key}: {type(exc).__name__}: {exc}"
                )
    if not _ASR:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "S4E ASR unavailable: install faster-whisper in the grader image"
            ) from exc
        try:
            _ASR.append(WhisperModel(A.asr, device="cpu", compute_type="int8"))
        except Exception as exc:
            raise RuntimeError(
                f"S4E ASR model {A.asr!r} failed to initialise: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    try:
        segs, _ = _ASR[0].transcribe(
            path,
            word_timestamps=words,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        # Decoding happens lazily during iteration; consume inside the try.
        if not words:
            kept = []
            for s in segs:
                if _is_canned_hallucination(s):
                    _HEALTH["asr"]["filtered_hallucination_segments"] += 1
                    continue
                kept.append(s.text)
            result = " ".join(kept).strip()
        else:
            result = []
            for s in segs:
                if _is_canned_hallucination(s):
                    _HEALTH["asr"]["filtered_hallucination_segments"] += 1
                    continue
                for w in (s.words or []):
                    if w.word.strip():
                        result.append({"w": w.word.strip(), "s": round(w.start, 3),
                                       "e": round(w.end, 3)})
    except Exception as exc:
        raise RuntimeError(
            f"S4E ASR failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _HEALTH["asr"]["runtime_transcriptions"] += 1
    if not result:
        _HEALTH["asr"]["empty_results"] += 1
    if shared_cache:
        os.makedirs(os.path.dirname(shared_cache), exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(shared_cache) + ".",
            suffix=".tmp", dir=os.path.dirname(shared_cache),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"contract": ASR_CONTRACT, "mode": mode, "result": result},
                      handle, ensure_ascii=False)
        os.replace(tmp, shared_cache)
        _HEALTH["asr"]["shared_cache_writes"] += 1
    return result


def has_audio_stream(path):
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        raise RuntimeError(f"ffprobe audio check failed for {path}: {run.stderr[-300:]}")
    return bool(run.stdout.strip())


def _audio_libs():
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "S4E audio analysis unavailable: install librosa and numpy in the grader image"
        ) from exc
    return np, librosa


def _parse_volumedetect(stderr, source="audio"):
    """Parse FFmpeg volume output without turning digital silence into 0 dB."""
    if re.search(r"n_samples:\s*0(?:\s|$)", stderr):
        return {"mean_db": None, "peak_db": None, "mean_status": "no_audio_samples",
                "peak_status": "no_audio_samples", "status": "no_audio_samples"}
    token = r"(?:-?inf|nan|-?(?:\d+(?:\.\d*)?|\.\d+))"

    def measured(label):
        match = re.search(rf"{label}:\s*({token})", stderr, flags=re.IGNORECASE)
        if not match:
            raise RuntimeError(f"volumedetect omitted {label} for {source}")
        raw = match.group(1).lower()
        if raw == "-inf":
            return None, "silence"
        if raw in ("inf", "nan"):
            raise RuntimeError(f"volumedetect returned invalid {label}: {raw}")
        value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError(f"volumedetect returned non-finite {label}: {raw}")
        return value, "measured"

    mean, mean_status = measured("mean_volume")
    peak, peak_status = measured("max_volume")
    return {"mean_db": mean, "peak_db": peak,
            "mean_status": mean_status, "peak_status": peak_status}


def level_db(path, ss=None, t=None):
    """Measure mean and peak audio levels; a declared +14 dB gain does not establish audibility."""
    path = _RENDER_AUDIO.get(str(path), path)
    c = ["ffmpeg", "-v", "info"]
    if ss is not None:
        c += ["-ss", f"{max(0.0, ss):.3f}"]
    if t is not None:
        c += ["-t", f"{max(0.05, t):.3f}"]
    c += ["-i", path, "-af", "volumedetect", "-f", "null", "-"]
    run = subprocess.run(c, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"volumedetect failed for {path}: {run.stderr[-300:]}")
    return _parse_volumedetect(run.stderr, path)


def luma_series(path, fps=24):
    """Measure frame luminance to inspect whether black frames occur and how long they last."""
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf",
                        f"fps={fps},scale=32:18", "-pix_fmt", "gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    k = 32 * 18
    if a.size < k:
        return None
    fr = a[: (a.size // k) * k].reshape(-1, k).mean(axis=1)
    return {"fps": fps, "v": [round(float(x), 1) for x in fr]}


def music_accents(path, offset, dur, top=14):
    """Map music accents to the task timeline.

    offset is the declared alignment: cue time offset corresponds to timeline
    time zero. Subtract it before comparing cue events with picture cuts."""
    np, librosa = _audio_libs()
    try:
        y, sr = librosa.load(path, sr=22050, offset=max(0.0, offset), duration=dur + 0.5)
    except Exception as exc:
        raise RuntimeError(
            f"music analysis failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _HEALTH["audio_analysis"]["librosa_calls"] += 1
    if y.size < sr // 4:
        _HEALTH["audio_analysis"]["empty_windows"] += 1
        return None
    hop = 256
    on = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    t = librosa.frames_to_time(np.arange(len(on)), sr=sr, hop_length=hop)
    if not len(on):
        return None
    idx = np.argsort(on)[::-1]
    picked, used = [], []
    for i in idx:
        if len(picked) >= top:
            break
        # An accent can span several frames; keep the strongest event within 60 ms.
        if any(abs(t[i] - u) < 0.06 for u in used):
            continue
        used.append(t[i])
        picked.append({"t_s": round(float(t[i]), 3),
                       "strength": round(float(on[i]), 2),
                       "pct": round(100 * float((on < on[i]).mean()))})
    picked.sort(key=lambda x: x["t_s"])
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    trms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    env = [{"t_s": round(float(trms[i]), 2),
            "db": round(float(20 * np.log10(rms[i] + 1e-9)), 1)}
           for i in range(0, len(rms), max(1, len(rms) // 24))]
    return {"accents": picked, "envelope": env,
            "note": "t_s uses the task timeline (file time minus the declared alignment offset)."}


def music_render_match(music_path, offset, render_path, dur):
    """Measure whether the declared cue segment is acoustically present in the render.

    The cue envelope alone only describes what the JSON *claims* was placed.  Compare the
    actual render against that source, allowing a small codec delay.  Correlation is exposed
    as evidence rather than converted to a hard pass/fail because dialogue and production
    sound legitimately reduce it.
    """
    np, librosa = _audio_libs()
    try:
        cue, sr = librosa.load(
            music_path, sr=8000, mono=True, offset=max(0.0, offset), duration=dur
        )
        rendered, _ = librosa.load(render_path, sr=8000, mono=True, duration=dur)
    except Exception as exc:
        raise RuntimeError(
            f"music/render comparison failed: {type(exc).__name__}: {exc}"
        ) from exc
    _HEALTH["audio_analysis"]["librosa_calls"] += 2
    n = min(len(cue), len(rendered))
    if n < sr // 2:
        _HEALTH["audio_analysis"]["empty_windows"] += 1
        return {"status": "insufficient_audio"}
    cue = cue[:n].astype(np.float64)
    rendered = rendered[:n].astype(np.float64)
    cue -= cue.mean()
    rendered -= rendered.mean()
    cue_energy = float(np.dot(cue, cue))
    render_energy = float(np.dot(rendered, rendered))
    if cue_energy < 1e-10 or render_energy < 1e-10:
        return {"status": "silent_source_or_render"}
    try:
        from scipy.signal import correlate, correlation_lags
    except ImportError as exc:
        raise RuntimeError("S4E music/render comparison requires scipy") from exc
    corr = correlate(rendered, cue, mode="full", method="fft")
    lags = correlation_lags(len(rendered), len(cue), mode="full")
    keep = np.abs(lags) <= int(0.25 * sr)
    if not np.any(keep):
        return {"status": "insufficient_audio"}
    local = corr[keep]
    local_lags = lags[keep]
    pos = int(np.argmax(np.abs(local)))
    best = float(local[pos])
    coeff = best / max(1e-12, (cue_energy * render_energy) ** 0.5)
    gain = best / cue_energy
    return {
        "status": "measured",
        "waveform_correlation": round(abs(coeff), 4),
        "signed_correlation": round(coeff, 4),
        "estimated_codec_lag_s": round(float(local_lags[pos]) / sr, 4),
        "least_squares_gain": round(gain, 5),
        "note": (
            "Waveform correlation between the render and declared cue, allowing +/-0.25s codec lag. "
            "Higher correlation supports the cue being present; dialogue/production sound can lower it, so no single threshold justifies a zero score."
        ),
    }


STOP = set("a an the and or but of to in on at is are was were be been it its this that "
            "i you he she they we me him her them my your his their our for with as so "
            "not no yes do does did have has had will would can could just now then there "
            "here what who how when why all any some one".split())


def words(t):
    """Content-word set for comparing take text with source-shot production sound."""
    return {w for w in re.findall(r"[a-z']+", (t or "").lower())
            if len(w) > 2 and w not in STOP}


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def wave(src, t0, dur, out_png, w=1200, h=110, spec=False):
    """Render waveform or spectrum images alongside picture frames.

    This supplies navigational audio evidence for image-based inspection;
    continuity and spectral content do not prove a particular bridge type."""
    f = (f"showspectrumpic=s={w}x{h}:legend=0:scale=log:color=intensity"
         if spec else f"showwavespic=s={w}x{h}:colors=0xdddddd:split_channels=0")
    src = _RENDER_AUDIO.get(str(src), src)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t0):.3f}",
                        "-t", f"{dur:.3f}", "-i", src, "-filter_complex", f,
                        "-frames:v", "1", out_png], capture_output=True)
    return os.path.exists(out_png) and r.returncode == 0


def cut_roughness(path, t, half=0.23):
    """Measure rendered-audio discontinuity from the waveform, not declarations.

    step_db: level difference between 0.23-second windows beside the join.
    spec_dist: cosine distance between 16 log-frequency energy bands.
    gap_db: quietest 20 ms within 50 ms of the join, relative to neighboring levels.

    These measurements cannot identify bridge type. A continuous music bed can
    dominate the spectrum of a hard cut, while entering/leaving dialogue can
    increase spectral distance during a J/L cut. They describe audio roughness,
    not whether a declared transition was implemented."""
    np, librosa = _audio_libs()
    try:
        y, sr = librosa.load(_RENDER_AUDIO.get(str(path), path), sr=22050, mono=True,
                             offset=max(0.0, t - half - 0.06), duration=2 * half + 0.12)
    except Exception as exc:
        raise RuntimeError(
            f"cut audio analysis failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _HEALTH["audio_analysis"]["librosa_calls"] += 1
    if len(y) < sr * 0.2:
        _HEALTH["audio_analysis"]["empty_windows"] += 1
        return None
    dbf = lambda x: float(20 * np.log10(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-9))

    def band(x):
        S = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        e = np.logspace(np.log10(80), np.log10(min(8000, sr / 2)), 17)
        v = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(e[:-1], e[1:])])
        return v / (v.sum() + 1e-9)

    c = int(min(t, half + 0.06) * sr)
    a = y[max(0, c - int(half * sr)): max(1, c - int(0.02 * sr))]
    b = y[c + int(0.02 * sr): c + int(half * sr)]
    if len(a) < sr * 0.05 or len(b) < sr * 0.05:
        return None
    mid = y[max(0, c - int(0.05 * sr)): c + int(0.05 * sr)]
    w = int(0.02 * sr)
    quiet = (min(dbf(mid[i:i + w]) for i in range(0, max(1, len(mid) - w), w // 2))
             if len(mid) > w else dbf(mid))
    ba, bb = band(a), band(b)
    return {"step_db": round(abs(dbf(a) - dbf(b)), 2),
            "spec_dist": round(float(1 - np.dot(ba, bb) /
                                     (np.linalg.norm(ba) * np.linalg.norm(bb) + 1e-9)), 4),
            "gap_db": round(float((dbf(a) + dbf(b)) / 2 - quiet), 2)}


def seg_db(path, a, b):
    """Return mean audio level for a segment, or None if it cannot be measured."""
    if not path or not os.path.exists(path) or b - a < 0.02:
        return None
    np, librosa = _audio_libs()
    try:
        y, _ = librosa.load(path, sr=22050, mono=True, offset=max(0.0, a), duration=b - a)
    except Exception as exc:
        raise RuntimeError(
            f"segment audio analysis failed for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    _HEALTH["audio_analysis"]["librosa_calls"] += 1
    if len(y) < 200:
        _HEALTH["audio_analysis"]["empty_windows"] += 1
        return None
    return round(float(20 * np.log10(np.sqrt((y.astype(np.float64) ** 2).mean()) + 1e-9)), 2)


def carry_content(is_dir, shot, past_s, is_prelap):
    """Measure whether the source handle extension contains sound.

    Extending a handle by 0.9 seconds can extend silence rather than production
    sound. Compare the extension level with the in-shot level; a much quieter
    extension suggests silence. This source measurement does not verify the render."""
    h = (shot.get("handle") or {})
    pin = h.get("picture_in_s")
    if pin is None or past_s <= 0.04:
        return None
    p = os.path.join(is_dir, "handles", f'{shot["id"]}.m4a')
    dur = shot.get("dur_s") or 0
    if is_prelap:
        a, b = max(0.0, pin - past_s), pin
    else:
        a, b = pin + dur, pin + dur + past_s
    ext, inner = seg_db(p, a, b), seg_db(p, pin, pin + dur)
    if ext is None or inner is None:
        return None
    return {"extended_s": round(past_s, 3), "ext_db": ext, "in_shot_db": inner,
            "rel_db": round(ext - inner, 1),
            "reading": ("The extended source handle is much quieter than the in-shot segment; "
                        "extending silence does not establish an L/J cut." if ext - inner < -12 else
                        "The extended source handle contains sound beyond the picture boundary; verify its use in the render.")}


def carry_opportunity(is_dir, leaving, arriving, win=0.6):
    """Measure source-handle opportunities independently of the delivered edit.

    An L-cut opportunity uses sound after the leaving picture; a J-cut opportunity
    uses sound before the arriving picture. Handles provide up to four seconds of
    context on each side; inspect 0.6-second windows by default. This tells the
    blind judge whether source sound is available to extend, not whether the
    submitted render includes it or whether extending it is artistically required."""
    out = {}
    for role, sh_ in (("carry_out", leaving), ("prelap_in", arriving)):
        h = (sh_ or {}).get("handle") or {}
        pin = h.get("picture_in_s")
        if pin is None:
            continue
        p = os.path.join(is_dir, "handles", f'{sh_["id"]}.m4a')
        dur = sh_.get("dur_s") or 0
        if role == "carry_out":
            a, b = pin + dur, pin + dur + win
        else:
            a, b = max(0.0, pin - win), pin
        ext, inner = seg_db(p, a, b), seg_db(p, pin, pin + dur)
        if ext is None or inner is None:
            continue
        rel = round(ext - inner, 1)
        out[role] = {"shot": sh_["id"], "handle_db": ext, "in_shot_db": inner,
                     "rel_db": rel,
                     "supports": rel > -12,
                     "reading": ("Source sound is available beyond the picture boundary and could be extended across the join."
                                 if rel > -12 else
                                 "The source handle beyond the picture boundary is mostly quiet, with little sound to extend.")}
    if out:
        out["note"] = ("This measures source material independently of the submission. supports=true indicates "
                       "a source-sound extension opportunity that a hard cut would leave unused; it does not prove a rendered bridge.")
    return out or None


def black_runs(path):
    """Use FFmpeg blackdetect to distinguish black intervals from ordinarily dark shots."""
    if not path or not os.path.exists(path):
        return []
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-an",
                        "-threads", "1", "-filter_threads", "1",
                        "-vf", "blackdetect=d=0.05:pix_th=0.06", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=90)
    if r.returncode:
        raise RuntimeError(f"black detector decode failed: {r.stderr[-200:]}")
    out = []
    for m in re.finditer(r"black_start:([0-9.]+) black_end:([0-9.]+)", r.stderr):
        a, b = float(m.group(1)), float(m.group(2))
        out.append({"start_s": round(a, 3), "end_s": round(b, 3), "dur_s": round(b - a, 3)})
    return out


def render_offsets(runs_in, joins, drift, expected_black_spans=None):
    """Map rough-cut joins to rendered time using extra black intervals.

    Only black time added beyond the rough cut can shift the clock. A title card
    already included in the task duration must not be counted again. Bound all
    offsets by positive duration drift; otherwise an existing three-second black
    title card could incorrectly shift all later evidence by three seconds."""
    zero = {j["join"]: 0.0 for j in joins}
    try:
        numeric_drift = float(drift)
    except (TypeError, ValueError):
        numeric_drift = 0.0
    positive_drift = max(0.0, numeric_drift)
    if not runs_in or positive_drift < 0.02:
        reason = ("The render is not longer than the rough cut; black frames cannot add inserted duration."
                  if numeric_drift < -0.02 else
                  "The render and rough cut have equal duration; waveform windows use rough-cut times.")
        return zero, reason

    expected = []
    for span in expected_black_spans or []:
        if isinstance(span, dict):
            start, end = span.get("start_s"), span.get("end_s")
        else:
            try:
                start, end = span[:2]
            except (TypeError, ValueError):
                continue
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end > start:
            expected.append((start, end))

    def subtract(segments, cut_start, cut_end):
        out = []
        for start, end in segments:
            if cut_end <= start or cut_start >= end:
                out.append((start, end))
                continue
            if start < cut_start:
                out.append((start, cut_start))
            if cut_end < end:
                out.append((cut_end, end))
        return out

    # Walk render-time black runs in order.  ``accepted`` is the amount of
    # previously detected insertion and therefore the shift of later expected
    # title-card spans.  A small edge tolerance absorbs encoder/frame rounding.
    extra_runs, accepted, ignored_expected = [], 0.0, 0.0
    for run in sorted(runs_in, key=lambda item: float(item.get("start_s", 0))):
        try:
            start = float(run["start_s"])
            end = float(run.get("end_s", start + float(run["dur_s"])))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        pieces = [(start, end)]
        before = end - start
        for card_start, card_end in expected:
            pieces = subtract(
                pieces,
                card_start + accepted - 0.08,
                card_end + accepted + 0.08,
            )
        remaining = sum(piece_end - piece_start for piece_start, piece_end in pieces)
        ignored_expected += max(0.0, before - remaining)
        for piece_start, piece_end in pieces:
            available = positive_drift - accepted
            if available <= 0:
                break
            duration = min(piece_end - piece_start, available)
            if duration >= 0.02:
                extra_runs.append((piece_start, duration))
                accepted += duration

    if not extra_runs:
        return zero, (
            f"All detected black intervals are accounted for by rough-cut title cards; no join offset added "
            f"({ignored_expected:.3f}s of already-budgeted black time excluded)."
        )

    off, used = {}, 0.0
    for join in sorted(joins, key=lambda item: item["t_s"]):
        acc = 0.0
        for _ in range(4):
            nxt = min(
                positive_drift,
                sum(duration for start, duration in extra_runs
                    if start < join["t_s"] + acc),
            )
            if abs(nxt - acc) < 0.01:
                break
            acc = nxt
        off[join["join"]] = round(acc, 3)
        used = max(used, acc)
    tail = round(numeric_drift - used, 3)
    return off, (f"The render is {numeric_drift:+.3f}s longer than the rough cut; offsets are bounded by this positive drift. "
                 f"After excluding {ignored_expected:.3f}s of rough-cut title-card black time, "
                 f"{used:.3f}s of extra black time precedes joins; the remaining {tail:+.3f}s does not shift joins.")


def bridge_decision_count(bridges, placed):
    """Count editorial decisions, deduplicating a card that touches two joins."""
    spanning_takes = {line["take"] for line in placed
                      if any(x.get("status") == "confirmed" for x in line.get("crossing_confirmations", {}).values())}
    title_card_shots = {
        (bridge.get("measurements") or {}).get("card", {}).get("shot")
        for bridge in bridges if "title_card" in bridge.get("bridge_kinds", [])
    } - {None}
    count = (len(spanning_takes)
             + sum(1 for bridge in bridges
                   if any(kind.startswith("picture:")
                          for kind in bridge.get("bridge_kinds", [])))
             + int(any("sound:music_bridge" in bridge.get("bridge_kinds", [])
                       for bridge in bridges))
             + len(title_card_shots))
    return count, spanning_takes, title_card_shots


def music_timeline(shots, joins, placed, mus, render_path, is_dir, out_png, w=1500):
    """Draw aligned picture, render waveform/spectrum, cue envelope/accents and dialogue.

    A shared timeline gives the judge observable evidence beyond authored notes:
    cue placement, accent timing and dialogue occupancy. The cue envelope comes
    from the declared source cue; it is not isolated music from the rendered mix."""
    from PIL import Image, ImageDraw
    tot = mus.get("render_duration_s") or max(s["end_s"] for s in shots) or 1.0
    FH, WH, SH, MH, DH, LH = 92, 96, 96, 116, 30, 22
    X = lambda t: int(max(0.0, min(1.0, t / tot)) * w)
    tmp = _tempdir("s4mt")
    im = Image.new("RGB", (w, FH + WH + SH + MH + DH + LH * 2 + 10), (14, 14, 14))
    d = ImageDraw.Draw(im)
    render_audio = bool(render_path and has_audio_stream(render_path))

    # 1. Picture strip: each shot's width represents its duration.
    y = 0
    for sh_ in shots:
        x0, x1 = X(sh_["start_s"]), X(sh_["end_s"])
        q = f'{tmp}/{sh_["id"]}.png'
        src = os.path.join(is_dir, "shots", f'{sh_["id"]}.mp4')
        seek = (sh_["end_s"] - sh_["start_s"]) * 0.5
        if not os.path.exists(src) and render_path:
            src = render_path
            seek = (sh_.get("render_start_s", sh_["start_s"]) +
                    sh_.get("render_end_s", sh_["end_s"])) * 0.5
        if os.path.exists(src):
            run = sh(["ffmpeg", "-v", "error", "-ss", f'{seek:.3f}',
                "-i", src, "-frames:v", "1",
                "-vf", f"scale={max(2, x1 - x0)}:{FH}:force_original_aspect_ratio=increase,"
                       f"crop={max(2, x1 - x0)}:{FH}", "-y", q])
            if run.returncode == 0 and os.path.isfile(q) and os.path.getsize(q) > 0:
                im.paste(Image.open(q).convert("RGB"), (x0, y))
            else:
                # Preserve the shot's timeline slot when its nominal midpoint
                # falls outside the actual encoded frames.  The red tile is
                # visible evidence; aborting here would misclassify a model or
                # media-boundary condition as a grader failure.
                d.rectangle([x0, y, max(x0 + 1, x1), y + FH], fill=(90, 16, 16))
                d.text(
                    (x0 + 3, y + FH // 2), "NO FRAME",
                    fill=(255, 210, 210),
                )
        d.text((x0 + 3, y + 3), sh_["id"], fill=(255, 255, 255))
    y += FH

    for is_spec, hh in ((False, WH), (True, SH)):
        q = f'{tmp}/a{int(is_spec)}.png'
        if render_audio:
            if not wave(render_path, 0.0, tot, q, w, hh, is_spec):
                d.rectangle([0, y, w, y + hh], fill=(45, 8, 8))
                d.text(
                    (8, y + 8), "NO DECODABLE AUDIO IN TIMELINE WINDOW",
                    fill=(255, 170, 170),
                )
            else:
                im.paste(Image.open(q).convert("RGB"), (0, y))
        else:
            d.rectangle([0, y, w, y + hh], fill=(8, 8, 8))
            d.text((8, y + 8), "NO AUDIO STREAM IN RENDER", fill=(255, 120, 120))
        y += hh

    # 4. Music bed: envelope line and accent markers scaled by strength percentile.
    my, env = y, (mus.get("envelope") or [])
    d.rectangle([0, my, w, my + MH], fill=(20, 20, 26))
    if env:
        lo = min(e["db"] for e in env); hi = max(e["db"] for e in env)
        rng = max(1e-6, hi - lo)
        pts = [(X(e["t_s"]), my + MH - 6 - int((e["db"] - lo) / rng * (MH - 14))) for e in env]
        d.line(pts, fill=(90, 200, 255), width=2)
        d.text((6, my + 3), f"declared source cue  {lo:.0f} to {hi:.0f} dB",
               fill=(140, 200, 240))
    for a in (mus.get("accents") or []):
        x = X(a["t_s"]); h = int((a.get("pct", 50) / 100.0) * (MH - 10))
        d.line([(x, my + MH), (x, my + MH - h)], fill=(255, 190, 60), width=2)
    y = my + MH

    # 5. Dialogue occupancy.
    d.rectangle([0, y, w, y + DH], fill=(18, 24, 18))
    for pl in placed:
        sp = pl.get("timeline_span_s") or []
        if len(sp) == 2:
            x0, x1 = X(sp[0]), X(sp[1])
            if x1 > x0:
                d.rectangle([x0, y + 4, x1, y + DH - 4], fill=(70, 160, 90))
                d.text((x0 + 4, y + 8), str(pl.get("take", "")),
                       fill=(240, 255, 240))
            else:
                # Invalid submissions (for example out_s < in_s) are evidence
                # for the judge, not a reason for the evidence renderer to
                # crash and turn an outcome into an infrastructure failure.
                d.line([(x0, y + 4), (x0, y + DH - 4)],
                       fill=(255, 80, 80), width=3)
                d.text((x0 + 4, y + 8),
                       f'{pl.get("take", "")} invalid span',
                       fill=(255, 150, 150))
    if not placed:
        d.text((6, y + 8), "no dialogue placed", fill=(150, 170, 150))
    y += DH

    # Join markers span every layer.
    for j in joins:
        x = X(j["t_s"])
        d.line([(x, 0), (x, y)], fill=(235, 60, 60), width=2)
        d.text((x + 2, FH - 15), f'j{j["join"]}', fill=(255, 130, 130))
    d.rectangle([0, y, w, y + LH * 2], fill=(14, 14, 14))
    d.text((6, y + 4), f'rows: picture / render waveform / render spectrogram / '
                       f'declared source-cue envelope (blue) with accents (yellow, height = strength '
                       f'percentile) / dialogue spans (green)', fill=(225, 225, 225))
    d.text((6, y + 4 + LH), f'red verticals = the {len(joins)} picture cuts.  '
                            f'full stretch = {tot:.3f}s.  blue/yellow use the declared cue offset; '
                            f'render_match separately checks whether that cue reached the mix.',
           fill=(225, 225, 225))
    im.save(out_png)
    return True


def render_panels(render_path, joins, runs, drift, out_png, w=1188,
                  expected_black_spans=None):
    """Build aligned picture/audio panels from the actual delivered render.

    Unlike source-based join_panels, these show implemented black frames, fades
    and timing shifts. Declared cut/under/prelap fields do not substitute for
    rendered evidence. Mapping remains heuristic and requires visual checking."""
    from PIL import Image, ImageDraw
    if not render_path or not joins:
        return False
    HALF, TH, SH = 0.55, 84, 84
    offs, _ = render_offsets(runs, joins, drift, expected_black_spans)
    render_audio = has_audio_stream(render_path)
    tw, th = w // 6, int(w // 6 * 9 / 16)
    tmp = _tempdir("s4rp")
    blocks = []
    for j in joins:
        rt = j["t_s"] + offs.get(j["join"], 0.0)
        row = Image.new("RGB", (w, th), (12, 12, 12))
        frame_failures = []
        # Six frames span each rendered join, sharing the waveform's timeline.
        for k, dt in enumerate((-0.30, -0.16, -0.04, 0.02, 0.14, 0.28)):
            q = f'{tmp}/{j["join"]}_{k}.png'
            tag = f'j{j["join"]}{"+" if dt > 0 else ""}{dt:+.2f}'.replace("++", "+")
            run = sh(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, rt + dt):.4f}",
                      "-i", render_path, "-frames:v", "1", "-vf",
                      f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                      f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
                      f"drawtext=text='{dt:+.2f}s':x=5:y=5:fontsize=17:fontcolor=white:"
                      f"box=1:boxcolor=black@0.7:boxborderw=4", "-y", q])
            if run.returncode == 0 and os.path.isfile(q) and os.path.getsize(q) > 0:
                row.paste(Image.open(q).convert("RGB"), (k * tw, 0))
            else:
                # A sample just beyond the render boundary is evidence about
                # the submitted render, not an infrastructure failure.  Keep
                # the six-slot geometry stable and make the missing frame
                # explicit to the visual judge.
                missing = Image.new("RGB", (tw, th), (90, 16, 16))
                md = ImageDraw.Draw(missing)
                md.text((5, 5), f"{dt:+.2f}s", fill=(255, 255, 255))
                md.text((5, th // 2), "NO RENDER FRAME", fill=(255, 210, 210))
                row.paste(missing, (k * tw, 0))
                frame_failures.append(f"{dt:+.2f}: {(run.stderr or '')[-120:]}")
        parts = [row]
        for is_spec, hh in ((False, TH), (True, SH)):
            wp = f'{tmp}/w{j["join"]}{int(is_spec)}.png'
            if render_audio:
                if not wave(render_path, rt - HALF, HALF * 2, wp, w, hh, is_spec):
                    blank = Image.new("RGB", (w, hh), (45, 8, 8))
                    ImageDraw.Draw(blank).text(
                        (8, 8), "NO DECODABLE AUDIO IN THIS JOIN WINDOW",
                        fill=(255, 170, 170),
                    )
                    parts.append(blank)
                else:
                    parts.append(Image.open(wp).convert("RGB").resize((w, hh)))
            else:
                blank = Image.new("RGB", (w, hh), (8, 8, 8))
                ImageDraw.Draw(blank).text(
                    (8, 8), "NO AUDIO STREAM IN RENDER", fill=(255, 120, 120)
                )
                parts.append(blank)
        H = sum(x.height for x in parts) + 24
        blk = Image.new("RGB", (w, H), (12, 12, 12))
        y = 0
        for x in parts:
            blk.paste(x, (0, y)); y += x.height
        d = ImageDraw.Draw(blk)
        d.line([(w // 2, 0), (w // 2, y)], fill=(235, 45, 45), width=3)
        d.rectangle([0, y, w, y + 24], fill=(12, 12, 12))
        d.text((7, y + 6), f'join {j["join"]}   render {rt:.3f}s'
                           + (f' (rough cut {j["t_s"]:.3f}s + {offs[j["join"]]:.3f} inserted '
                              f'black)' if offs.get(j["join"]) else '')
                           + f'   {j["leaving"]} -> {j["arriving"]}   '
                             f'all three rows are the RENDER   red line = the cut',
               fill=(235, 235, 235))
        blocks.append(blk)
    if not blocks:
        return False
    out = Image.new("RGB", (w, sum(b.height + 8 for b in blocks)), (24, 24, 24))
    y = 0
    for b in blocks:
        out.paste(b, (0, y)); y += b.height + 8
    out.save(out_png)
    return True


def join_panels(items, render_path, joins, out_png, w=1188, runs=None, drift=0.0,
                expected_black_spans=None):
    """Place six picture samples over a waveform/spectrum window for each join.

    A shared timeline with a red cut marker links picture positions to audio
    measurements without requiring the judge to align separate images mentally."""
    from PIL import Image, ImageDraw
    if not items or not joins:
        return False
    HALF, TH, SH = 0.55, 84, 84          # Window half-width / waveform height / spectrum height.
    tw = w // 6
    th = int(tw * 9 / 16)
    offs, off_note = render_offsets(runs, joins, drift, expected_black_spans)
    render_audio = bool(render_path and has_audio_stream(render_path))
    tmp = _tempdir("s4jp")
    got = {}
    for i, (path, t, tag) in enumerate(items):
        q = f"{tmp}/{i:03d}.png"
        # Near the clip end, extraction can seek beyond the final frame. Try earlier times;
        # otherwise the -1 tile (0.04 seconds before the join) can be missing.
        for back in (0.0, 0.03, 0.07, 0.12):
            run = sh(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t - back):.4f}",
                      "-i", path, "-frames:v", "1", "-vf",
                      f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                      f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
                      f"drawtext=text='{tag}':x=5:y=5:fontsize=17:fontcolor=white:"
                      f"box=1:boxcolor=black@0.7:boxborderw=4", "-y", q])
            if run.returncode == 0 and os.path.isfile(q) and os.path.getsize(q) > 0:
                break
        if os.path.isfile(q) and os.path.getsize(q) > 0:
            m = re.match(r"j(\d+)([+-]\d)", str(tag))
            if m:
                got.setdefault(int(m.group(1)), []).append((m.group(2), q))
        else:
            # Invalid/out-of-range model timecodes should remain gradeable.
            # Preserve the slot with an explicit red tile so the judge sees
            # that no source frame exists at the submitted point.
            missing = Image.new("RGB", (tw, th), (90, 16, 16))
            md = ImageDraw.Draw(missing)
            md.text((5, 5), str(tag), fill=(255, 255, 255))
            md.text((5, th // 2), "NO SOURCE FRAME", fill=(255, 210, 210))
            missing.save(q)
            m = re.match(r"j(\d+)([+-]\d)", str(tag))
            if m:
                got.setdefault(int(m.group(1)), []).append((m.group(2), q))
    blocks = []
    for j in joins:
        tiles = sorted(got.get(j["join"], []), key=lambda x: (x[0][0] == "+", x[0]))
        if not tiles or not any(tag.startswith("-") for tag, _ in tiles) or not any(
                tag.startswith("+") for tag, _ in tiles):
            raise RuntimeError(f"S4E join {j['join']} lacks source frames on one side")
        row = Image.new("RGB", (w, th), (12, 12, 12))
        # Keep three tiles on each side; missing tiles must not move the central cut marker.
        for tag, q in tiles:
            k = int(tag[1]) - 1
            x = (2 - k) * tw if tag[0] == "-" else w // 2 + k * tw
            row.paste(Image.open(q).convert("RGB"), (x, 0))
        parts = [row]
        rt = j["t_s"] + offs.get(j["join"], 0.0)
        for is_spec, hh in ((False, TH), (True, SH)):
            wp = f"{tmp}/w{j['join']}{int(is_spec)}.png"
            if render_audio:
                if not wave(render_path, rt - HALF, HALF * 2, wp, w, hh, is_spec):
                    blank = Image.new("RGB", (w, hh), (45, 8, 8))
                    ImageDraw.Draw(blank).text(
                        (8, 8), "NO DECODABLE AUDIO IN THIS JOIN WINDOW",
                        fill=(255, 170, 170),
                    )
                    parts.append(blank)
                else:
                    parts.append(Image.open(wp).convert("RGB").resize((w, hh)))
            else:
                blank = Image.new("RGB", (w, hh), (8, 8, 8))
                ImageDraw.Draw(blank).text(
                    (8, 8), "NO AUDIO STREAM IN RENDER", fill=(255, 120, 120)
                )
                parts.append(blank)
        H = sum(x.height for x in parts) + 24
        blk = Image.new("RGB", (w, H), (12, 12, 12))
        y = 0
        for x in parts:
            blk.paste(x, (0, y)); y += x.height
        d = ImageDraw.Draw(blk)
        d.line([(w // 2, 0), (w // 2, y)], fill=(235, 45, 45), width=3)
        d.rectangle([0, y, w, y + 24], fill=(12, 12, 12))
        sh_off = offs.get(j["join"], 0.0)
        d.text((7, y + 6), f'join {j["join"]}   cut at {j["t_s"]:.3f}s'
                           + (f' (render {rt:.3f}s, +{sh_off:.3f} black inserted before it)'
                              if sh_off else '')
                           + f'   {j["leaving"]} -> {j["arriving"]}   '
                             f'top=picture  middle=waveform  bottom=spectrogram   '
                             f'red line=the cut   window +/-{HALF}s',
               fill=(235, 235, 235))
        blocks.append(blk)
    if not blocks:
        return False
    out = Image.new("RGB", (w, sum(b.height + 8 for b in blocks)), (24, 24, 24))
    y = 0
    for b in blocks:
        out.paste(b, (0, y)); y += b.height + 8
    out.save(out_png)
    return True


def sheet(items, out_png, cols=6, w=280):
    """Create labeled contact sheets linked to evidence records.

    Extract each frame, scale/pad to a common size, and tile. Different source
    aspect ratios must not produce unequal tiles or silently break the sheet."""
    if not items:
        return None
    h = int(w * 9 / 16)
    tmp = _tempdir("s4sheet")
    n = 0
    failures = []
    for path, t, label in items:
        lab = str(label).replace(":", "\\:").replace("'", "")
        q = f"{tmp}/x{n:03d}.png"
        for back in (0.0, 0.03, 0.08, 0.15):
            run = sh(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t - back):.4f}",
                      "-i", path, "-frames:v", "1", "-vf",
                      f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                      f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
                      f"drawtext=text='{lab}':x=6:y=6:fontsize=21:fontcolor=white:"
                      f"box=1:boxcolor=black@0.7:boxborderw=5", "-y", q])
            if run.returncode == 0 and os.path.isfile(q) and os.path.getsize(q) > 0:
                break
        if run.returncode == 0 and os.path.isfile(q) and os.path.getsize(q) > 0:
            os.rename(q, f"{tmp}/t{n:03d}.png")
            n += 1
        else:
            # Keep an explicit placeholder for a model-selected time outside
            # the source clip.  The task corpus is separately integrity-audited;
            # an invalid submitted timestamp must lower outcome quality rather
            # than erase the entire trajectory as a grader error.
            from PIL import Image, ImageDraw
            missing = Image.new("RGB", (w, h), (90, 16, 16))
            md = ImageDraw.Draw(missing)
            md.text((6, 6), str(label), fill=(255, 255, 255))
            md.text((6, h // 2), "NO FRAME AT REQUESTED TIME", fill=(255, 210, 210))
            missing.save(f"{tmp}/t{n:03d}.png")
            failures.append(f"{label}@{t:.3f}: {(run.stderr or '')[-160:]}")
            n += 1
    if not n:
        return None
    rows = (n + cols - 1) // cols
    run = sh(["ffmpeg", "-v", "error", "-i", f"{tmp}/t%03d.png",
              "-vf", f"tile={cols}x{rows}:padding=4:margin=4:color=0x1a1a1a",
              "-frames:v", "1", "-y", out_png])
    if run.returncode != 0 or not os.path.isfile(out_png) or os.path.getsize(out_png) == 0:
        raise RuntimeError(f"S4E contact-sheet assembly failed: {(run.stderr or '')[-300:]}")
    return out_png


def build(task):
    validate_task(task, "s_4e")
    man = json.load(open(f"{task}/initial_state/manifest.json"))
    sub_p = f"{task}/output/edit.json"
    if not os.path.exists(sub_p):
        return {"error": "Missing output/edit.json"}
    sub = json.load(open(sub_p))
    ev_dir = f"{task}/evidence"
    os.makedirs(ev_dir, exist_ok=True)
    prev = f"{task}/output/preview.mp4"
    has_prev = os.path.isfile(prev) and os.path.getsize(prev) > 0
    if not has_prev:
        raise RuntimeError("S4E preview required")
    audio_meta = render_audio(prev, f"{ev_dir}/render_audio.wav")
    audio_path = f"{ev_dir}/render_audio.wav" if audio_meta["status"] == "ok" else None
    if audio_path:
        _RENDER_AUDIO[prev] = audio_path

    beat = None
    tp = f"{task}/task_prompt.md"
    if os.path.exists(tp):
        m = re.search(r"covers:\s*\n+>\s*(.+?)\n", open(tp).read())
        beat = m.group(1).strip() if m else None

    # ---------------------------------------------------------- Timeline and picture content
    # Production-sound transcripts supplement pictures: a silent shot can still contain a subject,
    # and a spoken line does not establish that its speaker is visible.
    t, shots = 0.0, []
    for s in man["shots"]:
        rec = {"id": s["id"], "start_s": round(t, 3), "end_s": round(t + s["dur_s"], 3),
               "dur_s": s["dur_s"], "origin": s.get("origin"),
               "has_production_sound": bool(s.get("has_sound"))}
        p = f"{task}/initial_state/shots/{s['id']}.mp4"
        if s.get("has_sound") and os.path.exists(p):
            if not has_audio_stream(p):
                raise RuntimeError(
                    f"manifest marks {s['id']} has_sound but its media has no audio stream"
                )
            rec["production_sound_level"] = level_db(p)
            transcript = (asr(p) or "")[:160]
            rec["on_screen_says"] = (
                transcript or "(ASR succeeded but recognized no dialogue; this does not establish absence of speech.)"
            )
        elif s.get("origin") == "to_build":
            rec["on_screen_says"] = "(Title card to be generated by the agent; no source asset.)"
        else:
            rec["on_screen_says"] = "(Trailer-only footage; muted.)"
        shots.append(rec)
        t += s["dur_s"]
    total = round(t, 3)
    joins = [{"join": j["join"], "t_s": j["t_s"],
              "leaving": j["leaving"], "arriving": j["arriving"]}
             for j in man["joins"]]

    # ---------------------------------------------------------- Audio observed in the render
    render = {"exists": has_prev, "audio_stream": False}
    if has_prev:
        render["audio_stream"] = has_audio_stream(prev)
        rw = asr(audio_path, words=True) if render["audio_stream"] else []
        render["heard_words"] = rw
        render["duration_s"] = None
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", prev], capture_output=True, text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"preview ffprobe failed: {probe.stderr[-300:]}")
        d = probe.stdout.strip()
        render["duration_s"] = round(float(d), 3) if d else None
        render["overall_level"] = (
            level_db(prev) if render["audio_stream"]
            else {"mean_db": None, "peak_db": None, "status": "no_audio_stream"}
        )
        render["luma"] = luma_series(prev)
        if render["luma"] is None:
            raise RuntimeError("preview video frames could not be decoded for luma analysis")

    # Establish the rough-cut -> delivered-render clock once, before any
    # render-derived dialogue or join measurement.  A dip-to-black lengthens
    # the delivered file, so comparing ASR words at unshifted rough times can
    # make every line after the first dip appear absent even when it is audible.
    blk_runs = black_runs(prev) if has_prev else []
    expected_black_spans = [
        {"shot": shot["id"], "start_s": shot["start_s"], "end_s": shot["end_s"]}
        for source, shot in zip(man["shots"], shots)
        if (source.get("build_it") is True or source.get("kind") == "card"
            or source.get("origin") == "to_build")
    ]
    # Dark source content already occupies rough-cut time; it is not an edit.
    for source, shot in zip(man["shots"], shots):
        path = f"{task}/initial_state/shots/{source['id']}.mp4"
        if os.path.isfile(path) and not (source.get("build_it") or source.get("origin") == "to_build"):
            for run in black_runs(path):
                expected_black_spans.append({"shot": shot["id"], "kind": "source_black",
                    "start_s": shot["start_s"] + run["start_s"],
                    "end_s": min(shot["end_s"], shot["start_s"] + run["end_s"])})
    jn_render_off, render_offset_note = render_offsets(
        blk_runs, joins, (render.get("duration_s") or 0) - (total or 0),
        expected_black_spans,
    )

    def to_render_time(rough_t):
        prior = [j for j in joins if j["t_s"] <= rough_t + 1e-6]
        offset = jn_render_off.get(prior[-1]["join"], 0.0) if prior else 0.0
        return float(rough_t) + offset

    # ---------------------------------------------------------- 1. Dialogue placement
    pool = {l["id"]: l for l in (man.get("dialogue") or [])}
    placed = []
    for x in (sub.get("dialogue") or []):
        tid = x.get("take")
        i_s, o_s, at = x.get("in_s"), x.get("out_s"), x.get("at_s")
        rec = {"take": tid, "declared": {k: x.get(k) for k in
                                         ("in_s", "out_s", "at_s", "gain_db", "why")}}
        if tid in pool:
            rec["take_text"] = (pool[tid].get("text") or "")[:120]
            # Match the actually selected source words, not the whole take label.
            source_words = asr(f"{task}/initial_state/{pool[tid]['file']}", words=True)
            selected_words = [w for w in source_words if w["e"] > float(i_s) and w["s"] < float(o_s)]
            rec["take_text"] = " ".join(w["w"] for w in selected_words)
            rec["take_spans"] = selected_words
            rec["take_text_status"] = "source_asr_selected_interval" if selected_words else "unknown"
            rec["manifest_text_not_used_for_matching"] = True
        if None not in (i_s, o_s, at):
            span = (float(at), float(at) + (float(o_s) - float(i_s)))
            rec["timeline_span_s"] = [round(span[0], 3), round(span[1], 3)]
            if span[1] <= span[0]:
                rec["invalid_timeline_span"] = (
                    f"out_s ({o_s}) must be greater than in_s ({i_s})"
                )
                rec["covers_shots"] = []
                rec["crosses_joins"] = []
            else:
                covers = [s["id"] for s in shots
                          if s["start_s"] < span[1] and s["end_s"] > span[0]]
                rec["covers_shots"] = covers
                crossed = [j for j in joins if span[0] < j["t_s"] < span[1]]
                rec["crosses_joins"] = [
                    {"join": j["join"], "t_s": j["t_s"],
                     "before_s": round(j["t_s"] - span[0], 3),
                     "after_s": round(span[1] - j["t_s"], 3)} for j in crossed]
                # Declared gain and measured rendered level are different evidence.
                if has_prev:
                    render_span = (to_render_time(span[0]), to_render_time(span[1]))
                    rec["render_timeline_span_s"] = [
                        round(render_span[0], 3), round(render_span[1], 3)
                    ]
                    rec["render_span_basis"] = "declared placement plus inferred rough-to-render offsets; not content alignment proof"
                    rec["measured_in_render"] = (
                        level_db(prev, render_span[0], render_span[1] - render_span[0])
                        if render["audio_stream"] else
                        {"mean_db": None, "peak_db": None,
                         "status": "no_audio_stream"}
                    )
                    heard = [w for w in (render.get("heard_words") or [])
                             if render_span[0] - 0.3 <= w["s"] <= render_span[1] + 0.3]
                    heard_text = " ".join(w["w"] for w in heard)
                    expected_words = words(rec.get("take_text") or "")
                    heard_words = words(heard_text)
                    overlap = sorted(expected_words & heard_words)
                    rec["heard_here"] = (
                        heard_text or
                        ("(The render has no audio stream.)" if not render["audio_stream"] else
                         "(ASR succeeded but recognized no words in this interval; this alone cannot prove dialogue is absent.)")
                    )
                    rec["asr_match"] = {
                        "expected_content_words": sorted(expected_words),
                        "heard_content_words": sorted(heard_words),
                        "overlap": overlap,
                        "expected_recall": round(
                            len(overlap) / max(1, len(expected_words)), 3
                        ),
                        "reading": (
                            "matched_expected_dialogue" if len(overlap) >= 2 else
                            "no_audio_stream" if not render["audio_stream"] else
                            "not_confirmed_by_asr"
                        ),
                        "note": "An ASR non-match is unconfirmed, not proof of absence; inspect interval levels and spectrum too.",
                    }
        placed.append(rec)
    used = {x.get("take") for x in (sub.get("dialogue") or [])}
    omitted = [{"id": k, "text": (v.get("text") or "")[:100]}
               for k, v in pool.items() if k not in used]

    # ---------------------------------------------------------- 2. Music accent timing
    mu = sub.get("music") or {}
    off = mu.get("starts_at_s")
    music = {"declared_start_s": off, "declared_note": str(mu.get("note", ""))[:200]}
    mp = f"{task}/initial_state/music.wav"
    if off is not None and not os.path.exists(mp):
        raise RuntimeError("submission declares music but initial_state/music.wav is missing")
    if off is not None and os.path.exists(mp):
        acc = music_accents(mp, float(off), total)
        music["render_match"] = {
            "status": "unknown" if audio_path else "no_render_audio_stream",
            "note": "Declared cue alignment is not a measured global match; judge the actual attached audio."
        }
        if acc:
            music["analysis_status"] = "ok"
            for a in acc["accents"]:
                near = min(joins, key=lambda j: abs(j["t_s"] - a["t_s"])) if joins else None
                if near:
                    a["nearest_join"] = near["join"]
                    a["dist_to_join_s"] = round(a["t_s"] - near["t_s"], 3)
            music |= acc
            strong = [a for a in acc["accents"] if a["pct"] >= 90]
            on_cut = [a for a in strong if abs(a.get("dist_to_join_s", 9)) <= 0.12]
            music["summary"] = {
                "n_accents": len(acc["accents"]),
                "n_strong_p90": len(strong),
                "strong_landing_on_a_join_within_0.12s": len(on_cut),
                "joins_total": len(joins)}
            for j in joins:
                a = min(acc["accents"], key=lambda x: abs(x["t_s"] - j["t_s"])) if acc["accents"] else None
                j["nearest_accent_s"] = round(a["t_s"] - j["t_s"], 3) if a else None
                j["nearest_accent_pct"] = a["pct"] if a else None
        else:
            music["analysis_status"] = "no_usable_audio_in_selected_window"
    else:
        music["analysis_status"] = "not_declared"

    # ------------------------------------------- 2.5 Production sound extending across picture joins
    # Track production_sound source handles and in_s/out_s/at_s placement
    # separately from declarations; a declared bridge is not measured evidence.
    SPOS = {s_["id"]: s_ for s_ in shots}
    psound = []
    for x in (sub.get("production_sound") or []):
        sid = x.get("shot")
        sh_ = SPOS.get(sid)
        if not sh_ or x.get("keep") is False:
            continue
        i_s, o_s, at = x.get("in_s"), x.get("out_s"), x.get("at_s")
        if None not in (i_s, o_s, at):
            span = (float(at), float(at) + (float(o_s) - float(i_s)))
            how = "handle"          # Source handles can extend beyond picture boundaries.
        else:
            span = (sh_["start_s"], sh_["end_s"])
            how = "sync"            # In-shot production sound alone does not form a bridge.
        rec = {"shot": sid, "how": how, "gain_db": x.get("gain_db"),
               "timeline_span_s": [round(span[0], 3), round(span[1], 3)],
               "picture_span_s": [sh_["start_s"], sh_["end_s"]],
               "note": str(x.get("note", ""))[:100]}
        if span[1] <= span[0]:
            rec["invalid_timeline_span"] = (
                f"out_s ({o_s}) must be greater than in_s ({i_s})"
            )
        # Measure how far audio extends beyond its source picture.
        rec["tail_past_picture_s"] = round(span[1] - sh_["end_s"], 3)
        rec["head_before_picture_s"] = round(sh_["start_s"] - span[0], 3)
        rec["crosses_joins"] = [
            {"join": j["join"], "t_s": j["t_s"],
             "before_s": round(j["t_s"] - span[0], 3),
             "after_s": round(span[1] - j["t_s"], 3)}
            for j in joins if span[0] < j["t_s"] < span[1]]
        psound.append(rec)

    # v5: declared plans never become measured bridge facts.
    for shot in shots:
        shot["render_start_s"] = round(to_render_time(shot["start_s"]), 6)
        shot["render_end_s"] = round(to_render_time(shot["end_s"] - 1e-4) + 1e-4, 6)
    extra_black = unexplained_black(blk_runs, expected_black_spans, joins, jn_render_off)
    bridges, reconcile = verified_bridges(task, man, sub, placed, joins, jn_render_off,
                                          audio_path, extra_black)
    jn_render_t = {j["join"]: round(to_render_time(j["t_s"]), 6) for j in joins}
    for bridge in bridges:
        rt = bridge["render_t_s"]
        result = cut_roughness(audio_path, rt) if audio_path else None
        bridge["measurements"]["cut_roughness"] = {
            "status": "measured" if result else ("unknown" if audio_path else "no_audio_stream"),
            "render_t_s": rt, "clock_mapping": "rough_to_render_heuristic",
            "interpretation": "local mixed-audio change; not bridge type/intelligibility or proof of a defect",
            **(result or {})}


    # Count decisions, not joins: one line crossing 8 cuts is one decision, not eight.
    # Define this in the evidence so reviewers use the same unit of counting.
    n_join_touched = sum(1 for b in bridges if b["bridge_kinds"])
    n_decisions, spanning_takes, title_card_shots = bridge_decision_count(
        bridges, placed
    )
    bridge_count = {
        "joins_touched": n_join_touched, "joins_total": len(bridges),
        "distinct_decisions": n_decisions,
        "spanning_takes": sorted(spanning_takes),
        "title_card_shots": sorted(title_card_shots),
        "note": "One dialogue line crossing N cuts counts as one decision. Count decisions, not joins: "
                "two lines spanning a montage are 2 decisions, not 8."}

    # ---------------------------------------------------------- Contact sheets
    # Contact sheets provide navigation for visual inspection.
    # Include pictures directly; production-sound transcripts cannot represent
    # silent or action-only shots.
    sheets = {}
    SP = lambda sid: f"{task}/initial_state/shots/{sid}.mp4"
    # 1. Six samples span each join to inspect how a bridge fits the adjacent shots.
    # Use source picture: inserted black time can shift the render relative to fixed task joins.
    jt = []
    by_id = {s_["id"]: s_ for s_ in shots}
    for j in joins:
        a, b = by_id.get(j["leaving"]), by_id.get(j["arriving"])
        if not (a and b):
            continue
        ad = a["end_s"] - a["start_s"]
        bd = b["end_s"] - b["start_s"]
        ap = SP(a["id"])
        bp = SP(b["id"])
        a_source = os.path.exists(ap)
        b_source = os.path.exists(bp)
        for back, tag in ((0.30, "-3"), (0.16, "-2"), (0.04, "-1")):
            if ad > back:
                jt.append((ap if a_source else prev,
                           ad - back if a_source else a["render_end_s"] - back,
                           f'j{j["join"]}{tag}'))
        for fwd, tag in ((0.0, "+1"), (0.12, "+2"), (0.26, "+3")):
            if bd > fwd:
                jt.append((bp if b_source else prev,
                           fwd if b_source else b["render_start_s"] + fwd,
                           f'j{j["join"]}{tag}'))
    _rd = (render.get("duration_s") or 0) - (total or 0)
    if join_panels(jt, prev if has_prev else None, joins,
                   f"{ev_dir}/joins_av.png",
                   runs=blk_runs, drift=_rd,
                   expected_black_spans=expected_black_spans):
        sheets["join_frames"] = {
            "file": "evidence/joins_av.png",
            "note": "SOURCE picture samples with actual RENDER audio waveform/spectrum. "
                    "Source frames do not prove delivered picture. Continuous audio/waveform does not "
                    "identify a particular source bridge; inspect the continuous delivered video."}
    elif sheet(jt, f"{ev_dir}/frames_joins.png", cols=6):
        sheets["join_frames"] = {         # Fall back to picture-only samples when the render cannot be read.
            "file": "evidence/frames_joins.png",
            "note": "Six tiles per join: -3 -2 -1 | +1 +2 +3, about 0.13 seconds apart. "
                    "The left three show the leaving shot (-1 nearest the cut); the right three show the arriving shot. "
                    "No readable render was available, so these are picture samples without waveform/spectrum."}
    render_shots = [{**x, "start_s": x["render_start_s"], "end_s": x["render_end_s"]} for x in shots]
    render_joins = [{**x, "t_s": to_render_time(x["t_s"])} for x in joins]
    render_placed = [{**x, "timeline_span_s": x.get("render_timeline_span_s", x.get("timeline_span_s"))} for x in placed]
    render_music = {**music,
                    "envelope": [{**x, "t_s": to_render_time(x["t_s"])} for x in music.get("envelope", [])],
                    "accents": [{**x, "t_s": to_render_time(x["t_s"])} for x in music.get("accents", [])],
                    "render_duration_s": render.get("duration_s")}
    if music_timeline(render_shots, render_joins, render_placed, render_music, prev if has_prev else None,
                      f"{task}/initial_state", f"{ev_dir}/music_timeline.png"):
        sheets["music_timeline"] = {
            "file": "evidence/music_timeline.png",
            "note": "One timeline with five layers: picture (ID labels and widths proportional to duration), "
                    "render waveform, render spectrum, cue envelope (blue; higher means louder) and accents "
                    "(yellow markers; height is strength percentile), then dialogue occupancy (green blocks labeled by take). "
                    "Red picture-cut markers span all layers.\n"
                    "Blue/yellow traces come from the source cue at its declared offset, not music separated from the render. "
                    "Check music.render_match and the render spectrum to assess whether the cue entered the final mix. "
                    "The shared axis supports inspection of bed entry, accent timing and space left for dialogue."}
    # 2. Midpoint pictures show what appears under dialogue; production-sound text alone cannot do this.
    st = [(SP(s_["id"]) if os.path.exists(SP(s_["id"])) else prev,
           ((s_["end_s"] - s_["start_s"]) * 0.5
            if os.path.exists(SP(s_["id"])) else
            (s_["render_start_s"] + s_["render_end_s"]) * 0.5),
           f'{s_["id"]} {s_["end_s"] - s_["start_s"]:.1f}s')
          for s_ in shots if os.path.exists(SP(s_["id"])) or has_prev]
    # Every source midpoint is paired with the corresponding DELIVERED frame.
    st = [item for shot in shots for item in (
        (SP(shot["id"]) if os.path.exists(SP(shot["id"])) else prev,
         (shot["end_s"]-shot["start_s"])*.5 if os.path.exists(SP(shot["id"])) else
         (shot["render_start_s"]+shot["render_end_s"])*.5, f'{shot["id"]} SOURCE'),
        (prev, (shot["render_start_s"]+shot["render_end_s"])*.5, f'{shot["id"]} RENDER'))]
    if sheet(st, f"{ev_dir}/frames_shots.png", cols=4, w=320):
        sheets["shot_frames"] = {
            "file": "evidence/frames_shots.png",
            "note": "Each SOURCE/RENDER pair shows supplied footage and the corresponding delivered moment. Judge the delivered result from RENDER; SOURCE is not a substitute."}
    # 3. Rendered join neighborhoods show whether declared black intervals or dissolves occurred.
    if has_prev:
        rt = []
        for j in joins:
            for off, tag in ((-0.20, "-2"), (-0.06, "-1"), (0.02, "+1"), (0.16, "+2")):
                t_ = j["t_s"] + off
                if 0 <= t_ <= (render.get("duration_s") or total):
                    rt.append((prev, t_, f'j{j["join"]}{tag}'))
        if render_panels(prev, joins, blk_runs,
                         (render.get("duration_s") or 0) - (total or 0),
                         f"{ev_dir}/render_av.png",
                         expected_black_spans=expected_black_spans):
            sheets["render_frames"] = {
                "file": "evidence/render_av.png",
                "note": "Actual RENDER picture, waveform and spectrum near heuristically mapped joins. "
                        "Use continuous render to verify timing. Source black/title cards can be intentional; "
                        "waveform continuity is not proof of a specific source or transition type."}

    required_sheets = ["music_timeline"]
    if shots and has_prev:
        required_sheets.append("shot_frames")
    if joins:
        required_sheets.extend(("join_frames", "render_frames"))
    missing_sheets = [name for name in required_sheets if name not in sheets]
    if missing_sheets:
        raise RuntimeError(
            "S4E evidence is incomplete; missing required contact sheets: "
            + ", ".join(missing_sheets)
        )

    ev = {"task": task.split("training_task/")[-1],
          "beat": beat,
          "task_prompt": open(tp).read() if os.path.exists(tp) else None,
          "render_audio": audio_meta,
          "continuous_video": s4_video(task, prev, audio_meta),
          "grading_contract": "s34-grading-v5",
          "unexplained_black_runs": extra_black,
          "contact_sheets": sheets, "production_sound": psound,
          "render_black_runs": blk_runs,
          "joins_render_time_s": jn_render_t,
          "render_time_mapping": render_offset_note,
          "rough_black_spans": expected_black_spans,
          "reconcile": reconcile,
          "bridge_count": bridge_count | {"scope": "lower bound of mechanically confirmed work; unknown is not absent"},
          "stretch_dur_s": total,
          "shots": shots, "joins": joins,
          "render": {k: v for k, v in render.items() if k not in ("heard_words", "luma")},
          "render_transcript": " ".join(w["w"] for w in (render.get("heard_words") or []))[:600],
          "dialogue_placed": placed,
          "dialogue_omitted": omitted,
          "music": music,
          "bridges": bridges,
          "measurement_health": _HEALTH | {
              "required_contact_sheets": required_sheets,
              "contact_sheets_status": "ok",
              "preview": {
                  "exists": has_prev,
                  "audio_stream": render.get("audio_stream", False),
                  "video_decoded": bool(render.get("luma")),
              },
          },
          "how_measured": {
              "on_screen_says": f"faster-whisper {A.asr} transcribes each shot's source production sound with VAD enabled. "
                                f"ASR provides positive semantic evidence; missing words do not establish non-membership. "
                                f"It supplements contact sheets, not the visual content itself.",
              "measured_in_render": "volumedetect measures the relevant preview.mp4 interval. "
                                    "Declared gain_db is intent; this is the rendered result.",
              "heard_here": "Word-timed transcription of the full preview.mp4, selected by interval. "
                            "Matching words support dialogue presence; no match alone cannot establish absence. "
                            "Inspect that interval's level and spectrum too.",
              "music accents": "Local peaks of librosa onset_strength, deduplicated within 60 ms. "
                               "t_s subtracts the declared alignment offset to use the task timeline.",
              "music render match": "The declared source-cue envelope does not confirm delivered music. Global correlation is unsuitable "
                                    "for inserted or misaligned timelines: mark unknown and judge the actual rendered audio.",
              "clock": "ASR, waveform and spectrum use PCM aligned to the video start, preserving timestamp gaps. "
                       "Local rough-to-render mapping remains a black-interval heuristic, not proof of video-content alignment.",
              "cut_roughness": "mapped render-clock PCM windows: step_db/spec_dist/gap_db; "
                               "status distinguishes measured, unknown and no_audio_stream; not proof of an editorial defect"}}
    json.dump(ev, open(f"{ev_dir}/s4e_evidence.json", "w"), ensure_ascii=False, indent=1)
    return ev


tasks = json.load(open(A.tasks)) if A.tasks else [A.task]
for t in tasks:
    if (not A.force and os.path.exists(f"{t}/evidence/s4e_evidence.json")
            and json.load(open(f"{t}/evidence/s4e_evidence.json")).get("continuous_video", {}).get("contract") == VIDEO_CONTRACT):
        print(f"  {t.split('training_task/')[-1]:<30} Already available; skipped", flush=True)
        continue
    e = build(t)
    if e.get("error"):
        print(f"  {t.split('training_task/')[-1]:<30} ★ {e['error']}", flush=True)
        continue
    nb = sum(1 for b in e["bridges"] if b["bridge_kinds"])
    print(f"  {e['task']:<30} dialogue placements {len(e['dialogue_placed'])}  "
          f"bridge {nb}/{len(e['bridges'])} joins  "
          f"music accents {(e['music'].get('summary') or {}).get('n_accents', '—')}", flush=True)
