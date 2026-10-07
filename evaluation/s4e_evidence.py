"""Build trusted render-clock audio, continuous video and navigational measurements.
Source observations/declarations never substitute for the actual delivered render.
"""
import argparse, atexit, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile, warnings
from grading_contract import validate_task
from grading_video import s4_video, VIDEO_CONTRACT
from grading_media import render_audio, map_time, unexplained_black, verified_bridges
_RENDER_AUDIO = {}

if __name__ != "__main__":
    raise ImportError(f"{__file__} 是命令行脚本，不能被 import（会写文件、起子进程）")

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
    """优先读预计算缓存 `<media>.asr.json`。

    ASR 只依赖媒体文件本身、和模型输出无关，离线算一次就够；放在 rollout 里
    现算等于给每条轨迹白加一两分钟。缓存同时存整段文本和词级两种形态。

    缓存缺失时必须真正运行 ASR。依赖、模型初始化或转录失败都抛错，绝不能
    伪装成“成片没有对白”；空转录只有在一次成功解码后才是有效测量。
    """
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
    """一段音频的平均电平和峰值。声明抬了 +14 dB 和实际听不听得见是两回事。"""
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
    """逐帧亮度。黑场是不是真的黑、黑了多久，只能从这里读。"""
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
    """音乐里的重击，换算到本段时间线上。

    offset 是 agent 声明的对齐位置：文件里的第 offset 秒 = 本段的 0 秒。卡点判得准不准，
    全看这个换算——用文件时间跟切点比是没有意义的。
    """
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
        # 同一记重击会连出好几帧，60ms 内只取最强的那个
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
            "note": "t_s 已换算到本段时间线（文件时间 − 声明的对齐位置）"}


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
            "render 与声明 cue 段的波形相关性，允许 ±0.25s 编码延迟；越高越能确认音乐"
            "确实进入成片，但混入对白/同期声会压低它，不能用单一阈值判零分"
        ),
    }


STOP = set("a an the and or but of to in on at is are was were be been it its this that "
            "i you he she they we me him her them my your his their our for with as so "
            "not no yes do does did have has had will would can could just now then there "
            "here what who how when why all any some one".split())


def words(t):
    """实词集合。比对 take 文本和镜头同期声用——判这条台词是不是这个镜头自己的话。"""
    return {w for w in re.findall(r"[a-z']+", (t or "").lower())
            if len(w) > 2 and w not in STOP}


def sh(c):
    return subprocess.run(c, capture_output=True, text=True)


def wave(src, t0, dur, out_png, w=1200, h=110, spec=False):
    """把一段声音画成图。判官这个部署只收 text 和 image_url——探过，input_audio 直接
    400。所以声音要么以数字到它面前（电平、秒数、转录），要么画成画面跟帧一起看。
    showwavespic 看连不连得上，showspectrumpic 看连过去的是人声还是乐器。"""
    f = (f"showspectrumpic=s={w}x{h}:legend=0:scale=log:color=intensity"
         if spec else f"showwavespic=s={w}x{h}:colors=0xdddddd:split_channels=0")
    src = _RENDER_AUDIO.get(str(src), src)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t0):.3f}",
                        "-t", f"{dur:.3f}", "-i", src, "-filter_complex", f,
                        "-frames:v", "1", out_png], capture_output=True)
    return os.path.exists(out_png) and r.returncode == 0


def cut_roughness(path, t, half=0.23):
    """量这一刀在成片里听起来有多糙。不看声明，只看波形。

    三个数：
      step_db    切点两侧 0.23 秒窗口的电平差
      spec_dist  两侧 16 个对数频带能量向量的余弦距离——同一个声源继续响则小
      gap_db     切点前后各 50ms 里最安静的 20ms，相对两侧平均电平掉多少

    **这把尺子量不出桥的类型。** 拿 90 段真预告片按 golden 的声明分组量过：hard 刀的
    频谱距离(0.119)反而比 J(0.208)、L(0.221)更小，因为成片里音乐床连续铺着、占了频谱
    主导，干接时音乐没变所以"连续"，而 J/L 是台词在进出、频谱变化更大。

    它量的是另一件事：这一刀有多糙。真预告片的干接中位 0.119，agent 提交的中位 0.166——
    比预告片的干接还糙 39%。这是"声音是断的"这个印象的机械对应物。
    """
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
    """一段音频的平均电平。取不到就 None——不猜。"""
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
    """手柄铺过切点了——铺出去的那一段里到底有没有声音。

    L cut 的意思是离开镜自己的声音继续响。把手柄拉长 0.9 秒、拉到的是一段室内静音，
    那不是 L cut，是把静音接到了下一个镜头上。声明和手柄记账都看不出这个区别，
    只有量那一段的电平能看出来。

    返回铺出去那段相对镜内的电平差。低太多就是铺了个寂静。
    """
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
            "reading": ("铺出去的是静音——手柄拉长了，拉到的那段没有声音，"
                        "这不是 L/J cut" if ext - inner < -12 else
                        "铺出去的那段有内容，是真的把声音带过了切点")}


def carry_opportunity(is_dir, leaving, arriving, win=0.6):
    """这一刀撑不撑得住一个过渡——不用成片也答得出。

    L cut 要的是离开镜的声音在画面切掉之后还在响。手柄文件比画面前后各多 4 秒，
    所以"画面出点之后那 0.6 秒里有没有声音"是直接量得到的。有内容就是这一刀本来
    可以做 L；作业干接了，那是放掉了一个现成的过渡。J cut 同理，看到达镜画面入点
    之前那 0.6 秒。

    盲判占最终分的 0.6，但它没有成片可比。参照判官那边靠 join_alignment 数"成片铺了
    作业没铺"（成片 2%、agent 23%、拆平对照 38%），盲判这边靠这个——素材自己就能
    证明那里有声音可以延过去。
    """
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
                     "reading": ("这一刀撑得住：手柄里画面之外那段有声音，延过去是现成的"
                                 if rel > -12 else
                                 "撑不住：手柄里画面之外那段基本是静音，没东西可延")}
    if out:
        out["note"] = ("量的是素材本身，跟作业怎么做无关。supports=true 而作业在这一刀"
                       "干接，就是放掉了一个现成的过渡")
    return out or None


def black_runs(path):
    """用 ffmpeg 的 blackdetect 找黑场，避免把正常的暗镜误判为黑场。"""
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
    """粗剪的接点时间 → 成片里的时间。

    只有成片相对 rough cut **多出来**的黑场才能移动时钟。题目里本来就占时长的标题卡
    即使整段为黑，也绝不能再次累加；所有偏移总和还必须受正 duration drift 的硬上限
    约束。否则一张 3 秒黑底标题卡会把后续所有声画证据错移 3 秒。
    """
    zero = {j["join"]: 0.0 for j in joins}
    try:
        numeric_drift = float(drift)
    except (TypeError, ValueError):
        numeric_drift = 0.0
    positive_drift = max(0.0, numeric_drift)
    if not runs_in or positive_drift < 0.02:
        reason = ("成片没有比粗剪更长，黑帧不能作为插入时长累加"
                  if numeric_drift < -0.02 else
                  "成片与粗剪等长，波形按粗剪时间截取")
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
            f"检测到的黑场均已由 rough-cut 标题卡解释；未增加接点偏移"
            f"（忽略已编入时长的黑场 {ignored_expected:.3f}s）"
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
    return off, (f"成片比粗剪长 {numeric_drift:+.3f}s；最多只允许使用这段正 drift。"
                 f"排除 rough-cut 标题卡黑场 {ignored_expected:.3f}s 后，"
                 f"接点前确认额外黑场 {used:.3f}s；剩余 {tail:+.3f}s 不移动接点")


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
    """整段一条时间轴：画面 / 成片波形 / 成片频谱 / 音乐床包络与重击 / 台词占位。

    music 这条 rubric 原来完全靠作业自己写的 note 打分——同一个入点数字，换一段有论证
    的文字，十份的 band 从均 1.20 涨到 2.90。判官手里除了数字没有别的，不给 note 就
    什么都不剩。这张图是给它的东西：床从哪儿进、在哪儿起、重击落在哪一刀上、台词占着
    哪几段，全在同一条轴上，自己看。
    """
    from PIL import Image, ImageDraw
    tot = mus.get("render_duration_s") or max(s["end_s"] for s in shots) or 1.0
    FH, WH, SH, MH, DH, LH = 92, 96, 96, 116, 30, 22
    X = lambda t: int(max(0.0, min(1.0, t / tot)) * w)
    tmp = _tempdir("s4mt")
    im = Image.new("RGB", (w, FH + WH + SH + MH + DH + LH * 2 + 10), (14, 14, 14))
    d = ImageDraw.Draw(im)
    render_audio = bool(render_path and has_audio_stream(render_path))

    # ① 画面条：每个镜头占的宽度就是它占的时间
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

    # ④ 音乐床：包络折线 + 重击竖线，高度按强度百分位
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

    # ⑤ 台词占位
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

    # 接点：贯穿所有层的竖线
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
    """跟 join_panels 同样的三层结构，但画面取自**成片**。

    join_panels 的画面来自素材，答的是"这两个镜头在干什么，这一刀配不配得上"。
    这一张答的是另一件事：成片这一刀实际发生了什么。插进去的黑、真做出来的淡、
    时间轴的位移，只有在成片里看得见——素材里永远没有黑场。

    bridges 那条 rubric 原来只有 declared_* 和 reconcile 可读，问的全是"作业说的对不对"。
    散文摘掉之后剩下的 cut / under / prelap 仍然是自述。给它成片本身。
    """
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
        # 六格跨过成片里的切点，和下面的波形共用同一条时间轴
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
    """每刀一块：六帧一排，底下压同一时间窗的波形和频谱，红线标切点。

    分两张图（帧一张、声音一张）判官对不上号——它得自己记住 j05 的第几格对应哪一段声音。
    压在一起就不用记：同一条时间轴，上面是看见的，下面是听见的。
    """
    from PIL import Image, ImageDraw
    if not items or not joins:
        return False
    HALF, TH, SH = 0.55, 84, 84          # 窗口半宽 / 波形高 / 频谱高
    tw = w // 6
    th = int(tw * 9 / 16)
    offs, off_note = render_offsets(runs, joins, drift, expected_black_spans)
    render_audio = bool(render_path and has_audio_stream(render_path))
    tmp = _tempdir("s4jp")
    got = {}
    for i, (path, t, tag) in enumerate(items):
        q = f"{tmp}/{i:03d}.png"
        # 贴着片尾抽帧常常落到最后一帧之后，什么都不出。往回退着试——-1 那格
        # （切点前 0.04 秒）在旧的联络表里从来没出现过，就是这个原因。
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
        # 左三格贴左半，右三格贴右半——中缝永远是切点，缺格也不会把红线错位
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
    """联络表。每格左上角压标签，判官才能把画面跟证据里的行对上。

    每帧单独抽、统一 pad 到同一尺寸再用 tile 拼。xstack 的 `w0*n_h0*n` 布局要求瓦片
    等尺寸，各镜头宽高比不同会让整张静默失败——S3 那边踩过这个坑。
    """
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
        return {"error": "没有 output/edit.json"}
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

    # ---------------------------------------------------------- 时间线与画面内容
    # 同期声是画面内容的文字侧写。判官现在也看得见帧了，但这两样答的不是一回事：
    # 静默镜头转录是空的，画面里却有主体；一句同期声也不说明说话的人在不在画内。
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
                transcript or "(ASR 成功，但没有识别出对白；不等同于无人声)"
            )
        elif s.get("origin") == "to_build":
            rec["on_screen_says"] = "(要 agent 生成的字幕卡，没有素材)"
        else:
            rec["on_screen_says"] = "(预告片专有素材，已静音)"
        shots.append(rec)
        t += s["dur_s"]
    total = round(t, 3)
    joins = [{"join": j["join"], "t_s": j["t_s"],
              "leaving": j["leaving"], "arriving": j["arriving"]}
             for j in man["joins"]]

    # ---------------------------------------------------------- 渲染里实际听到什么
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

    # ---------------------------------------------------------- ① 台词落点
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
                # 声明抬了多少是一回事，渲染里量出来是另一回事
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
                        ("(成片没有音轨)" if not render["audio_stream"] else
                         "(ASR 成功，但这一段没有识别出词；不能单独证明对白缺失)")
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
                        "note": "ASR 未匹配只是不确认；需结合区间电平和频谱，不能单独判缺失",
                    }
        placed.append(rec)
    used = {x.get("take") for x in (sub.get("dialogue") or [])}
    omitted = [{"id": k, "text": (v.get("text") or "")[:100]}
               for k, v in pool.items() if k not in used]

    # ---------------------------------------------------------- ② 音乐卡点
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

    # ------------------------------------------- ②·5 同期声铺过接点（真正的 J/L cut）
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
            how = "handle"          # 用手柄铺出来的，可以越过画面
        else:
            span = (sh_["start_s"], sh_["end_s"])
            how = "sync"            # 只是保留本镜同步声，跟画面等长，不构成 bridge
        rec = {"shot": sid, "how": how, "gain_db": x.get("gain_db"),
               "timeline_span_s": [round(span[0], 3), round(span[1], 3)],
               "picture_span_s": [sh_["start_s"], sh_["end_s"]],
               "note": str(x.get("note", ""))[:100]}
        if span[1] <= span[0]:
            rec["invalid_timeline_span"] = (
                f"out_s ({o_s}) must be greater than in_s ({i_s})"
            )
        # 声音比画面多出来的部分——正是 J/L cut 的量
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


    # 决定数 ≠ 接点数。一条台词跨 8 刀是一个决定，不是八个——两轮判分在这上面差了
    # 三档（4 分 vs 1 分），读的是同一份证据。口径必须在证据里就算清楚。
    n_join_touched = sum(1 for b in bridges if b["bridge_kinds"])
    n_decisions, spanning_takes, title_card_shots = bridge_decision_count(
        bridges, placed
    )
    bridge_count = {
        "joins_touched": n_join_touched, "joins_total": len(bridges),
        "distinct_decisions": n_decisions,
        "spanning_takes": sorted(spanning_takes),
        "title_card_shots": sorted(title_card_shots),
        "note": "一条台词跨 N 刀算一个决定。数决定，不数接点——"
                "两条台词铺满一段蒙太奇是 2 个决定，不是 8 个"}

    # ---------------------------------------------------------- 联络表
    # Contact sheets provide navigation for visual inspection.
    # 这套 S4 证据原先建在"看不了"的假设上，用同期声转录当画面代理；那个假设不成立，
    # 而同期声在静音镜头和纯动作镜头上是全盲的。
    sheets = {}
    SP = lambda sid: f"{task}/initial_state/shots/{sid}.mp4"
    # ① 接点附近密采样：每刀六格，跨过切点。判 bridge 配不配得上这两个镜头。
    #    取自素材而非渲染——渲染里如果加了黑场，时间线会偏移，而接点位置是题目定死的。
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
        sheets["join_frames"] = {         # 没成片可读时退回纯画面
            "file": "evidence/frames_joins.png",
            "note": "每个接点六格：-3 -2 -1 | +1 +2 +3，跨过切点，间隔约 0.13 秒。"
                    "左三格是离开的镜头（-1 最接近切点），右三格是到达的镜头。"
                    "（这份没有渲染成品可读，所以只有画面，没有配套的波形和频谱）"}
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
            "note": "整段一条时间轴，五层从上到下：画面（每个镜头占的宽度＝它占的时间，"
                    "标了 id）／成片波形／成片频谱／音乐床的包络（蓝线，越高越响）和重击"
                    "（黄竖线，高度＝强度百分位）／台词占位（绿块，标 take）。"
                    "红竖线是画面切点，贯穿所有层。\n"
                    "蓝线/黄线来自声明 offset 对应的原始 cue，不是从成片分离出的音乐；"
                    "是否真的混进成片必须同时看 music.render_match 和成片频谱。"
                    "床从哪儿进、重击落在哪一刀上、有没有给台词让位，可以在同轴上核对"}
    # ② 每镜一帧：判台词落在什么画面上。同期声只在有人说话的镜头上有效。
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
            "note": "每对 SOURCE/RENDER 分别是提供的素材与成品对应时刻；判断实际交付用 RENDER，不能拿 SOURCE 冒充成品。"}
    # ③ 渲染里接点附近：核对声明的黑场/叠化到底有没有发生
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
              "on_screen_says": f"faster-whisper {A.asr} 转每个镜头自带的正片同期声。"
                                f"VAD 已开启；ASR 只作正向语义证据，不以缺词证明不属于。"
                                f"它补充画面联络表，不是画面内容的替代品。",
              "measured_in_render": "对 preview.mp4 的那一段跑 volumedetect；"
                                    "声明的 gain_db 是意图，这个是结果",
              "heard_here": "对 preview.mp4 整条做词级转录后按时间取词——"
                            "有匹配词可确认台词存在；无匹配词不能单独证明台词缺失，"
                            "必须结合该区间电平和频谱",
              "music accents": "librosa onset_strength 的局部峰，60ms 内去重；"
                               "t_s 已减去声明的对齐位置，换算到本段时间线",
              "music render match": "声明源 cue 的包络不是成品音乐的确认；全局相关不适用于"
                                    "有插入/错位的时间线，标 unknown，由实际成品听音评判。",
              "clock": "ASR、波形、频谱均使用按视频起点对齐并保留时间戳空洞的 PCM。"
                       "rough→render 的局部映射仍是黑段启发式，不是视频内容对齐证明。",
              "cut_roughness": "mapped render-clock PCM windows: step_db/spec_dist/gap_db; "
                               "status distinguishes measured, unknown and no_audio_stream; not proof of an editorial defect"}}
    json.dump(ev, open(f"{ev_dir}/s4e_evidence.json", "w"), ensure_ascii=False, indent=1)
    return ev


tasks = json.load(open(A.tasks)) if A.tasks else [A.task]
for t in tasks:
    if (not A.force and os.path.exists(f"{t}/evidence/s4e_evidence.json")
            and json.load(open(f"{t}/evidence/s4e_evidence.json")).get("continuous_video", {}).get("contract") == VIDEO_CONTRACT):
        print(f"  {t.split('training_task/')[-1]:<30} 已有，跳过", flush=True)
        continue
    e = build(t)
    if e.get("error"):
        print(f"  {t.split('training_task/')[-1]:<30} ★ {e['error']}", flush=True)
        continue
    nb = sum(1 for b in e["bridges"] if b["bridge_kinds"])
    print(f"  {e['task']:<30} 台词 {len(e['dialogue_placed'])} 处  "
          f"bridge {nb}/{len(e['bridges'])} 接点  "
          f"音乐重击 {(e['music'].get('summary') or {}).get('n_accents', '—')}", flush=True)
