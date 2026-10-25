"""Versioned, CPU-only observation tools, executed inside each rollout sandbox.

No grader, answer, network or training process is reachable here. The parent
enforces episode budgets; this worker independently bounds every request.
"""
from __future__ import annotations

import base64
import gzip
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import uuid

VERSION = "s1234-tools-v2"
MEDIA_STAGES = ("s_1", "s_2", "s_3", "s_4e")
FREE_CLIP_STAGES = ("s_1", "s_2")
# This is the current inference engine's image-context capacity, not a
# per-clip sampling quota. S1/S2 have no separate dynamic-image/audio quota.
MAX_EPISODE_IMAGES = 192
MAX_DYNAMIC_IMAGES = 96
MAX_VIDEO_FRAMES = 8
MAX_AUDIO_SECONDS = 30.0
MAX_EPISODE_AUDIO_SECONDS = 180.0
MAX_EPISODE_AUDIO_CALLS = 12
IMAGE_EDGE = 448
MAX_IMAGE_BYTES = 256 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
WORK = Path("/work")
ASR_MODEL = "/opt/s12-whisper-small.en"


def number(value, name, *, minimum=0.0, maximum=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return result


def flag(value):
    if value in (True, "true", "True"):
        return True
    if value in (False, "false", "False"):
        return False
    raise ValueError("transcribe must be a boolean")


def public_path(value):
    raw = Path(str(value))
    path = raw if raw.is_absolute() else WORK / raw
    if ".." in path.parts or not path.is_relative_to(WORK):
        raise ValueError("path must be inside /work, without '..'")
    current = WORK
    for part in path.relative_to(WORK).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink media paths are not accepted")
    if not path.is_file():
        raise ValueError(f"not a file: {value}")
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("media exceeds the 512 MiB inspection limit")
    return path


def run_media(argv, timeout=35):
    result = subprocess.run(argv, capture_output=True, timeout=timeout)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", "replace")[-1500:])
    return result.stdout


def probe(path):
    data = json.loads(run_media([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_entries", "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels,duration",
        "-of", "json", str(path),
    ]))
    duration = number(data.get("format", {}).get("duration"), "media duration")
    return {"duration_s": duration, "streams": data.get("streams", [])}


def jpeg_payload(path, label):
    # Pillow is only invoked inside the CPU sandbox, never on the GPU host.
    from PIL import Image
    import io
    Image.MAX_IMAGE_PIXELS = 24_000_000
    with Image.open(path) as source:
        if source.width * source.height > Image.MAX_IMAGE_PIXELS:
            raise ValueError("image exceeds 24 megapixels")
        source.thumbnail((IMAGE_EDGE, IMAGE_EDGE))
        image = source.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=82)
    body = out.getvalue()
    if len(body) > MAX_IMAGE_BYTES:
        raise ValueError("sanitized image exceeds byte budget")
    return {"label": label, "jpeg_b64": base64.b64encode(body).decode("ascii")}


def requested_video_frames(args):
    """Validate image cardinality before any remote execution/decoding."""
    times = args.get("timestamps_s")
    if times is not None:
        if any(args.get(key) is not None for key in ("start_s", "end_s", "num_frames")):
            raise ValueError("use timestamps_s OR start_s/end_s/num_frames, not both")
        if not isinstance(times, list) or not 1 <= len(times) <= MAX_VIDEO_FRAMES:
            raise ValueError(f"timestamps_s must contain 1–{MAX_VIDEO_FRAMES} timestamps")
        for t in times:
            number(t, "timestamp_s")
        return len(times)
    count = number(args.get("num_frames", MAX_VIDEO_FRAMES), "num_frames", minimum=1, maximum=MAX_VIDEO_FRAMES)
    if int(count) != count:
        raise ValueError("num_frames must be an integer")
    start = number(args.get("start_s", 0), "start_s")
    if args.get("end_s") is not None and number(args["end_s"], "end_s") <= start:
        raise ValueError("end_s must be greater than start_s")
    return int(count)


def video_timestamps(args, duration):
    count = requested_video_frames(args)
    if args.get("timestamps_s") is not None:
        times = [number(t, "timestamp_s") for t in args["timestamps_s"]]
        if any(t >= duration for t in times):
            raise ValueError(f"timestamps must be < actual duration {duration:.3f}s")
        return times
    start = number(args.get("start_s", 0), "start_s")
    end = number(args.get("end_s", duration), "end_s")
    if not 0 <= start < end <= duration:
        raise ValueError(f"require 0 <= start_s < end_s <= actual duration {duration:.3f}s")
    return [start + (end-start) * (i+.5)/count for i in range(count)]


def inspect_video(args):
    requested_video_frames(args)
    path = public_path(args.get("path", ""))
    meta = probe(path)
    if not any(s.get("codec_type") == "video" for s in meta["streams"]):
        raise ValueError("no video stream")
    times = video_timestamps(args, meta["duration_s"])
    images = []
    with tempfile.TemporaryDirectory(prefix="s12-video-") as temp:
        for index, t in enumerate(times):
            dest = Path(temp) / f"{index}.jpg"
            run_media([
                "ffmpeg", "-v", "error", "-nostdin", "-threads", "1",
                "-filter_threads", "1", "-protocol_whitelist", "file,pipe",
                "-ss", str(t), "-i", str(path), "-map", "0:v:0",
                "-frames:v", "1", "-vf",
                f"scale={IMAGE_EDGE}:{IMAGE_EDGE}:force_original_aspect_ratio=decrease",
                "-threads", "1", "-y", str(dest),
            ])
            images.append(jpeg_payload(dest, f"{args['path']} requested t={t:.3f}s (nearest decoded frame)"))
    return {"status": "ok", "media": meta, "timestamps_s": times, "images": images,
            "sampling": "explicit_timestamps" if args.get("timestamps_s") is not None else "uniform_window",
            "note": "These are sampled frames, not continuous video playback. Revisit any timestamp/window or use bash/FFmpeg for custom sampling."}


def inspect_audio(args):
    import numpy as np
    path = public_path(args.get("path", ""))
    start = number(args.get("start_s", 0), "start_s")
    duration = number(args.get("duration_s", 20), "duration_s", minimum=0.05, maximum=MAX_AUDIO_SECONDS)
    transcribe = flag(args.get("transcribe", True))
    meta = probe(path)
    result = {"media": meta, "start_s": start, "requested_duration_s": duration,
              "transcript": [], "transcript_status": "not_requested"}
    if not any(s.get("codec_type") == "audio" for s in meta["streams"]):
        return {**result, "status": "no_audio_stream", "transcript_status": "unavailable"}
    if start >= meta["duration_s"]:
        raise ValueError(f"start_s must be < actual duration {meta['duration_s']:.3f}s")
    duration = min(duration, meta["duration_s"] - start)
    raw = run_media([
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-filter_threads", "1",
        "-protocol_whitelist", "file,pipe", "-ss", str(start), "-i", str(path),
        "-t", str(duration), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "pipe:1",
    ])
    samples = np.frombuffer(raw, dtype="<f4")
    if not len(samples):
        return {**result, "status": "no_audio_samples", "sample_count": 0,
                "transcript_status": "unavailable"}
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    def db(value):
        return round(20 * math.log10(value), 2) if value > 0 else None
    result.update({"status": "silence" if peak < 1e-6 else "ok", "sample_count": len(samples),
                   "decoded_duration_s": len(samples) / 16000, "window_duration_s": duration,
                   "peak_dbfs": db(peak), "rms_dbfs": db(rms),
                   "near_clip_fraction": round(float(np.mean(np.abs(samples) >= .999)), 6),
                   "below_minus_50db_fraction": round(float(np.mean(np.abs(samples) < .0031623)), 6),
                   "note": "Mono-downmixed sample metrics, not LUFS. ASR is approximate evidence, not a score; no transcript does not prove no dialogue."})
    if transcribe and peak >= 1e-6:
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8", cpu_threads=2,
                                 num_workers=1, local_files_only=True)
            segments, _ = model.transcribe(samples.copy(), language="en", beam_size=1,
                                            condition_on_previous_text=False, vad_filter=True)
            result["transcript"] = [
                {"start_s": round(start + s.start, 3), "end_s": round(start + s.end, 3),
                 "text": s.text.strip(), "no_speech_prob": round(s.no_speech_prob, 3)}
                for s in segments
            ]
            result["transcript_status"] = "recognized" if result["transcript"] else "no_speech_recognized"
        except Exception as exc:
            # Infrastructure errors must be distinguishable from actual silence.
            result["transcript_status"] = "error"
            result["transcript_error"] = f"{type(exc).__name__}: {exc}"[:1000]
    elif transcribe:
        result["transcript_status"] = "digital_silence"
    return result


def bash(args):
    """Drain all output; truncating the observation must never SIGPIPE the job."""
    command = args.get("cmd", "")
    timeout = number(args.get("timeout_s", 120), "timeout_s", minimum=.05, maximum=120)
    logs = WORK / "tool_logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{uuid.uuid4().hex}.log"
    total, head, tail = 0, bytearray(), bytearray()
    with log.open("xb") as sink:
        process = subprocess.Popen(["/bin/bash", "-c", command], cwd=WORK,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        def drain():
            nonlocal total, tail
            while True:
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                if total < MAX_LOG_BYTES:
                    sink.write(chunk[:MAX_LOG_BYTES-total])
                if len(head) < 5200:
                    head.extend(chunk[:5200-len(head)])
                tail = (tail + chunk)[-1200:]
                total += len(chunk)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # A tool command cannot leave background children keeping stdout open.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join(timeout=5)
        if reader.is_alive():
            raise RuntimeError("command output drain did not finish")
        process.stdout.close()
    truncated = total > len(head)
    output = head.decode("utf-8", "replace")
    if truncated:
        output += "\n… [middle omitted; see log file] …\n" + tail.decode("utf-8", "replace")
    if total > MAX_LOG_BYTES:
        output += f"\n[log capped at {MAX_LOG_BYTES} bytes; process output was still drained]"
    if truncated or process.returncode or timed_out:
        output += f"\n[exit={process.returncode}; timeout={timed_out}; output_bytes={total}; log={log}]"
    return {"status": "ok", "output": output, "exit_code": process.returncode,
            "timed_out": timed_out, "output_bytes": total, "log_path": str(log),
            "log_complete": total <= MAX_LOG_BYTES,
            # Short outputs already live in the conversation. Persist omitted
            # output separately so sandbox cleanup does not destroy diagnostics.
            "log_gzip_b64": base64.b64encode(gzip.compress(log.read_bytes(), mtime=0)).decode("ascii") if truncated else None}


def dispatch(request):
    name, args = request["name"], request.get("arguments", {})
    started = time.monotonic()
    if name == "bash":
        result = bash(args)
    elif name == "inspect_video":
        result = inspect_video(args)
    elif name == "inspect_audio":
        result = inspect_audio(args)
    elif name == "view_image":
        paths = args.get("paths", [])
        if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_VIDEO_FRAMES:
            raise ValueError(f"generated image inspection takes 1–{MAX_VIDEO_FRAMES} paths")
        result = {"status": "ok", "images": [jpeg_payload(public_path(p), str(p)) for p in paths]}
    else:
        raise ValueError("unknown observation tool")
    return {**result, "version": VERSION, "elapsed_s": round(time.monotonic() - started, 3)}


if __name__ == "__main__":
    import sys
    try:
        payload = dispatch(json.loads(sys.stdin.read(128 * 1024)))
    except Exception as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:2000], "version": VERSION}
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
