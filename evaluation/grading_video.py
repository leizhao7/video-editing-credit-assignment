"""Trusted, bounded continuous-video evidence; never fetch policy URLs.

S3 is a picture-only exercise. Its two silent videos show the submitted assembly
and separately labelled source context, not an invented finished sound mix.
S4 uses the same render-clock PCM as ASR/measurements, including real PTS holes.
"""
import base64
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from grading_contract import InvalidSubmission, judge_only, media_info, source_file, validate_task
from grading_media import run

VIDEO_CONTRACT = "s34-v5-continuous-video-1"
MAX_VIDEO_BYTES = 2_000_000
MAX_SECONDS = 180
ENCODE_FPS = 24


@lru_cache(maxsize=256)
def _frame_times(path, size, mtime):
    data = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json", path]).stdout)
    origin = float(next(s for s in media_info(path)["streams"] if s["codec_type"] == "video").get("start_time", 0))
    return tuple(float(f["best_effort_timestamp_time"]) - origin for f in data["frames"])


def frame_in_span(path, requested, lo, hi):
    """Return an actual decoded frame index/PTS in [lo, hi), or no frame.

    Never back-seek into discarded footage or return the next frame past out_s.
    Both contact sheets and similarity measurements use this exact selection.
    """
    # Evidence sheets use the task symlink while continuous reels use the
    # resolved source path. Canonicalize before keying the frame-time cache so
    # both names share one decode. Still stat every call: file replacement,
    # mutation, and symlink retargeting must invalidate the cached timestamps.
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    times = _frame_times(str(path), stat.st_size, stat.st_mtime_ns)
    candidates = [(i, t) for i, t in enumerate(times) if lo - 1e-7 <= t < hi - 1e-7]
    return min(candidates, key=lambda pair: abs(pair[1] - requested)) if candidates else None


def clamped_time(t, lo, hi):
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        if judge_only() and math.isfinite(lo) and math.isfinite(hi):
            # The sampler still receives the ORIGINAL empty/reversed bounds
            # and emits its labelled no-frame tile; never invent a valid trim.
            return lo
        raise ValueError("empty/invalid sampling interval")
    return min(max(t, lo), math.nextafter(hi, lo))


def _encode(video, output, duration, *, audio=None, origin=0.):
    """Normalize only clocks/resolution/codec; no editorial transitions added."""
    if not 0 < duration <= MAX_SECONDS:
        raise RuntimeError("continuous evidence exceeds the supported 180-second envelope")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bitrate = min(900_000, max(24_000, int(MAX_VIDEO_BYTES * 8 * .77 / duration) - (48_000 if audio else 0)))
    args = ["ffmpeg", "-v", "error", "-nostdin", "-copyts", "-i", str(video)]
    if audio:
        args += ["-i", str(audio)]
    args += ["-map", "0:v:0", "-vf", f"setpts=PTS-({origin})/TB,fps={ENCODE_FPS}:start_time=0,"
             "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
             "-c:v", "libx264", "-preset", "fast", "-b:v", str(bitrate), "-maxrate", str(bitrate),
             "-bufsize", str(2 * bitrate), "-pix_fmt", "yuv420p", "-threads", "1"]
    args += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "48k"] if audio else ["-an"]
    args += ["-t", f"{duration:.9f}", "-movflags", "+faststart", "-y", str(output)]
    run(args, timeout=180)
    if not output.stat().st_size or output.stat().st_size > MAX_VIDEO_BYTES:
        raise RuntimeError("continuous video size contract failed; no evidence may be dropped")
    run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], timeout=90)


def _meta(task, path, role, timeline, *, fps=24, audio="none_picture_only"):
    raw = Path(path).read_bytes()
    return {"file": str(Path(path).relative_to(task)), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "duration_s": media_info(path)["av_end_s"], "role": role,
            "sampling_fps": fps, "encoded_fps": ENCODE_FPS, "audio": audio, "timeline": timeline}


def _reel(task, segments, destination, role, *, describe_absence=False):
    """Lossless temporary pieces, then one bounded encoding; each piece labelled in metadata."""
    timeline, unavailable, clock = [], [], 0.
    with tempfile.TemporaryDirectory(prefix="grader-video-") as tmp:
        paths = []
        for index, seg in enumerate(segments):
            src, lo, hi = seg["path"], seg["start_s"], seg["end_s"]
            description = {k: v for k, v in seg.items() if k != "path"}
            if describe_absence and clock >= MAX_SECONDS - 1 / ENCODE_FPS:
                unavailable.append({**description, "reason": "outside_180s_review_excerpt"})
                continue
            selected = frame_in_span(src, lo, lo, hi)
            if selected is None:
                if describe_absence:
                    unavailable.append({**description, "reason": "no_frame_in_requested_interval"})
                    continue
                raise RuntimeError(f"No displayable frame in {role}: {Path(src).name} [{lo}, {hi})")
            if describe_absence and hi - lo > MAX_SECONDS - clock:
                unavailable.append({**description, "reason": "outside_180s_review_excerpt",
                                    "unobserved_span_s": [lo + MAX_SECONDS - clock, hi]})
                hi = lo + MAX_SECONDS - clock
            origin = float(next(s for s in media_info(src)["streams"] if s["codec_type"] == "video").get("start_time", 0))
            path = Path(tmp)/f"{index:03d}.mp4"
            # Select only retained frame PTS, then preserve their relative timing.
            run(["ffmpeg", "-v", "error", "-nostdin", "-copyts", "-i", str(src),
                "-vf", f"setpts=PTS-({origin})/TB,trim=start={lo:.9f}:end={hi:.9f},setpts=PTS-STARTPTS,"
                "fps=24,scale=640:360:force_original_aspect_ratio=decrease,"
                "pad=640:360:(ow-iw)/2:(oh-ih)/2,setsar=1",
                "-an", "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-threads", "1", "-y", str(path)], timeout=90)
            length = media_info(path)["av_end_s"]
            timeline.append({k:v for k,v in seg.items() if k != "path"} |
                            {"video_start_s": clock, "video_end_s": clock + length,
                             "quantization_delta_s": length - (hi-lo)})
            clock += length
            paths.append(path)
        if not paths and describe_absence:
            return {"role": role, "status": "no_renderable_segments", "timeline": [],
                    "coverage": {"complete": False, "unavailable": unavailable},
                    "note": "No source frames exist in the submitted intervals, or no clips were submitted. "
                            "No replacement footage or synthetic black video was made. Judge the original submission."}
        if not paths or clock > MAX_SECONDS:
            raise RuntimeError("empty or overlong continuous video reel")
        listing = Path(tmp)/"concat.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in paths))  # internal generated paths only
        assembled = Path(tmp)/"assembled.mp4"
        run(["ffmpeg", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-y", str(assembled)])
        _encode(assembled, destination, clock)
    result = _meta(task, destination, role, timeline)
    if describe_absence:
        result['coverage'] = {'complete': not unavailable, 'unavailable': unavailable}
        if unavailable:
            result['note'] = ('This proxy contains ONLY the renderable reviewed pieces. Gaps listed in unavailable '
                              'were NOT repaired. Joins across omitted pieces are proxy seams, NOT submitted edits.')
    return result


def s3_videos(task, man, sub, *, reference=False):
    task = Path(task)
    try:
        validate_task(task, "s_3", {"cuts": [{k: c[k] for k in ("clip", "in_s", "out_s")} for c in sub["cuts"]]})
    except InvalidSubmission as exc:
        if reference:
            raise RuntimeError("trusted S3 reference media ranges are invalid") from exc
        raise
    cuts = {c["clip"]: c for c in sub["cuts"]}
    kept, context = [], []
    for c in man["clips"]:
        if judge_only() and not reference and c['id'] not in cuts:
            continue  # Missing clips remain visible in original_submission below.
        cut = cuts[c["id"]]
        src = source_file(task, c["file"])
        lo, hi = float(cut["in_s"]), float(cut["out_s"])
        kept.append({"path": src, "clip": c["id"], "start_s": lo, "end_s": hi})
        duration = float(next(s for s in media_info(src)["streams"] if s["codec_type"] == "video")["duration"])
        for label, t in (("IN", lo), ("OUT", hi)):
            context.append({"path": src, "clip": c["id"], "edge": label, "cut_source_s": t,
                            "kept_source_span_s": [lo, hi],
                            "start_s": max(0., t-.5), "end_s": min(duration, t+.5)})
    label = "reference" if reference else "submission"
    describe = judge_only() and not reference
    videos = [_reel(task, kept, task/f"evidence/{label}_continuous.mp4", label,
                    describe_absence=describe)]
    if not reference:
        videos.append(_reel(task, context, task/"evidence/source_cut_context.mp4", "source_context_not_submission",
                            describe_absence=describe))
    return {"contract": VIDEO_CONTRACT, "stage": "s_3", "videos": videos,
            "original_submission": sub,
            "supplied_order": [c['id'] for c in man['clips']],
            "clock_note": "Silent picture-only review at 24fps. Per-piece source/reel mapping includes frame quantization. "
                          "Context reel concatenation seams are NOT submitted edits; pre/post material was discarded. "
                          "The proxy is in supplied clip order. Read original_submission for missing, repeated or reordered cuts. "
                          "Coverage omissions are observations, not an automatic score or a repaired submission."}


def s4_video(task, preview, audio):
    task = Path(task)
    destination = task/"evidence/delivered_continuous.mp4"
    pcm = task/audio["file"] if audio["status"] == "ok" else None
    _encode(preview, destination, audio["duration_s"], audio=pcm, origin=audio["video_origin_s"])
    return {"contract": VIDEO_CONTRACT, "stage": "s_4e",
            'coverage': audio.get('coverage', {'complete': True}),
            "videos": [_meta(task, destination, "delivered_render", [],
                             audio="canonical_render_pcm" if pcm else "no_audio_stream")],
            "clock_note": "Render seconds from video origin. Audio uses the identical gap-preserving PCM as ASR. "
                          "Compressed proxy video/audio is not source material; no transition was added. "
                          "Check coverage: an excerpt is NOT the full delivery; unseen time is unknown."}


def video_parts(task, ev):
    pack = ev.get("continuous_video")
    if not pack or pack.get("contract") != VIDEO_CONTRACT or not pack.get("videos"):
        raise RuntimeError("continuous video evidence is missing/stale; rebuild before judging")
    parts = [{"type": "text", "text": "CONTINUOUS VIDEO EVIDENCE AND TIME MAP:\n" + json.dumps(pack, ensure_ascii=False)}]
    for meta in pack["videos"]:
        if meta.get('status') == 'no_renderable_segments' and judge_only() and pack.get('stage') == 's_3':
            if (meta.get('role') not in ('submission', 'source_context_not_submission')
                    or meta.get('timeline') != [] or meta.get('coverage', {}).get('complete') is not False):
                raise RuntimeError('invalid explicit no-frame evidence')
            parts.append({'type': 'text', 'text': 'NO RENDERABLE VIDEO FOR ROLE ' + meta['role'] + '\n' +
                          json.dumps(meta, ensure_ascii=False)})
            continue
        path = (Path(task)/meta["file"]).resolve()
        if not path.is_relative_to((Path(task)/"evidence").resolve()):
            raise RuntimeError("untrusted video path")
        raw = path.read_bytes()
        if not raw or len(raw) > MAX_VIDEO_BYTES or hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise RuntimeError("continuous video integrity check failed")
        parts += [{"type": "text", "text": "Next video role: " + meta["role"]},
                  {"type": "grading_video", "mime_type": "video/mp4", "data": base64.b64encode(raw).decode(),
                   "fps": meta["sampling_fps"]}]
    return parts
