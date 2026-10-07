"""Shot detection and clip export for explicitly supplied private media.

Optional Modal adapter. Importing it defines functions but does not call a service.
Set VIDEO_EDITING_MODAL_VOLUME to an existing user-owned volume before execution.
All input media and generated artifacts remain runtime data, outside this source tree.
"""
import os
import sys

if __name__ == "__main__" and ("--help" in sys.argv[1:] or "-h" in sys.argv[1:]):
    print("See docs/data_construction.md for explicit Modal and local preprocessing commands.")
    raise SystemExit(0)

import modal

VOLUME_NAME = os.environ.get("VIDEO_EDITING_MODAL_VOLUME", "").strip()
GPU = os.environ.get("VIDEO_EDITING_PREPROCESS_GPU", "A10G")


def _require_volume():
    if not VOLUME_NAME:
        raise RuntimeError("Set VIDEO_EDITING_MODAL_VOLUME to your existing volume before execution")


# Keep these small helpers local: Modal can load each adapter directly by filename.
# Sharing a new local package would require an additional image/mount contract.
DATA_ROOT = "/data"


def _component(value, label="identifier"):
    import re
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {label}: expected a single safe path component")
    return value


def _data_path(*parts):
    from pathlib import Path
    root = Path(DATA_ROOT).resolve()
    path = root.joinpath(*(_component(part) for part in parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Media path escapes the configured volume mount")
    return str(path)


def _write_json(path, payload):
    import json
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=Path(path).parent,
                                prefix=".metadata-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            json.dump(payload, stream, ensure_ascii=False, indent=1, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


app = modal.App(os.environ.get("VIDEO_EDITING_SHOTS_APP", "video-editing-shots"))
vol = (modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
       if VOLUME_NAME else None)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("torch", "numpy", "transnetv2-pytorch"))
image = image.env({"VIDEO_EDITING_MODAL_VOLUME": VOLUME_NAME})


def _parse_fps(text):
    import math
    from fractions import Fraction
    rows = [line.strip().strip(",") for line in text.splitlines() if line.strip().strip(",")]
    try:
        fps = float(Fraction(rows[0]))
    except (IndexError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("ffprobe returned an invalid frame rate") from error
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be finite and positive")
    return fps


def _decode_frames(src, crop):
    import subprocess
    import numpy as np
    from tempfile import TemporaryFile
    frame_bytes, chunks, pending = 48 * 27 * 3, [], b""
    with TemporaryFile() as errors:
        command = ["ffmpeg", "-v", "error", "-i", src, "-vf", f"{crop}scale=48:27",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                chunk = process.stdout.read(frame_bytes * 4096)
                if not chunk:
                    break
                pending += chunk
                complete = len(pending) // frame_bytes * frame_bytes
                if complete:
                    chunks.append(np.frombuffer(pending[:complete], np.uint8).reshape(-1, 27, 48, 3))
                    pending = pending[complete:]
            code = process.wait()
            if code:
                errors.seek(0)
                raise subprocess.CalledProcessError(code, command, stderr=errors.read(8192).decode(errors="replace"))
            if pending:
                raise RuntimeError("ffmpeg returned an incomplete RGB frame")
            if not chunks:
                raise RuntimeError("ffmpeg decoded no video frames")
            return np.concatenate(chunks)
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


@app.function(image=image, gpu=GPU, memory=65536, volumes=({"/data": vol} if vol is not None else {}), timeout=7200)
def detect(film: str, name: str, crop: float = 0.0, crop_expr: str = "",
           th: float = 0.20,
           export: bool = True, scale: str = "854:-2", crf: str = "24"):
    _require_volume()
    film, name = _component(film, "film"), _component(name, "name")
    import subprocess, json, os, shutil, csv, time, numpy as np, torch
    from transnetv2_pytorch import TransNetV2

    t0 = time.time(); base = _data_path(film); src = _data_path(film, name + ".mp4")
    if not os.path.isfile(src): src = _data_path(film, name + ".mkv")
    if not os.path.isfile(src): raise FileNotFoundError(src)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src],
                           capture_output=True, text=True, check=True)
    fps = _parse_fps(probe.stdout)
    # pillarboxed trailers (content narrower than the frame) need the other expression;
    # iw/1.5 on a 1920x1080 frame asks for 1280 rows out of 1080 and ffmpeg emits nothing
    CROP = (crop_expr + ",") if crop_expr else (
        f"crop=iw:iw/{crop}:0:(ih-iw/{crop})/2," if crop else "")
    print(f"{name}: fps={fps:.4f}  crop={CROP or 'none'}", flush=True)

    frames = _decode_frames(src, CROP)
    N = len(frames)
    print(f"  decoded {N} frames  {time.time()-t0:.0f}s", flush=True)

    model = TransNetV2().to("cuda").eval()
    pad = 25 + 50 - (N % 50 if N % 50 else 50)
    padded = np.concatenate([np.repeat(frames[:1], 25, 0), frames, np.repeat(frames[-1:], pad, 0)])
    out = []
    with torch.no_grad():
        for i in range(0, len(padded) - 99, 50 * 32):
            wins = np.stack([padded[p_:p_ + 100]
                             for p_ in range(i, min(i + 50 * 32, len(padded) - 99), 50)])
            s, _ = model(torch.from_numpy(wins).cuda())
            out.append(torch.sigmoid(s)[:, 25:75, 0].cpu().numpy().reshape(-1))
    pred = np.concatenate(out)[:N]
    if len(pred) != N or not np.isfinite(pred).all():
        raise RuntimeError("Shot detector returned invalid frame probabilities")
    np.save(_data_path(film, name + "_tnpred.npy"), pred)
    print(f"  inference done  {time.time()-t0:.0f}s", flush=True)

    m = pred > th
    b, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            b.append((i + j - 1) / 2 / fps); i = j
        else: i += 1
    bounds = np.unique(np.concatenate([[0.0], np.array(b), [N / fps]]))
    d = np.diff(bounds)
    print(f"\n{len(d)} shots @ thr={th}   中位 {np.median(d):.2f}s  最长 {d.max():.1f}s  "
          f"<1s {int((d<1).sum())}  >10s {int((d>10).sum())}", flush=True)
    _write_json(_data_path(film, name + "_shots.json"),
                {"fps": fps, "n": int(len(d)), "bounds": [round(x, 3) for x in bounds.tolist()]})

    if export:
        outdir = _data_path(film, name + "_clips")
        shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir, exist_ok=True)
        ts = ",".join(f"{t:.3f}" for t in bounds[1:-1])
        command = ["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", f"scale={scale}",
                   "-c:v", "libx264", "-crf", crf, "-preset", "veryfast",
                   "-c:a", "aac", "-b:a", "96k"]
        if ts:
            command += ["-force_key_frames", ts, "-f", "segment", "-segment_times", ts,
                        "-reset_timestamps", "1", f"{outdir}/_s%04d.mp4"]
        else:
            command += [f"{outdir}/_s0000.mp4"]
        subprocess.run(command, check=True)
        rows = []
        for k in range(len(bounds) - 1):
            f0 = f"{outdir}/_s{k:04d}.mp4"
            if not os.path.isfile(f0) or os.path.getsize(f0) == 0:
                raise RuntimeError(f"Clip export missing expected shot {k}")
            a, bb = bounds[k], bounds[k + 1]
            nm = (f"s{k:04d}__{int(a//60):03d}m{a%60:05.2f}s__dur{bb-a:06.2f}s.mp4")
            os.rename(f0, f"{outdir}/{nm}")
            rows.append([f"s{k:04d}", round(a, 3), round(bb, 3), round(bb - a, 3), nm])
        with open(f"{outdir}/manifest.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "start_s", "end_s", "dur_s", "file"]); w.writerows(rows)
        sz = sum(os.path.getsize(os.path.join(outdir, x)) for x in os.listdir(outdir))
        print(f"  exported {len(rows)} clips  {sz/1e6:.0f} MB -> {outdir}", flush=True)

    vol.commit()
    print(f"total {time.time()-t0:.0f}s")
    return {"shots": int(len(d))}
