"""Private, cached visual aids for script annotation; no annotations are generated."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


_VERSION = "script-contact-sheets-v3"
_COLUMNS, _ROWS = 4, 3
_CELL_WIDTH, _IMAGE_HEIGHT, _LABEL_HEIGHT = 384, 216, 44
_FRAME_LONG_EDGE = 1024
_SHORT_SHOT_SECONDS = 0.75


def _read_shots(input_bytes):
    try:
        inputs = json.loads(input_bytes)
        shots = inputs["shots"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("script_inputs.json must contain a shots list") from exc
    if not isinstance(shots, list) or not shots:
        raise ValueError("script_inputs.json must contain a nonempty shots list")
    result, seen = [], set()
    for shot in shots:
        if not isinstance(shot, dict):
            raise ValueError("Each script input shot must be an object")
        sid = shot.get("id")
        if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", sid):
            raise ValueError(f"Invalid ASCII shot ID: {sid!r}")
        if sid in seen:
            raise ValueError(f"Duplicate shot ID: {sid}")
        seen.add(sid)
        times = []
        for key in ("start_s", "dur_s"):
            value = shot.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Shot {sid}: {key} must be a finite number")
            times.append(float(value))
        start, duration = times
        midpoint = start + duration / 2
        if start < 0 or duration <= 0 or not math.isfinite(start + duration):
            raise ValueError(f"Shot {sid}: start_s must be nonnegative and dur_s positive, with a finite end")
        # Existing shot manifests round seconds to three decimals. An upward
        # rounding can put a one-frame shot's only PTS just before its start.
        # Do not broaden starts supplied at finer precision or the midpoint/end.
        tolerance = 0.0005 if start == round(start, 3) else 0.0
        result.append({"id": sid, "start_s": start, "dur_s": duration,
                       "requested_midpoint_s": midpoint,
                       "start_rounding_tolerance_s": tolerance,
                       "frame_window_start_s": max(0.0, start - tolerance),
                       "frame_window_end_s": min(start + duration, midpoint + 1e-9)})
    return result


def _video_metadata(video):
    resolved = video.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"Reference video is not a file: {video}")
    stat = resolved.stat()
    if stat.st_size == 0:
        raise ValueError(f"Reference video is empty: {video}")
    return {"resolved_path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_image(cv2, path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _extract_frame(cv2, video, shot, path, ffmpeg):
    with tempfile.NamedTemporaryFile(prefix=".frame-", suffix=".jpg", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        # A direct seek to the midpoint chooses the next frame and can cross a
        # single-frame shot. Decode only its first half and select the last frame.
        window_start, window_end = shot["frame_window_start_s"], shot["frame_window_end_s"]
        window = window_end - window_start
        command = [ffmpeg, "-v", "error", "-nostdin", "-copyts", "-start_at_zero",
                   "-ss", format(window_start, ".17g"),
                   "-t", format(window, ".17g"), "-i", str(video), "-map", "0:v:0", "-frames:v", "1",
                   "-vf", (f"trim=start={window_start:.17g}:end={window_end:.17g},"
                            f"scale=w='min({_FRAME_LONG_EDGE},iw)':h='min({_FRAME_LONG_EDGE},ih)':"
                            "force_original_aspect_ratio=decrease,reverse"),
                   "-q:v", "2", "-y", str(temporary)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Frame extraction failed for {shot['id']}: {type(exc).__name__}: {exc}") from exc
        frame = _read_image(cv2, temporary)
        if completed.returncode or frame is None:
            reason = completed.stderr.strip().splitlines()
            detail = reason[-1][-300:] if reason else "FFmpeg did not produce a readable frame"
            raise RuntimeError(f"Frame extraction failed for {shot['id']} at "
                               f"{shot['requested_midpoint_s']:.6f}s: {detail}")
        os.replace(temporary, path)
        return frame
    finally:
        temporary.unlink(missing_ok=True)


def prepare_script_visuals(task_dir, ffmpeg="ffmpeg"):
    """Prepare one midpoint frame per shot and 4x3 contact sheets.

    Returns a dict with Path values ``manifest_path`` and ``draft_path``, a list
    of Paths ``sheet_paths``, and a string ``fingerprint``. The draft path is
    reserved for the annotation caller; this function never creates a draft.
    Manifest media paths are relative to task_dir. The fingerprint covers input
    bytes and video path/size/mtime metadata, not a hash of the video's contents.
    Millisecond-precision starts allow 0.5 ms of earlier numerical tolerance;
    requested midpoints remain unchanged and are not claims of exact frame PTS.
    """
    task = Path(task_dir).resolve(strict=True)
    inputs_path = task / "stages/script_inputs.json"
    input_bytes = inputs_path.read_bytes()
    shots = _read_shots(input_bytes)
    video = task / "reference/golden.mp4"
    video_metadata = _video_metadata(video)
    basis = json.dumps({"version": _VERSION, "video": video_metadata}, sort_keys=True).encode()
    fingerprint = hashlib.sha256(input_bytes + b"\0" + basis).hexdigest()
    output = task / "work/script_visuals" / fingerprint
    if not output.resolve().is_relative_to(task):
        raise ValueError("Script visual output must remain inside the private task directory")
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    manifest_path, draft_path = output / "manifest.json", output / "script.draft.json"
    sheet_paths = [output / f"sheet_{i + 1:03d}.jpg" for i in range((len(shots) + 11) // 12)]
    relative = lambda path: path.relative_to(task).as_posix()
    records = []
    for index, shot in enumerate(shots):
        records.append({**shot, "frame_path": relative(frames / f"{shot['id']}.jpg"),
                        "sheet_path": relative(sheet_paths[index // 12]),
                        "requires_individual_frame_read": shot["dur_s"] < _SHORT_SHOT_SECONDS,
                        "row": index % 12 // _COLUMNS, "column": index % _COLUMNS})
    manifest = {
        "version": _VERSION, "fingerprint": fingerprint,
        "fingerprint_basis": "script_inputs bytes plus video resolved path, size, mtime_ns, and version; not a video content hash",
        "inputs_path": relative(inputs_path), "video_path": relative(video), "video_metadata": video_metadata,
        "layout": {"columns": _COLUMNS, "rows": _ROWS, "cell_width": _CELL_WIDTH,
                   "image_height": _IMAGE_HEIGHT, "label_height": _LABEL_HEIGHT},
        "individual_frames": {"max_long_edge": _FRAME_LONG_EDGE, "upscale": False,
                              "short_shot_threshold_s": _SHORT_SHOT_SECONDS,
                              "review_instruction": "Read each short shot's individual frame_path at full available resolution before describing it; the contact sheet alone is insufficient."},
        "notes": ["Each shot receives its own requested midpoint; this is not uniform whole-video sampling.",
                  "Select the last decodable frame in the recorded frame window, ending at the requested midpoint (1 ns inclusion tolerance, bounded by shot end).",
                  "Starts with at most three decimal places allow up to 0.5 ms earlier for existing millisecond-rounded shot manifests. Higher-precision starts are not expanded; input times and midpoint/end limits are unchanged.",
                  "The decoded frame is limited by video frame timestamps; its timestamp need not exactly equal the requested time. A window without frames fails explicitly.",
                  "Individual frames preserve aspect ratio with the longest edge at most 1024 pixels, without upscaling. Contact-sheet cells use smaller 384x216 previews.",
                  "Shots shorter than 0.75 seconds require reading their individual frame. One still does not establish motion or all content in a shot; inspect the source video when needed."],
        "sheets": [{"path": relative(path), "shot_ids": [s["id"] for s in shots[i * 12:(i + 1) * 12]]}
                   for i, path in enumerate(sheet_paths)],
        "shots": records, "draft_path": relative(draft_path),
    }
    # OpenCV and NumPy are existing optional construction dependencies, loaded only here.
    import cv2
    import numpy as np

    result = {"manifest_path": manifest_path, "sheet_paths": sheet_paths,
              "draft_path": draft_path, "fingerprint": fingerprint}
    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        image_paths = [task / record["frame_path"] for record in records] + sheet_paths
        if cached == manifest and all(_read_image(cv2, path) is not None for path in image_paths):
            return result
    except (OSError, ValueError, TypeError):
        pass
    manifest_path.unlink(missing_ok=True)

    cell_height = _IMAGE_HEIGHT + _LABEL_HEIGHT
    for sheet_index, sheet_path in enumerate(sheet_paths):
        canvas = np.full((_ROWS * cell_height, _COLUMNS * _CELL_WIDTH, 3), 24, dtype=np.uint8)
        for index, shot in enumerate(shots[sheet_index * 12:(sheet_index + 1) * 12]):
            frame = _extract_frame(cv2, video, shot, frames / f"{shot['id']}.jpg", ffmpeg)
            height, width = frame.shape[:2]
            scale = min(_CELL_WIDTH / width, _IMAGE_HEIGHT / height)
            fitted = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            row, column = divmod(index, _COLUMNS)
            x = column * _CELL_WIDTH + (_CELL_WIDTH - fitted.shape[1]) // 2
            y = row * cell_height + (_IMAGE_HEIGHT - fitted.shape[0]) // 2
            canvas[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
            for line, label in enumerate((shot["id"], f"t={shot['requested_midpoint_s']:.6f}s")):
                font_scale = 0.55
                text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
                if text_width > _CELL_WIDTH - 16:
                    font_scale *= (_CELL_WIDTH - 16) / text_width
                cv2.putText(canvas, label, (column * _CELL_WIDTH + 8,
                            row * cell_height + _IMAGE_HEIGHT + 17 + line * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(sheet_path), canvas) or _read_image(cv2, sheet_path) is None:
            raise RuntimeError(f"Could not write readable contact sheet: {sheet_path}")

    if _video_metadata(video) != video_metadata or inputs_path.read_bytes() != input_bytes:
        raise RuntimeError("Script visual inputs changed during extraction; retry with the current inputs")
    temporary_manifest = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".manifest-", suffix=".json",
                                         dir=output, delete=False) as stream:
            temporary_manifest = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest is not None:
            temporary_manifest.unlink(missing_ok=True)
    return result
