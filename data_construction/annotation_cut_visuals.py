"""Cached boundary pictures for annotation, without transition labels or measurements."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


_VERSION = "cut-boundary-visuals-v2"
_INPUT_NAMES = ("cut_list.json", "s3_units.json", "audio_measured.json", "picture_measured.json")
_WINDOW_S = 0.25
_MAX_PER_SIDE = 3
_FRAME_EDGE = 1024
_CELL_W, _CELL_H, _LABEL_H, _HEADER_H = 384, 216, 26, 38
_SHEET_CUTS = 6


def _number(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value <= 0)):
        raise ValueError(f"{label} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def _split(value):
    # Existing cut lists round reference seconds to milliseconds. Moving the
    # shared split (both sides) avoids assigning an upward-rounded boundary's
    # first frame to the preceding unit. Higher-precision values are unchanged.
    return max(0.0, value - (0.0005 if value == round(value, 3) else 0.0))


def _read_cuts(snapshots):
    docs = {name: json.loads(raw) for name, raw in snapshots.items()}
    if any(not isinstance(document, dict) for document in docs.values()):
        raise ValueError("All four cut-visual inputs must be JSON objects")
    cuts = docs["cut_list.json"].get("cuts")
    units = docs["s3_units.json"].get("units")
    if not isinstance(cuts, list) or not cuts or not isinstance(units, list) or len(units) != len(cuts) + 1:
        raise ValueError("cut_list needs nonempty cuts and s3_units exactly one more ordered unit")
    ids, durations = [], []
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get("id"), str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", unit["id"]):
            raise ValueError("Unit IDs must be safe nonempty ASCII strings")
        ids.append(unit["id"])
        durations.append(_number(unit.get("dur"), f"{unit['id']}.dur", positive=True))
    if len(set(ids)) != len(ids):
        raise ValueError("Unit IDs must be unique")
    indices, times = [], []
    for i, cut in enumerate(cuts):
        if not isinstance(cut, dict):
            raise ValueError("Every cut must be an object")
        idx = cut.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx in indices:
            raise ValueError("Cut indices must be unique nonnegative integers")
        if (cut.get("after"), cut.get("before")) != (ids[i], ids[i + 1]):
            raise ValueError(f"Cut {idx}: adjacency does not match ordered s3 units")
        indices.append(idx)
        times.append(_number(cut.get("timeline_s"), f"Cut {idx}.timeline_s", positive=True))
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Reference timeline_s values must be strictly increasing")
    for name in ("audio_measured.json", "picture_measured.json"):
        rows = docs[name].get("cuts")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) or [
                row.get("idx") for row in rows] != indices:
            raise ValueError(f"{name} indices must match cut_list in order")
    records = []
    for i, (cut, center) in enumerate(zip(cuts, times)):
        split = _split(center)
        # Adjacent cuts are in reference time. The final unit's supplied duration
        # bounds the last right window; assembled_s and accumulated durations are
        # never used to locate a boundary.
        left = _split(times[i - 1]) if i else 0.0
        right = _split(times[i + 1]) if i + 1 < len(times) else center + durations[-1]
        start, end = max(left, split - _WINDOW_S), min(right, split + _WINDOW_S)
        if not math.isfinite(end) or not start < split < end:
            raise ValueError(f"Cut {cut['idx']}: no positive extraction window on both sides")
        records.append({"idx": cut["idx"], "after": cut["after"], "before": cut["before"],
                        "timeline_s": center, "boundary_split_s": split,
                        "rounding_tolerance_s": center - split,
                        "window_start_s": start, "window_end_s": end,
                        "requires_individual_strip_read": True})
    return records


def _video_metadata(video):
    resolved = video.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError("Reference video must be a nonempty regular file")
    stat = resolved.stat()
    return {"resolved_path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _read_image(cv2, path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() and path.stat().st_size else None


def _decode_window(cv2, video, cut, directory, ffmpeg):
    """Decode only the boundary window; return images with FFmpeg's actual PTS."""
    start, end = cut["window_start_s"], cut["window_end_s"]
    command = [ffmpeg, "-hide_banner", "-v", "info", "-nostdin", "-copyts", "-start_at_zero",
               "-ss", format(start, ".17g"), "-t", format(end - start, ".17g"), "-i", str(video),
               "-map", "0:v:0", "-an", "-vf",
               (f"trim=start={start:.17g}:end={end:.17g},"
                f"scale=w='min({_FRAME_EDGE},iw)':h='min({_FRAME_EDGE},ih)':"
                "force_original_aspect_ratio=decrease,showinfo"),
               "-vsync", "0", "-q:v", "2", "-y", str(directory / "%06d.jpg")]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cut {cut['idx']} frame extraction failed: {type(exc).__name__}: {exc}") from exc
    if result.returncode:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1][-300:] if lines else "FFmpeg returned no diagnostic"
        raise RuntimeError(f"Cut {cut['idx']} frame extraction failed: {detail}")
    timebase = re.search(r"config in time_base:\s*(\d+)/(\d+)", result.stderr)
    timestamps = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", result.stderr)
    paths = sorted(directory.glob("*.jpg"))
    if not timebase or not paths or len(paths) != len(timestamps):
        raise RuntimeError(f"Cut {cut['idx']}: empty frame window or incomplete decoded PTS records")
    num, den = map(int, timebase.groups())
    if num <= 0 or den <= 0 or [int(n) for n, _ in timestamps] != list(range(len(paths))):
        raise RuntimeError(f"Cut {cut['idx']}: invalid decoded PTS records")
    decoded = []
    for path, (_, ticks) in zip(paths, timestamps):
        pts = int(ticks) * num / den
        frame = _read_image(cv2, path)
        if frame is None or not math.isfinite(pts) or not start <= pts < end:
            raise RuntimeError(f"Cut {cut['idx']}: unreadable or out-of-window frame")
        decoded.append({"pts_s": pts, "pts_ticks": int(ticks), "time_base": [num, den], "image": frame})
    if any(b["pts_s"] <= a["pts_s"] for a, b in zip(decoded, decoded[1:])):
        raise RuntimeError(f"Cut {cut['idx']}: decoded frame times are not strictly increasing")
    before = [frame for frame in decoded if frame["pts_s"] < cut["boundary_split_s"]][-_MAX_PER_SIDE:]
    after = [frame for frame in decoded if frame["pts_s"] >= cut["boundary_split_s"]][:_MAX_PER_SIDE]
    if not before or not after:
        raise RuntimeError(f"Cut {cut['idx']}: no decodable frame on {'before' if not before else 'after'} side; cannot borrow from an adjacent unit")
    return [dict(frame, side=side) for side, frames in (("before", before), ("after", after)) for frame in frames]


def _fit(cv2, image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    return cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))


def _label(cv2, canvas, text, x, y, width, scale=0.55):
    measured = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    if measured > width:
        scale *= width / measured
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)


def _save_image(cv2, path, image):
    if not cv2.imwrite(str(path), image) or _read_image(cv2, path) is None:
        raise RuntimeError(f"Could not write readable boundary image: {path}")


def _cached_frames_valid(row):
    frames = row.get("frames")
    if (not isinstance(frames, list) or not 2 <= len(frames) <= 6
            or any(not isinstance(frame, dict) for frame in frames)):
        return False
    if row.get("frame_paths") != [frame.get("path") for frame in frames if isinstance(frame, dict)]:
        return False
    counts = {side: sum(frame.get("side") == side for frame in frames) for side in ("before", "after")}
    if any(not 1 <= n <= 3 for n in counts.values()) or counts != row.get("frame_counts"):
        return False
    previous = -math.inf
    for frame in frames:
        pts, ticks, timebase = frame.get("pts_s"), frame.get("pts_ticks"), frame.get("time_base")
        if (isinstance(pts, bool) or not isinstance(pts, (int, float)) or not math.isfinite(pts)
                or not row["window_start_s"] <= pts < row["window_end_s"] or pts <= previous
                or isinstance(ticks, bool) or not isinstance(ticks, int)
                or not isinstance(timebase, list) or len(timebase) != 2
                or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in timebase)
                or pts != ticks * timebase[0] / timebase[1]
                or frame.get("side") != ("before" if pts < row["boundary_split_s"] else "after")):
            return False
        previous = pts
    return True


def prepare_cut_visuals(task_dir, ffmpeg="ffmpeg"):
    """Return Paths manifest_path/draft_path/sheet_paths and string fingerprint.

    Each boundary has up to three consecutive decoded frames per side, confined
    to neighboring reference boundaries and a 0.25-second window per side.
    A short unit can provide fewer frames; zero on either side is an error.
    Actual PTS are recorded in the normalized reference-video timeline. These
    pictures do not establish motion, transition duration, or no_change labels.
    """
    task = Path(task_dir).resolve(strict=True)
    paths = {name: task / "stages" / name for name in _INPUT_NAMES}
    snapshots = {name: path.read_bytes() for name, path in paths.items()}
    records = _read_cuts(snapshots)
    video = task / "reference/golden.mp4"
    metadata = _video_metadata(video)
    basis = {"version": _VERSION, "video": metadata,
             "inputs_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in snapshots.items()}}
    fingerprint = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
    output = task / "work/cut_visuals" / fingerprint
    if not output.resolve().is_relative_to(task):
        raise ValueError("Cut visual output must remain inside the private task directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, draft_path = output / "manifest.json", output / "s4.draft.json"
    sheet_paths = [output / f"sheet_{i + 1:03d}.jpg" for i in range((len(records) + _SHEET_CUTS - 1) // _SHEET_CUTS)]
    relative = lambda path: path.relative_to(task).as_posix()
    result = {"manifest_path": manifest_path, "draft_path": draft_path,
              "sheet_paths": sheet_paths, "fingerprint": fingerprint}
    import cv2
    import numpy as np

    try:
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (cached["fingerprint"] == fingerprint and cached["version"] == _VERSION
                and cached["fingerprint_metadata"] == basis and len(cached["cuts"]) == len(records)
                and cached["sheets"] == [{"path": relative(path), "cut_indices": [c["idx"] for c in records[i * 6:(i + 1) * 6]]}
                                        for i, path in enumerate(sheet_paths)]
                and all(all(row.get(k) == value for k, value in expected.items())
                        and _cached_frames_valid(row)
                        for row, expected in zip(cached["cuts"], records))):
            media = sheet_paths + [task / row["strip_path"] for row in cached["cuts"]]
            media += [task / name for row in cached["cuts"] for name in row["frame_paths"]]
            if all(path.resolve().is_relative_to(output.resolve()) and _read_image(cv2, path) is not None for path in media):
                return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    manifest_path.unlink(missing_ok=True)
    strip_h = _HEADER_H + 2 * (_CELL_H + _LABEL_H)
    for index, cut in enumerate(records):
        directory = output / f"cut_{cut['idx']:04d}"
        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".decode-", dir=directory) as temporary:
            decoded = _decode_window(cv2, video, cut, Path(temporary), ffmpeg)
        strip = np.full((strip_h, _CELL_W * 3, 3), 24, dtype=np.uint8)
        _label(cv2, strip, f"CUT {cut['idx']}  {cut['after']} -> {cut['before']}  reference={cut['timeline_s']:.6f}s",
               8, 25, strip.shape[1] - 16, 0.65)
        # Empty slots must not look like decoded black video frames.
        for row, side in enumerate(("BEFORE", "AFTER")):
            for column in range(3):
                x, y = column * _CELL_W, _HEADER_H + row * (_CELL_H + _LABEL_H)
                strip[y:y + _CELL_H, x:x + _CELL_W] = 72
                _label(cv2, strip, "NO FRAME", x + 92, y + _CELL_H // 2 + 10,
                       _CELL_W - 104, 1.0)
                _label(cv2, strip, f"nominal {side} - no sample", x + 8,
                       y + _CELL_H + 19, _CELL_W - 16)
        frames, side_counts = [], {"before": 0, "after": 0}
        for frame in decoded:
            side = frame["side"]
            column = side_counts[side]
            side_counts[side] += 1
            row = 0 if side == "before" else 1
            path = directory / f"{side}_{column + 1}.jpg"
            _save_image(cv2, path, frame["image"])
            fitted = _fit(cv2, frame["image"], _CELL_W, _CELL_H)
            x = column * _CELL_W + (_CELL_W - fitted.shape[1]) // 2
            y = _HEADER_H + row * (_CELL_H + _LABEL_H) + (_CELL_H - fitted.shape[0]) // 2
            slot_x, slot_y = column * _CELL_W, _HEADER_H + row * (_CELL_H + _LABEL_H)
            strip[slot_y:slot_y + _CELL_H + _LABEL_H, slot_x:slot_x + _CELL_W] = 24
            strip[y:y + fitted.shape[0], x:x + fitted.shape[1]] = fitted
            _label(cv2, strip, f"nominal {side.upper()} PTS={frame['pts_s']:.6f}s", column * _CELL_W + 8,
                   _HEADER_H + row * (_CELL_H + _LABEL_H) + _CELL_H + 19, _CELL_W - 16)
            frames.append({k: v for k, v in frame.items() if k != "image"} | {"path": relative(path)})
        strip_path = directory / "strip.jpg"
        _save_image(cv2, strip_path, strip)
        cut.update(strip_path=relative(strip_path), frames=frames,
                   frame_paths=[frame["path"] for frame in frames],
                   frame_counts=side_counts, sheet_path=relative(sheet_paths[index // 6]))
    for i, sheet_path in enumerate(sheet_paths):
        height, width = strip_h // 2, _CELL_W * 3 // 2
        sheet = np.full((height * 3, width * 2, 3), 24, dtype=np.uint8)
        for j, cut in enumerate(records[i * 6:(i + 1) * 6]):
            strip = _read_image(cv2, task / cut["strip_path"])
            preview = cv2.resize(strip, (width, height))
            row, column = divmod(j, 2)
            sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = preview
        _save_image(cv2, sheet_path, sheet)
    if _video_metadata(video) != metadata or any(path.read_bytes() != snapshots[name] for name, path in paths.items()):
        raise RuntimeError("Cut visual inputs changed during extraction; success manifest not published")
    manifest = {"version": _VERSION, "fingerprint": fingerprint, "fingerprint_metadata": basis,
                "fingerprint_basis": "Four input byte hashes plus video resolved path/size/mtime and version; not a video content hash",
                "inputs_paths": {name: relative(path) for name, path in paths.items()},
                "video_path": relative(video), "draft_path": relative(draft_path),
                "notes": ["Boundary positions use cut_list.timeline_s in the reference video, never assembled_s.",
                          "Each nominal side shows up to three consecutive decoded frames nearest the supplied boundary, within 0.25 seconds and neighboring supplied boundaries; short units can yield fewer frames. No frames outside the recorded extraction window are added. Unfilled gray NO FRAME slots are placeholders, not black video frames.",
                          "Millisecond-rounded reference boundaries use a shared split up to 0.5 ms earlier for numerical rounding. Higher-precision boundaries are unchanged. Original measured times are not modified.",
                          "BEFORE/AFTER denote position relative to the supplied numerical split, not a certified visual edit or shot identity. Input precision and boundary-estimation limits can leave the first nominal AFTER frame visually in the preceding shot; inspect the recorded PTS and source video rather than treating side labels as ground truth.",
                          "PTS are decoded FFmpeg showinfo ticks/time_base after normalizing the reference start to zero; requested boundaries do not imply exact frame timing.",
                          "Read every individual strip_path; six-boundary sheets are navigation previews. Individual frame_paths preserve aspect ratio, longest edge at most 1024, without upscaling.",
                          "A few stills cannot establish motion, a full dissolve/transition duration, or continuous picture. Inspect reference video as needed; no_change requires the supplied picture_measured continuity support."],
                "layout": {"cuts_per_sheet": 6, "sheet_columns": 2, "sheet_rows": 3,
                           "frames_per_side_max": _MAX_PER_SIDE, "individual_frame_max_edge": _FRAME_EDGE},
                "sheets": [{"path": relative(path), "cut_indices": [c["idx"] for c in records[i * 6:(i + 1) * 6]]}
                           for i, path in enumerate(sheet_paths)], "cuts": records}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".manifest-", suffix=".json",
                                         dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
