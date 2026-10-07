"""Public construction draft: s1_media_reuse.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""
try:
    from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)
except ImportError:  # Direct script invocation.
    from release_config import (CODE_ROOT, DATA_ROOT, require_target,
        require_annotation_model, cloud_volume, run_legacy_rubric_hook)

import hashlib
import json
import math
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(candidate):
    return (candidate["src"], round(float(candidate["t0"]), 3),
            round(float(candidate["dur"]), 3), candidate["crop"])


TIMEBASE_POLICY = "native-cfr24-source-legacy-packets-v1"


def timebase_reuse_allowed(reference):
    """Legacy dur*24 native packets are time-correct only for native 24 fps.

    This provenance is produced from the pinned source probe, not inferred from
    the already-normalized candidate's 24 fps container (which hides the defect).
    Conservatively recut 23.976 as well as 25/30/unknown source rates.
    """
    fps = reference.get("native_source_fps")
    return (reference.get("timebase_policy") == TIMEBASE_POLICY
            and type(fps) in (int, float) and math.isfinite(fps) and abs(fps - 24.) <= 1e-6)


def _timebase(probe, origin):
    return {"native_source_fps": probe.get("src" if origin == "film" else "gold", {}).get("fps"),
            "timebase_policy": TIMEBASE_POLICY}


def candidate_index(root, slug, probe, verified):
    root = Path(root)
    gcrop = probe.get("crop_expr") or (f"crop=iw:iw/{probe['crop']}:0:(ih-iw/{probe['crop']})/2"
                                       if probe.get("crop") else "")
    result = {}
    for directory in sorted((root / "training_task" / slug / "s_1").glob("c*")):
        audit_path = root / "review_evidence/s1/inputs/tasks" / (f"s_1_{slug}_{directory.name}.json")
        if not audit_path.exists():
            continue  # No audit identity: normal encoder only (also synthetic tests).
        audit = json.loads(audit_path.read_text())
        original = json.loads((directory / "verifier/answer.json").read_text())
        if original != audit["existing_answer"]:
            raise ValueError("Original candidate metadata changed since visual audit")
        for candidate in original["key"]:
            cid, origin = candidate["id"], candidate["origin"]
            clock = _timebase(probe, origin)
            if not timebase_reuse_allowed(clock):
                continue
            path = directory / "initial_state/clips" / (cid + ".mp4")
            if path.resolve() != path or not path.is_file():
                raise ValueError("Unsafe reuse candidate path")
            entry = {"src": str(Path(probe["dir"]) / probe["source" if origin == "film" else "golden"]),
                     "crop": probe.get("src_crop_expr", "") if origin == "film" else gcrop,
                     "t0": candidate["src_t0"], "dur": candidate["dur_s"]}
            result.setdefault(identity(entry), {"path": str(path), "sha256": audit["candidate_sha256"][cid],
                "method": "audited_video_stream_copy_uniform_silent_audio", **clock})
    # A visually adopted old candidate takes precedence over an equivalent recipe
    # from another task. Thus the positive's exact reviewed video packets survive.
    for record in verified["audit"]:
        ref = record["reviewed_reference"]
        clock = _timebase(probe, record["adopted_span"]["origin"])
        if not timebase_reuse_allowed(clock):
            continue
        result[identity(record["adopted_span"])] = {
            "path": record["original_media_path"], "sha256": ref["original_media_sha256"],
            "method": "exact_reviewed_video_stream_copy_uniform_silent_audio", **clock}
    return result


def remux_silent(reference, output, duration, require_timebase_certification=False):
    if require_timebase_certification and not timebase_reuse_allowed(reference):
        raise ValueError("Legacy candidate lacks native 24fps source-timebase certification; recut source")
    source, output = Path(reference["path"]), Path(output)
    if source.resolve() != source or output.resolve() == source.resolve() or output.exists():
        raise ValueError("Reuse must write a new, separate output file")
    if sha(source) != reference["sha256"]:
        raise ValueError("Audited candidate SHA changed before reuse")
    metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-of", "json", str(source)], capture_output=True, text=True, check=True).stdout)["streams"]
    frames = max(1, round(float(duration) * 24))
    if (len(metadata) != 1 or metadata[0].get("width") != 854 or metadata[0].get("height") != 480
            or metadata[0].get("pix_fmt") != "yuv420p" or metadata[0].get("r_frame_rate") != "24/1"
            or int(metadata[0].get("nb_frames", 0)) != frames):
        raise ValueError("Audited reuse candidate is not exactly the normalized planned video")
    command = ["ffmpeg", "-v", "error", "-i", str(source), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
               "-map", "0:v:0", "-map", "1:a:0", "-sn", "-dn", "-map_metadata", "-1",
               "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-t", str(frames / 24),
               "-metadata:s:v:0", "handler_name=", "-metadata:s:v:0", "language=und",
               "-metadata", "encoder=", "-video_track_timescale", "12288", "-movflags", "+faststart", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    if result.returncode or result.stderr.strip() or sha(source) != reference["sha256"]:
        raise RuntimeError("Silent reuse failed or source changed: " + result.stderr[-1000:])
    return True
