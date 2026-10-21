"""Separate a supplied reference mix into stems for private preprocessing.

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


app = modal.App(os.environ.get("VIDEO_EDITING_STEMS_APP", "video-editing-stems"))
vol = (modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
       if VOLUME_NAME else None)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("demucs", "torch", "torchaudio", "numpy", "soundfile"))
image = image.env({"VIDEO_EDITING_MODAL_VOLUME": VOLUME_NAME})


@app.function(image=image, gpu=GPU, volumes=({"/data": vol} if vol is not None else {}), timeout=3600)
def split(film: str, name: str = "golden", model: str = "htdemucs"):
    _require_volume()
    film, name = _component(film, "film"), _component(name, "name")
    model = _component(model, "Demucs model")
    import subprocess, os, time, json
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="video-editing-preprocess-") as workdir:
        t0 = time.time()
        src = _data_path(film, name + ".mp4")
        wav = f"{workdir}/mix.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "2",
                        "-ar", "44100", wav], check=True)

        out = f"{workdir}/stems"
        subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-o", out,
                        "--device", "cuda", wav], check=True)
        d = f"{out}/{model}/mix"
        got = sorted(os.listdir(d))
        if set(got) != {"vocals.wav", "drums.wav", "bass.wav", "other.wav"}:
            raise RuntimeError("Demucs did not produce the expected four stems")
        print(f"Separated {got}   {time.time()-t0:.0f}s", flush=True)

        dst = _data_path(film, name + "_stems")
        os.makedirs(dst, exist_ok=True)
        info = {}
        for f in got:
            # 16k mono is all the analysis needs and keeps the download small
            o = _data_path(film, name + "_stems", f)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{d}/{f}",
                            "-ac", "1", "-ar", "16000", o], check=True)
            info[f.replace(".wav", "")] = os.path.getsize(o)
        # music = everything that is not the voice
        music = _data_path(film, name + "_stems", "music.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y",
                        "-i", f"{dst}/drums.wav", "-i", f"{dst}/bass.wav", "-i", f"{dst}/other.wav",
                        "-filter_complex", "amix=inputs=3:normalize=0", music], check=True)
        info["music"] = os.path.getsize(f"{dst}/music.wav")
        _write_json(_data_path(film, name + "_stems", "manifest.json"), info)
        vol.commit()
        print(f"-> {dst}   " + "  ".join(f"{k} {v/1e6:.1f}MB" for k, v in info.items()))
        print(f"total {time.time()-t0:.0f}s")
        return list(info)
