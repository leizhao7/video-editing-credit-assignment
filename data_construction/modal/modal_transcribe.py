"""Word-level transcription of an explicitly supplied reference edit.

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


app = modal.App(os.environ.get("VIDEO_EDITING_TRANSCRIBE_APP", "video-editing-transcribe"))
vol = (modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
       if VOLUME_NAME else None)
# faster-whisper needs the CUDA runtime libs that debian_slim does not carry
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("faster-whisper", "numpy",
                      "nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
         .env({"VIDEO_EDITING_MODAL_VOLUME": VOLUME_NAME, "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/"
                                  "cublas/lib:/usr/local/lib/python3.11/site-packages/"
                                  "nvidia/cudnn/lib"}))


@app.function(image=image, gpu=GPU, volumes=({"/data": vol} if vol is not None else {}), timeout=3600)
def run(film: str, name: str = "golden", model: str = "large-v3"):
    _require_volume()
    film, name = _component(film, "film"), _component(name, "name")
    import subprocess, json, time
    from faster_whisper import WhisperModel

    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="video-editing-preprocess-") as workdir:
        t0 = time.time()
        src = _data_path(film, name + ".mp4")
        wav = f"{workdir}/a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1",
                        "-ar", "16000", wav], check=True)

        m = WhisperModel(model, device="cuda", compute_type="float16")
        segs, info = m.transcribe(wav, word_timestamps=True, vad_filter=True,
                                  beam_size=5, language="en")
        out = []
        for s in segs:
            out.append({"start": round(s.start, 3), "end": round(s.end, 3),
                        "text": s.text.strip(),
                        "words": [{"w": w.word.strip(), "s": round(w.start, 3),
                                   "e": round(w.end, 3), "p": round(w.probability, 2)}
                                  for w in (s.words or [])]})
        nw = sum(len(s["words"]) for s in out)
        print(f"{len(out)} 段 / {nw} 词   语言 {info.language} ({info.language_probability:.2f})"
              f"   {time.time()-t0:.0f}s", flush=True)
        for s in out[:6]:
            print(f"  {s['start']:7.2f}-{s['end']:7.2f}  {s['text'][:78]}", flush=True)

        p = _data_path(film, name + "_transcript.json")
        _write_json(p, {"film": film, "source": name, "model": model,
                        "n_segments": len(out), "n_words": nw, "segments": out})
        vol.commit()
        print(f"-> {p}")
        return {"segments": len(out), "words": nw}
