"""Transcribe user-supplied source films to support private dialogue location.

Optional Modal adapter. Importing it defines functions but does not call a service.
Set VIDEO_EDITING_MODAL_VOLUME to an existing user-owned volume before execution.
All input media and generated artifacts remain runtime data, outside this source tree.
"""
import os
import sys

if __name__ == "__main__" and ("--help" in sys.argv[1:] or "-h" in sys.argv[1:]):
    print("Usage: python data_construction/modal/modal_film_transcribe.py "
          "(--extract | --upload | --fetch) (--film SLUG | --all) [-j WORKERS]\n\n"
          "--film selects one exact film slug; --all explicitly selects every available input.\n"
          "--extract uses the raw source named in each film's probe.json.\n"
          "Remote: modal run data_construction/modal/modal_film_transcribe.py "
          "[--only EXACT_SLUG] [--model large-v3]\n"
          "See docs/data_construction.md for configuration and preprocessing commands.")
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


def _filename(value, label="filename"):
    if (not isinstance(value, str) or not value.strip() or value in {".", ".."}
            or any(char in value for char in ("/", "\\", "\0"))):
        raise ValueError(f"{label} must be a single safe filename")
    return value


def _local_path(root, *parts):
    """Keep generated artifacts inside the configured private data root."""
    from pathlib import Path
    root = Path(root).resolve()
    path = root.joinpath(*(_filename(part) for part in parts))
    if not path.resolve().is_relative_to(root):
        raise ValueError("Film output path escapes the private data root")
    return path


def _local_probe(root, slug):
    import json
    slug = _component(slug, "slug")
    path = _local_path(root, "exports", slug, "probe.json")
    with path.open(encoding="utf-8") as stream:
        probe = json.load(stream)
    if not isinstance(probe, dict) or _component(probe.get("slug"), "slug") != slug:
        raise ValueError(f"Probe slug does not match its export directory: {path}")
    return probe


def _probe_source(probe):
    """Use the exact raw input recorded by probing, never a sorted video guess."""
    from pathlib import Path
    source = _filename(probe.get("source"), "Probe source")
    directory = probe.get("dir")
    if not isinstance(directory, str) or not directory.strip() or "\0" in directory:
        raise ValueError("Probe dir must name the source input directory")
    path = Path(directory).expanduser() / source
    # Raw media may be a user-owned symlink outside the private output root.
    if not path.is_file():
        raise FileNotFoundError(f"Source named by probe.json does not exist: {path}")
    return str(path)


app = modal.App(os.environ.get("VIDEO_EDITING_FILM_TRANSCRIBE_APP", "video-editing-film-transcribe"))
vol = (modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
       if VOLUME_NAME else None)

# faster-whisper needs the CUDA runtime libs debian_slim does not carry
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("ffmpeg")
         .pip_install("faster-whisper==1.2.1", "av>=11,<19", "numpy",
                      "nvidia-cublas-cu12", "nvidia-cudnn-cu12==9.*")
         .env({"VIDEO_EDITING_MODAL_VOLUME": VOLUME_NAME, "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/"
                                  "cublas/lib:/usr/local/lib/python3.11/site-packages/"
                                  "nvidia/cudnn/lib"}))


@app.function(image=image, gpu=GPU, volumes=({"/data": vol} if vol is not None else {}), timeout=10800,
              max_containers=21)
def transcribe(slug: str, model: str = "large-v3"):
    _require_volume()
    slug = _component(slug, "slug")
    import json, os, subprocess, time
    src = _data_path(slug, "source_audio.opus")
    out = _data_path(slug, "source_transcript.json")
    if not os.path.exists(src):
        return {"slug": slug, "ok": False, "why": "没有 source_audio.opus"}
    if os.path.exists(out) and os.path.getsize(out) > 10000:
        with open(out, encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("model") == model and cached.get("film") == slug and isinstance(cached.get("segments"), list):
            return {"slug": slug, "ok": True, "why": "已有，跳过", "words": cached.get("n_words", 0)}

    from faster_whisper import WhisperModel
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix="video-editing-preprocess-") as workdir:
        t0 = time.time()
        wav = f"{workdir}/a.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1",
                        "-ar", "16000", wav], check=True)
        m = WhisperModel(model, device="cuda", compute_type="float16")
        segs, info = m.transcribe(wav, word_timestamps=True, vad_filter=True,
                                  beam_size=5, language="en")
        o = []
        for s in segs:
            o.append({"start": round(s.start, 3), "end": round(s.end, 3),
                      "text": s.text.strip(),
                      "words": [{"w": w.word.strip(), "s": round(w.start, 3),
                                 "e": round(w.end, 3), "p": round(w.probability, 2)}
                                for w in (s.words or [])]})
        nw = sum(len(x["words"]) for x in o)
        dur = o[-1]["end"] if o else 0.0
        _write_json(out, {"film": slug, "source": "source", "model": model,
                         "audio_dur_s": round(dur, 1), "n_segments": len(o), "n_words": nw,
                         "segments": o})
        vol.commit()
        dt = (time.time() - t0) / 60
        print(f"[{slug}] {dt:.1f}分  {len(o)} 段 / {nw} 词  覆盖 {dur/60:.0f} 分钟", flush=True)
        return {"slug": slug, "ok": True, "segments": len(o), "words": nw, "min": round(dt, 1)}


@app.function(image=image, volumes=({"/data": vol} if vol is not None else {}), timeout=600)
def list_ready():
    _require_volume()
    import os
    return sorted(d for d in os.listdir(DATA_ROOT)
                  if os.path.isdir(_data_path(d)) and os.path.isfile(_data_path(d, "source_audio.opus")))


@app.local_entrypoint()
def main(model: str = "large-v3", only: str = ""):
    _require_volume()
    if only:
        only = _component(only, "slug")
    slugs = list_ready.remote()
    if only:
        slugs = [s for s in slugs if only == s]
    if not slugs:
        raise RuntimeError("No source_audio.opus inputs match; extract and upload the intended inputs first")
    print(f"转录 {len(slugs)} 部正片  模型 {model}\n")
    res = list(transcribe.starmap([(s, model) for s in slugs]))
    ok = sum(1 for r in res if r.get("ok"))
    print(f"\n成功 {ok}/{len(res)}")
    for r in sorted(res, key=lambda x: -(x.get("words") or 0)):
        print(f"  {r['slug']:<24} " + (f"{r.get('words',0):>6} 词  {r.get('min','—')}分"
                                       if r.get("ok") else f"★ {r.get('why')}"))
    if ok != len(res):
        raise RuntimeError(f"Source transcription failed for {len(res) - ok} input(s)")


def _local_main(argv=None):
    import argparse, json, os, subprocess
    from concurrent.futures import ThreadPoolExecutor

    ROOT = os.path.abspath(os.path.expanduser(os.environ.get("VIDEO_EDITING_DATA_ROOT", "").strip() or "data/private"))
    CACHE = _local_path(ROOT, "exports", "_film_audio")
    P = argparse.ArgumentParser(description="Extract, upload, or fetch source-film audio metadata.")
    actions = P.add_mutually_exclusive_group(required=True)
    actions.add_argument("--extract", action="store_true")
    actions.add_argument("--upload", action="store_true")
    actions.add_argument("--fetch", action="store_true")
    targets = P.add_mutually_exclusive_group(required=True)
    targets.add_argument("--film", help="one exact film slug")
    targets.add_argument("--all", action="store_true", help="explicitly process every available input")
    P.add_argument("-j", type=int, default=4)
    a = P.parse_args(argv)
    if a.j < 1:
        P.error("-j must be positive")
    if a.film is not None:
        try:
            a.film = _component(a.film, "slug")
        except ValueError as exc:
            P.error(str(exc))

    def probes():
        exports = _local_path(ROOT, "exports")
        slugs = [a.film] if a.film else sorted(p.parent.name for p in exports.glob("*/probe.json"))
        # Validate the whole selected batch before running any media/service commands.
        return [(slug, _local_probe(ROOT, slug)) for slug in slugs]

    if a.extract:
        inputs = [(slug, _probe_source(probe)) for slug, probe in probes()]
        if not inputs:
            raise RuntimeError("No private source inputs found")
        os.makedirs(CACHE, exist_ok=True)

        def eng_stream(src):
            """Select an English-tagged audio track; otherwise use the default track."""
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                                "-show_entries", "stream=index:stream_tags=language",
                                "-of", "json", src], capture_output=True, text=True, check=True).stdout
            st = json.loads(r).get("streams", [])
            for i, x in enumerate(st):
                if (x.get("tags") or {}).get("language", "").lower() in ("eng", "en"):
                    return i
            return None

        def one(t):
            slug, src = t
            out = _local_path(ROOT, "exports", "_film_audio", f"{slug}.opus")
            if os.path.exists(out) and os.path.getsize(out) > 1e6:
                return slug, os.path.getsize(out), True
            i = eng_stream(src)
            pick = ["-map", f"0:a:{i}"] if i is not None else []
            from tempfile import TemporaryDirectory
            with TemporaryDirectory(prefix=".extract-", dir=CACHE) as temporary:
                candidate = os.path.join(temporary, "audio.opus")
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src] + pick +
                               ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                                "-b:a", "24k", candidate], capture_output=True, text=True, check=True)
                if not os.path.isfile(candidate) or os.path.getsize(candidate) == 0:
                    raise RuntimeError("Audio extraction produced no output")
                os.replace(candidate, out)
            print(f"  {slug:<24} {os.path.getsize(out)/1e6:6.1f} MB", flush=True)
            return slug, os.path.getsize(out), True

        with ThreadPoolExecutor(max_workers=a.j) as ex:
            res = list(ex.map(one, inputs))
        tot = sum(r[1] for r in res)
        print(f"\n{sum(1 for r in res if r[2])}/{len(res)} 部，共 {tot/1e6:.0f} MB")
    elif a.upload:
        _require_volume()
        paths = [CACHE / f"{a.film}.opus"] if a.film else sorted(CACHE.glob("*.opus"))
        inputs = []
        for path in paths:
            slug = _component(path.stem, "slug")
            path = _local_path(ROOT, "exports", "_film_audio", f"{slug}.opus")
            if not path.is_file():
                raise FileNotFoundError(f"No extracted source audio for {slug}: {path}")
            inputs.append((slug, path))
        if not inputs:
            raise RuntimeError("No extracted source audio to upload")
        for slug, p in inputs:
            subprocess.run(["modal", "volume", "put", "--force", VOLUME_NAME,
                            str(p), f"/{slug}/source_audio.opus"], check=True)
            print(f"  {slug} 传完", flush=True)
    elif a.fetch:
        _require_volume()
        inputs = probes()
        if not inputs:
            raise RuntimeError("No private source inputs to fetch")
        outputs = [(slug, _local_path(ROOT, "exports", slug, "source_transcript.json"))
                   for slug, _ in inputs]
        for slug, output in outputs:
            subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME,
                            f"/{slug}/source_transcript.json",
                            str(output)],
                           capture_output=True, check=True)
        print(f"取回 {len(outputs)} 份正片转录")
    return 0


if __name__ == "__main__":
    raise SystemExit(_local_main())
