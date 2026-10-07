"""Trusted PRE-rollout music preparation. Never installed as an agent tool.

Manifest: {"assets": [{"music_id": "music_0", "source_path": "/trusted/music.wav",
"source_sha256": "...", "public_path": "initial_state/music.wav"}]}.
The launcher must supply only explicitly designated original music, NOT discover
files from a policy-writable /work or guess that every audio file is music.
Output is a complete task-scoped catalogue; failures never become fake labels.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import wave

from .contract import HASH, ID, ParityError, canonical, digest
from .music_observation import (MODEL, PROMPT, OUTPUT_SCHEMA, WINDOW_SECONDS, MAX_ASSETS_PER_TASK,
                               MAX_WINDOWS_PER_ASSET, policy_sha256, public_music_path,
                               validate_catalog, validate_labels)
from .store import atomic_bytes

ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
MAX_SOURCE_BYTES = 512 * 1024 * 1024
PCM_RATE = 22050


def fixed_payload(wav):
    # Audio must be canonical PCM WAV with no metadata. Even a trusted caller
    # cannot smuggle an arbitrary text part or File/URL into this request builder.
    with wave.open(io.BytesIO(wav), "rb") as audio:
        frames = audio.getnframes()
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (2, 2, PCM_RATE, "NONE"):
            raise ParityError("Expected bounded canonical music PCM")
        if not 0 < frames <= WINDOW_SECONDS * PCM_RATE:
            raise ParityError("Music segment exceeds bound")
        pcm = audio.readframes(frames)
        if len(pcm) != frames * 4:
            raise ParityError("Truncated music PCM")
    clean = io.BytesIO()
    with wave.open(clean, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(PCM_RATE)
        audio.writeframes(pcm)
    if clean.getvalue() != wav:
        raise ParityError("Noncanonical WAV/metadata is forbidden")
    return {
        "systemInstruction": {"parts": [{"text": PROMPT}]},
        "contents": [{"role": "user", "parts": [{"inlineData": {
            "mimeType": "audio/wav", "data": base64.b64encode(wav).decode()}}]}],
        "generationConfig": {"temperature": 0, "candidateCount": 1, "maxOutputTokens": 4096,
                             "responseMimeType": "application/json", "responseJsonSchema": OUTPUT_SCHEMA},
    }


def source_info(asset):
    if type(asset) is not dict or set(asset) != {"music_id", "source_path", "source_sha256", "public_path"}:
        raise ParityError("Music sources must be explicitly registered by the trusted launcher")
    if type(asset["music_id"]) is not str or not ID.fullmatch(asset["music_id"]):
        raise ParityError("Invalid music ID")
    public_music_path(asset["public_path"])
    sha = asset["source_sha256"]
    if type(sha) is not str or not HASH.fullmatch(sha):
        raise ParityError("Original source hash is required")
    path = Path(asset["source_path"])
    if (not path.is_absolute() or path.is_symlink() or any(p.is_symlink() for p in path.parents)
            or not path.is_file() or not 0 < path.stat().st_size <= MAX_SOURCE_BYTES):
        raise ParityError("Invalid original music file")
    if path.is_relative_to("/work"):
        raise ParityError("Never prepare music from the policy-writable sandbox")
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != sha:
            raise ParityError("Original music was changed")
    return path


def media_command(argv, *, timeout=40):
    result = subprocess.run(argv, capture_output=True, timeout=timeout)
    if result.returncode:
        # Raw stderr may contain source paths/metadata; do not send it to policy.
        raise ParityError("Original music decoding failed")
    return result.stdout


def source_duration(path):
    body = media_command([shutil.which("ffprobe") or "ffprobe", "-v", "error",
        "-protocol_whitelist", "file,pipe", "-select_streams", "a:0", "-show_entries",
        "stream=index,duration:format=duration", "-of", "json", str(path)])
    data = json.loads(body)
    if not data.get("streams"):
        raise ParityError("Registered music has no audio stream")
    value = data["streams"][0].get("duration", data.get("format", {}).get("duration"))
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ParityError("Cannot establish original music duration") from exc
    if not math.isfinite(seconds) or not 0 < seconds <= WINDOW_SECONDS * MAX_WINDOWS_PER_ASSET:
        raise ParityError("Original music duration exceeds bound")
    return seconds


def extract_window(path, index):
    pcm = media_command([shutil.which("ffmpeg") or "ffmpeg", "-nostdin", "-v", "error", "-threads", "2",
        "-protocol_whitelist", "file,pipe", "-ss", str(index * WINDOW_SECONDS), "-i", str(path),
        "-t", str(WINDOW_SECONDS), "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        "-ac", "2", "-ar", str(PCM_RATE), "-threads", "2", "-f", "s16le", "pipe:1"])
    if len(pcm) % 4 or not 0 < len(pcm) <= WINDOW_SECONDS * PCM_RATE * 4:
        raise ParityError("Invalid bounded music decode")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(PCM_RATE)
        audio.writeframes(pcm)
    return stream.getvalue(), len(pcm) / (PCM_RATE * 4)


def parse_response(raw):
    body = json.loads(raw)
    candidates = body.get("candidates", [])
    if len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
        raise ParityError("Music response incomplete/blocked")
    parts = candidates[0].get("content", {}).get("parts", [])
    # Reasoning, free-form preambles, tool calls and extra parts are not policy observations.
    text = [p["text"] for p in parts if set(p) <= {"text", "thought", "thoughtSignature"}
            and not p.get("thought") and type(p.get("text")) is str]
    if len(text) != 1 or any("functionCall" in p for p in parts):
        raise ParityError("Unexpected music response parts")
    return validate_labels(json.loads(text[0])), body.get("modelVersion", MODEL)


class GeminiMusicPreparer:
    """Share one instance for bounded concurrency and coalescing across tasks."""
    def __init__(self, api_key, cache_dir, *, concurrency=16):
        if type(concurrency) is not int or not 1 <= concurrency <= 64 or not api_key:
            raise ValueError("Invalid trusted music preparation configuration")
        self._api_key = api_key
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.gate = asyncio.Semaphore(concurrency)
        self.decode_gate = asyncio.Semaphore(2)
        self.locks = {}

    async def observe(self, client, wav):
        payload = fixed_payload(wav)
        audio_sha = hashlib.sha256(wav).hexdigest()
        key = digest({"policy": policy_sha256(), "audio": audio_sha})
        folder = self.cache / key
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            ready = folder / "accepted.json"
            if ready.exists():
                saved = json.loads(ready.read_bytes())
                if saved["policy_sha256"] != policy_sha256() or saved["audio_sha256"] != audio_sha:
                    raise ParityError("Music cache identity mismatch")
                raw = (folder / saved["response_file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != saved["response_sha256"]:
                    raise ParityError("Music cache response changed")
                return parse_response(raw)[0]
            atomic_bytes(folder / "audio.wav", wav)
            atomic_bytes(folder / "request.json", canonical(payload))
            async with self.gate:
                # 429/temporary server failures get at most two retries; no
                # unbounded fallback model, endpoint, or larger prompt.
                for attempt in range(3):
                    response_file = f"response_{os.urandom(8).hex()}.json"
                    async with client.post(ENDPOINT, json=payload,
                            headers={"x-goog-api-key": self._api_key}, allow_redirects=False) as response:
                        chunks, size = [], 0
                        async for chunk in response.content.iter_chunked(65536):
                            size += len(chunk)
                            if size > 1024 * 1024:
                                raise ParityError("Music response exceeds bound")
                            chunks.append(chunk)
                        raw = b"".join(chunks)
                        atomic_bytes(folder / response_file, raw)
                        status = response.status
                    if status == 200:
                        labels, model_version = parse_response(raw)
                        atomic_bytes(ready, canonical({"policy_sha256": policy_sha256(), "audio_sha256": audio_sha,
                            "response_sha256": hashlib.sha256(raw).hexdigest(), "response_file": response_file,
                            "model_version": model_version}))
                        return labels
                    if status not in (429, 500, 502, 503, 504) or attempt == 2:
                        raise ParityError(f"Music preparation HTTP status {status}")
                    await asyncio.sleep(2 ** attempt)

    async def prepare(self, client, manifest):
        if type(manifest) is not dict or set(manifest) != {"assets"}:
            raise ParityError("Expected explicit original-music manifest")
        sources = manifest["assets"]
        if type(sources) is not list or len(sources) > MAX_ASSETS_PER_TASK:
            raise ParityError("Invalid task music source count")
        # Reject an invalid manifest before making any billable request.
        ids, paths = set(), set()
        for source in sources:
            await asyncio.to_thread(source_info, source)
            public_path = public_music_path(source["public_path"])
            if source["music_id"] in ids or public_path in paths:
                raise ParityError("Duplicate registered music")
            ids.add(source["music_id"])
            paths.add(public_path)
        # Decode one window per asset at a time; avoid a 240-window PCM memory
        # burst. Parallelism is across original tracks/tasks, not ffmpeg threads.
        async def prepare_asset(source):
            path = await asyncio.to_thread(source_info, source)
            seconds = await asyncio.to_thread(source_duration, path)
            windows = []
            for index in range(math.ceil(seconds / WINDOW_SECONDS)):
                async with self.decode_gate:
                    wav, decoded_seconds = await asyncio.to_thread(extract_window, path, index)
                labels = await self.observe(client, wav)
                windows.append({"index": index, "start_s": index * WINDOW_SECONDS,
                    "duration_s": decoded_seconds, "labels": labels})
            # Source must still be identical after reading every window.
            await asyncio.to_thread(source_info, source)
            return {"music_id": source["music_id"], "public_path": public_music_path(source["public_path"]),
                    "source_sha256": source["source_sha256"], "windows": windows}
        assets = await asyncio.gather(*(prepare_asset(source) for source in sources))
        return validate_catalog({"policy_sha256": policy_sha256(), "assets": assets})


async def main():
    import aiohttp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    if Path("/opt/OCRL_SANDBOX_IMAGE").exists():
        raise RuntimeError("Gemini music preparation must never run inside a rollout sandbox")
    if Path(args.output).exists():
        raise ParityError("Refuse to overwrite a frozen music catalogue")
    preparer = GeminiMusicPreparer(os.environ["GOOGLE_AI_STUDIO_API_KEY"], args.cache,
                                  concurrency=args.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as client:
        catalog = await preparer.prepare(client, json.loads(Path(args.manifest).read_bytes()))
    atomic_bytes(Path(args.output), canonical(catalog))
    print(json.dumps({"status": "prepared", "catalog_sha256": digest(catalog),
                      "assets": len(catalog["assets"]),
                      "windows": sum(len(a["windows"]) for a in catalog["assets"])}))


if __name__ == "__main__":
    asyncio.run(main())
