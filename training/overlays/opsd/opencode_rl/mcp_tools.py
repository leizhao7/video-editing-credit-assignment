"""Expose the existing video/audio/image observation worker to real OpenCode."""
from __future__ import annotations

import importlib.util
import copy
import json
import os
from pathlib import Path
import signal
import sys
import threading

from .contract import digest, ParityError
from .music_observation import MusicLookup, TOOL_SCHEMA
from .editing_tools import EditingTools, SCHEMAS as EDITING_SCHEMAS, number as editing_number


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MediaTools:
    def __init__(self, worker, schemas, stage, *, music_catalog=None, editing=None):
        if stage not in ("s_1", "s_2", "s_3", "s_4e"):
            raise ValueError("Unknown task stage")
        self.worker, self.stage = worker, stage
        self.schemas = [copy.deepcopy(item) for item in schemas if item["name"] in ("view_image", "inspect_video", "inspect_audio")]
        descriptions = {
            'view_image': '查看预抽帧或自制图片，每次1–32张，其中自制图片最多8张。所有stage均不设累计新增图片配额。单次模型请求最多999张图片，包括保留的历史图片；无独立总像素额度，仍受模型token上下文及传输大小上限约束。大图按实际像素记录用量，单图最长边1792。',
            'inspect_video': '自由查看完整源clip或自制视频。path默认给全片8张均匀抽样图；可传timestamps_s或start_s/end_s/num_frames，不混用。每次1–8帧、长边最多448，按实际解码PTS返回最近帧和真实时间；不是连续播放。所有stage均不设累计新增图片配额，单次模型请求含历史最多999张图；无独立总像素额度，保留模型token上下文与传输大小保护。',
        }
        for item in self.schemas:
            if item['name'] in descriptions:
                item['description'] = descriptions[item['name']]
        self.images = self.audio_calls = 0
        self.audio_seconds = 0.0
        self.scan_seconds = 0.0
        self._call_lock = threading.RLock()
        self.editing = editing if editing is not None else EditingTools()
        self.schemas.extend(EDITING_SCHEMAS)
        self.music = MusicLookup(music_catalog) if music_catalog is not None else None
        if self.music is not None:
            self.schemas.append(TOOL_SCHEMA)

    def call(self, name, args):
        # One MCP worker is serial. Keep direct callers serial too, and restore
        # all reservations if decoding, timeout or response construction fails.
        with self._call_lock:
            before = (self.images, self.audio_calls, self.audio_seconds, self.scan_seconds)
            try:
                return self._call(name, args)
            except BaseException:
                self.images, self.audio_calls, self.audio_seconds, self.scan_seconds = before
                raise

    def _call(self, name, args):
        bounded = self.stage in ("s_3", "s_4e")
        units = 0
        if name == "listen_music" and self.music is not None:
            result = self.music.call(args)
        elif name in ('contact_sheet', 'inspect_frames'):
            # Reject malformed/oversized single calls before decoding. Cumulative
            # visual use is logged, not capped; the common request gate remains.
            self.editing.visual_source_minimum(name, args)
            result = getattr(self.editing, name)(args)
            units = result.pop('_visual_source_units')
            self.images += units
        elif name == 'validate_media':
            requested = editing_number(args.get('duration_s', 180), 'duration_s', .05, 180)
            start = editing_number(args.get('start_s', 0), 'start_s')
            seconds = self.editing.probe(self.editing.path(args['path']))['duration_s'] if 'path' in args else 0
            seconds = min(requested, max(0, seconds - start))
            if bounded and self.scan_seconds + seconds > 360:
                raise ValueError('Technical media scan quota exceeded')
            self.scan_seconds += seconds
            result = self.editing.validate_media(args)
        elif name == "view_image":
            paths = args.get("paths", [])
            if not isinstance(paths, list) or not 1 <= len(paths) <= 32:
                raise ValueError("view_image requires 1–32 paths")
            predefined = [self.worker.public_path(p).is_relative_to(Path("/work/frames")) for p in paths]
            custom = sum(not item for item in predefined)
            if custom > 8:
                raise ValueError("At most 8 custom images per view_image call")
            self.images += custom
            units = custom
            # Existing observation worker accepts at most 8 images per call;
            # fan out deterministically so a 32-frame request is never truncated.
            images = []
            for start in range(0, len(paths), 8):
                images.extend(self.worker.dispatch({"name": name, "arguments": {"paths": paths[start:start + 8]}})["images"])
            result = {"status": "ok", "images": images}
        elif name == "inspect_video":
            count = self.worker.requested_video_frames(args)
            self.images += count
            units = count
            # OpenCode overlay only: keep the legacy RL fallback worker intact.
            # Exact decoded PTS avoids successful seeks that produce no EOF JPEG.
            result = self.editing.inspect_video(args)
        elif name in ("inspect_audio", "analyze_audio"):
            seconds = editing_number(args.get("duration_s", 20), "duration_s", .05, 30)
            if bounded and (self.audio_calls + 1 > 12 or self.audio_seconds + seconds > 180):
                raise ValueError("Audio inspection quota exceeded")
            self.audio_calls += 1
            self.audio_seconds += seconds
            result = (self.editing.analyze_audio(args) if name == 'analyze_audio'
                      else self.worker.dispatch({"name": name, "arguments": args}))
        else:
            raise ValueError("Unknown media tool")
        images = result.pop("images", [])
        content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False, allow_nan=False)}]
        for image in images:
            content += [{"type": "text", "text": image["label"]},
                        {"type": "image", "data": image["jpeg_b64"], "mimeType": "image/jpeg"}]
        return {"content": content, "isError": False, '_meta': {'ocrl_visual_source_units': units}}


def main():
    if not Path("/opt/OCRL_SANDBOX_IMAGE").is_file():
        raise RuntimeError("Media tools must run inside the isolated sandbox")
    worker = load_module("ocrl_observation_worker", "/opt/s12-tools/worker.py")
    schema = load_module("ocrl_observation_schema", "/opt/s12-tools/schema.py")
    catalog = None
    if os.environ.get("OCRL_MUSIC_POLICY_SHA256"):
        from .music_observation import policy_sha256
        if os.environ["OCRL_MUSIC_POLICY_SHA256"] != policy_sha256():
            raise ParityError("Music tool policy differs from pinned config")
        catalog = json.loads(Path(os.environ["OCRL_MUSIC_CATALOG"]).read_bytes())
        if digest(catalog) != os.environ["OCRL_MUSIC_CATALOG_SHA256"]:
            raise ParityError("Task music catalogue changed")
    tools = MediaTools(worker, schema.FUNCTIONS, os.environ["S12_STAGE"], music_catalog=catalog)
    def deadline_expired(signum, frame):
        raise TimeoutError('Media tool exceeded 120 seconds; split the request into smaller windows/groups')
    # Bound the WHOLE tool, not just each ffmpeg subprocess. A 16-source sheet
    # must not continue decoding after OpenCode's 150-second MCP timeout.
    signal.signal(signal.SIGALRM, deadline_expired)
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        try:
            method = request["method"]
            if method == "initialize":
                result = {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}},
                          "serverInfo": {"name": "s12-media", "version": "1.0.0"}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [{"name": item["name"], "description": item["description"], "inputSchema": item["parameters"]}
                                    for item in tools.schemas]}
            elif method == "tools/call":
                try:
                    signal.setitimer(signal.ITIMER_REAL, 120)
                    result = tools.call(request["params"]["name"], request["params"].get("arguments", {}))
                except Exception as exc:
                    result = {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            else:
                raise ValueError("Unsupported MCP method")
            response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            response = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32602, "message": str(exc)}}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
