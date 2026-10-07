"""Sandbox-side HTTP-to-stdio relay. Requires no network access or API key.

The trusted Modal parent forwards only the fixed chat-completions endpoint.
Actual HTTP/SSE bytes are transported unchanged; stderr/CLI logs have a separate
frame type. Never execute this runtime on a developer/training host.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pwd
import queue
import subprocess
import sys
import threading
import uuid

from .contract import Contract, ID
from .runtime import prepare_runtime, verify_binary

MAX_BODY = 64 * 1024 * 1024


class Relay:
    def __init__(self, emit, timeout=600, *, visual_budget=None):
        self.emit, self.timeout = emit, timeout
        self.pending = {}
        self.lock = threading.Lock()
        self.visual_budget = visual_budget

    def accept(self, frame):
        with self.lock:
            target = self.pending.get(frame.get("id"))
        if target is not None:
            target.put(frame, timeout=self.timeout)

    def handler(self):
        relay = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path not in ("/v1/chat/completions", "/ocrl/images") or self.headers.get("Transfer-Encoding"):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(400)
                    return
                if not 0 < length <= MAX_BODY:
                    self.send_error(413)
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self.close_connection = True
                    return
                if self.path == '/ocrl/images':
                    try:
                        if relay.visual_budget is None:
                            raise ValueError('Shared image gate is not enabled')
                        result = relay.visual_budget.register(json.loads(body))
                        relay.emit({'kind': 'runtime_media_observation', 'observation': result})
                        payload, status = json.dumps(result).encode(), 200
                    except Exception as exc:
                        relay.emit({'kind': 'runtime_media_error', 'error': str(exc)[:1000]})
                        payload, status = json.dumps({'error': str(exc)[:1000]}).encode(), 400
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if relay.visual_budget is not None:
                    try:
                        relay.visual_budget.check_request(json.loads(body))
                    except Exception as exc:
                        relay.emit({'kind': 'runtime_media_error', 'error': str(exc)[:1000]})
                        payload = json.dumps({'error': {'message': str(exc)[:1000], 'type': 'invalid_image_observation'}}).encode()
                        self.send_response(400)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Content-Length', str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                        return
                request_id = uuid.uuid4().hex
                inbox = queue.Queue(maxsize=128)
                with relay.lock:
                    relay.pending[request_id] = inbox
                try:
                    relay.emit({"kind": "request", "id": request_id,
                        "headers": {key: value for key, value in self.headers.items()
                                    if key.lower().startswith("x-ocrl-")},
                        "body_b64": base64.b64encode(body).decode()})
                    first = inbox.get(timeout=relay.timeout)
                    if first["kind"] != "response_start":
                        raise ValueError("Missing HTTP response start")
                    self.send_response(int(first["status"]))
                    self.send_header("Content-Type", first["content_type"])
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        frame = inbox.get(timeout=relay.timeout)
                        if frame["kind"] == "response_end":
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                            break
                        if frame["kind"] != "response_chunk":
                            raise ValueError("Invalid response frame")
                        chunk = base64.b64decode(frame["body_b64"], validate=True)
                        if chunk:
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                            self.wfile.flush()
                except (OSError, ValueError, KeyError, queue.Empty) as exc:
                    self.close_connection = True
                    relay.emit({"kind": "cancel", "id": request_id, "error": type(exc).__name__})
                finally:
                    with relay.lock:
                        relay.pending.pop(request_id, None)
        return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--stage", required=True, choices=("s_1", "s_2", "s_3", "s_4e"))
    parser.add_argument("--music-catalog", help="Trusted precomputed task music JSON, never policy-supplied")
    args = parser.parse_args()
    if not ID.fullmatch(args.attempt):
        raise ValueError("Invalid attempt ID")
    if os.geteuid() != 0 or not Path("/opt/OCRL_SANDBOX_IMAGE").is_file() or Path.cwd() != Path("/work"):
        raise RuntimeError("OpenCode may only run inside the dedicated isolated sandbox")
    from .eval_contract import load_contract
    contract = load_contract(json.loads(Path(args.contract).read_bytes()))
    verify_binary("/usr/local/bin/opencode", contract)
    runtime = Path("/var/lib/ocrl") / args.attempt
    music_catalog = json.loads(Path(args.music_catalog).read_bytes()) if args.music_catalog else None
    environment = prepare_runtime(runtime, json.loads(Path(args.config).read_bytes()), contract,
                                  attempt_id=args.attempt, stage=args.stage, music_catalog=music_catalog)
    user = pwd.getpwnam("rollout")
    # Policy can write its DB/cache, but cannot replace the pinned config.
    runtime.chmod(0o755)
    for name in ("home", "data", "cache", "state"):
        os.chown(runtime / name, user.pw_uid, user.pw_gid)
    # Pinned core/npm.ts skips dependency installs when config is read-only.
    # Our plugins use built-in modules/fetch only; provider SDK is in the binary.
    # This also prevents generated global agents/plugins from being discovered.
    (runtime / "config").chmod(0o555)
    (runtime / "opencode.json").chmod(0o444)
    lock = threading.Lock()

    def emit(frame):
        with lock:
            sys.stdout.write(json.dumps(frame, separators=(",", ":")) + "\n")
            sys.stdout.flush()

    if music_catalog is not None:
        # Parent stores both the exact catalogue and its identity with the
        # complete attempt; train/eval observations are replayed, never rejudged.
        from .contract import digest
        emit({"kind": "runtime_music_catalog", "catalog": music_catalog,
              "catalog_sha256": digest(music_catalog)})

    from .visual_budget import VisualBudget, prepared_frame_bytes
    relay = Relay(emit, visual_budget=VisualBudget(args.stage, preframes=prepared_frame_bytes()))
    server = ThreadingHTTPServer(("127.0.0.1", 18080), relay.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def read_frames():
        for line in sys.stdin:
            relay.accept(json.loads(line))
        # Parent loss is not a license to keep generating without a journal.
        with relay.lock:
            targets = list(relay.pending.items())
        for request_id, _ in targets:
            relay.accept({"kind": "parent_disconnected", "id": request_id})

    threading.Thread(target=read_frames, daemon=True).start()

    from .submission_policy import submission_prompt
    from .policy_limits import resource_prompt
    command = ["/usr/sbin/runuser", "--preserve-environment", "-u", "rollout", "--",
               "/usr/local/bin/python", "/opt/opencode_rl/policy_limits.py", "/usr/local/bin/opencode",
               "run", "--format", "json", "--agent", "build", "--",
               resource_prompt(submission_prompt(Path(args.task).read_text()))]
    # Never preexec_fn after starting threads (fork lock deadlock). Never let
    # OpenCode inherit relay stdin: run.ts reads non-TTY stdin to EOF.
    process = subprocess.Popen(command, cwd="/work", env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    def drain(stream, kind):
        while chunk := stream.read1(65536):
            emit({"kind": kind, "body_b64": base64.b64encode(chunk).decode()})

    readers = [threading.Thread(target=drain, args=(stream, kind))
               for stream, kind in ((process.stdout, "runtime_stdout"), (process.stderr, "runtime_stderr"))]
    for reader in readers:
        reader.start()
    returncode = process.wait()
    for reader in readers:
        reader.join()
    emit({"kind": "runtime_exit", "returncode": returncode, "runtime_path": str(runtime)})
    server.shutdown()
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
