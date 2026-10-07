"""Root-owned tool transport; policy has no chat route, secrets or host access."""
import argparse
import base64
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import pwd
import select
import subprocess
import sys
import threading
import time

from opencode_rl.contract import ID, ParityError, canonical, digest
from opencode_rl.runtime import prepare_runtime, verify_binary
from opencode_rl.stdio_relay import Relay
from opencode_rl.visual_budget import VisualBudget, prepared_frame_bytes
from opencode_rl.store import atomic_bytes
from opencode_rl.submission_policy import submission_prompt
from opencode_rl.policy_limits import resource_prompt
from .contract import AppendContract, validate_runtime


def stderr_chunks(stream, *, max_bytes=65536, max_delay=.1):
    """Losslessly coalesce diagnostic fragments before the durable wire journal.

    Native metadata updates can emit tens of thousands of ~80-byte log lines
    during one Bash command. Each wire frame requires durable blob+journal I/O
    at the parent, so forwarding every read1 fragment can strand tool results
    behind minutes of diagnostic traffic after the command has already exited.
    Bound both memory and flush latency; never drop, decode or truncate stderr.
    This stream is diagnostic-only and is never rendered into policy context.
    """
    if type(max_bytes) is not int or max_bytes < 1 or not 0 < max_delay <= 1:
        raise ValueError('Positive bounded stderr batching limits required')
    fd = stream.fileno()
    pending = bytearray()
    deadline = None
    while True:
        remaining = None if deadline is None else max(0., deadline - time.monotonic())
        try:
            ready, _, _ = select.select([fd], [], [], remaining)
        except InterruptedError:
            continue
        if not ready:
            if pending:
                yield bytes(pending)
                pending.clear()
                deadline = None
            continue
        body = os.read(fd, max_bytes - len(pending))
        if not body:
            if pending:
                yield bytes(pending)
            return
        if not pending:
            deadline = time.monotonic() + max_delay
        pending.extend(body)
        if len(pending) == max_bytes or time.monotonic() >= deadline:
            yield bytes(pending)
            pending.clear()
            deadline = None


def media_handler(visual, emit):
    # Reuse the exact protected normalization/registration implementation, but
    # remove its chat path. The tool-only process cannot ask another model.
    native = Relay(emit, visual_budget=visual).handler()
    class MediaOnly(native):
        def do_POST(self):
            if self.path != '/ocrl/images':
                self.send_error(403)
                return
            super().do_POST()
    return MediaOnly


def validate_output(frame, visual):
    if frame.get('kind') == 'profile':
        profile = frame['profile']
        if (profile.get('format') != 'append-opencode-tools-v1' or profile.get('native_agent_loop') is not False or
                not ID.fullmatch(profile.get('session_id', ''))):
            raise ParityError('Unexpected native tool profile')
    elif frame.get('kind') == 'tool_result':
        messages = []
        for result in frame['result']['results']:
            content = [{'type': 'image_url', 'image_url': {'url': p['url']}}
                       for p in result['attachments']]
            if any(not p.get('mime', '').startswith('image/') for p in result['attachments']):
                raise ParityError('Unsupported tool attachment')
            messages.append({'role': 'tool', 'content': content})
        # A plugin omission/tampered URL cannot reach inference. This validates
        # NEW observations; the engine separately enforces total-context count.
        visual.check_request({'messages': messages})
    else:
        raise ParityError('Unexpected tool-host stdout frame')
    return frame


def main():
    parser = argparse.ArgumentParser()
    for name in ('contract', 'config', 'attempt', 'task', 'stage'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--music-catalog')
    args = parser.parse_args()
    if (os.geteuid() != 0 or Path.cwd() != Path('/work') or
            not Path('/opt/OCRL_SANDBOX_IMAGE').is_file() or not ID.fullmatch(args.attempt)):
        raise ParityError('Broker requires a new isolated rollout sandbox')
    contract = AppendContract(**json.loads(Path(args.contract).read_bytes()))
    config = json.loads(Path(args.config).read_bytes())
    validate_runtime(config, contract)
    verify_binary('/usr/local/bin/opencode-tools', contract)
    catalog = json.loads(Path(args.music_catalog).read_bytes()) if args.music_catalog else None
    root = Path('/var/lib/ocrl') / args.attempt
    environment = prepare_runtime(root, config, contract, attempt_id=args.attempt, stage=args.stage, music_catalog=catalog)
    user = pwd.getpwnam('rollout')
    root.chmod(0o755)
    for name in ('home', 'data', 'cache', 'state'):
        os.chown(root / name, user.pw_uid, user.pw_gid)
    (root / 'config').chmod(0o555)
    (root / 'opencode.json').chmod(0o444)
    task = root / 'public_task.txt'
    atomic_bytes(task, resource_prompt(submission_prompt(Path(args.task).read_text())).encode())
    task.chmod(0o444)
    lock = threading.Lock()
    def emit(frame):
        with lock:
            sys.stdout.write(canonical(frame).decode() + '\n')
            sys.stdout.flush()
    if catalog is not None:
        emit({'kind': 'runtime_music_catalog', 'catalog': catalog, 'catalog_sha256': digest(catalog)})
    visual = VisualBudget(args.stage, preframes=prepared_frame_bytes())
    server = ThreadingHTTPServer(('127.0.0.1', 18080), media_handler(visual, emit))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    process = subprocess.Popen(['/usr/sbin/runuser', '--preserve-environment', '-u', 'rollout', '--',
        '/usr/local/bin/python', '-m', 'append_agent_rl.policy_limits', str(task)],
        cwd='/work', env={**environment, 'PYTHONPATH': '/opt'}, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    errors = []
    def inputs():
        try:
            for line in sys.stdin:
                if len(line) > 16 * 1024**2:
                    raise ParityError('Oversized tool command')
                frame = json.loads(line)
                if (set(frame) != {'id', 'method', 'action'} or frame['method'] != 'step' or
                        not ID.fullmatch(frame['id'])):
                    raise ParityError('Unknown broker command')
                process.stdin.write(canonical(frame) + b'\n')
                process.stdin.flush()
        except BaseException as exc:
            errors.append(str(exc))
            process.terminate()
        finally:
            process.stdin.close()
    def stderr():
        try:
            for body in stderr_chunks(process.stderr):
                emit({'kind': 'runtime_stderr', 'body_b64': base64.b64encode(body).decode()})
        except BaseException as exc:
            errors.append('stderr transport failed: ' + str(exc))
            process.terminate()
    reader = threading.Thread(target=inputs, daemon=True)
    err = threading.Thread(target=stderr, daemon=True)
    reader.start()
    err.start()
    try:
        for line in process.stdout:
            if len(line) > 100_000_000:
                raise ParityError('Oversized tool observation')
            emit(validate_output(json.loads(line), visual))
        code = process.wait()
        err.join(timeout=5)
        if err.is_alive():
            raise ParityError('Tool-host stderr transport did not drain after exit')
        if errors or code:
            raise ParityError(f'Tool-host failed: rc={code}; {errors[:1]}')
        emit({'kind': 'runtime_exit', 'returncode': code, 'runtime_path': str(root)})
    finally:
        if process.poll() is None:
            process.terminate()
        server.shutdown()


if __name__ == '__main__':
    main()
