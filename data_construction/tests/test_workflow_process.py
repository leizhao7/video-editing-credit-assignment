"""Exercise workflow subprocess cleanup using real local processes, never services."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="workflow process groups require POSIX")


def test_ctrl_c_unwinds_annotation_and_stops_its_detached_cli(tmp_path):
    """The annotation CLI has a different session from the workflow child.

    Killing only the workflow child's group with SIGTERM bypasses its Python
    finally block and leaves the CLI running. Keep this test in a separate
    harness process so its deliberate SIGINT never interrupts the test runner.
    """
    worker_pid = tmp_path / "worker.pid"
    cli_pid = tmp_path / "cli.pid"
    cleaned = tmp_path / "annotation-cleaned"
    worker = tmp_path / "annotation_worker.py"
    cli_source = (
        "import os,pathlib,time\n"
        f"pathlib.Path({str(cli_pid)!r}).write_text(str(os.getpid()))\n"
        "print('annotation-ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    worker.write_text(
        "import os,pathlib,sys\n"
        "from data_construction.release_config import annotation_process\n"
        f"pathlib.Path({str(worker_pid)!r}).write_text(str(os.getpid()))\n"
        "try:\n"
        f" with annotation_process([sys.executable, '-u', '-c', {cli_source!r}], "
        f"cwd={str(tmp_path)!r}, stderr_path={str(tmp_path / 'cli.stderr')!r}) as child:\n"
        "  for line in child.stdout:\n"
        "   print(line, end='', flush=True)\n"
        "  child.wait()\n"
        "finally:\n"
        f" pathlib.Path({str(cleaned)!r}).write_text('cleaned')\n",
        encoding="utf-8",
    )
    harness = (
        "import os,pathlib,signal,sys,threading,time\n"
        "from data_construction.workflow import run_command\n"
        "def interrupt_when_running():\n"
        " deadline = time.monotonic() + 10\n"
        f" while not pathlib.Path({str(cli_pid)!r}).exists():\n"
        "  if time.monotonic() > deadline:\n"
        "   raise RuntimeError('annotation CLI did not start')\n"
        "  time.sleep(0.02)\n"
        " os.kill(os.getpid(), signal.SIGINT)\n"
        "threading.Thread(target=interrupt_when_running, daemon=True).start()\n"
        "try:\n"
        f" run_command([sys.executable, '-u', {str(worker)!r}], dict(os.environ), "
        f"pathlib.Path({str(tmp_path / 'workflow.log')!r}))\n"
        "except KeyboardInterrupt:\n"
        " print('workflow-interrupted', flush=True)\n"
        " sys.exit(130)\n"
        "raise RuntimeError('workflow unexpectedly returned without interruption')\n"
    )
    env = dict(os.environ)
    env.pop("VIDEO_EDITING_ANNOTATION_TIMEOUT_S", None)
    env.pop("GOOGLE_AI_STUDIO_API_KEY", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", harness], cwd=REPO, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=25)
        assert process.returncode == 130, output
        assert "workflow-interrupted" in output
        assert cleaned.read_text() == "cleaned"
        for path in (worker_pid, cli_pid):
            pid = int(path.read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        # A regression must not leave the intentionally long-lived test CLI.
        for path in (cli_pid, worker_pid):
            if path.exists():
                try:
                    os.killpg(int(path.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)


def test_command_output_redacts_key_from_console_and_private_log(tmp_path, capsys):
    from data_construction.workflow import run_command

    fake_key = "test-only-secret-not-a-real-provider-key"
    env = dict(os.environ, GOOGLE_AI_STUDIO_API_KEY=fake_key)
    log = tmp_path / "step.log"
    code = run_command(
        [sys.executable, "-c", "import os;print(os.environ['GOOGLE_AI_STUDIO_API_KEY'])"],
        env, log,
    )
    assert code == 0
    for output in (capsys.readouterr().out, log.read_text()):
        assert fake_key not in output
        assert "[REDACTED]" in output
