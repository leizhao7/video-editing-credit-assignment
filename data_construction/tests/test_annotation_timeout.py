"""Real local processes exercise total deadlines independently of stdout activity."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_config as config


TIMEOUT_ENV = "VIDEO_EDITING_ANNOTATION_TIMEOUT_S"


@unittest.skipUnless(os.name == "posix", "annotation process groups require POSIX")
class AnnotationTimeoutTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="annotation deadline ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(TIMEOUT_ENV, None)

    def process(self, source):
        return config.annotation_process([sys.executable, "-u", "-c", source],
                                         cwd=self.root, stderr_path=self.root / "stderr.log")

    def assert_cleaned(self, process):
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed)
        self.assertNotIn(f"annotation-timeout-{process.pid}",
                         {thread.name for thread in threading.enumerate()})

    def test_unset_timeout_does_not_create_a_timer(self):
        with mock.patch.object(config.threading, "Thread") as thread:
            with self.process("print('finished')") as process:
                self.assertEqual(process.stdout.read().strip(), "finished")
                self.assertEqual(process.wait(timeout=5), 0)
            thread.assert_not_called()
        self.assert_cleaned(process)

    def test_normal_completion_cancels_timer_without_waiting_for_deadline(self):
        os.environ[TIMEOUT_ENV] = "60"
        started = time.monotonic()
        with self.process("print('finished')") as process:
            self.assertEqual(process.stdout.read().strip(), "finished")
            self.assertEqual(process.wait(timeout=5), 0)
        self.assertLess(time.monotonic() - started, 5)
        self.assert_cleaned(process)

    def test_stdout_heartbeats_cannot_extend_the_total_deadline(self):
        os.environ[TIMEOUT_ENV] = "0.3"
        started = time.monotonic()
        lines = []
        with self.assertRaisesRegex(TimeoutError, "0.3s total runtime limit"):
            with self.process("import time\nwhile True:\n print('heartbeat', flush=True)\n time.sleep(0.02)") as process:
                lines.extend(process.stdout)
                process.wait(timeout=5)
        self.assertGreaterEqual(len(lines), 2)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(process.returncode, -signal.SIGTERM)
        self.assert_cleaned(process)

    def test_timeout_reaches_children_escalates_and_leaves_other_group_alive(self):
        os.environ[TIMEOUT_ENV] = "0.5"
        marker = self.root / "child-terminated"
        child = (
            "import pathlib,signal,sys,time\n"
            f"def stop(*args):\n pathlib.Path({str(marker)!r}).write_text('TERM'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\nprint('child-ready', flush=True)\ntime.sleep(60)"
        )
        parent = (
            "import signal,subprocess,sys,time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGCHLD, signal.SIG_IGN)\n"
            f"subprocess.Popen([sys.executable, '-u', '-c', {child!r}])\n"
            "while True:\n print('heartbeat', flush=True)\n time.sleep(0.02)"
        )
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            with self.assertRaisesRegex(TimeoutError, "process group was terminated"):
                with self.process(parent) as process:
                    lines = list(process.stdout)
                    process.wait(timeout=5)
            self.assertIn("child-ready\n", lines)
            self.assertEqual(marker.read_text(), "TERM")
            self.assertEqual(process.returncode, -signal.SIGKILL)
            self.assertIsNone(unrelated.poll())
            self.assert_cleaned(process)
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_caller_exception_cancels_timer_and_preserves_the_original_error(self):
        os.environ[TIMEOUT_ENV] = "60"
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with self.process("import time;time.sleep(60)") as process:
                raise RuntimeError("caller failure")
        self.assert_cleaned(process)

    def test_exited_leader_does_not_exempt_a_child_holding_stdout(self):
        os.environ[TIMEOUT_ENV] = "0.5"
        marker = self.root / "orphan-terminated"
        child = (
            "import pathlib,signal,sys,time\n"
            f"def stop(*args):\n pathlib.Path({str(marker)!r}).write_text('TERM'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "while True:\n print('child heartbeat', flush=True)\n time.sleep(0.02)"
        )
        parent = ("import subprocess,sys\n"
                  f"subprocess.Popen([sys.executable, '-u', '-c', {child!r}])")
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "total runtime limit"):
            with self.process(parent) as process:
                self.assertEqual(process.wait(timeout=5), 0)
                lines = list(process.stdout)
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(marker.read_text(), "TERM")
        self.assertLess(time.monotonic() - started, 5)
        self.assert_cleaned(process)

    def test_early_exit_cleans_children_even_when_the_leader_already_exited(self):
        os.environ[TIMEOUT_ENV] = "60"
        child = "import os,time;print(os.getpid(), flush=True);time.sleep(60)"
        parent = ("import subprocess,sys\n"
                  f"subprocess.Popen([sys.executable, '-u', '-c', {child!r}])")
        for caller_error in (False, True):
            with self.subTest(caller_error=caller_error):
                process = None
                try:
                    try:
                        with self.process(parent) as process:
                            child_pid = int(process.stdout.readline())
                            self.assertEqual(process.wait(timeout=5), 0)
                            if caller_error:
                                raise RuntimeError("early caller failure")
                    except RuntimeError as exc:
                        self.assertTrue(caller_error)
                        self.assertEqual(str(exc), "early caller failure")
                    else:
                        self.assertFalse(caller_error)
                    self.assert_cleaned(process)
                    # Orphan reaping is OS-dependent; a zombie is terminated, not alive.
                    deadline = time.monotonic() + 2
                    while True:
                        state = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="],
                                               capture_output=True, text=True, timeout=5).stdout.strip()
                        if not state or state.startswith("Z"):
                            break
                        self.assertLess(time.monotonic(), deadline,
                                        "the exited leader's child survived context cleanup")
                        time.sleep(0.02)
                finally:
                    # Keep this regression test from leaking the child if cleanup breaks.
                    if process is not None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_invalid_timeouts_fail_before_starting_a_process(self):
        for value in ("", " ", "no", "0", "-1", "nan", "NaN", "inf", "-inf", "1e999"):
            with self.subTest(value=value), mock.patch.object(config.subprocess, "Popen") as popen:
                os.environ[TIMEOUT_ENV] = value
                with self.assertRaisesRegex(ValueError, "finite positive number of seconds"):
                    with self.process("print('must not run')"):
                        self.fail("invalid timeout entered the process context")
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
