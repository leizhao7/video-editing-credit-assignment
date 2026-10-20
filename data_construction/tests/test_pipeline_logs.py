"""Stage-log tests using only temporary files and synthetic Python subprocesses."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


class PipelineLogs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pipeline logs ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(pipeline, "ROOT", str(self.root)))
        self.stack.enter_context(mock.patch.object(pipeline, "SRC", str(self.root / "sources")))
        self.console = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.console))

    def test_complete_streams_and_parsed_stdout_are_preserved(self):
        path = self.root / "logs" / "probe.log"
        code = ("import json,sys; "
                "print(json.dumps({'text':'synthetic 中文'})); "
                "sys.stderr.write('stderr start\\n'+'x'*12000+'\\nstderr end\\n')")
        with pipeline._stage_logging(path):
            result = pipeline.sh([sys.executable, "-c", code, "UNPRINTED_ARGUMENT"],
                                 env=dict(os.environ, SYNTHETIC_SECRET="UNPRINTED_ENVIRONMENT"))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout), {"text": "synthetic 中文"})
        self.assertEqual(result.stderr, "stderr start\n" + "x" * 12000 + "\nstderr end\n")
        log = path.read_text(encoding="utf-8")
        self.assertIn(result.stdout, log)
        self.assertIn("x" * 12000, log)
        self.assertIn("stderr start", log)
        self.assertIn("stderr end", log)
        self.assertNotIn("UNPRINTED_ARGUMENT", log)
        self.assertNotIn("UNPRINTED_ENVIRONMENT", log)
        self.assertIsNone(pipeline._STAGE_LOG.get())

    def test_failed_stage_prints_final_error_and_retains_entire_log(self):
        film = pipeline.Film("Synthetic failure")
        def fail():
            pipeline.sh_checked([sys.executable, "-c",
                "import sys;print('BUILD LOG START');print('b'*9000);"
                "sys.stderr.write('trace start\\n'+'e'*3000+'\\nFINAL BUILD FAILURE\\n');sys.exit(7)"])
        film.shots = fail
        self.assertFalse(pipeline._run(film, ["shots"]))
        path = Path(film.E) / "logs/shots.log"
        log = path.read_text(encoding="utf-8")
        self.assertIn("BUILD LOG START", log)
        self.assertIn("b" * 9000, log)
        self.assertIn("e" * 3000, log)
        self.assertIn("--- exit 7 ---", log)
        self.assertIn("Traceback", log)
        self.assertIn("FINAL BUILD FAILURE", self.console.getvalue())
        self.assertIn(str(path), self.console.getvalue())
        self.assertFalse(film.state["shots"]["ok"])
        self.assertEqual(film.state["shots"]["log"], str(path))
        self.assertIsNone(pipeline._STAGE_LOG.get())

    def test_attempts_append_and_do_not_capture_later_unrelated_commands(self):
        film = pipeline.Film("Synthetic retry")
        film.cards = lambda: pipeline.sh_checked([sys.executable, "-c", "print('FIRST ATTEMPT');raise SystemExit(1)"])
        self.assertFalse(pipeline._run(film, ["cards"]))
        def succeed():
            pipeline.sh_checked([sys.executable, "-c", "print('SECOND ATTEMPT')"])
            film.mark("cards")
            return "done"
        film.cards = succeed
        self.assertTrue(pipeline._run(film, ["cards"]))
        path = Path(film.E) / "logs/cards.log"
        before = path.read_text(encoding="utf-8")
        self.assertEqual(before.count("=== attempt "), 2)
        self.assertIn("FIRST ATTEMPT", before)
        self.assertIn("SECOND ATTEMPT", before)
        result = pipeline.sh([sys.executable, "-c", "print('unrelated command')"])
        self.assertIn("unrelated command", result.stdout)
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_parallel_films_keep_their_logs_separate(self):
        barrier = threading.Barrier(2)
        films = [pipeline.Film("Synthetic alpha"), pipeline.Film("Synthetic beta")]
        def make_stage(film, marker):
            def stage():
                barrier.wait(timeout=5)
                pipeline.sh_checked([sys.executable, "-c",
                    "import sys,time;print(sys.argv[1],flush=True);time.sleep(.03);"
                    "print(sys.argv[1],file=sys.stderr)", marker])
                film.mark("audio")
                return marker
            return stage
        for film, marker in zip(films, ("ALPHA_ONLY", "BETA_ONLY")):
            film.audio = make_stage(film, marker)
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(list(pool.map(lambda film: pipeline._run(film, ["audio"]), films)), [True, True])
        for film, own, other in ((films[0], "ALPHA_ONLY", "BETA_ONLY"),
                                 (films[1], "BETA_ONLY", "ALPHA_ONLY")):
            log = (Path(film.E) / "logs/audio.log").read_text(encoding="utf-8")
            self.assertEqual(log.count(own), 2)
            self.assertNotIn(other, log)
        self.assertIsNone(pipeline._STAGE_LOG.get())

    def test_output_is_visible_in_log_before_child_finishes(self):
        film = pipeline.Film("Synthetic streaming")
        gate = self.root / "finish gate"
        code = ("import pathlib,sys,time;print('CHILD READY',flush=True);"
                "gate=pathlib.Path(sys.argv[1]);deadline=time.monotonic()+5\n"
                "while not gate.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                "print('CHILD DONE')")
        film.probe = lambda: pipeline.sh_checked([sys.executable, "-c", code, str(gate)])
        path = Path(film.E) / "logs/probe.log"
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pipeline._run, film, ["probe"])
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if path.exists() and "CHILD READY" in path.read_text(encoding="utf-8"):
                        break
                    time.sleep(.01)
                self.assertIn("CHILD READY", path.read_text(encoding="utf-8"))
                self.assertFalse(future.done())
                self.assertIn(str(path), self.console.getvalue())
            finally:
                gate.touch()
            self.assertTrue(future.result(timeout=5))
        self.assertIn("CHILD DONE", path.read_text(encoding="utf-8"))

    def test_logged_timeout_keeps_partial_output_and_resets_context(self):
        path = self.root / "timeout.log"
        with self.assertRaises(subprocess.TimeoutExpired):
            with pipeline._stage_logging(path):
                pipeline.sh([sys.executable, "-c", "import time;print('before timeout',flush=True);time.sleep(10)"],
                            timeout=.2)
        self.assertIn("before timeout", path.read_text(encoding="utf-8"))
        self.assertIsNone(pipeline._STAGE_LOG.get())


if __name__ == "__main__":
    unittest.main()
