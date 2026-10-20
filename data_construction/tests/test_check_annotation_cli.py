"""Smoke-checker regression tests using synthetic inputs and fake child CLIs."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_annotation_cli as checker


# Only the fake process decodes RGB bytes. A real agent is instructed to use vision.
FAKE = r'''
import json, os, pathlib, struct, sys, time, zlib
mode = os.environ.get("SMOKE_TEST_MODE", "ok")
if mode == "timeout":
    time.sleep(20)
    raise SystemExit(0)
payload = json.loads(pathlib.Path("input.json").read_text())
image = pathlib.Path("colors.png").read_bytes()
cursor, compressed = 8, b""
while cursor < len(image):
    length = struct.unpack(">I", image[cursor:cursor+4])[0]
    kind = image[cursor+4:cursor+8]
    if kind == b"IDAT": compressed += image[cursor+8:cursor+8+length]
    cursor += 12 + length
row = zlib.decompress(compressed)
names = {(255,0,0): "red", (0,200,0): "green", (0,0,255): "blue", (255,255,0): "yellow"}
result = {"nonce": payload["nonce"], "total": sum(payload["numbers"]),
          "left_color": names[tuple(row[1:4])], "right_color": names[tuple(row[193:196])]}
if mode == "wrong_result": result["total"] += 1
pathlib.Path("result.json").write_text(json.dumps(result))
if mode == "input_changed": pathlib.Path("input.json").write_text("{}")
if mode == "extra_file": pathlib.Path("extra.txt").write_text("synthetic")
backend = {"exec": "codex", "run": "opencode"}.get(sys.argv[1], "claude")
if mode != "no_completion":
    failure = mode == "error_zero"
    event = ({"type": "result", "is_error": failure} if backend == "claude" else
             {"type": "turn.failed" if failure else "turn.completed"} if backend == "codex" else
             {"type": "error"} if failure else {"type": "step_finish", "part": {"reason": "stop"}})
    print(json.dumps(event), flush=True)
print("synthetic-secret-do-not-print", file=sys.stderr)
raise SystemExit(7 if mode == "exit7" else 0)
'''


class CheckAnnotationCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="fake smoke cli ")
        self.addCleanup(directory.cleanup)
        self.cli = Path(directory.name) / "fake cli"
        self.cli.write_text("#!" + sys.executable + "\n" + FAKE, encoding="utf-8")
        self.cli.chmod(0o700)

    def invoke(self, backend="claude", mode="ok", timeout=5):
        with mock.patch.dict(os.environ, {"SMOKE_TEST_MODE": mode}):
            result, directory = checker.run_check(backend, "synthetic/model", cli=str(self.cli), timeout=timeout)
        self.addCleanup(shutil.rmtree, directory, True)
        return result, directory

    def test_generated_png_has_valid_chunks_and_two_panels(self):
        data = checker.two_color_png((255, 0, 0), (0, 0, 255))
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        cursor, compressed, kinds = 8, b"", []
        while cursor < len(data):
            length = struct.unpack(">I", data[cursor:cursor+4])[0]
            kind = data[cursor+4:cursor+8]
            content = data[cursor+8:cursor+8+length]
            crc = struct.unpack(">I", data[cursor+8+length:cursor+12+length])[0]
            self.assertEqual(crc, zlib.crc32(kind + content) & 0xffffffff)
            kinds.append(kind)
            if kind == b"IDAT":
                compressed += content
            cursor += length + 12
        self.assertEqual(kinds, [b"IHDR", b"IDAT", b"IEND"])
        rows = zlib.decompress(compressed)
        self.assertEqual(len(rows), 64 * (1 + 128 * 3))
        self.assertEqual(rows[:385], b"\x00" + bytes((255, 0, 0)) * 64 + bytes((0, 0, 255)) * 64)

    def test_three_backends_validate_generated_outputs_and_keep_logs_outside_repo(self):
        for backend in ("claude", "codex", "opencode"):
            with self.subTest(backend=backend):
                result, directory = self.invoke(backend)
                self.assertTrue(result["ok"], result)
                self.assertIn(" ", directory.name)
                self.assertNotIn(Path(checker.__file__).resolve().parents[1], directory.parents)
                self.assertEqual(json.loads((directory / "logs/summary.json").read_text()), result)
                self.assertTrue((directory / "logs/session.jsonl").is_file())
                self.assertTrue((directory / "logs/stderr.log").is_file())

    def test_valid_file_cannot_override_session_failure_or_nonzero_exit(self):
        for backend in ("claude", "codex", "opencode"):
            for mode in ("error_zero", "exit7", "no_completion"):
                with self.subTest(backend=backend, mode=mode):
                    result, _ = self.invoke(backend, mode)
                    self.assertFalse(result["ok"])
                    self.assertTrue(result["errors"])

    def test_wrong_result_changed_input_and_extra_file_fail(self):
        for mode in ("wrong_result", "input_changed", "extra_file"):
            with self.subTest(mode=mode):
                result, _ = self.invoke(mode=mode)
                self.assertFalse(result["ok"])

    def test_total_timeout_terminates_a_silent_child(self):
        result, _ = self.invoke(mode="timeout", timeout=0.15)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertNotEqual(result["returncode"], 0)
        self.assertLess(result["elapsed_seconds"], 5)

    def test_help_does_not_start_a_cli_and_timeout_must_be_positive_and_finite(self):
        with mock.patch.object(checker, "run_check") as run, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                checker.main(["--help"])
            self.assertEqual(result.exception.code, 0)
            run.assert_not_called()
        for value in ("0", "-1", "nan", "inf", "bad"):
            with self.subTest(value=value), self.assertRaises(Exception):
                checker.positive_seconds(value)

    def test_console_never_prints_provider_logs(self):
        actual = checker.run_check
        directories = []
        def capture(*args, **kwargs):
            result, directory = actual(*args, **kwargs)
            directories.append(directory)
            return result, directory
        output = io.StringIO()
        try:
            with mock.patch.dict(os.environ, {"SMOKE_TEST_MODE": "error_zero"}), \
                    mock.patch.object(checker, "run_check", side_effect=capture), redirect_stdout(output):
                status = checker.main(["--backend", "codex", "--model", "synthetic/model", "--cli", str(self.cli)])
            self.assertEqual(status, 1)
            self.assertIn("Local inputs and logs:", output.getvalue())
            self.assertNotIn("synthetic-secret", output.getvalue())
        finally:
            for directory in directories:
                shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
