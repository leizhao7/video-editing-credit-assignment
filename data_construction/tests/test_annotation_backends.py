"""Annotation adapters tested with synthetic events and a local fake executable."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from annotation_backend import AnnotationEvents, build_annotation_command
import release_config as config


FAKE_CLI = r'''
import json
import os
import sys

backend = os.environ["VIDEO_EDITING_ANNOTATION_BACKEND"]
failed = os.environ.get("ANNOTATION_TEST_FAILURE") == "1"
print(json.dumps({"type": "test.capture", "argv": sys.argv[1:],
                  "cwd": os.getcwd(), "stdin": sys.stdin.read()}), flush=True)
if backend == "claude":
    terminal = {"type": "result", "is_error": failed}
    if failed:
        terminal["errors"] = ["synthetic provider failure"]
elif backend == "codex":
    terminal = ({"type": "turn.failed", "error": {"message": "synthetic provider failure"}}
                if failed else {"type": "turn.completed", "usage": {"input_tokens": 123}})
else:
    terminal = ({"type": "error", "error": {"name": "SyntheticError",
                 "data": {"message": "synthetic provider failure"}}}
                if failed else {"type": "step_finish", "part": {"id": "end",
                 "reason": "stop", "tokens": {"input": 123, "output": 4}}})
print(json.dumps(terminal), flush=True)
print("synthetic stderr", file=sys.stderr)
sys.exit(int(os.environ.get("ANNOTATION_TEST_EXIT", "0")))
'''


class AnnotationBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="annotation adapter ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "task with spaces"
        self.task.mkdir()
        self.cli = self.root / "fake tools" / "annotation cli"
        self.cli.parent.mkdir()
        self.cli.write_text("#!" + sys.executable + "\n" + FAKE_CLI, encoding="utf-8")
        self.cli.chmod(0o700)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {
            "VIDEO_EDITING_ANNOTATION_BACKEND": "claude",
            "VIDEO_EDITING_ANNOTATION_CLI": str(self.cli),
            "VIDEO_EDITING_ANNOTATION_MODEL": "synthetic/provider-model",
            "VIDEO_EDITING_ANNOTATION_EFFORT": "",
            "ANNOTATION_TEST_FAILURE": "0",
            "ANNOTATION_TEST_EXIT": "0",
        }))

    def build(self, backend, prompt="synthetic prompt", **kwargs):
        os.environ["VIDEO_EDITING_ANNOTATION_BACKEND"] = backend
        return build_annotation_command(prompt, self.task, tools="Read,Write,Bash(ffmpeg:*)",
                                        effort="max", **kwargs)

    def invoke(self, command):
        parser = AnnotationEvents(command.backend)
        with config.annotation_process(command.argv, cwd=self.task,
                                       stderr_path=self.root / "stderr log.txt") as process:
            stdout, _ = process.communicate(timeout=5)
            returncode = process.returncode
        lines = stdout.splitlines()
        for line in lines:
            parser.feed(line)
        capture = next(json.loads(line) for line in lines
                       if json.loads(line).get("type") == "test.capture")
        return returncode, capture, parser

    def test_real_child_preserves_arguments_cwd_and_stdin_eof_for_all_backends(self):
        prompt = 'Keep "quotes" and 中文\n$(touch shell_marker); `touch backtick_marker` $HOME --model fake'
        for backend in ("claude", "codex", "opencode"):
            with self.subTest(backend=backend):
                command = self.build(backend, prompt)
                returncode, capture, parser = self.invoke(command)
                self.assertEqual(returncode, 0)
                self.assertEqual(capture["argv"], command.argv[1:])
                self.assertEqual(capture["argv"].count(prompt), 1)
                self.assertEqual(capture["argv"][capture["argv"].index("--model") + 1],
                                 "synthetic/provider-model")
                self.assertEqual(Path(capture["cwd"]), self.task.resolve())
                self.assertEqual(capture["stdin"], "")
                self.assertFalse((self.task / "shell_marker").exists())
                self.assertFalse((self.task / "backtick_marker").exists())
                self.assertEqual(parser.failures(), [])
                self.assertIsNone(parser.cost)
                self.assertIn("synthetic stderr", (self.root / "stderr log.txt").read_text())
                if backend == "claude":
                    self.assertEqual(capture["argv"][0], "-p")
                    self.assertIn("--allowedTools", capture["argv"])
                else:
                    self.assertEqual(capture["argv"][0], "exec" if backend == "codex" else "run")
                    self.assertNotIn("--allowedTools", capture["argv"])
                    self.assertNotIn("--permission-mode", capture["argv"])
                    self.assertNotIn("--auto", capture["argv"])
                    self.assertNotIn("--dangerously-skip-permissions", capture["argv"])

    def test_exit_zero_does_not_hide_reported_provider_failure(self):
        os.environ["ANNOTATION_TEST_FAILURE"] = "1"
        for backend in ("claude", "codex", "opencode"):
            with self.subTest(backend=backend):
                returncode, _, parser = self.invoke(self.build(backend))
                self.assertEqual(returncode, 0)
                self.assertTrue(parser.failures())
                self.assertIn("synthetic provider failure", " ".join(parser.failures()))

    def test_process_preserves_nonzero_exit_even_after_completion_event(self):
        os.environ["ANNOTATION_TEST_EXIT"] = "7"
        returncode, _, parser = self.invoke(self.build("codex"))
        self.assertTrue(parser.completed)
        self.assertEqual(returncode, 7)

    def test_backend_and_model_errors_fail_before_launch(self):
        with self.assertRaisesRegex(ValueError, "BACKEND"):
            self.build("unknown-backend")
        os.environ["VIDEO_EDITING_ANNOTATION_MODEL"] = " \t"
        with self.assertRaisesRegex(RuntimeError, "MODEL"):
            self.build("claude")

    def test_claude_mcp_path_is_one_argument_and_other_backends_reject_it(self):
        path = self.root / "native config" / "server settings.json"
        command = self.build("claude", mcp_config=str(path))
        self.assertEqual(command.argv[command.argv.index("--mcp-config") + 1], str(path.resolve()))
        self.assertIn("--strict-mcp-config", command.argv)
        for backend in ("codex", "opencode"):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "Claude-only"):
                self.build(backend, mcp_config=str(path))

    def test_effort_uses_native_flags_without_inheriting_claude_max(self):
        for backend in ("codex", "opencode"):
            command = self.build(backend)
            self.assertNotIn("--variant", command.argv)
            self.assertFalse(any("model_reasoning_effort" in value for value in command.argv))
        os.environ["VIDEO_EDITING_ANNOTATION_EFFORT"] = " high "
        claude = self.build("claude").argv
        codex = self.build("codex").argv
        opencode = self.build("opencode").argv
        self.assertEqual(claude[claude.index("--effort") + 1], "high")
        self.assertIn('model_reasoning_effort="high"', codex)
        self.assertEqual(opencode[opencode.index("--variant") + 1], "high")

    def test_claude_tools_deduplicate_and_report_only_explicit_cost(self):
        parser = AnnotationEvents("claude")
        tool = {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"file_path": "local"}}]}}
        self.assertEqual(parser.feed(json.dumps(tool)), [{"name": "Read", "input": {"file_path": "local"}}])
        self.assertEqual(parser.feed(json.dumps(tool)), [])
        parser.feed(json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}}))
        parser.feed(json.dumps({"type": "result", "is_error": False, "total_cost_usd": 0}))
        self.assertEqual(parser.rate_limit, {"status": "allowed"})
        self.assertEqual(parser.cost, 0)
        self.assertEqual(parser.failures(), [])

    def test_codex_started_and_completed_count_once_and_reconnect_can_recover(self):
        parser = AnnotationEvents("codex")
        item = {"id": "command-1", "type": "command_execution", "command": "ffprobe input.mp4"}
        self.assertEqual(parser.feed(json.dumps({"type": "item.started", "item": item})),
                         [{"name": "Bash", "input": {"command": "ffprobe input.mp4"}}])
        self.assertEqual(parser.feed(json.dumps({"type": "item.completed", "item": item})), [])
        parser.feed(json.dumps({"type": "error", "message": "reconnecting"}))
        self.assertTrue(parser.failures())
        parser.feed(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1000,
                                                                    "output_tokens": 10}}))
        self.assertEqual(parser.failures(), [])
        self.assertIsNone(parser.cost)
        self.assertEqual(parser.rate_limit, {})

    def test_codex_terminal_failure_is_not_cleared_by_completion(self):
        parser = AnnotationEvents("codex")
        parser.feed(json.dumps({"type": "turn.failed", "error": {"message": "provider stopped"}}))
        parser.feed(json.dumps({"type": "turn.completed"}))
        self.assertIn("provider stopped", " ".join(parser.failures()))

    def test_codex_file_and_mcp_progress_are_normalized(self):
        parser = AnnotationEvents("codex")
        changes = [{"path": "stages/script.json", "kind": "update"}]
        self.assertEqual(parser.feed(json.dumps({"type": "item.completed", "item": {
            "id": "change-1", "type": "file_change", "changes": changes}})),
            [{"name": "file_change", "input": {"changes": changes}}])
        self.assertEqual(parser.feed(json.dumps({"type": "item.started", "item": {
            "id": "mcp-1", "type": "mcp_tool_call", "tool": "inspect", "arguments": {"frame": 1}}})),
            [{"name": "inspect", "input": {"frame": 1}}])

    def test_opencode_cost_sums_unique_steps_and_tool_errors_can_recover(self):
        parser = AnnotationEvents("opencode")
        tool = {"type": "tool_use", "part": {"id": "part-1", "callID": "call-1", "tool": "bash",
                "state": {"status": "error", "input": {"command": "ffprobe missing.mp4"},
                          "error": "synthetic missing file"}}}
        self.assertEqual(parser.feed(json.dumps(tool)),
                         [{"name": "Bash", "input": {"command": "ffprobe missing.mp4"}}])
        self.assertEqual(parser.feed(json.dumps(tool)), [])
        step = {"type": "step_finish", "part": {"id": "step-1", "reason": "tool-calls", "cost": 0.03}}
        parser.feed(json.dumps(step))
        parser.feed(json.dumps(step))
        self.assertFalse(parser.completed)
        parser.feed(json.dumps({"type": "step_finish", "part": {
            "id": "step-2", "reason": "stop", "cost": 0.02}}))
        self.assertAlmostEqual(parser.cost, 0.05)
        self.assertEqual(parser.failures(), [])

    def test_opencode_session_error_remains_fatal_even_with_stop(self):
        parser = AnnotationEvents("opencode")
        parser.feed(json.dumps({"type": "error", "error": {"name": "APIError",
                         "data": {"message": "provider failed"}}}))
        parser.feed(json.dumps({"type": "step_finish", "part": {"id": "end", "reason": "stop"}}))
        self.assertIn("provider failed", " ".join(parser.failures()))

    def test_incomplete_streams_and_non_event_lines_never_imply_success(self):
        for backend in ("claude", "codex", "opencode"):
            with self.subTest(backend=backend):
                parser = AnnotationEvents(backend)
                for line in ("", "not json", "null", "[]", '{"type":"unknown"}',
                             '{"type":"assistant","message":null}',
                             '{"type":"item.started","item":"bad payload"}',
                             '{"type":"step_finish","part":"bad payload"}'):
                    self.assertEqual(parser.feed(line), [])
                self.assertFalse(parser.completed)
                self.assertTrue(parser.failures())
                self.assertIsNone(parser.cost)


if __name__ == "__main__":
    unittest.main()
