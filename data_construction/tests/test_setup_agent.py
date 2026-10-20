"""Local-only setup tests, including actual shell sourcing of generated profiles."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_agent as setup


class SetupAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="agent setup tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "profile with spaces.env"

    def invoke(self, args, *, found=True, incompatible=False, check_result=True):
        stdout, stderr = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            inspect = stack.enter_context(mock.patch.object(setup, "inspect_cli", return_value=(found, incompatible)))
            check = stack.enter_context(mock.patch.object(setup, "run_check", return_value=({"ok": check_result}, self.root / "logs")))
            try:
                result = setup.main(args)
            except SystemExit as exc:
                result = exc.code
        return result, stdout.getvalue(), stderr.getvalue(), inspect, check

    def arguments(self, backend="claude", model="test/model", output=None):
        return ["--backend", backend, "--model", model, "--data-root", str(self.root / "private data"),
                "--output", str(output or self.output)]

    def sourced(self, *profiles):
        keys = ["VIDEO_EDITING_DATA_ROOT", "VIDEO_EDITING_ANNOTATION_BACKEND", "VIDEO_EDITING_ANNOTATION_CLI",
                "VIDEO_EDITING_ANNOTATION_MODEL", "VIDEO_EDITING_ANNOTATION_EFFORT", "VIDEO_EDITING_MCP_CONFIG"]
        program = "import json, os; print(json.dumps({key: os.environ.get(key) for key in " + repr(keys) + "}))"
        commands = [f"source {shlex.quote(str(profile))}" for profile in profiles]
        commands.append(shlex.quote(sys.executable) + " -c " + shlex.quote(program))
        result = subprocess.run(["/bin/bash", "-c", " && ".join(commands)], capture_output=True,
                                text=True, check=True, env={**os.environ, "VIDEO_EDITING_MCP_CONFIG": "stale-config"})
        return json.loads(result.stdout)

    def test_real_shell_preserves_metacharacters_and_switch_clears_backend_settings(self):
        marker = self.root / "must not exist"
        hostile_model = "provider/model 'quoted' $(touch " + shlex.quote(str(marker)) + ") `false` $HOME"
        status, _, _, _, check = self.invoke(self.arguments(model=hostile_model)
                                             + ["--cli", "./fake tool", "--effort", "high"])
        self.assertEqual(status, 0)
        check.assert_not_called()
        first = self.sourced(self.output)
        self.assertEqual(first["VIDEO_EDITING_ANNOTATION_MODEL"], hostile_model)
        self.assertEqual(first["VIDEO_EDITING_ANNOTATION_CLI"], str(Path("./fake tool").resolve()))
        self.assertEqual(first["VIDEO_EDITING_ANNOTATION_EFFORT"], "high")
        self.assertEqual(first["VIDEO_EDITING_MCP_CONFIG"], "")
        self.assertFalse(marker.exists())
        second_path = self.root / "second.env"
        self.assertEqual(self.invoke(self.arguments("codex", "second/model", second_path))[0], 0)
        second = self.sourced(self.output, second_path)
        self.assertEqual(second["VIDEO_EDITING_ANNOTATION_BACKEND"], "codex")
        self.assertEqual(second["VIDEO_EDITING_ANNOTATION_MODEL"], "second/model")
        self.assertEqual(second["VIDEO_EDITING_ANNOTATION_CLI"], "")
        self.assertEqual(second["VIDEO_EDITING_ANNOTATION_EFFORT"], "")
        self.assertEqual(second["VIDEO_EDITING_MCP_CONFIG"], "")
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        self.assertEqual(sum(line.startswith("export ") for line in self.output.read_text().splitlines()), 6)

    def test_existing_profile_requires_force_and_force_restores_private_mode(self):
        self.output.write_text("original")
        self.output.chmod(0o644)
        status, _, error, inspect, check = self.invoke(self.arguments())
        self.assertEqual(status, 2)
        self.assertIn("--force", error)
        self.assertEqual(self.output.read_text(), "original")
        inspect.assert_not_called()
        check.assert_not_called()
        self.assertEqual(self.invoke(self.arguments() + ["--force"])[0], 0)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)

    def test_repo_paths_and_symlink_output_are_rejected(self):
        target = self.root / "precious.txt"
        target.write_text("unchanged")
        self.output.symlink_to(target)
        self.assertEqual(self.invoke(self.arguments() + ["--force"])[0], 2)
        self.assertEqual(target.read_text(), "unchanged")
        self.output.unlink()
        for option, value in (("--data-root", setup.REPO / "private"), ("--output", setup.REPO / "profile.env")):
            args = self.arguments()
            args[args.index(option) + 1] = str(value)
            self.assertEqual(self.invoke(args)[0], 2)
        alias = self.root / "repo alias"
        alias.symlink_to(setup.REPO, target_is_directory=True)
        args = self.arguments()
        args[args.index("--output") + 1] = str(alias / "profile.env")
        self.assertEqual(self.invoke(args)[0], 2)

    def test_check_is_explicit_and_forwards_exact_values(self):
        args = self.arguments("codex") + ["--cli", "./custom cli", "--effort", "high", "--timeout", "12.5", "--check"]
        status, output, _, _, check = self.invoke(args)
        self.assertEqual(status, 0)
        self.assertIn("may incur provider usage", output)
        check.assert_called_once_with("codex", "test/model", cli=str(Path("./custom cli").resolve()),
                                      effort="high", timeout=12.5)
        self.assertIn('--cli "$VIDEO_EDITING_ANNOTATION_CLI"', output)
        self.assertIn('--effort "$VIDEO_EDITING_ANNOTATION_EFFORT"', output)

    def test_missing_or_incompatible_cli_prevents_requested_real_check(self):
        for found, incompatible in ((False, False), (True, True)):
            status, output, _, _, check = self.invoke(self.arguments("opencode") + ["--check", "--force"],
                                                      found=found, incompatible=incompatible)
            self.assertEqual(status, 1)
            check.assert_not_called()
            self.assertIn("not started", output)
            self.assertTrue(self.output.exists())

    def test_missing_cli_reports_login_and_guide_but_still_saves(self):
        output = io.StringIO()
        with mock.patch.object(setup.shutil, "which", return_value=None), \
                mock.patch.object(setup, "run_check") as check, redirect_stdout(output):
            self.assertEqual(setup.main(self.arguments("opencode")), 0)
        check.assert_not_called()
        self.assertIn("CLI not found", output.getvalue())
        self.assertIn("opencode auth login", output.getvalue())
        self.assertIn("https://opencode.ai/docs/", output.getvalue())

    def test_default_probes_only_version_and_prints_no_raw_process_output(self):
        stdout = io.StringIO()
        fake = subprocess.CompletedProcess([], 0, "opencode v2.1.3\nsecret-stdout", "secret-stderr")
        with mock.patch.object(setup.shutil, "which", return_value="/fake/opencode"), \
                mock.patch.object(setup.subprocess, "run", return_value=fake) as run, \
                mock.patch.object(setup, "run_check") as check, redirect_stdout(stdout):
            self.assertEqual(setup.main(self.arguments("opencode")), 0)
        check.assert_not_called()
        self.assertEqual(run.call_args.args[0], ["/fake/opencode", "--version"])
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        self.assertIn("2.1.3", stdout.getvalue())
        self.assertIn("Incompatible", stdout.getvalue())
        self.assertNotIn("secret-", stdout.getvalue())

    def test_help_and_non_tty_missing_model_never_probe_or_call(self):
        with mock.patch.object(setup.sys.stdin, "isatty", return_value=False):
            for args, expected in ((["--help"], 0), ([], 2), (["--backend", "codex"], 2)):
                status, _, _, inspect, check = self.invoke(args)
                self.assertEqual(status, expected)
                inspect.assert_not_called()
                check.assert_not_called()
        status, _, error, inspect, check = self.invoke(self.arguments("opencode", "missing-provider"))
        self.assertEqual(status, 2)
        self.assertIn("opencode models", error)
        self.assertFalse(self.output.exists())
        inspect.assert_not_called()
        check.assert_not_called()

    def test_no_argument_tty_requires_model_and_uses_backend_specific_default_profile(self):
        with mock.patch.object(setup.Path, "home", return_value=self.root), \
                mock.patch.object(setup.sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", side_effect=["codex", "", "chosen/model", "", "", ""]):
            status, _, _, _, check = self.invoke([])
        self.assertEqual(status, 0)
        check.assert_not_called()
        profile = self.root / ".config/video-editing-credit-assignment/codex.env"
        values = self.sourced(profile)
        self.assertEqual(values["VIDEO_EDITING_ANNOTATION_MODEL"], "chosen/model")
        self.assertEqual(values["VIDEO_EDITING_DATA_ROOT"], str(self.root / "video-editing-data"))


if __name__ == "__main__":
    unittest.main()
