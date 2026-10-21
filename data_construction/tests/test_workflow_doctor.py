"""Doctor checks remain local and usable without installed pipeline packages."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workflow_doctor as doctor


class WorkflowDoctorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="workflow doctor ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.movie = self.root / "movie with spaces.mkv"
        self.trailer = self.root / "trailer.mp4"
        for path in (self.movie, self.trailer):
            path.write_bytes(b"local fixture, not real video")
        self.config = {"project": str(self.root), "film": "example", "inputs": {
            "movie": {"path": str(self.movie)}, "trailer": {"path": str(self.trailer)}},
            "agent": {"backend": "claude", "model": "sonnet", "cli": "", "effort": "high"},
            "modal_volume": "private-demo", "quality_model": "quality-test", "judge_model": "judge-test",
            "annotation_timeout_s": 900, "quality_timeout_s": 600}

    def local_run(self, command, **kwargs):
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["capture_output"])
        self.assertFalse(kwargs["check"])
        self.assertLessEqual(kwargs["timeout"], 30)
        self.assertNotIn("shell", kwargs)
        name = Path(command[0]).name
        if name == "ffprobe":
            self.assertEqual(command[-1], str(self.movie) if "movie" in command[-1] else str(self.trailer))
            output = json.dumps({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                                 "format": {"duration": "25.25"}})
        elif name == "tesseract":
            self.assertEqual(command[1:], ["--list-langs"])
            output = "List of available languages (2):\neng\nosd\n"
        elif name in doctor.BACKENDS:
            self.assertEqual(command[1:], ["--version"])
            output = name + " v1.2.3\nraw secret must not be printed"
        else:
            self.fail("Unexpected executable: " + name)
        return subprocess.CompletedProcess(command, 0, output, "private raw stderr")

    def invoke(self, scope="all", *, which=None, run=None, specs=True, key="test-secret-value"):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(doctor.shutil, "which", side_effect=which or (lambda name: "/local/" + name)))
            finder = stack.enter_context(mock.patch.object(doctor.importlib.util, "find_spec", return_value=object() if specs else None))
            runner = stack.enter_context(mock.patch.object(doctor.subprocess, "run", side_effect=run or self.local_run))
            stack.enter_context(mock.patch.dict(os.environ, {"GOOGLE_AI_STUDIO_API_KEY": key}))
            rows = doctor.check(self.config, scope)
        return {r["name"]: r for r in rows}, runner, finder

    def test_all_reports_success_with_only_allowed_local_commands_and_no_secrets(self):
        rows, runner, finder = self.invoke()
        self.assertTrue(all(row["ok"] for row in rows.values()), rows)
        self.assertEqual(sum(Path(call.args[0][0]).name == "ffprobe" for call in runner.call_args_list), 2)
        self.assertIn("dependency:PIL", rows)
        self.assertIn("dependency:modal", rows)
        self.assertIn("no cloud/model requests", rows["authentication_scope"]["detail"])
        encoded = json.dumps(rows)
        for secret in ("test-secret-value", "private raw stderr", "raw secret must not be printed"):
            self.assertNotIn(secret, encoded)
        self.assertEqual(set(next(iter(rows.values()))), {"name", "ok", "detail", "fix"})

    def test_build_needs_no_gemini_key_or_scoring_imports(self):
        rows, _, finder = self.invoke("build", key="")
        self.assertTrue(all(row["ok"] for row in rows.values()))
        self.assertNotIn("gemini_key", rows)
        self.assertNotIn("quality_model", rows)
        self.assertEqual({call.args[0] for call in finder.call_args_list}, set(doctor.BUILD_IMPORTS))

    def test_probe_is_a_cheap_local_check_without_cloud_annotation_or_key(self):
        self.config["agent"] = {}
        self.config["modal_volume"] = ""
        rows, runner, finder = self.invoke("probe", key="")
        self.assertTrue(all(row["ok"] for row in rows.values()))
        self.assertEqual({call.args[0] for call in finder.call_args_list}, {"numpy", "cv2"})
        self.assertEqual(len(runner.call_args_list), 2)
        self.assertTrue(all(Path(call.args[0][0]).name == "ffprobe" for call in runner.call_args_list))
        for name in ("annotation_cli", "modal_volume", "gemini_key", "tesseract"):
            self.assertNotIn(name, rows)

    def test_review_needs_key_but_never_probes_annotation_auth_or_media(self):
        self.config["agent"] = {}
        rows, runner, finder = self.invoke("review", key="")
        runner.assert_not_called()
        finder.assert_not_called()
        self.assertFalse(rows["gemini_key"]["ok"])
        self.assertNotIn("annotation_cli", rows)
        self.assertNotIn("judge_model", rows)
        self.assertIn("quality_model", rows)

    def test_score_checks_own_packages_only_without_local_subprocesses(self):
        rows, runner, finder = self.invoke("score")
        runner.assert_not_called()
        self.assertEqual({call.args[0] for call in finder.call_args_list}, set(doctor.SCORE_IMPORTS))
        self.assertIn("judge_model", rows)
        self.assertNotIn("quality_model", rows)
        self.assertTrue(all(row["ok"] for row in rows.values()))

    def test_preflight_needs_no_api_key_annotation_modal_or_external_packages(self):
        self.config["agent"] = {}
        self.config["modal_volume"] = ""
        self.config["quality_model"] = self.config["judge_model"] = ""
        rows, runner, finder = self.invoke("preflight", key="", specs=False)
        runner.assert_not_called()
        finder.assert_not_called()
        self.assertTrue(all(row["ok"] for row in rows.values()))
        self.assertEqual(set(rows), {"python", "project", "ffmpeg", "ffprobe", "media:movie",
                                    "media:trailer", "authentication_scope"})

    def test_missing_dependencies_are_reported_without_importing_them(self):
        rows, runner, _ = self.invoke(which=lambda name: None, specs=False)
        runner.assert_not_called()
        for name in ("ffmpeg", "ffprobe", "tesseract", "tesseract_eng", "annotation_cli", "dependency:numpy"):
            self.assertFalse(rows[name]["ok"])
            self.assertTrue(rows[name]["fix"])
        self.assertIn("python -m pip install", rows["dependency:PIL"]["fix"])

    def test_missing_or_empty_media_is_not_probed(self):
        self.movie.unlink()
        self.trailer.write_bytes(b"")
        rows, runner, _ = self.invoke("build")
        self.assertFalse(rows["media:movie"]["ok"])
        self.assertFalse(rows["media:trailer"]["ok"])
        self.assertFalse(any(Path(call.args[0][0]).name == "ffprobe" for call in runner.call_args_list))

    def test_incomplete_media_and_invalid_duration_do_not_pass(self):
        for document, expected in (({"streams": [{"codec_type": "video"}], "format": {"duration": "15"}}, "audio"),
                                   ({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                                     "format": {"duration": "nan"}}, "duration"),
                                   ({"streams": "invalid"}, "metadata")):
            def run(command, **kwargs):
                if Path(command[0]).name == "ffprobe":
                    return subprocess.CompletedProcess(command, 0, json.dumps(document), "")
                return self.local_run(command, **kwargs)
            rows, _, _ = self.invoke("build", run=run)
            self.assertFalse(rows["media:movie:probe"]["ok"])
            self.assertIn(expected, rows["media:movie:probe"]["detail"])

    def test_stream_duration_fallback_is_accepted(self):
        def run(command, **kwargs):
            if Path(command[0]).name == "ffprobe":
                return subprocess.CompletedProcess(command, 0, json.dumps({"streams": [
                    {"codec_type": "video", "duration": "30"}, {"codec_type": "audio"}]}), "")
            return self.local_run(command, **kwargs)
        rows, _, _ = self.invoke("build", run=run)
        self.assertTrue(rows["media:movie:probe"]["ok"])
        self.assertIn("30.000", rows["media:movie:probe"]["detail"])

    def test_local_tool_failure_does_not_leak_process_output(self):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 7, "secret stdout", "secret stderr")
        rows, _, _ = self.invoke("build", run=run)
        self.assertFalse(rows["annotation_cli"]["ok"])
        self.assertFalse(rows["tesseract_eng"]["ok"])
        self.assertFalse(rows["media:movie:probe"]["ok"])
        self.assertNotIn("secret", json.dumps(rows))

    def test_timeouts_and_unicode_errors_are_reported_as_local_failures(self):
        for error in (subprocess.TimeoutExpired("private tool", 30, output="secret"),
                      UnicodeDecodeError("utf8", b"\xff", 0, 1, "private"), OSError("secret")):
            rows, _, _ = self.invoke("build", run=lambda *args, **kwargs: (_ for _ in ()).throw(error))
            self.assertFalse(rows["annotation_cli"]["ok"])
            self.assertFalse(rows["media:movie:probe"]["ok"])
            self.assertNotIn("secret", json.dumps(rows))

    def test_opencode_requires_one_x_and_provider_model(self):
        self.config["agent"] = {"backend": "opencode", "model": "provider/model", "cli": ""}
        for version, expected in (("1.2.3", True), ("2.1.0", False), ("unknown", False)):
            def run(command, **kwargs):
                if Path(command[0]).name == "opencode":
                    return subprocess.CompletedProcess(command, 0, version, "")
                return self.local_run(command, **kwargs)
            rows, _, _ = self.invoke("build", run=run)
            self.assertEqual(rows["annotation_cli"]["ok"], expected)
        self.config["agent"]["model"] = "missing-provider"
        rows, _, _ = self.invoke("build")
        self.assertFalse(rows["annotation_model"]["ok"])

    def test_timeouts_must_be_finite_positive_numbers(self):
        for value in (0, -1, float("nan"), float("inf"), True, "900"):
            self.config["annotation_timeout_s"] = self.config["quality_timeout_s"] = value
            rows, _, _ = self.invoke()
            self.assertFalse(rows["annotation_timeout"]["ok"])
            self.assertFalse(rows["quality_timeout"]["ok"])
        self.config["annotation_timeout_s"] = None
        rows, _, _ = self.invoke("build")
        self.assertTrue(rows["annotation_timeout"]["ok"])

    def test_invalid_scope_and_configuration_fail_before_process_calls(self):
        with mock.patch.object(doctor.subprocess, "run") as run:
            for config, scope in ((self.config, "typo"), (None, "all")):
                with self.assertRaises(ValueError):
                    doctor.check(config, scope)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
