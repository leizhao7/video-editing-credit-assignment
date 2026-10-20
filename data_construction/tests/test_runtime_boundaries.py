"""Boundary regression tests using generated temporary metadata and local child processes."""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_config as config
import pipeline
import run_script_queue as script_queue
import run_s4_queue as s4_queue
import translate_script as translation


class RuntimeBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="construction boundary ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "private directory"
        self.root.mkdir()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (pipeline, script_queue, s4_queue, translation):
            self.stack.enter_context(mock.patch.object(module, "ROOT", str(self.root)))
        self.stack.enter_context(mock.patch.object(pipeline, "SRC", str(self.root / "sources")))
        self.stack.enter_context(mock.patch.dict(os.environ, {"VIDEO_EDITING_ANNOTATION_MODEL": "test-model", "VIDEO_EDITING_ANNOTATION_BACKEND": "claude", "VIDEO_EDITING_MCP_CONFIG": "", "VIDEO_EDITING_ANNOTATION_EFFORT": ""}))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def annotation_inputs(self, stage):
        state = self.write_json("exports/sample/state.json", {stage: {"ok": True}})
        task = self.root / "edit/task_sample"
        (task / "stages").mkdir(parents=True)
        return state, task

    @staticmethod
    def fake_annotation(task, output, returncode):
        @contextmanager
        def process(*args, **kwargs):
            (task / "stages" / output).write_text("{}")
            yield mock.Mock(stdout=io.StringIO(json.dumps({"type": "result", "is_error": False}) + "\n"), poll=lambda: returncode,
                            wait=lambda: returncode, pid=0)
        return process

    def test_blank_optional_environment_uses_default(self):
        with mock.patch.dict(os.environ, {"VIDEO_EDITING_DATA_ROOT": " \t",
                                         "VIDEO_EDITING_ANNOTATION_CLI": "  "}):
            self.assertEqual(config.env_text("VIDEO_EDITING_DATA_ROOT", "data/private"), "data/private")
            self.assertEqual(config.annotation_cli(), "claude")

    def test_blank_data_root_defaults_during_real_import(self):
        env = dict(os.environ, VIDEO_EDITING_DATA_ROOT="  ",
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", "import release_config;print(release_config.DATA_ROOT)"],
                                cwd=self.root, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(Path(result.stdout.strip()), (self.root / "data/private").resolve())

    def test_cache_cleanup_retains_independent_parameter_semantics(self):
        args = argparse.Namespace(gc=True, film="", all=True, workers=0)
        config.require_target(argparse.ArgumentParser(), args)

    def test_proxy_concurrency_rejects_deadlocking_zero(self):
        for value in ("0", "-1", "not-an-int"):
            with mock.patch.dict(os.environ, PROXY_LANES=value), self.assertRaises(ValueError):
                config.env_positive_int("PROXY_LANES", 2)
        with mock.patch.dict(os.environ, PROXY_LANES=" "):
            self.assertEqual(config.env_positive_int("PROXY_LANES", 2), 2)

    def test_cli_path_with_spaces_remains_one_path(self):
        with mock.patch.dict(os.environ, {"VIDEO_EDITING_ANNOTATION_CLI": " /custom tools/provider "}):
            self.assertEqual(config.annotation_cli(), "/custom tools/provider")

    def test_target_and_workers_are_validated(self):
        for values in ({"film": " "}, {"film": "sample", "all": True},
                       {"all": True, "workers": 0}):
            with self.subTest(values=values), self.assertRaises(SystemExit) as error:
                config.require_target(argparse.ArgumentParser(), argparse.Namespace(**values))
            self.assertEqual(error.exception.code, 2)

    def test_slug_rejects_paths_and_symlink_escape(self):
        for slug in ("../outside", "/absolute", "..", "", "a/b", "a\\b"):
            with self.subTest(slug=slug), self.assertRaises(ValueError):
                config.validate_data_slug(self.root, slug)
        (self.root / "exports").mkdir()
        (self.root / "exports/sample").symlink_to(Path(self.temp.name))
        with self.assertRaises(ValueError):
            config.validate_data_slug(self.root, "sample")

    def test_probe_cannot_redirect_to_another_slug(self):
        with self.assertRaises(ValueError):
            config.export_slug({"slug": "other"}, self.root / "exports/sample/probe.json")

    def test_selection_exact_slug_wins_and_ambiguous_title_fails(self):
        films = {"sample": "Shared title one", "other": "Shared title two"}
        self.assertEqual(config.select_films(films, argparse.Namespace(film="sample")),
                         {"sample": films["sample"]})
        with self.assertRaises(ValueError):
            config.select_films(films, argparse.Namespace(film="Shared title"))

    def test_list_unknown_entry_fails_instead_of_silently_skipping(self):
        listing = self.root / "input list.txt"
        listing.write_text("sample\nmissing\n")
        with self.assertRaises(ValueError):
            config.select_films({"sample": "Sample"}, argparse.Namespace(films=str(listing)))
        listing.write_text("sample\nsample\n")
        self.assertEqual(len(config.select_films({"sample": "Sample"},
                            argparse.Namespace(films=str(listing)))), 1)

    def test_invalid_stage_is_rejected_before_any_work(self):
        with mock.patch.object(sys, "argv", ["pipeline", "--all", "--only", "probe,unknown"]), \
             self.assertRaises(SystemExit) as error:
            pipeline.main()
        self.assertEqual(error.exception.code, 2)

    def test_colliding_source_names_do_not_share_exports(self):
        for name in ("Sample Name", "SampleName"):
            (self.root / "sources" / name).mkdir(parents=True)
        with mock.patch.object(sys, "argv", ["pipeline", "--all", "--only", "probe"]), \
             self.assertRaises(SystemExit) as error:
            pipeline.main()
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "exports").exists())

    def test_fetch_failure_does_not_mark_success(self):
        film = pipeline.Film("Sample Name")
        with mock.patch.object(pipeline, "cloud_volume", return_value="test-volume"), \
             mock.patch.object(pipeline, "sh", return_value=subprocess.CompletedProcess([], 2, "", "download failed")), \
             self.assertRaises(RuntimeError):
            film.fetch()
        self.assertFalse(film.done("fetch"))

    def test_audio_download_failure_does_not_mark_success(self):
        film = pipeline.Film("Sample Name")
        results = [subprocess.CompletedProcess([], 0, json.dumps({"type": "result", "is_error": False}), "") for _ in range(2)]
        results.append(subprocess.CompletedProcess([], 2, "", "download failed"))
        with mock.patch.object(pipeline, "cloud_volume", return_value="test-volume"), \
             mock.patch.object(pipeline, "sh", side_effect=results), self.assertRaises(RuntimeError):
            film.audio()
        self.assertFalse(film.done("audio"))

    def test_probe_failure_is_not_a_zero_duration(self):
        for result in (subprocess.CompletedProcess([], 1, "", "bad input"),
                       subprocess.CompletedProcess([], 0, "nan", "")):
            with mock.patch.object(pipeline, "sh", return_value=result), self.assertRaises(RuntimeError):
                pipeline._dur("synthetic path")

    def test_same_process_cannot_claim_film_twice(self):
        film = pipeline.Film("Sample Name")
        self.assertIsNotNone(pipeline._claim(film))
        self.assertIsNone(pipeline._claim(film))

    def test_redo_reaches_annotation_instead_of_skipping(self):
        state, task = self.annotation_inputs("script")
        state.write_text(json.dumps({"script": {"ok": True}, "cutlist": {"ok": True}, "acuts": {"ok": True}}))
        with mock.patch.object(script_queue, "prep", return_value=(str(task), 0)), \
             mock.patch.object(script_queue, "_once", return_value=(True, {}, [])) as once:
            self.assertTrue(script_queue.run("sample", "Sample", redo=True))
        once.assert_called_once()

    def test_failed_annotation_cannot_pass_with_valid_output(self):
        for module, stage, output in ((script_queue, "script", "script.json"),
                                      (s4_queue, "s4", "s4_cuts.json")):
            with self.subTest(stage=stage):
                state, task = self.annotation_inputs(stage) if stage == "script" else (
                    self.write_json("exports/sample/state.json", {stage: {"ok": True}}),
                    self.root / "edit/task_sample")
                with mock.patch.object(module, "annotation_process", self.fake_annotation(task, output, 7)), \
                     mock.patch.object(module, "check", return_value=[]):
                    ok, _, errors = module._once("sample", "Sample", str(task), 0, 0)
                self.assertFalse(ok)
                self.assertIn("status 7", errors[0])
                self.assertFalse(json.loads(state.read_text())[stage]["ok"])

    def test_failed_report_keeps_s4_state_incomplete(self):
        state, task = self.annotation_inputs("s4")
        with mock.patch.object(s4_queue, "annotation_process", self.fake_annotation(task, "s4_cuts.json", 0)), \
             mock.patch.object(s4_queue, "check", return_value=[]), \
             mock.patch.object(s4_queue.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "report failed")):
            ok, _, _ = s4_queue._once("sample", "Sample", str(task), 0, 0)
        self.assertFalse(ok)
        self.assertFalse(json.loads(state.read_text())["s4"]["ok"])

    def test_cli_failure_event_cannot_pass_even_with_zero_exit_and_valid_file(self):
        failures = {
            "claude": {"type": "result", "is_error": True, "errors": ["synthetic failure"]},
            "codex": {"type": "turn.failed", "error": {"message": "synthetic failure"}},
            "opencode": {"type": "error", "error": {"data": {"message": "synthetic failure"}}},
        }
        task = self.root / "edit/task_sample"
        (task / "stages").mkdir(parents=True)
        for backend, failure in failures.items():
            for module, stage, output in ((script_queue, "script", "script.json"),
                                          (s4_queue, "s4", "s4_cuts.json")):
                with self.subTest(backend=backend, stage=stage):
                    state = self.write_json("exports/sample/state.json", {stage: {"ok": True}})

                    @contextmanager
                    def process(*args, **kwargs):
                        (task / "stages" / output).write_text("{}")
                        yield mock.Mock(stdout=io.StringIO(json.dumps(failure)),
                                        poll=lambda: 0, wait=lambda: 0, pid=0)

                    with mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_BACKEND=backend), \
                         mock.patch.object(module, "annotation_process", process), \
                         mock.patch.object(module, "check", return_value=[]):
                        ok, _, errors = module._once("sample", "Sample", str(task), 0, 0)
                    self.assertFalse(ok)
                    self.assertIn("synthetic failure", errors[0])
                    self.assertFalse(json.loads(state.read_text())[stage]["ok"])

    def test_successful_cli_cannot_reuse_stale_annotation(self):
        success = {"claude": {"type": "result", "is_error": False},
                   "codex": {"type": "turn.completed"},
                   "opencode": {"type": "step_finish", "part": {"reason": "stop"}}}
        task = self.root / "edit/task_sample"
        for backend, event in success.items():
            for module, stage, output in ((script_queue, "script", "script.json"),
                                          (s4_queue, "s4", "s4_cuts.json")):
                with self.subTest(backend=backend, stage=stage):
                    self.write_json("exports/sample/state.json", {stage: {"ok": True}})
                    self.write_json(f"edit/task_sample/stages/{output}", {})

                    @contextmanager
                    def process(*args, **kwargs):
                        yield mock.Mock(stdout=io.StringIO(json.dumps(event)),
                                        poll=lambda: 0, wait=lambda: 0, pid=0)

                    with mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_BACKEND=backend), \
                         mock.patch.object(module, "annotation_process", process), \
                         mock.patch.object(module, "check", return_value=[]):
                        ok, _, errors = module._once("sample", "Sample", str(task), 0, 0)
                    self.assertFalse(ok)
                    self.assertIn("没被本次运行写过", errors[0])

    def test_pending_marker_requires_success_state_not_just_old_file(self):
        self.write_json("exports/sample/state.json", {"s4": {"ok": False}})
        self.write_json("edit/task_sample/stages/s4_cuts.json", {})
        marker = self.write_json("edit/S4_PENDING.json", {"films": ["sample"]})
        s4_queue._clear_sentinel({"sample": "Sample"})
        self.assertTrue(marker.exists())

    def test_stderr_larger_than_pipe_capacity_does_not_deadlock(self):
        log = self.root / "stderr log.txt"
        command = [sys.executable, "-c", "import sys;sys.stderr.write('x'*200000);print('done')"]
        with config.annotation_process(command, cwd=self.root, stderr_path=log) as process:
            self.assertEqual(process.stdout.readline().strip(), "done")
            self.assertEqual(process.wait(timeout=5), 0)
        self.assertEqual(log.stat().st_size, 200000)

    def test_annotation_exception_reaps_child(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with config.annotation_process([sys.executable, "-c", "import time;time.sleep(60)"],
                                           cwd=self.root, stderr_path=self.root / "stderr.txt") as process:
                raise RuntimeError("synthetic failure")
        self.assertIsNotNone(process.poll())

    def translation_documents(self):
        original = {"logline": "source", "shots": [{"id": "one", "desc": "source"}],
                    "beats": [{"id": "beat", "name": "source", "what": "source"}],
                    "chunks": [{"id": "chunk", "what": "source"}]}
        translated = deepcopy(original)
        translated["logline_en"] = "translation"
        translated["beats"][0].update(name_en="translation", what_en="translation")
        translated["chunks"][0]["what_en"] = "translation"
        return original, translated

    def test_failed_translation_process_restores_original(self):
        original, translated = self.translation_documents()
        path = self.write_json("edit/task_sample/stages/script.json", original)
        before = path.read_text()
        def provider(*args, **kwargs):
            path.write_text(json.dumps(translated))
            return subprocess.CompletedProcess([], 9, "", "provider failed")
        with mock.patch.object(translation.subprocess, "run", side_effect=provider):
            self.assertFalse(translation.run("sample", "Sample"))
        self.assertEqual(path.read_text(), before)

    def test_translation_rejects_dropped_entries_or_changed_source(self):
        for change in (lambda d: d["chunks"].clear(),
                       lambda d: d["shots"][0].update(desc="changed"),
                       lambda d: d.pop("logline_en")):
            original, translated = self.translation_documents()
            change(translated)
            path = self.write_json("edit/task_sample/stages/script.json", original)
            before = path.read_text()
            def provider(*args, **kwargs):
                path.write_text(json.dumps(translated))
                return subprocess.CompletedProcess([], 0, json.dumps({"type": "result", "is_error": False}), "")
            with mock.patch.object(translation.subprocess, "run", side_effect=provider):
                self.assertFalse(translation.run("sample", "Sample"))
            self.assertEqual(path.read_text(), before)

    def test_translation_allows_only_expected_new_fields(self):
        events = {"claude": {"type": "result", "is_error": False},
                  "codex": {"type": "turn.completed"},
                  "opencode": {"type": "step_finish", "part": {"reason": "stop"}}}
        for backend, event in events.items():
            with self.subTest(backend=backend):
                original, translated = self.translation_documents()
                path = self.write_json("edit/task_sample/stages/script.json", original)

                def provider(*args, **kwargs):
                    path.write_text(json.dumps(translated))
                    return subprocess.CompletedProcess([], 0, json.dumps(event), "")

                with mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_BACKEND=backend), \
                     mock.patch.object(translation.subprocess, "run", side_effect=provider):
                    self.assertTrue(translation.run("sample", "Sample"))
                self.assertEqual(json.loads(path.read_text()), translated)


if __name__ == "__main__":
    unittest.main()
