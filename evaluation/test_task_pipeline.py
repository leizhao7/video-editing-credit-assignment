"""Exercise the public task-to-scoring boundary with synthetic private tasks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
import io
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import task_pipeline as pipeline


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TaskPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / "s_1" / "sample"
        self.cs()

    def tearDown(self):
        self.temporary.cleanup()

    def cs(self, selected=None):
        dump(self.task / "initial_state/manifest.json", {"candidates": [{"id": x} for x in ("a", "b", "c")]})
        dump(self.task / "verifier/answer.json", {"task": "synthetic", "answer": ["a", "b"]})
        dump(self.task / "output/selection.json", {"selected": ["a", "b"] if selected is None else selected})

    def run_cli(self, *args):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", GOOGLE_AI_STUDIO_API_KEY="",
                   HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")
        return subprocess.run([sys.executable, str(ROOT / "task_pipeline.py"), "--task", str(self.task), *args],
                              text=True, capture_output=True, env=env, timeout=20)

    def result(self):
        return json.loads((self.task / "evidence/task_result.json").read_text())

    def test_cs_exact_partial_and_invalid(self):
        for selected, expected, status in [(["a", "b"], 1.0, "scored"), (["a"], .6667, "scored"),
                                            (["a", "unknown"], 0.0, "invalid_submission"),
                                            (["a", "a"], 0.0, "invalid_submission")]:
            self.cs(selected)
            run = self.run_cli()
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(self.result()["final"], expected)
            self.assertEqual(self.result()["status"], status)

    def test_preserve_task_specific_repaired_grader(self):
        script = self.task / "verifier/grade.py"
        custom = 'import json\nprint(json.dumps({"score": 0.75, "accepted_equivalence": True}))\n'
        script.write_text(custom)
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertEqual(self.result()["final"], .75)
        self.assertEqual(script.read_text(), custom)
        metadata = json.loads((self.task / "verifier/evaluation.json").read_text())
        self.assertFalse(metadata["task_grader"]["matches_released_default"])

    def test_failed_grader_replaces_previous_success_with_unscored(self):
        self.assertEqual(self.run_cli().returncode, 0)
        (self.task / "verifier/grade.py").write_text('print(\'{"score": 0.9}\')\nraise SystemExit(1)\n')
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertIsNone(self.result()["final"])
        self.assertEqual(self.result()["status"], "incomplete")

    def test_ns_exact_and_reverse(self):
        self.task = self.root / "s_2" / "sample"
        ids = ["a", "b", "c", "d"]
        dump(self.task / "initial_state/manifest.json", {"clips": [{"id": x} for x in ids]})
        dump(self.task / "verifier/answer.json", {"task": "synthetic", "order": ids})
        for order, expected in [(ids, 1.0), (ids[::-1], 0.0)]:
            dump(self.task / "output/order.json", {"order": order})
            run = self.run_cli()
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(self.result()["final"], expected)

    def test_prepare_is_idempotent_and_keeps_original_rubric_backup(self):
        previous = {"old_rubric": "synthetic"}
        dump(self.task / "verifier/rubric.json", previous)
        self.assertEqual(self.run_cli("--prepare-only").returncode, 0)
        state = {str(p.relative_to(self.task)): p.read_bytes() for p in self.task.rglob("*") if p.is_file()}
        self.assertEqual(self.run_cli("--prepare-only").returncode, 0)
        self.assertEqual(state, {str(p.relative_to(self.task)): p.read_bytes() for p in self.task.rglob("*") if p.is_file()})
        backups = list((self.task / "verifier/.evaluation-backups").glob("rubric.json.*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text()), previous)
        self.assertFalse((self.task / "evidence/task_result.json").exists())

    def test_preflight_invalid_is_nonzero_without_scoring(self):
        self.cs(["unknown"])
        self.assertNotEqual(self.run_cli("--preflight-only").returncode, 0)
        self.assertEqual(self.result()["status"], "invalid_submission")
        self.cs()
        self.assertEqual(self.run_cli("--preflight-only").returncode, 0)
        self.assertIsNone(self.result()["final"])
        self.assertEqual(self.result()["status"], "preflight_passed")

    def test_conflicting_stage_does_not_leave_stale_score(self):
        self.assertEqual(self.run_cli().returncode, 0)
        self.assertNotEqual(self.run_cli("--stage", "NS").returncode, 0)
        self.assertIsNone(self.result()["final"])
        self.assertEqual(self.result()["status"], "error")

    def test_symlink_verifier_escape_is_rejected(self):
        other = self.root / "outside"
        (self.task / "verifier").rename(other)
        (self.task / "verifier").symlink_to(other, target_is_directory=True)
        run = self.run_cli("--prepare-only")
        self.assertNotEqual(run.returncode, 0)
        self.assertFalse((other / "rubric.json").exists())

    def test_rubric_weights_and_duration_channel(self):
        for stage in pipeline.WEIGHTS:
            descriptor = pipeline.rubric_descriptor(stage)
            self.assertIn("supersede legacy construction-only", descriptor["metadata_precedence"])
            self.assertAlmostEqual(sum(descriptor["weights"].values()), 1.0)
            if stage in ("s_3", "s_4e"):
                for channel in ("blind", "referenced"):
                    self.assertAlmostEqual(sum(r["weight"] for r in descriptor[channel]), 1.0)
        rc = pipeline.rubric_descriptor("s_3")
        self.assertEqual(rc["referenced"][-1]["id"], "duration")
        self.assertEqual(rc["referenced"][-1]["weight"], .2)

    def test_fc_preflight_requires_preview(self):
        self.task = self.root / "s_4e" / "sample"
        dump(self.task / "initial_state/manifest.json", {"shots": [{"id": "p1", "dur_s": 1}],
                                                       "joins": [], "dialogue": [], "stretch_dur_s": 1})
        dump(self.task / "verifier/answer.json", {"task": "synthetic"})
        dump(self.task / "output/edit.json", {"joins": [], "dialogue": []})
        run = self.run_cli("--preflight-only")
        self.assertNotEqual(run.returncode, 0)
        self.assertEqual(self.result()["status"], "invalid_submission")
        self.assertIn("preview", self.result()["reason"])


class JudgedCompletenessTests(unittest.TestCase):
    """Exercise wrapper boundaries without changing or invoking provider code."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {
            "GOOGLE_AI_STUDIO_API_KEY": "synthetic-test-key", "S12_JUDGE_PROVIDER": "gemini"}))
        self.stack.enter_context(mock.patch("s12_task.load_and_validate_deliverable", return_value={}))
        self.stack.enter_context(mock.patch("grading_contract.validate_task", return_value={}))

    def task(self, stage):
        task = self.root / stage / "sample"
        dump(task / "output" / ("cuts.json" if stage == "s_3" else "edit.json"), {})
        dump(task / "initial_state/manifest.json", {})
        dump(task / "verifier/answer.json", {})
        return task

    def components(self, stage):
        description = pipeline.rubric_descriptor(stage)
        return {channel: {"score": .8, "partial": False, "missing": [],
                          "per_rubric": {row["id"]: 3 for row in description[channel]}}
                for channel in ("blind", "referenced")}

    def publish(self, task, stage, components):
        for channel, path in pipeline.judge_component_paths(task, stage).items():
            if channel in components:
                dump(path, components[channel])

    def run_grade(self, task, stage, components, value=.8):
        def grade(*args):
            self.publish(task, stage, components)
            return value, json.dumps({"final": value, "judgement_mode": "direct-judge"})
        with mock.patch("s12_task.grade", side_effect=grade):
            return pipeline.score(task, stage)

    def test_complete_fresh_channels_preserve_rc_and_fc_scores(self):
        for stage in ("s_3", "s_4e"):
            result = self.run_grade(self.task(stage), stage, self.components(stage))
            self.assertEqual(result["final"], .8)
            self.assertEqual(result["status"], "scored")
            self.assertNotIn("incomplete_components", result)

    def test_partial_missing_images_or_missing_bands_with_finite_aggregate_are_incomplete(self):
        for stage in ("s_3", "s_4e"):
            task = self.task(stage)
            for name, change in (
                ("partial", {"partial": True}),
                ("missing", {"missing": ["unanswered"]}),
                ("images", {"required_images_missing": ["image.jpg"]}),
                ("no-score", {"score": None}),
                ("no-bands", {"per_rubric": {}}),
            ):
                with self.subTest(stage=stage, failure=name):
                    components = self.components(stage)
                    components["referenced"].update(change)
                    result = self.run_grade(task, stage, components)
                    self.assertEqual(result["status"], "incomplete")
                    self.assertIsNone(result["final"])
                    self.assertEqual(result["raw_aggregate"], .8)
                    self.assertEqual(result["detail"]["final"], .8)
                    self.assertEqual(set(result["incomplete_components"]), {"referenced"})

    def test_unreported_missing_rubric_or_invalid_band_is_not_accepted(self):
        stage = "s_3"; task = self.task(stage)
        for band in (None, True, -1, 5, float("inf")):
            with self.subTest(band=band):
                components = self.components(stage)
                components["blind"]["per_rubric"]["placement"] = band
                result = self.run_grade(task, stage, components)
                self.assertEqual(result["status"], "incomplete")
                self.assertIn("blind", result["incomplete_components"])
        components = self.components(stage)
        del components["referenced"]["per_rubric"]["duration"]
        result = self.run_grade(task, stage, components)
        self.assertIsNone(result["final"])
        self.assertIn("referenced", result["incomplete_components"])

    def test_old_channels_are_removed_before_early_core_return(self):
        for stage in ("s_3", "s_4e"):
            task = self.task(stage)
            self.publish(task, stage, self.components(stage))
            def early_return(*args):
                self.assertTrue(all(not p.exists() for p in pipeline.judge_component_paths(task, stage).values()))
                return .9, '{"final":0.9}'
            with mock.patch("s12_task.grade", side_effect=early_return):
                result = pipeline.score(task, stage)
            self.assertEqual(result["status"], "incomplete")
            self.assertIsNone(result["final"])
            self.assertEqual(set(result["incomplete_components"]), {"blind", "referenced"})

    def test_one_new_channel_cannot_reuse_a_previous_reference_component(self):
        stage = "s_3"; task = self.task(stage)
        complete = self.components(stage)
        self.publish(task, stage, complete)
        result = self.run_grade(task, stage, {"blind": complete["blind"]})
        self.assertIsNone(result["final"])
        self.assertEqual(set(result["incomplete_components"]), {"referenced"})

    def test_unreadable_new_component_stays_incomplete(self):
        stage = "s_4e"; task = self.task(stage)
        def grade(*args):
            self.publish(task, stage, self.components(stage))
            pipeline.judge_component_paths(task, stage)["blind"].write_text("{")
            return .8, '{"final":0.8}'
        with mock.patch("s12_task.grade", side_effect=grade):
            result = pipeline.score(task, stage)
        self.assertIsNone(result["final"])
        self.assertEqual(set(result["incomplete_components"]), {"blind"})

    def test_complete_channels_do_not_override_failed_core_aggregate(self):
        stage = "s_3"
        result = self.run_grade(self.task(stage), stage, self.components(stage), value=None)
        self.assertIsNone(result["final"])
        self.assertEqual(result["status"], "incomplete")

    def test_preflight_does_not_clear_or_score_channels(self):
        stage = "s_4e"; task = self.task(stage)
        self.publish(task, stage, self.components(stage))
        before = {p: p.read_bytes() for p in pipeline.judge_component_paths(task, stage).values()}
        with mock.patch("s12_task.grade") as grade:
            result = pipeline.score(task, stage, preflight_only=True)
        grade.assert_not_called()
        self.assertEqual(result["status"], "preflight_passed")
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_cli_replaces_prior_final_and_returns_nonzero_for_partial_result(self):
        stage = "s_3"; task = self.task(stage)
        dump(task / "evidence/task_result.json", {"status": "scored", "final": 1})
        components = self.components(stage)
        components["blind"]["partial"] = True
        def grade(*args):
            self.publish(task, stage, components)
            return .8, '{"final":0.8}'
        with mock.patch("s12_task.grade", side_effect=grade), redirect_stdout(io.StringIO()):
            code = pipeline.main(["--task", str(task)])
        result = json.loads((task / "evidence/task_result.json").read_text())
        self.assertEqual(code, 1)
        self.assertIsNone(result["final"])
        self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
