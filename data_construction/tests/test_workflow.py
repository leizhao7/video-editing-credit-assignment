"""Project isolation, resume and paid-call boundaries; no external services."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_construction import workflow as w


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / "project with spaces"
        self.movie, self.trailer = self.root / "movie one.mkv", self.root / "trailer.mp4"
        self.movie.write_bytes(b"original movie")
        self.trailer.write_bytes(b"original trailer")
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))
        self.assertEqual(w.main(["init", "--project", str(self.project), "--movie", str(self.movie),
                                 "--trailer", str(self.trailer), "--backend", "claude", "--model", "sonnet"]), 0)
        _, self.config, self.state = w.open_project(self.project)

    def task(self, stage="s_1", name="c1"):
        task = self.project / "training_task/film" / stage / name
        (task / "initial_state/clips").mkdir(parents=True)
        (task / "verifier").mkdir()
        (task / "task.json").write_text("{}")
        (task / "task_prompt.md").write_text("Choose a clip.")
        (task / "initial_state/clips/c01.mp4").write_bytes(b"synthetic media for fake transport")
        items = "candidates" if stage == "s_1" else "shots" if stage == "s_4e" else "clips"
        w.save(task / "initial_state/manifest.json", {items: [{"id": "c01", "file": "clips/c01.mp4"}]})
        w.save(task / "verifier/answer.json", {"answer": ["c01"]})
        return str(task)

    def audit_runner(self, actions=None, calls=None, interrupt_after_receipt=False):
        actions, calls = actions or {}, calls if calls is not None else []
        def runner(argv, env, log):
            task = argv[argv.index("--task") + 1]
            calls.append(task)
            q = w.quality_module()
            value = q.prepare(task)
            action = actions.get(task, "keep_candidate")
            result = {"task": value["task"], "action": action, "reason": "Synthetic test assessment.",
                      "media_observations": [{"path": a["path"], "modality": a["mime"],
                         "observed": "A colored frame.", "limitations": "Synthetic.", "inspected": True} for a in value["assets"]],
                      "issues": [], "repair_plan": [], "additional_evidence_needed": [], "scope_change_needed": False}
            if action == "repair_needed":
                result["repair_plan"] = ["Clarify the supplied public instruction."]
            class Client:
                def request(self, payload):
                    return {"modelVersion": q.MODEL, "candidates": [{"finishReason": "STOP",
                            "content": {"parts": [{"text": json.dumps(result)}]}}]}
            output = Path(argv[argv.index("--output") + 1]) / "task-00000"
            terminal = q.audit(task, output, client=Client())
            if interrupt_after_receipt:
                raise KeyboardInterrupt()
            return 0 if terminal["status"] == "complete" else 1
        return runner

    def ready(self):
        self.state["steps"]["scoring-setup"] = {"status": "complete"}

    def test_init_links_originals_and_is_non_destructive(self):
        self.assertTrue((self.project / "sources/film/source.mkv").is_symlink())
        self.assertEqual((self.project / "sources/film/source.mkv").resolve(), self.movie)
        before = (self.project / "project.json").read_bytes()
        self.assertEqual(w.main(["init", "--project", str(self.project), "--movie", str(self.movie),
                               "--trailer", str(self.trailer), "--backend", "claude", "--model", "sonnet"]), 1)
        self.assertEqual(before, (self.project / "project.json").read_bytes())
        self.assertEqual(self.movie.read_bytes(), b"original movie")

    def test_fingerprints_detect_same_size_same_mtime_change(self):
        stamp = self.movie.stat()
        self.movie.write_bytes(b"modified movie")
        os.utime(self.movie, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "bytes or source link changed"):
            w.open_project(self.project, verify=True)

    def test_config_and_source_changes_do_not_reuse_caches(self):
        with mock.patch.object(w, "source_identity", return_value="different"):
            with self.assertRaisesRegex(ValueError, "code/dependencies changed"):
                w.open_project(self.project, verify=True)
        config = dict(self.config, quality_model="changed-model")
        w.save(self.project / "project.json", config)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            w.open_project(self.project)

    def test_runtime_prompts_affect_source_identity_but_docs_do_not(self):
        fake = self.root / "checkout"
        (fake / "data_construction/prompts").mkdir(parents=True)
        (fake / "video_editing.py").write_text("# entry")
        prompt = fake / "data_construction/prompts/PROMPT_SCRIPT.md"
        prompt.write_text("original instructions")
        with mock.patch.object(w, "REPO", fake):
            first = w.source_identity()
            (fake / "data_construction/README.md").write_text("new docs")
            self.assertEqual(first, w.source_identity())
            prompt.write_text("changed instructions")
            self.assertNotEqual(first, w.source_identity())

    def test_output_symlink_cannot_escape_private_project(self):
        (self.project / "external").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            w.owned(self.project, "external/should-not-write.json")

    def test_plan_and_status_are_read_only(self):
        before = {p.relative_to(self.project): p.read_bytes() for p in self.project.rglob("*") if p.is_file()}
        with mock.patch.object(w.subprocess, "Popen", side_effect=AssertionError("Must not run")), \
             mock.patch.object(w, "doctor", side_effect=AssertionError("Must not probe")):
            self.assertEqual(w.main(["run", "--project", str(self.project), "--plan"]), 0)
            self.assertEqual(w.main(["status", "--project", str(self.project)]), 0)
        after = {p.relative_to(self.project): p.read_bytes() for p in self.project.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_scoped_environment_ignores_ambient_profiles_without_saving_keys(self):
        with mock.patch.dict(os.environ, {"GOOGLE_AI_STUDIO_API_KEY": "test-only-secret",
                "VIDEO_EDITING_RUBRIC_PROGRAM": "unwanted", "S12_GRADING_DEADLINE": "expired"}):
            env = w.environment(self.config)
        self.assertNotIn("VIDEO_EDITING_RUBRIC_PROGRAM", env)
        self.assertNotIn("S12_GRADING_DEADLINE", env)
        self.assertEqual(env["GOOGLE_AI_STUDIO_API_KEY"], "test-only-secret")
        self.assertNotIn("test-only-secret", (self.project / "project.json").read_text())

    def test_failed_step_resumes_and_successful_step_is_skipped(self):
        step = w.Step("example", "probe", "Example", ["unused"], ("artifact.json",))
        fail = mock.Mock(return_value=1)
        with self.assertRaises(RuntimeError):
            w.execute_step(self.project, self.config, self.state, step, fail)
        self.assertEqual(self.state["steps"]["example"]["status"], "failed")
        def succeed(*args):
            w.save(self.project / "artifact.json", {"ok": True})
            return 0
        success = mock.Mock(side_effect=succeed)
        w.execute_step(self.project, self.config, self.state, step, success)
        w.execute_step(self.project, self.config, self.state, step, success)
        self.assertEqual(success.call_count, 1)
        self.assertEqual(self.state["steps"]["example"]["attempt"], 2)
        (self.project / "artifact.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            w.execute_step(self.project, self.config, self.state, step, success)
        self.assertEqual(success.call_count, 1)

    def test_empty_builder_exit_zero_is_not_success(self):
        step = next(s for s in w.steps(self.config) if s.name == "build-cs")
        with self.assertRaisesRegex(ValueError, "No CS tasks"):
            w.execute_step(self.project, self.config, self.state, step, lambda *a: 0)
        self.assertEqual(self.state["steps"]["build-cs"]["status"], "failed")

    def test_discover_requires_all_four_types_and_complete_tasks(self):
        self.task()
        with self.assertRaisesRegex(ValueError, "No NS tasks"):
            w.discover_tasks(self.project, "film")
        for stage in ("s_2", "s_3", "s_4e"):
            self.task(stage)
        self.assertEqual(len(w.discover_tasks(self.project, "film")), 4)
        (self.project / "training_task/film/s_3/c1/verifier/answer.json").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete task"):
            w.discover_tasks(self.project, "film")

    def test_review_limit_and_resume_reuse_only_verified_receipts(self):
        tasks = [self.task(name=name) for name in ("c1", "c2")]
        self.ready()
        calls = []
        runner = self.audit_runner(calls=calls)
        with mock.patch.object(w, "discover_tasks", return_value=tasks):
            self.assertEqual(w.review_tasks(self.project, self.config, self.state, limit=1, runner=runner), tasks[:1])
            self.assertFalse(w.load(self.project / "review_summary.json")["complete"])
            self.assertEqual(w.review_tasks(self.project, self.config, self.state, limit=1, runner=runner), tasks)
            w.review_tasks(self.project, self.config, self.state, runner=runner)
        self.assertEqual(calls, tasks)
        self.assertTrue(w.load(self.project / "review_summary.json")["complete"])

    def test_completed_child_receipt_recovered_after_parent_interruption(self):
        task = self.task()
        self.ready()
        calls = []
        with mock.patch.object(w, "discover_tasks", return_value=[task]):
            with self.assertRaises(KeyboardInterrupt):
                w.review_tasks(self.project, self.config, self.state,
                               runner=self.audit_runner(calls=calls, interrupt_after_receipt=True))
            self.assertEqual(self.state["reviews"][task]["status"], "interrupted")
            no_request = mock.Mock(side_effect=AssertionError("Duplicate paid call"))
            self.assertEqual(w.review_tasks(self.project, self.config, self.state, runner=no_request), [task])
            no_request.assert_not_called()
        self.assertEqual(calls, [task])

    def test_repair_decision_stays_out_of_accepted_tasks_without_repeated_review(self):
        task = self.task()
        self.ready()
        calls = []
        runner = self.audit_runner({task: "repair_needed"}, calls)
        with mock.patch.object(w, "discover_tasks", return_value=[task]):
            self.assertEqual(w.review_tasks(self.project, self.config, self.state, runner=runner), [])
            self.assertEqual(w.review_tasks(self.project, self.config, self.state, runner=runner), [])
        self.assertEqual(calls, [task])
        self.assertEqual(w.load(self.project / "review_summary.json")["tasks"][0]["action"], "repair_needed")

    def test_failed_review_preserves_prior_acceptances_and_retries_only_incomplete(self):
        tasks = [self.task(name=name) for name in ("c1", "c2")]
        self.ready()
        calls = []
        real_runner = self.audit_runner(calls=calls)
        def fail_second(argv, env, log):
            if argv[argv.index("--task") + 1] == tasks[1]:
                return 1
            return real_runner(argv, env, log)
        with mock.patch.object(w, "discover_tasks", return_value=tasks):
            with self.assertRaisesRegex(RuntimeError, "Quality review failed"):
                w.review_tasks(self.project, self.config, self.state, runner=fail_second)
            self.assertEqual(w.load(self.project / "accepted_tasks.json"), tasks[:1])
            self.assertEqual(w.review_tasks(self.project, self.config, self.state, runner=real_runner), tasks)
        self.assertEqual(calls, tasks)
        self.assertEqual(self.state["reviews"][tasks[1]]["attempt"], 2)

    def test_complete_orchestration_orders_dependencies_and_resumes_without_calls(self):
        ordered = w.steps(self.config)
        calls = []
        review = self.audit_runner()
        def runner(argv, env, log):
            if "--task" in argv:
                return review(argv, env, log)
            step = ordered[len(calls)]
            self.assertEqual(argv, step.argv)
            calls.append(step.name)
            for relative in step.required:
                path = self.project / relative
                if path.suffix == ".json":
                    w.save(path, {})
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"synthetic artifact")
            if step.name == "media":
                w.save(self.project / "exports/film/state.json", {name: {"ok": True} for name in
                       ("probe", "proxy", "upload", "shots", "fetch", "p2", "cards", "audio", "s3", "cutlist", "acuts", "report")})
            if step.name == "translation":
                w.save(self.project / step.required[0], {"logline_en": "A test story."})
            for stage in {"build-cs": ("s_1",), "build-ns-rc": ("s_2", "s_3"), "build-fc": ("s_4e",)}.get(step.name, ()):
                self.task(stage)
            if step.name == "scoring-setup":
                for task in w.load(self.project / "candidate_tasks.json"):
                    w.save(Path(task) / "verifier/rubric.json", {})
                    w.save(Path(task) / "verifier/evaluation.json", {})
            return 0
        w.run_workflow(self.project, self.config, self.state, runner=runner)
        self.assertEqual(calls, [s.name for s in ordered])
        self.assertEqual(len(w.load(self.project / "accepted_tasks.json")), 4)
        no_requests = mock.Mock(side_effect=AssertionError("Completed workflow must not repeat calls"))
        w.run_workflow(self.project, self.config, self.state, runner=no_requests)
        no_requests.assert_not_called()

    def reviewed_task(self):
        task = self.task()
        self.ready()
        with mock.patch.object(w, "discover_tasks", return_value=[task]):
            w.review_tasks(self.project, self.config, self.state, runner=self.audit_runner())
        return task

    def test_changed_task_is_not_silently_reaudited_or_scored(self):
        task = self.reviewed_task()
        (Path(task) / "task_prompt.md").write_text("Different instruction")
        no_request = mock.Mock(side_effect=AssertionError("Must not pay"))
        with mock.patch.object(w, "discover_tasks", return_value=[task]):
            with self.assertRaises(ValueError):
                w.review_tasks(self.project, self.config, self.state, runner=no_request)
            with self.assertRaises(ValueError):
                w.score_tasks(self.project, self.config, self.state, True, no_request)
        no_request.assert_not_called()

    def test_result_tampering_cannot_override_bound_response(self):
        task = self.reviewed_task()
        evidence = Path(self.state["reviews"][task]["evidence"])
        terminal, result = w.load(evidence / "terminal.json"), w.load(evidence / "result.json")
        terminal["action"], result["action"] = "filter", "filter"
        w.save(evidence / "terminal.json", terminal)
        w.save(evidence / "result.json", result)
        with self.assertRaisesRegex(ValueError, "decision differs"):
            w.verify_review(self.project, self.config, task, evidence)

    def test_submission_changes_allow_preflight_but_stale_score_does_not_survive_failure(self):
        task = self.reviewed_task()
        (Path(task) / "output").mkdir()
        (Path(task) / "output/selection.json").write_text('["c01"]')
        calls = []
        def preflight(argv, env, log):
            self.assertIn("--preflight-only", argv)
            calls.append(argv)
            w.save(Path(argv[argv.index("--summary") + 1]), {"tasks": [{"task": task, "status": "preflight_passed"}]})
            return 0
        with mock.patch.object(w, "discover_tasks", return_value=[task]):
            self.assertEqual(w.score_tasks(self.project, self.config, self.state, runner=preflight), 0)
            self.assertEqual(self.state["last_score"]["status"], "complete")
            with self.assertRaises(KeyboardInterrupt):
                w.score_tasks(self.project, self.config, self.state,
                              runner=mock.Mock(side_effect=KeyboardInterrupt()))
        self.assertEqual(self.state["last_score"]["status"], "interrupted")
        self.assertEqual(w.load(self.project / w.STATE)["last_score"]["status"], "interrupted")

    def test_foreign_score_result_not_accepted(self):
        task = self.reviewed_task()
        def foreign(argv, env, log):
            w.save(Path(argv[argv.index("--summary") + 1]), {"tasks": [{"task": "different", "status": "preflight_passed"}]})
            return 0
        with mock.patch.object(w, "discover_tasks", return_value=[task]):
            self.assertEqual(w.score_tasks(self.project, self.config, self.state, runner=foreign), 1)
        self.assertEqual(self.state["last_score"]["status"], "incomplete")

    def test_default_score_uses_key_free_preflight_scope(self):
        with mock.patch.object(w, "doctor", return_value=True) as doctor, \
             mock.patch.object(w, "score_tasks", return_value=0) as score:
            self.assertEqual(w.main(["score", "--project", str(self.project)]), 0)
            self.assertEqual(doctor.call_args.args[1], "preflight")
            self.assertFalse(score.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
