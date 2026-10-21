"""Bounded S4 lifecycle tests; fake native sessions never call a model."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_s4_queue as queue
from annotation_backend import AnnotationCommand
from test_s4_stages import synthetic_s4, read_events


class S4ProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="S4 lifecycle ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "edit/task_sample"
        (self.task / "stages").mkdir(parents=True)
        self.inputs, self.document, self.manifest = synthetic_s4()
        self.source_paths = {}
        for key, filename in queue.INPUT_FILES.items():
            path = self.task / "stages" / filename
            path.write_text(json.dumps(self.inputs[key]))
            self.source_paths[key] = path
        self.original_inputs = {key: p.read_bytes() for key, p in self.source_paths.items()}
        self.state = self.root / "exports/sample/state.json"
        self.state.parent.mkdir(parents=True)
        self.state.write_text(json.dumps({"cutlist": {"ok": True}, "acuts": {"ok": True}, "s3": {"ok": True}}))
        visual_dir = self.task / "work/cut_visuals/fingerprint"
        visual_dir.mkdir(parents=True)
        self.draft = visual_dir / "s4.draft.json"
        manifest_path = visual_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest))
        self.visuals = {"manifest_path": manifest_path, "draft_path": self.draft,
                        "fingerprint": "fixture", "sheet_paths": []}
        self.final = self.task / "stages/s4_cuts.json"

    def current_batch(self):
        return max(self.draft.parent.glob("attempt-*/cuts-*/manifest.json"),
                   key=lambda p: p.stat().st_mtime_ns).parent

    def batch_output(self):
        directory = self.current_batch()
        indices = [c["idx"] for c in json.loads((directory / "cut_list.json").read_text())["cuts"]]
        document = {"stage": "S4", "cuts": [deepcopy(c) for c in self.document["cuts"] if c["idx"] in indices]}
        (directory / "cuts.json").write_text(json.dumps(document))
        return document

    def invoke(self, *, actions=None, endings=None, read_mode="success", backend="claude",
               resume=True, report_status=0, why=None):
        actions, endings, calls = actions or {}, endings or {}, []

        @contextmanager
        def process(command, **kwargs):
            stage = Path(kwargs["stderr_path"]).parent.name
            calls.append({"stage": stage, "command": command, "stderr": kwargs["stderr_path"]})
            if backend == "claude":
                directory = self.current_batch()
                strips = [c["strip_path"] for c in json.loads((directory / "manifest.json").read_text())["cuts"]]
                if stage in actions:
                    actions[stage]()
                elif "batch" in actions:
                    actions["batch"]()
                else:
                    self.batch_output()
                events = read_events(strips[:-1] if read_mode == "missing" else strips,
                                     failed=read_mode == "failed", results=read_mode != "no_result",
                                     image=read_mode != "text_only")
                events += endings.get(stage, [{"type": "result", "is_error": False}])
            else:
                self.final.write_text(json.dumps(self.document))
                events = ([{"type": "turn.completed"}] if backend == "codex" else
                          [{"type": "step_finish", "part": {"reason": "stop"}}])
            yield mock.Mock(stdout=io.StringIO("".join(json.dumps(e) + "\n" for e in events)),
                            poll=lambda: 0, wait=lambda: 0, pid=0)

        with mock.patch.object(queue, "ROOT", str(self.root)), \
             mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_BACKEND=backend,
                             VIDEO_EDITING_ANNOTATION_MODEL="synthetic-model", VIDEO_EDITING_ANNOTATION_EFFORT="high",
                             VIDEO_EDITING_MCP_CONFIG=""), \
             mock.patch.object(queue, "prepare_cut_visuals", return_value=self.visuals) as prepare, \
             mock.patch.object(queue, "annotation_process", process), \
             mock.patch.object(queue.subprocess, "run", return_value=subprocess.CompletedProcess([], report_status, "", "synthetic report error")) as report, \
             redirect_stdout(io.StringIO()):
            result = queue._once("sample", "Sample", str(self.task), len(self.document["cuts"]), 0,
                                 resume=resume, why=why)
        return result, calls, prepare.call_count, report.call_count

    def test_multiple_fresh_sessions_publish_ordered_python_merge(self):
        observed = []
        def inspect_subset():
            directory = self.current_batch()
            batch = {k: json.loads((directory / f).read_text()) for k, f in queue.INPUT_FILES.items()}
            indices = [c["idx"] for c in batch["cut_list"]["cuts"]]
            self.assertLessEqual(len(indices), 12)
            self.assertEqual(len(batch["units"]["units"]), len(indices) + 1)
            self.assertEqual([c["idx"] for c in batch["audio"]["cuts"]], indices)
            self.assertEqual([c["idx"] for c in batch["picture"]["cuts"]], indices)
            visual = json.loads((directory / "manifest.json").read_text())
            self.assertEqual((self.task / visual["draft_path"]).resolve(), (directory / "cuts.json").resolve())
            self.assertEqual({(self.task / p).resolve() for p in visual["inputs_paths"].values()},
                             {(directory / filename).resolve() for filename in queue.INPUT_FILES.values()})
            observed.append(indices)
            self.batch_output()
        result, calls, _, reports = self.invoke(actions={"batch": inspect_subset})
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["cuts-001", "cuts-002", "cuts-003"])
        self.assertEqual([len(b) for b in observed], [12, 12, 1])
        self.assertEqual(reports, 1)
        self.assertEqual(json.loads(self.final.read_text()), self.document)
        self.assertEqual(queue.check("sample", str(self.task)), [])
        self.assertEqual({k: p.read_bytes() for k, p in self.source_paths.items()}, self.original_inputs)
        for call in calls:
            command = call["command"]
            self.assertEqual(command[command.index("--model") + 1], "synthetic-model")
            self.assertEqual(command[command.index("--effort") + 1], "high")
            self.assertNotIn("--resume", command)
        self.assertEqual(len({str(c["stderr"]) for c in calls}), 3)

    def test_failed_batch_preserves_prefix_and_retry_does_not_repeat_it(self):
        result, calls, _, _ = self.invoke(endings={"cuts-002": []})
        self.assertFalse(result[0])
        self.assertEqual([c["stage"] for c in calls], ["cuts-001", "cuts-002"])
        self.assertEqual(json.loads(self.draft.read_text())["cuts"], self.document["cuts"][:12])
        self.assertFalse(self.final.exists())
        result, calls, _, _ = self.invoke(why=result[2])
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["cuts-002", "cuts-003"])
        self.assertIn("Previous attempt was rejected", calls[0]["command"][calls[0]["command"].index("-p") + 1])

    def test_partial_prefix_resume_and_redo_semantics(self):
        self.draft.write_text(json.dumps({"stage": "S4", "cuts": self.document["cuts"][:13]}))
        observed = []
        def inspect():
            observed.append([c["idx"] for c in self.batch_output()["cuts"]])
        result, calls, _, _ = self.invoke(actions={"batch": inspect})
        self.assertTrue(result[0], result[2])
        self.assertEqual(observed[0], list(range(14, 25)))
        self.assertEqual([c["stage"] for c in calls], ["cuts-002", "cuts-003"])
        result, calls, _, _ = self.invoke(resume=False)
        self.assertTrue(result[0], result[2])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(list((self.root / "edit/runs").rglob("previous-draft.json"))), 1)

    def test_invalid_prefix_is_not_silently_repaired(self):
        invalid = deepcopy(self.document); invalid["cuts"][0]["idx"] = 9
        self.draft.write_text(json.dumps(invalid)); original = self.draft.read_bytes()
        result, calls, _, _ = self.invoke()
        self.assertFalse(result[0]); self.assertEqual(calls, [])
        self.assertEqual(self.draft.read_bytes(), original)

    def test_missing_failed_unanswered_and_text_only_image_reads_reject_batch(self):
        for mode in ("missing", "failed", "no_result", "text_only"):
            with self.subTest(mode=mode):
                result, calls, _, _ = self.invoke(read_mode=mode)
                self.assertFalse(result[0]); self.assertEqual(len(calls), 1)
                self.assertTrue(any("visual access incomplete" in e for e in result[2]))
                self.assertEqual(json.loads(self.draft.read_text())["cuts"], [])
                self.assertFalse(self.final.exists())

    def test_original_inputs_are_restored_and_retry_uses_original_measurements(self):
        def corrupt_input():
            self.batch_output()
            self.source_paths["picture"].write_text('{"changed":true}')
        result, _, _, _ = self.invoke(actions={"batch": corrupt_input})
        self.assertFalse(result[0]); self.assertFalse(self.final.exists())
        self.assertEqual(self.source_paths["picture"].read_bytes(), self.original_inputs["picture"])
        self.assertEqual(json.loads(self.draft.read_text())["cuts"], [])
        self.assertTrue(self.invoke()[0][0])

    def test_batch_input_and_manifest_mutations_are_rejected(self):
        for filename in (*queue.INPUT_FILES.values(), "manifest.json"):
            with self.subTest(filename=filename):
                def corrupt():
                    self.batch_output()
                    (self.current_batch() / filename).write_text('{"changed":true}')
                result, _, _, _ = self.invoke(actions={"batch": corrupt})
                self.assertFalse(result[0])
                self.assertTrue(any("Frozen batch input" in e for e in result[2]))
                self.assertEqual(json.loads(self.draft.read_text())["cuts"], [])

    def test_model_cannot_replace_validated_prefix_or_previous_final(self):
        self.draft.write_text(json.dumps({"stage": "S4", "cuts": self.document["cuts"][:12]}))
        prefix = self.draft.read_bytes()
        original = b'{"previous":true}\n'; self.final.write_bytes(original)
        def corrupt():
            self.batch_output()
            self.draft.write_text('{"stage":"S4","cuts":[]}')
            self.final.write_text('{"model":true}')
        result, _, _, _ = self.invoke(actions={"batch": corrupt})
        self.assertFalse(result[0])
        self.assertEqual(self.draft.read_bytes(), prefix)
        self.assertEqual(self.final.read_bytes(), original)

    def test_invalid_or_missing_output_never_advances_prefix(self):
        for mode in ("missing", "wrong_idx", "no_change", "negative", "nan"):
            with self.subTest(mode=mode):
                def invalid_output():
                    document = self.batch_output()
                    path = self.current_batch() / "cuts.json"
                    if mode == "missing":
                        path.unlink(); return
                    if mode == "wrong_idx": document["cuts"][0]["idx"] = 100
                    if mode == "no_change": document["cuts"][0]["transition"]["type"] = "no_change"
                    if mode == "negative": document["cuts"][0]["transition"]["dur"] = -1
                    if mode == "nan": document["cuts"][0]["transition"]["dur"] = float("nan")
                    path.write_text(json.dumps(document))
                result, _, _, _ = self.invoke(actions={"batch": invalid_output})
                self.assertFalse(result[0])
                self.assertEqual(json.loads(self.draft.read_text())["cuts"], [])

    def test_diversity_is_not_enforced_until_all_batches_finish(self):
        for cut in self.document["cuts"]: cut["technique"] = "plain"
        result, calls, _, reports = self.invoke()
        self.assertFalse(result[0]); self.assertEqual(len(calls), 3); self.assertEqual(reports, 0)
        self.assertEqual(len(json.loads(self.draft.read_text())["cuts"]), 25)
        self.assertTrue(any("whole-film" in e for e in result[2]))
        self.assertFalse(self.final.exists())

    def test_failed_report_keeps_state_incomplete_but_validated_draft_resumable(self):
        result, _, _, reports = self.invoke(report_status=1)
        self.assertFalse(result[0]); self.assertEqual(reports, 1)
        self.assertFalse(json.loads(self.state.read_text())["s4"]["ok"])
        self.assertEqual(queue.check("sample", str(self.task)), [])
        result, calls, _, reports = self.invoke()
        self.assertTrue(result[0], result[2]); self.assertEqual(calls, []); self.assertEqual(reports, 1)

    def test_codex_and_opencode_keep_their_single_session_path(self):
        for backend in ("codex", "opencode"):
            with self.subTest(backend=backend):
                result, calls, prepared, _ = self.invoke(backend=backend)
                self.assertTrue(result[0], result[2]); self.assertEqual(len(calls), 1)
                self.assertEqual(prepared, 0)

    def test_real_continuous_fake_cli_obeys_total_deadline_even_after_completion(self):
        directory = self.root / "native-timeout"; directory.mkdir()
        script = ("import json,time\n"
                  "print(json.dumps({'type':'result','is_error':False}),flush=True)\n"
                  "for i in range(150):\n"
                  " print(json.dumps({'type':'system','subtype':'thinking_tokens','estimated_tokens':i}),flush=True)\n"
                  " time.sleep(.03)\n")
        command = AnnotationCommand("claude", "synthetic", [sys.executable, "-c", script])
        coverage = queue.VisualReadCoverage(self.task, [])
        started = time.monotonic()
        with mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_TIMEOUT_S="0.3"):
            phase = queue._cut_phase(command, str(self.task), directory, coverage)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(any("total runtime limit" in e for e in phase["errors"]))
        self.assertGreater(len((directory / "session.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
