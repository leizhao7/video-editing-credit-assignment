"""Incremental Claude drafts are observable but cannot count as final output."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_script_queue as queue


class ScriptProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="script progress ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = self.root / "edit/task_sample"
        (self.task / "stages").mkdir(parents=True)
        self.state = self.root / "exports/sample/state.json"
        self.state.parent.mkdir(parents=True)
        self.state.write_text(json.dumps({"cutlist": {"ok": True}, "acuts": {"ok": True}}))
        self.draft = self.task / "work/script.draft.json"
        self.draft.parent.mkdir()
        self.visuals = {"manifest_path": self.draft.parent / "manifest.json",
                        "draft_path": self.draft, "sheet_paths": [], "fingerprint": "fixture"}
        self.inputs = {"film": "sample", "shots": [
            {"id": "s1", "kind": "source", "line": {"text": "Wait."}},
            {"id": "s2", "kind": "source", "line": None}]}
        self.input_path = self.task / "stages/script_inputs.json"
        self.input_path.write_text(json.dumps(self.inputs))
        self.good_draft = {"film": "sample", "shots": [
            {"id": "s1", "desc": "A generated red shape.", "line": "Wait."},
            {"id": "s2", "desc": "A generated blue shape.", "line": None}]}
        self.structure = {"logline": "Two colored shapes appear.", "beats": [
            {"id": f"b{i}", "name": "A shape", "shots": [sid, sid], "what": "A shape appears.",
             "function": "Introduce a shape.", "why_here": "The color changes."}
            for i, sid in enumerate(("s1", "s2"), 1)], "chunks": [
            {"id": f"c{i}", "beat": f"b{i}", "shots": [sid, sid],
             "what": "Show this shape.", "why_here": "A single action."}
            for i, sid in enumerate(("s1", "s2"), 1)]}
        self.write_manifest()

    def write_manifest(self):
        ids = [shot["id"] for shot in self.inputs["shots"]]
        manifest = {"fingerprint": "fixture", "shots": [
            {"id": sid, "frame_path": f"work/frames/{sid}.jpg", "dur_s": 0.5,
             "requires_individual_frame_read": True} for sid in ids], "sheets": [
            {"path": f"work/sheet_{i // 12 + 1}.jpg", "shot_ids": ids[i:i + 12]}
            for i in range(0, len(ids), 12)]}
        self.visuals["manifest_path"].write_text(json.dumps(manifest))

    def use_many_shots(self, count=25):
        self.inputs["shots"] = [{"id": f"s{i:03d}", "kind": "source",
                                 "line": {"text": f"Measured line {i}."} if i % 2 else None}
                                for i in range(1, count + 1)]
        self.input_path.write_text(json.dumps(self.inputs))
        self.good_draft["shots"] = [{"id": s["id"], "desc": f"Visible shape {i}.",
                                     "line": s["line"]["text"] if s["line"] else None}
                                    for i, s in enumerate(self.inputs["shots"])]
        self.structure["beats"] = []
        self.structure["chunks"] = []
        for i, shot in enumerate(self.inputs["shots"], 1):
            interval = [shot["id"], shot["id"]]
            self.structure["beats"].append({"id": f"b{i}", "name": "Shape",
                "shots": interval, "what": "A shape appears.", "function": "Introduce shape.",
                "why_here": "The color changes."})
            self.structure["chunks"].append({"id": f"c{i}", "beat": f"b{i}",
                "shots": interval, "what": "Show shape.", "why_here": "Single action."})
        self.write_manifest()

    def test_progress_tolerates_partial_writes_and_counts_descriptions_only(self):
        self.assertIsNone(queue._draft_count(self.draft, "sample"))
        self.draft.write_text('{"shots":')
        self.assertIsNone(queue._draft_count(self.draft, "sample"))
        self.draft.write_text(json.dumps({"film": "sample", "shots": [
            {"id": "s1", "desc": "A generated shape"},
            {"id": "s1", "desc": "Duplicate"}, {"id": "s2", "desc": " "}, None]}))
        self.assertEqual(queue._draft_count(self.draft, "sample"), 1)
        self.assertIsNone(queue._draft_count(self.draft, "another-film"))

    def test_draft_is_never_accepted_as_final_annotation(self):
        self.draft.write_text(json.dumps({"film": "sample", "shots": [{"id": "s1", "desc": "Shape"}]}))
        self.assertEqual(queue.check("sample", str(self.task)), ["script.json 未生成"])

    def test_visual_preparation_is_only_used_for_claude(self):
        endings = {"claude": {"type": "result", "is_error": False},
                   "codex": {"type": "turn.completed"},
                   "opencode": {"type": "step_finish", "part": {"reason": "stop"}}}
        for backend, event in endings.items():
            with self.subTest(backend=backend):
                captured = []

                @contextmanager
                def process(command, **kwargs):
                    captured.extend(command)
                    yield mock.Mock(stdout=io.StringIO(json.dumps(event) + "\n"),
                                    poll=lambda: 0, wait=lambda: 0, pid=0)

                with mock.patch.object(queue, "ROOT", str(self.root)), \
                     mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_BACKEND=backend,
                                     VIDEO_EDITING_ANNOTATION_MODEL="test-model"), \
                     mock.patch.object(queue, "prepare_script_visuals", return_value=self.visuals) as prepare, \
                     mock.patch.object(queue, "annotation_process", process), \
                     redirect_stdout(io.StringIO()):
                    ok, _, errors = queue._once("sample", "Sample", str(self.task), 1, 0)
                self.assertFalse(ok)  # A successful CLI without final output is still a failure.
                if backend == "claude":
                    self.assertTrue(any("Shot batch output missing or invalid" in error for error in errors))
                    prepare.assert_called_once_with(str(self.task))
                    self.assertTrue(any("shots-001/shots.json" in part for part in captured))
                else:
                    self.assertIn("script.json 未生成", errors)
                    prepare.assert_not_called()
                    self.assertFalse(any(str(self.draft) in part for part in captured))

    def test_timeout_preserves_draft_and_records_actionable_failure(self):
        self.draft.write_text('{"film":"sample","shots":[]}')
        before = self.draft.read_bytes()
        with mock.patch.object(queue, "ROOT", str(self.root)), \
             mock.patch.object(queue, "prep", return_value=(str(self.task), 1)), \
             mock.patch.object(queue, "_once", side_effect=TimeoutError("test total runtime limit")), \
             redirect_stdout(io.StringIO()):
            self.assertFalse(queue.run("sample", "Sample"))
        self.assertEqual(self.draft.read_bytes(), before)
        state = json.loads(self.state.read_text())["script"]
        self.assertFalse(state["ok"])
        self.assertIn("TimeoutError: test total runtime limit", state["errors"][0])

    def invoke_stages(self, actions=None, endings=None, resume=True, why=None):
        actions, endings = actions or {}, endings or {}
        calls = []

        @contextmanager
        def process(command, **kwargs):
            stage = Path(kwargs["stderr_path"]).parent.name
            calls.append({"stage": stage, "command": command, "log": kwargs["stderr_path"]})
            alias = "shots" if stage.startswith("shots-") else stage
            action = actions.get(stage, actions.get(alias))
            if action:
                action()
            elif alias == "shots":
                self.write_batch_output()
            else:
                frozen = self.current_frozen()
                (frozen.parent / "script.structure.json").write_text(json.dumps(self.structure))
            events = endings.get(stage, endings.get(alias, [{"type": "result", "is_error": False}]))
            yield mock.Mock(stdout=io.StringIO("".join(json.dumps(e) + "\n" for e in events)),
                            poll=lambda: 0, wait=lambda: 0, pid=0)

        with mock.patch.object(queue, "ROOT", str(self.root)), \
             mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_BACKEND="claude",
                             VIDEO_EDITING_ANNOTATION_MODEL="test-model", VIDEO_EDITING_ANNOTATION_EFFORT="high"), \
             mock.patch.object(queue, "prepare_script_visuals", return_value=self.visuals), \
             mock.patch.object(queue, "annotation_process", process), redirect_stdout(io.StringIO()):
            result = queue._once("sample", "Sample", str(self.task), len(self.inputs["shots"]), 0,
                                 why=why, resume=resume)
        return result, calls

    def current_batch_input(self):
        return max(self.draft.parent.glob("attempt-*/shots-*/inputs.json"),
                   key=lambda p: p.stat().st_mtime_ns)

    def write_batch_output(self):
        batch_input = self.current_batch_input()
        ids = [s["id"] for s in json.loads(batch_input.read_text())["shots"]]
        by_id = {s["id"]: s for s in self.good_draft["shots"]}
        # The production prompt asks the model to omit measured dialogue.
        raw = {"film": "sample", "shots": [{k: v for k, v in by_id[sid].items() if k != "line"}
                                            for sid in ids]}
        (batch_input.parent / "shots.json").write_text(json.dumps(raw))

    def current_frozen(self):
        return max(self.draft.parent.glob("attempt-*/shots.frozen.json"), key=lambda p: p.stat().st_mtime_ns)

    def test_two_fresh_sessions_publish_only_a_valid_python_merge(self):
        result, calls = self.invoke_stages()
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["shots-001", "structure"])
        for call in calls:
            command = call["command"]
            self.assertEqual(command[command.index("--model") + 1], "test-model")
            self.assertEqual(command[command.index("--effort") + 1], "high")
            self.assertNotIn("--resume", command)
        self.assertNotEqual(calls[0]["log"], calls[1]["log"])
        final = json.loads((self.task / "stages/script.json").read_text())
        self.assertEqual(final["shots"], self.good_draft["shots"])
        self.assertEqual(queue.check("sample", str(self.task)), [])
        self.assertTrue(json.loads(self.state.read_text())["script"]["ok"])
        self.assertTrue(self.current_frozen().resolve().is_relative_to(self.task.resolve()))

    def test_shot_output_without_completion_does_not_advance_or_publish(self):
        result, calls = self.invoke_stages(endings={"shots": []})
        self.assertFalse(result[0])
        self.assertEqual([c["stage"] for c in calls], ["shots-001"])
        self.assertFalse((self.task / "stages/script.json").exists())
        self.assertTrue(self.draft.exists())
        self.assertEqual(json.loads(self.draft.read_text())["shots"], [])

    def test_structure_output_without_completion_is_not_published(self):
        result, _ = self.invoke_stages(endings={"structure": []})
        self.assertFalse(result[0])
        self.assertFalse((self.task / "stages/script.json").exists())

    def test_complete_matching_draft_resumes_structure_in_new_attempt(self):
        self.draft.write_text(json.dumps(self.good_draft))
        result, calls = self.invoke_stages()
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["structure"])
        self.assertTrue(json.loads(self.state.read_text())["script"]["phases"]["shots_resume"]["resumed"])

    def test_redo_does_not_reuse_old_complete_draft(self):
        self.draft.write_text(json.dumps(self.good_draft))
        def write_fresh():
            self.assertEqual(json.loads(self.draft.read_text())["shots"], [])
            self.write_batch_output()
        result, calls = self.invoke_stages(actions={"shots": write_fresh}, resume=False)
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["shots-001", "structure"])
        self.assertEqual(len(list((self.root / "edit/runs").rglob("previous-draft.json"))), 1)

    def test_bad_structure_cannot_replace_previous_final(self):
        final = self.task / "stages/script.json"
        original = b'{"previous":true}\n'
        final.write_bytes(original)
        def bad_structure():
            invalid = deepcopy(self.structure)
            invalid["chunks"][0]["shots"] = ["s1", "s2"]
            (self.current_frozen().parent / "script.structure.json").write_text(json.dumps(invalid))
        result, _ = self.invoke_stages(actions={"structure": bad_structure})
        self.assertFalse(result[0])
        self.assertEqual(final.read_bytes(), original)
        self.assertFalse(json.loads(self.state.read_text())["script"]["ok"])

    def test_model_input_mutation_blocks_structure(self):
        original = self.input_path.read_bytes()
        def corrupt_inputs():
            self.write_batch_output()
            self.input_path.write_text('{"changed":true}')
        result, calls = self.invoke_stages(actions={"shots": corrupt_inputs})
        self.assertFalse(result[0])
        self.assertTrue(any("script_inputs.json was changed" in e for e in result[2]))
        self.assertEqual([c["stage"] for c in calls], ["shots-001"])
        self.assertEqual(self.input_path.read_bytes(), original)
        result, calls = self.invoke_stages()
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["shots-001", "structure"])
        final = json.loads((self.task / "stages/script.json").read_text())
        self.assertEqual(final["shots"][0]["line"], "Wait.")

    def test_frozen_draft_mutation_blocks_publication(self):
        def corrupt_frozen():
            frozen = self.current_frozen()
            (frozen.parent / "script.structure.json").write_text(json.dumps(self.structure))
            frozen.write_text('{"changed":true}')
        result, _ = self.invoke_stages(actions={"structure": corrupt_frozen})
        self.assertFalse(result[0])
        self.assertTrue(any("Frozen shot draft was changed" in e for e in result[2]))
        self.assertFalse((self.task / "stages/script.json").exists())

    def test_unauthorized_final_from_model_is_removed_on_failure(self):
        def write_final():
            self.write_batch_output()
            (self.task / "stages/script.json").write_text('{"unvalidated":true}')
        result, _ = self.invoke_stages(actions={"shots": write_final})
        self.assertFalse(result[0])
        self.assertFalse((self.task / "stages/script.json").exists())

    def test_stage_timeout_retains_draft_and_records_failed_state(self):
        def timeout():
            raise TimeoutError("synthetic structure total deadline")
        result, _ = self.invoke_stages(actions={"structure": timeout})
        self.assertFalse(result[0])
        self.assertTrue(self.draft.exists())
        self.assertFalse(json.loads(self.state.read_text())["script"]["ok"])
        self.assertTrue(any("total deadline" in e for e in result[2]))

    def test_retry_feedback_reaches_actual_new_phase_prompts(self):
        result, calls = self.invoke_stages(why=["synthetic missing chunk boundary"])
        self.assertTrue(result[0], result[2])
        for call in calls:
            prompt = call["command"][call["command"].index("-p") + 1]
            self.assertIn("synthetic missing chunk boundary", prompt)

    def test_multiple_batches_get_only_their_ids_and_merge_in_original_order(self):
        self.use_many_shots()
        observed = []
        def inspect_batch():
            path = self.current_batch_input()
            inputs = json.loads(path.read_text())
            manifest = json.loads((path.parent / "manifest.json").read_text())
            ids = [s["id"] for s in inputs["shots"]]
            self.assertEqual([s["id"] for s in manifest["shots"]], ids)
            self.assertEqual([sid for sheet in manifest["sheets"] for sid in sheet["shot_ids"]], ids)
            self.assertLessEqual(len(ids), 12)
            observed.append(ids)
            self.write_batch_output()
        result, calls = self.invoke_stages(actions={"shots": inspect_batch})
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["shots-001", "shots-002", "shots-003", "structure"])
        self.assertEqual([len(ids) for ids in observed], [12, 12, 1])
        self.assertEqual([sid for ids in observed for sid in ids], [s["id"] for s in self.inputs["shots"]])
        self.assertEqual(json.loads((self.task / "stages/script.json").read_text())["shots"], self.good_draft["shots"])

    def test_failed_batch_preserves_validated_prefix_and_retry_starts_there(self):
        self.use_many_shots()
        result, calls = self.invoke_stages(endings={"shots-002": []})
        self.assertFalse(result[0])
        self.assertEqual([c["stage"] for c in calls], ["shots-001", "shots-002"])
        self.assertEqual(json.loads(self.draft.read_text())["shots"], self.good_draft["shots"][:12])
        self.assertFalse((self.task / "stages/script.json").exists())
        result, calls = self.invoke_stages()
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["shots-002", "shots-003", "structure"])

    def test_partial_sheet_resume_keeps_prefix_and_sends_only_remaining_ids(self):
        self.use_many_shots()
        prefix = {"film": "sample", "shots": self.good_draft["shots"][:13]}
        self.draft.write_text(json.dumps(prefix))
        observed = []
        def inspect_batch():
            observed.append([s["id"] for s in json.loads(self.current_batch_input().read_text())["shots"]])
            self.write_batch_output()
        result, calls = self.invoke_stages(actions={"shots": inspect_batch})
        self.assertTrue(result[0], result[2])
        self.assertEqual([c["stage"] for c in calls], ["shots-002", "shots-003", "structure"])
        self.assertEqual(observed[0], [s["id"] for s in self.inputs["shots"][13:24]])

    def test_existing_bad_prefix_is_rejected_without_silent_dialogue_repair(self):
        bad = deepcopy(self.good_draft)
        bad["shots"][0]["line"] = "Misquoted measured dialogue"
        self.draft.write_text(json.dumps(bad))
        before = self.draft.read_bytes()
        result, calls = self.invoke_stages()
        self.assertFalse(result[0])
        self.assertEqual(calls, [])
        self.assertEqual(self.draft.read_bytes(), before)
        self.assertTrue(any("Cannot resume invalid shot prefix" in e for e in result[2]))

    def test_batch_cannot_replace_an_accepted_prefix(self):
        self.use_many_shots()
        prefix = {"film": "sample", "shots": self.good_draft["shots"][:12]}
        self.draft.write_text(json.dumps(prefix))
        before = self.draft.read_bytes()
        def corrupt_prefix():
            self.write_batch_output()
            self.draft.write_text('{"film":"sample","shots":[]}')
        result, _ = self.invoke_stages(actions={"shots": corrupt_prefix})
        self.assertFalse(result[0])
        self.assertEqual(self.draft.read_bytes(), before)
        self.assertTrue(any("global shot prefix was changed" in e for e in result[2]))

    def test_batch_input_and_manifest_mutation_reject_output(self):
        for filename in ("inputs.json", "manifest.json"):
            with self.subTest(filename=filename):
                def corrupt_batch():
                    self.write_batch_output()
                    (self.current_batch_input().parent / filename).write_text('{"changed":true}')
                result, _ = self.invoke_stages(actions={"shots": corrupt_batch})
                self.assertFalse(result[0])
                self.assertEqual(json.loads(self.draft.read_text())["shots"], [])
                self.assertTrue(any(f"Frozen batch {filename}" in e for e in result[2]))

    def test_incorrect_model_dialogue_is_replaced_by_measured_input(self):
        def wrong_line():
            self.write_batch_output()
            output = self.current_batch_input().parent / "shots.json"
            data = json.loads(output.read_text())
            data["shots"][0]["line"] = "Model misquote"
            data["shots"][1]["line"] = "Model invented speech"
            output.write_text(json.dumps(data))
        result, _ = self.invoke_stages(actions={"shots": wrong_line})
        self.assertTrue(result[0], result[2])
        self.assertEqual(json.loads((self.task / "stages/script.json").read_text())["shots"], self.good_draft["shots"])


if __name__ == "__main__":
    unittest.main()
