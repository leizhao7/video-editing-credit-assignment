"""Staged annotation contracts and publication, using synthetic data only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotation_script_stages as stages


class ScriptStagesTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {"film": "sample", "shots": [
            {"id": "s1", "kind": "source", "line": {
                "text": " Wait here! ", "from_prev": 0.12, "into_next": None}},
            {"id": "s2", "kind": "card", "line": None},
        ]}
        self.draft = {"film": "sample", "shots": [
            {"id": "s1", "desc": "A person waits near a doorway.",
             "line": " Wait here! ", "card_text": None},
            {"id": "s2", "desc": "A title card appears.", "line": None,
             "card_text": "A NEW JOURNEY"},
        ]}
        self.structure = {"logline": "A person prepares for a journey.",
                          "beats": [{"id": "b1", "shots": ["s1", "s2"]}],
                          "chunks": [{"id": "c1", "beat": "b1", "shots": ["s1", "s2"]}]}

    def test_valid_draft_preserves_exact_text_null_and_card_evidence(self):
        before = deepcopy((self.inputs, self.draft))
        self.assertEqual(stages.validate_draft(self.inputs, self.draft), [])
        self.assertEqual((self.inputs, self.draft), before)
        self.draft["shots"][1]["card_text"] = None
        self.draft["shots"][1]["card_misflag"] = "The image shows a printed sign in a street scene."
        self.assertEqual(stages.validate_draft(self.inputs, self.draft), [])

    def test_draft_rejects_missing_duplicate_reordered_and_unknown_shots(self):
        variants = [self.draft["shots"][:1], self.draft["shots"][::-1],
                    [self.draft["shots"][0]] * 2,
                    self.draft["shots"] + [{"id": "s3", "desc": "Another scene.", "line": None}]]
        for shots in variants:
            with self.subTest(shots=shots):
                draft = {"film": "sample", "shots": deepcopy(shots)}
                self.assertTrue(stages.validate_draft(self.inputs, draft))

    def test_draft_rejects_missing_changed_invented_and_object_dialogue(self):
        for index, value in [(0, "Wait here!"), (0, None), (0, self.inputs["shots"][0]["line"]),
                             (1, "Invented dialogue"), (1, "")]:
            with self.subTest(index=index, value=value):
                draft = deepcopy(self.draft)
                draft["shots"][index]["line"] = value
                self.assertTrue(stages.validate_draft(self.inputs, draft))
        for index in (0, 1):
            draft = deepcopy(self.draft)
            del draft["shots"][index]["line"]
            self.assertTrue(stages.validate_draft(self.inputs, draft))

    def test_draft_rejects_wrong_film_empty_description_and_unexplained_card(self):
        variants = []
        draft = deepcopy(self.draft); draft["film"] = "other"; variants.append(draft)
        draft = deepcopy(self.draft); draft.pop("film"); variants.append(draft)
        for value in (" ", None, 123):
            draft = deepcopy(self.draft); draft["shots"][0]["desc"] = value; variants.append(draft)
        draft = deepcopy(self.draft); draft["shots"][1]["card_text"] = None; variants.append(draft)
        for draft in variants:
            with self.subTest(draft=draft):
                self.assertTrue(stages.validate_draft(self.inputs, draft))

    def test_malformed_documents_return_errors_instead_of_crashing(self):
        for inputs, draft in [(None, self.draft), (self.inputs, []),
                              ({"film": "sample", "shots": []}, self.draft),
                              (self.inputs, {"film": "sample", "shots": [None, {"id": []}]}),
                              ({"film": "sample", "shots": [None]}, self.draft),
                              ({"film": "sample", "shots": [self.inputs["shots"][0]] * 2}, self.draft)]:
            with self.subTest(inputs=inputs, draft=draft):
                self.assertTrue(stages.validate_draft(inputs, draft))
        inputs = deepcopy(self.inputs)
        inputs["shots"][0]["line"] = {"from_prev": 0.1}
        self.assertTrue(stages.validate_draft(inputs, self.draft))

    def test_merge_preserves_shots_and_does_not_share_mutable_data(self):
        original_draft, original_structure = deepcopy(self.draft), deepcopy(self.structure)
        merged = stages.merge_structure(self.draft, self.structure)
        self.assertEqual(merged["shots"], self.draft["shots"])
        self.assertEqual(merged["film"], self.draft["film"])
        self.assertEqual(merged["logline"], self.structure["logline"])
        merged["shots"][0]["line"] = "Changed after merge"
        merged["beats"][0]["shots"].append("s3")
        self.assertEqual(self.draft, original_draft)
        self.assertEqual(self.structure, original_structure)

    def test_prefix_accepts_only_exact_leading_shots_with_strict_dialogue(self):
        for length in (0, 1, 2):
            prefix = {"film": "sample", "shots": deepcopy(self.draft["shots"][:length])}
            self.assertEqual(stages.validate_prefix(self.inputs, prefix), [])
        for shots in (self.draft["shots"][1:], self.draft["shots"][::-1],
                      [self.draft["shots"][0]] * 2, self.draft["shots"] * 2):
            self.assertTrue(stages.validate_prefix(self.inputs, {"film": "sample", "shots": shots}))
        prefix = {"film": "sample", "shots": deepcopy(self.draft["shots"][:1])}
        prefix["shots"][0]["line"] = "Misquote"
        self.assertTrue(stages.validate_prefix(self.inputs, prefix))

    def test_canonicalize_inserts_measured_lines_without_changing_visual_fields(self):
        raw = deepcopy(self.draft)
        raw["shots"][0].pop("line")
        raw["shots"][1]["line"] = "Invented speech"
        before = deepcopy((self.inputs, raw))
        result = stages.canonicalize_batch_lines(self.inputs, raw)
        self.assertEqual(result, self.draft)
        self.assertEqual((self.inputs, raw), before)
        self.assertEqual(stages.validate_draft(self.inputs, result), [])

    def test_canonicalize_requires_exact_film_and_ids_before_projection(self):
        variants = [{**self.draft, "film": "other"},
                    {"film": "sample", "shots": self.draft["shots"][:1]},
                    {"film": "sample", "shots": self.draft["shots"][::-1]},
                    {"film": "sample", "shots": [self.draft["shots"][0]] * 2}]
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                stages.canonicalize_batch_lines(self.inputs, raw)

    def test_batch_subsets_keep_selected_measurements_and_visual_requirements(self):
        inputs = {"film": "sample", "shots": [
            {"id": f"s{i}", "kind": "source", "line": {"text": str(i)}} for i in range(25)]}
        ids = [s["id"] for s in inputs["shots"]]
        manifest = {"fingerprint": "fixture", "shots": [
            {"id": sid, "dur_s": 0.5, "frame_path": f"{sid}.jpg",
             "requires_individual_frame_read": True} for sid in ids],
            "sheets": [{"path": f"sheet_{i}.jpg", "shot_ids": ids[i:i + 12]}
                       for i in range(0, 25, 12)]}
        before = deepcopy((inputs, manifest))
        for start, stop in ((0, 12), (13, 24), (24, 25)):
            subset, visuals = stages.make_shot_batch(inputs, manifest, start, stop,
                task_dir="/private/task", inputs_path="/private/task/batch/inputs.json",
                output_path="/private/task/batch/shots.json")
            self.assertEqual(subset["shots"], inputs["shots"][start:stop])
            self.assertEqual(visuals["shots"], manifest["shots"][start:stop])
            self.assertEqual([sid for s in visuals["sheets"] for sid in s["shot_ids"]], ids[start:stop])
            subset["shots"][0]["line"]["text"] = "mutated copy"
            visuals["shots"][0]["frame_path"] = "mutated copy"
        self.assertEqual((inputs, manifest), before)
        with self.assertRaises(ValueError):
            stages.make_shot_batch(inputs, manifest, 0, 13, task_dir="/private/task",
                inputs_path="/private/task/input", output_path="/private/task/output")

    def test_structure_cannot_replace_film_or_shots_or_supply_unknown_fields(self):
        for key, value in [("film", "other"), ("shots", []), ("unexpected", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                stages.merge_structure(self.draft, {**self.structure, key: value})
        for key, value in [("logline", " "), ("beats", {}), ("chunks", None)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                stages.merge_structure(self.draft, {**self.structure, key: value})
        incomplete = deepcopy(self.structure); incomplete.pop("chunks")
        with self.assertRaises(ValueError):
            stages.merge_structure(self.draft, incomplete)

    def test_prompts_target_separate_outputs_and_forbid_rewriting_shots(self):
        # Spaces and quotes must be represented unambiguously, without filesystem I/O.
        inputs = Path('/private/example space/"task"/stages/script_inputs.json')
        visuals = {"manifest_path": Path("/private/manifest.json"),
                   "draft_path": Path("/private/shot draft.json")}
        shot_prompt = stages.build_shot_prompt(inputs, visuals)
        self.assertIn(json.dumps(str(inputs)), shot_prompt)
        self.assertIn(json.dumps(str(visuals["draft_path"])), shot_prompt)
        self.assertIn("at most 12", shot_prompt)
        self.assertIn("Do not output `line`", shot_prompt)
        self.assertIn("For EVERY shot with `dur_s < 0.75`", shot_prompt)
        self.assertIn("Do not write stages/script.json", shot_prompt)
        prompt = stages.build_structure_prompt("/private/frozen.json", inputs, "/private/structure.json")
        self.assertIn('"/private/frozen.json"', prompt)
        self.assertIn('"/private/structure.json"', prompt)
        self.assertIn("no `film`, `shots` or other fields", prompt)
        self.assertIn("at most 15 shots", prompt)


class AtomicJSONTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="staged json ")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "script.json"
        self.path.write_text('{"old": true}\n', encoding="utf-8")

    def test_success_replaces_target_with_complete_utf8_json(self):
        document = {"film": "synthetic", "label": "字幕", "shots": []}
        stages.atomic_write_json(self.path, document)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), document)
        self.assertEqual(list(self.path.parent.glob(".script.json-*.tmp")), [])

    def test_serialization_error_keeps_existing_output(self):
        before = self.path.read_bytes()
        for document in ({"bad": float("nan")}, {"bad": object()}):
            with self.subTest(document=document), self.assertRaises((TypeError, ValueError)):
                stages.atomic_write_json(self.path, document)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(list(self.path.parent.glob(".script.json-*.tmp")), [])

    def test_failed_atomic_replace_keeps_target_and_cleans_temporary_file(self):
        before = self.path.read_bytes()
        with mock.patch.object(stages.os, "replace", side_effect=OSError("synthetic replace failure")):
            with self.assertRaisesRegex(OSError, "synthetic replace failure"):
                stages.atomic_write_json(self.path, {"new": True})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(".script.json-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
