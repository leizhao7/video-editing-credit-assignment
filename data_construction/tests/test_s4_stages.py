"""S4 batch contracts on synthetic measured metadata and native event shapes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotation_s4_stages as stages


def synthetic_s4(count=25):
    cuts = [{"idx": i, "after": f"u{i}", "before": f"u{i+1}", "timeline_s": float(i)}
            for i in range(1, count + 1)]
    inputs = {"units": {"film": "sample", "fps": 25, "counts": {"source": count + 1},
                        "units": [{"id": f"u{i}", "kind": "source", "dur": 1.0}
                                  for i in range(1, count + 2)]},
              "cut_list": {"film": "sample", "fps": 25, "n_cuts": count, "cuts": cuts},
              "audio": {"cuts": [{"idx": i, "speech": {"type": "hard"}, "music": []}
                                 for i in range(1, count + 1)]},
              "picture": {"n": count, "n_continuous": 0,
                          "cuts": [{"idx": i, "continuous": False} for i in range(1, count + 1)]}}
    document = {"stage": "S4", "cuts": [{"idx": c["idx"], "after": c["after"], "before": c["before"],
        "transition": {"type": "cut"}, "technique": ("plain", "reaction", "reveal")[index % 3],
        "note": f"Synthetic visible contrast {index}; supplied speech boundary is hard."}
        for index, c in enumerate(cuts)]}
    manifest = {"fingerprint": "synthetic", "video_path": "reference/golden.mp4",
                "inputs_paths": {filename: f"stages/{filename}" for filename in stages.INPUT_FILES.values()},
                "draft_path": "work/global-s4.draft.json",
                "cuts": [{**deepcopy(c), "strip_path": f"work/strips/cut_{c['idx']}.jpg",
                          "frame_paths": [f"work/frames/cut_{c['idx']}.jpg"]} for c in cuts]}
    return inputs, document, manifest


def read_events(paths, *, failed=False, results=True, image=True):
    events = []
    for i, path in enumerate(paths):
        identifier = f"synthetic_read_{i}"
        events.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": identifier, "name": "Read", "input": {"file_path": str(path)}}]}})
        if results:
            content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "fixture"}}] if image else "text only"
            result = {"type": "tool_result", "tool_use_id": identifier, "content": content}
            if failed:
                result["is_error"] = True
            events.append({"type": "user", "message": {"content": [result]}})
    return events


class S4StageTests(unittest.TestCase):
    def setUp(self):
        self.inputs, self.document, self.manifest = synthetic_s4()

    def test_valid_document_and_inputs_are_not_changed(self):
        before = deepcopy((self.inputs, self.document))
        self.assertEqual(stages.validate_inputs(self.inputs, "sample"), [])
        self.assertEqual(stages.validate_document(self.inputs, self.document), [])
        self.assertEqual((self.inputs, self.document), before)

    def test_indices_and_adjacency_must_be_exact_ordered_and_complete(self):
        for field, value in (("idx", True), ("idx", 2), ("idx", 1.0), ("after", "other"), ("before", "u1")):
            document = deepcopy(self.document)
            document["cuts"][0][field] = value
            self.assertTrue(stages.validate_document(self.inputs, document))
        for cuts in (self.document["cuts"][:-1], self.document["cuts"][::-1], self.document["cuts"] + self.document["cuts"][:1]):
            self.assertTrue(stages.validate_document(self.inputs, {"stage": "S4", "cuts": cuts}))
        self.assertTrue(stages.validate_document(self.inputs, {"cuts": self.document["cuts"]}))

    def test_transition_parameters_must_be_finite_nonnegative_and_not_boolean(self):
        for key in ("dur", "fade_out", "black_hold", "fade_in", "new_numeric_parameter"):
            for value in (-0.1, float("nan"), float("inf"), True):
                document = deepcopy(self.document)
                document["cuts"][0]["transition"][key] = value
                self.assertTrue(stages.validate_document(self.inputs, document), (key, value))
        for value in (None, "0.2", [], {}):
            document = deepcopy(self.document)
            document["cuts"][0]["transition"]["dur"] = value
            self.assertTrue(stages.validate_document(self.inputs, document))
        self.document["cuts"][0]["transition"] = {"type": "wipe", "direction": "left", "dur": 0.0}
        self.assertEqual(stages.validate_document(self.inputs, self.document), [])

    def test_no_change_requires_literal_measured_true(self):
        self.document["cuts"][0]["transition"] = {"type": "no_change"}
        for value in (False, None, 1, "true"):
            self.inputs["picture"]["cuts"][0]["continuous"] = value
            self.assertTrue(stages.validate_document(self.inputs, self.document))
        self.inputs["picture"]["cuts"][0]["continuous"] = True
        self.assertEqual(stages.validate_document(self.inputs, self.document), [])

    def test_required_fields_and_malformed_documents_fail_cleanly(self):
        for field in ("technique", "note", "transition"):
            for value in (None, "", " ", []):
                document = deepcopy(self.document)
                document["cuts"][0][field] = value
                self.assertTrue(stages.validate_document(self.inputs, document))
        for document in (None, [], {"stage": "S4", "cuts": [None]}):
            self.assertTrue(stages.validate_document(self.inputs, document))

    def test_measured_indices_and_unit_ids_cannot_be_missing_or_duplicated(self):
        for key in ("audio", "picture"):
            inputs = deepcopy(self.inputs)
            inputs[key]["cuts"].reverse()
            self.assertTrue(stages.validate_inputs(inputs))
            inputs[key]["cuts"] = inputs[key]["cuts"][:1]
            self.assertTrue(stages.validate_inputs(inputs))
        for value in (None, [], ""):
            inputs = deepcopy(self.inputs); inputs["units"]["units"][0]["id"] = value
            self.assertTrue(stages.validate_inputs(inputs))
        inputs = deepcopy(self.inputs); inputs["units"]["units"].append(inputs["units"]["units"][0])
        self.assertTrue(stages.validate_inputs(inputs))
        self.assertTrue(stages.validate_inputs(self.inputs, "wrong-film"))

    def test_diversity_is_a_whole_film_check_and_prefix_is_exact(self):
        for cut in self.document["cuts"]:
            cut["technique"] = "plain"
        self.assertEqual(stages.validate_document(self.inputs, self.document, prefix=True), [])
        self.assertTrue(stages.validate_document(self.inputs, self.document))
        prefix = {"stage": "S4", "cuts": self.document["cuts"][:12]}
        self.assertEqual(stages.validate_document(self.inputs, prefix, prefix=True), [])
        prefix["cuts"] = self.document["cuts"][1:13]
        self.assertTrue(stages.validate_document(self.inputs, prefix, prefix=True))

    def test_batch_subsets_only_relevant_units_measurements_and_visuals(self):
        before = deepcopy((self.inputs, self.manifest))
        batch, manifest = stages.make_cut_batch(self.inputs, self.manifest, 12, 24)
        self.assertEqual(len(batch["units"]["units"]), 13)
        self.assertEqual([c["idx"] for c in batch["cut_list"]["cuts"]], list(range(13, 25)))
        for key in ("audio", "picture"):
            self.assertEqual(batch[key]["cuts"], self.inputs[key]["cuts"][12:24])
        self.assertEqual(manifest["cuts"], self.manifest["cuts"][12:24])
        self.assertNotIn("inputs_paths", manifest)
        self.assertNotIn("draft_path", manifest)
        batch["units"]["units"][0]["dur"] = 99
        manifest["cuts"][0]["strip_path"] = "other"
        self.assertEqual((self.inputs, self.manifest), before)
        with self.assertRaises(ValueError):
            stages.make_cut_batch(self.inputs, self.manifest, 0, 13)


class S4VisualReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.task = Path(temporary.name)
        self.paths = ["work/strip1.jpg", "work/strip2.jpg"]

    def feed(self, events):
        coverage = stages.VisualReadCoverage(self.task, self.paths)
        for event in events:
            coverage.feed(json.dumps(event))
        return coverage

    def test_matched_successful_native_image_results_cover_strips(self):
        coverage = self.feed(read_events([self.task / p for p in self.paths]))
        self.assertEqual(coverage.failures(), [])

    def test_missing_failed_unanswered_or_text_only_reads_do_not_cover_strips(self):
        variants = [read_events(self.paths[:1]), read_events(self.paths, failed=True),
                    read_events(self.paths, results=False), read_events(self.paths, image=False)]
        for events in variants:
            self.assertTrue(self.feed(events).failures())

    def test_unmatched_results_and_outside_paths_do_not_satisfy_coverage(self):
        events = read_events(self.paths)
        events[1]["message"]["content"][0]["tool_use_id"] = "wrong_id"
        self.assertTrue(self.feed(events).failures())
        self.assertTrue(self.feed(read_events(["/outside/strip1.jpg", self.paths[1]])).failures())
        with self.assertRaises(ValueError):
            stages.VisualReadCoverage(self.task, [self.paths[0], self.paths[0]])


if __name__ == "__main__":
    unittest.main()
