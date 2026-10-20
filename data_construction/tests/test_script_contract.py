"""Annotation validation must not accept altered or invented supplied dialogue."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_script_queue as queue


class ScriptContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="script contract ")
        self.addCleanup(temporary.cleanup)
        self.task = Path(temporary.name)
        (self.task / "stages").mkdir()
        self.inputs = {
            "film": "sample",
            "shots": [
                {"id": "s0001", "kind": "source",
                 "line": {"text": " Wait, here! ", "from_prev": 0.12,
                          "into_next": 0.34}},
                {"id": "s0002", "kind": "source", "line": None},
            ],
        }
        self.annotation = {
            "film": "sample", "logline": "Two people wait beside a doorway.",
            "shots": [
                {"id": "s0001", "desc": "A person points at a doorway.",
                 "line": " Wait, here! ", "card_text": None},
                {"id": "s0002", "desc": "Another person stops at the doorway.",
                 "line": None, "card_text": None},
            ],
            "beats": [
                {"id": f"b{i}", "name": f"Moment {i}", "shots": [sid, sid],
                 "what": "A person waits.", "function": "Establish the action.",
                 "why_here": "The visible action changes."}
                for i, sid in enumerate(("s0001", "s0002"), 1)
            ],
            "chunks": [
                {"id": f"c{i}", "beat": f"b{i}", "shots": [sid, sid],
                 "what": "Show the person waiting.", "why_here": "One action."}
                for i, sid in enumerate(("s0001", "s0002"), 1)
            ],
        }

    def check(self, annotation=None, inputs=None, slug="sample"):
        for name, document in (("script_inputs", self.inputs if inputs is None else inputs),
                               ("script", self.annotation if annotation is None else annotation)):
            (self.task / "stages" / f"{name}.json").write_text(
                json.dumps(document), encoding="utf-8")
        return queue.check(slug, str(self.task))

    def test_exact_text_projection_and_explicit_null_pass(self):
        self.assertEqual(self.check(), [])

    def test_missing_line_is_rejected_even_when_input_is_null(self):
        for index in (0, 1):
            with self.subTest(index=index):
                annotation = copy.deepcopy(self.annotation)
                del annotation["shots"][index]["line"]
                self.assertTrue(any("line" in error for error in self.check(annotation)))

    def test_rewritten_removed_or_object_dialogue_is_rejected(self):
        for value in ("Wait, here!", " Wait here. ", None,
                      self.inputs["shots"][0]["line"],
                      json.dumps(self.inputs["shots"][0]["line"])):
            with self.subTest(value=value):
                annotation = copy.deepcopy(self.annotation)
                annotation["shots"][0]["line"] = value
                self.assertTrue(any("line" in error for error in self.check(annotation)))

    def test_invented_dialogue_or_empty_string_instead_of_null_is_rejected(self):
        for value in ("I am waiting.", ""):
            with self.subTest(value=value):
                annotation = copy.deepcopy(self.annotation)
                annotation["shots"][1]["line"] = value
                self.assertTrue(any("line" in error for error in self.check(annotation)))

    def test_wrong_or_missing_output_film_is_rejected(self):
        for value in ("other-film", None):
            with self.subTest(value=value):
                annotation = copy.deepcopy(self.annotation)
                if value is None:
                    del annotation["film"]
                else:
                    annotation["film"] = value
                self.assertTrue(any("film" in error for error in self.check(annotation)))

    def test_input_and_output_film_must_match_requested_slug(self):
        self.assertTrue(any("film" in error for error in self.check(slug="other-film")))
        inputs = copy.deepcopy(self.inputs)
        inputs["film"] = "other-film"
        self.assertTrue(any("film" in error for error in self.check(inputs=inputs)))


if __name__ == "__main__":
    unittest.main()
