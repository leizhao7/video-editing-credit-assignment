"""Sampling tests use synthetic identities and never ship historical task rows."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_selection as selection


def rows():
    return [{"task": f"{film}-{index}", "film": film, "stage": "s_1", "original_split": "train",
             "task_path": f"/private/{film}/s_1/c{index}"}
            for film, n in (("alpha", 5), ("beta", 2), ("gamma", 1)) for index in range(n)]


class SelectionTests(unittest.TestCase):
    def test_largest_remainder_and_coverage(self):
        self.assertEqual(selection.apportion({"alpha": 5, "beta": 2, "gamma": 1}, 5, True),
                         {"alpha": 3, "beta": 1, "gamma": 1})
        self.assertEqual(selection.apportion({"b": 1, "a": 1}, 1, True), {"b": 0, "a": 1})

    def test_stable_across_input_order_and_not_quality_rejection(self):
        source = rows(); snapshot = copy.deepcopy(source)
        result = selection.select(source, {"train": {"s_1": 5}}, "sample-seed")
        reversed_result = selection.select(source[::-1], {"train": {"s_1": 5}}, "sample-seed")
        self.assertEqual({r["task"] for r in result["tasks"]}, {r["task"] for r in reversed_result["tasks"]})
        self.assertEqual(len(result["sampled_out"]), 3)
        self.assertTrue(result["sampled_out_tasks_are_not_quality_rejections"])
        self.assertEqual(source, snapshot)
        self.assertEqual({r["film"] for r in result["tasks"]}, {"alpha", "beta", "gamma"})

    def test_rejects_split_leakage_duplicates_bad_targets(self):
        source = rows()
        for bad in [source + [source[0]], [dict(r, original_split="val") if i == 0 else r for i, r in enumerate(source)]]:
            with self.assertRaises(ValueError):
                selection.select(bad, {"train": {"s_1": 5}}, "seed")
        for target in [{}, {"train": {"s_1": 9}}, {"train": {"s_1": True}}]:
            with self.assertRaises(ValueError):
                selection.select(source, target, "seed")

    def test_historical_path_film_inference(self):
        source = rows()
        for row in source:
            row.pop("film")
        result = selection.select(source, {"train": {"s_1": 5}}, "seed")
        self.assertEqual(result["count"], 5)
        source[0]["task_path"] = "/unstructured/path"
        with self.assertRaises(ValueError):
            selection.select(source, {"train": {"s_1": 5}}, "seed")


if __name__ == "__main__":
    unittest.main()
