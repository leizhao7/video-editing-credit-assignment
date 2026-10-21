"""Dependency-free checks of the real OPSD checkpoint identity guard."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "opsd/examples/train/answer_self_distillation"
PACKAGE = "_release_test_opsd_resume"


def load_launch():
    """Load the original stdlib-only modules without shadowing SkyRL examples."""
    package_spec = importlib.util.spec_from_file_location(
        PACKAGE, SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)])
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[PACKAGE] = package
    package_spec.loader.exec_module(package)
    spec = importlib.util.spec_from_file_location(PACKAGE + ".launch", SOURCE / "launch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ResumeIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def unload():
            for name in list(sys.modules):
                if name == PACKAGE or name.startswith(PACKAGE + "."):
                    sys.modules.pop(name, None)
        cls.addClassCleanup(unload)
        cls.launch = load_launch()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opsd-resume-test-")
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name)
        self.receipt = self.checkpoint / "opsd_resume.json"
        self.plan = {
            "opsd": {
                "loss": "teacher_topk_tail_forward_kl", "kl_direction": "forward",
                "top_k": 64, "teacher_mode": "fixed_initial", "recipe_sha256": "recipe-v1",
                "teacher_model_path": "/synthetic/model-revision",
            },
            "contract": {"model_revision": "synthetic-revision", "max_context": 131072},
            "prepared_sha256": "synthetic-prepared-tasks-v1",
            "grader_hashes": {"grader.py": "synthetic-grader-v1"},
            "requested_steps": 50,
        }
        self.write_receipt(step=10)

    def write_receipt(self, step):
        self.receipt.write_text(json.dumps({
            "step": step, "identity": self.launch.objective_identity(self.plan),
        }), encoding="utf-8")

    def reject_changed(self, changed):
        with self.assertRaisesRegex(ValueError, "same-identity resume only"):
            self.launch.check_resume(self.checkpoint, changed)

    def test_matching_identity_accepts_resume_without_mutation(self):
        original_plan, original_receipt = deepcopy(self.plan), self.receipt.read_bytes()
        self.assertIsNone(self.launch.check_resume(self.checkpoint, self.plan))
        self.assertEqual(self.plan, original_plan)
        self.assertEqual(self.receipt.read_bytes(), original_receipt)

    def test_changed_objective_is_rejected(self):
        for changes in ({"loss": "teacher_topk_tail_reverse_kl", "kl_direction": "reverse"},
                        {"top_k": 32}):
            with self.subTest(changes=changes):
                changed = deepcopy(self.plan)
                changed["opsd"].update(changes)
                self.reject_changed(changed)

    def test_changed_teacher_or_recipe_is_rejected(self):
        for key, value in (("teacher_model_path", "/synthetic/other-model"),
                           ("recipe_sha256", "recipe-v2")):
            with self.subTest(key=key):
                changed = deepcopy(self.plan)
                changed["opsd"][key] = value
                self.reject_changed(changed)

    def test_changed_prepared_data_is_rejected(self):
        changed = deepcopy(self.plan)
        changed["prepared_sha256"] = "synthetic-prepared-tasks-v2"
        self.reject_changed(changed)

    def test_changed_harness_contract_is_rejected(self):
        changed = deepcopy(self.plan)
        changed["contract"]["max_context"] = 65536
        self.reject_changed(changed)

    def test_changed_grader_identity_is_rejected(self):
        changed = deepcopy(self.plan)
        changed["grader_hashes"]["grader.py"] = "synthetic-grader-v2"
        self.reject_changed(changed)

    def test_missing_receipt_cannot_resume_old_grpo(self):
        self.receipt.unlink()
        with self.assertRaisesRegex(ValueError, "Cannot resume an old GRPO optimizer as OPSD"):
            self.launch.check_resume(self.checkpoint, self.plan)

    def test_nonpositive_steps_are_rejected(self):
        for step in (-1, 0):
            with self.subTest(step=step):
                self.write_receipt(step)
                with self.assertRaisesRegex(ValueError, "Resume step already finished or invalid"):
                    self.launch.check_resume(self.checkpoint, self.plan)

    def test_completed_or_later_steps_are_rejected(self):
        for step in (50, 51):
            with self.subTest(step=step):
                self.write_receipt(step)
                with self.assertRaisesRegex(ValueError, "Resume step already finished or invalid"):
                    self.launch.check_resume(self.checkpoint, self.plan)


if __name__ == "__main__":
    unittest.main()
