"""Run the selected synthetic CPU checks; no trainer or service is launched."""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime")]
TESTS = ROOT / "examples/train/answer_self_distillation/tests"
SELECTION = [
    ("test_topk.py", None),
    ("test_opsd.py", "test_teacher_only_insertion_exact_suffix_media_masks_and_no_mutation"),
    ("test_opsd.py", "test_teacher_fails_closed"),
    ("test_opsd.py", "test_hint_budget_and_special_token_rejection"),
    ("test_opsd.py", "test_teacher_packing_equals_separate_turn_guidance_and_gradients"),
    ("test_turn_demonstrations.py", "test_suffix_after_every_exact_context_not_system_no_previous_or_future_demo"),
    ("test_turn_demonstrations.py", "test_turn_teacher_rejects_misalignment_and_overflow"),
    ("test_turn_demonstrations.py", "test_length_bucketing_bounded_including_dp_padding"),
    ("test_turn_demonstrations.py", "test_real_worker_per_turn_targets_map_back_to_unchanged_packed_student"),
    ("test_turn_demonstrations.py", "test_no_current_action_future_observation_or_other_demo_reaches_gemini"),
    ("test_turn_teacher_trim.py", None),
]


def main():
    import pytest
    targets = [str(TESTS / filename) + ("::" + test if test else "") for filename, test in SELECTION]
    targets += [str(ROOT / "tests/support"), str(ROOT / "tests/test_paper_top64.py")]
    return pytest.main(["-q", "-p", "no:cacheprovider", *targets])


if __name__ == "__main__":
    raise SystemExit(main())
