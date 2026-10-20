"""Missing-input and missing-evidence checks with synthetic metadata and PGM frames."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_config
# Load module-level paths before tests temporarily replace release_config.DATA_ROOT.
import build_s1_tasks
import shot_luma
import verify_matches


class MeasurementInputs(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="measurement inputs ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(release_config, "DATA_ROOT", str(self.root)))
        self.console = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.console))

    def write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def inputs(self, *, count=1, prepared=True, matched=True, slug="synthetic"):
        self.write_json(f"exports/{slug}/probe.json", {
            "slug": slug, "dir": str(self.root / "source directory"),
            "source": "source.mp4", "golden": "trailer.mp4"})
        if prepared:
            self.write_json(f"edit/task_{slug}/stages/script_inputs.json", {
                "shots": [{"id": f"s{i}", "start_s": i * 10, "dur_s": 2}
                          for i in range(count)]})
        rows = [f"s{i},{i * 10 if matched else ''}" for i in range(count)]
        (self.root / f"exports/{slug}/p2_refined.csv").write_text(
            "golden_id,refined_src_s\n" + "\n".join(rows) + "\n", encoding="utf-8")

    def invoke(self, module):
        with mock.patch.object(sys, "argv", [module.__name__, "--all", "-j", "1"]):
            module.main()

    @staticmethod
    def frame(command, *, flat=False):
        pixels = bytes(64 * 36) if flat else bytes(range(256)) * 9
        Path(command[-1]).write_bytes(b"P5\n64 36\n255\n" + pixels)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def test_empty_root_reports_preparation_command_without_creating_evidence(self):
        for module, output in ((shot_luma, "shot_luma.json"), (verify_matches, "match_confidence.json")):
            with self.subTest(module=module.__name__), mock.patch("subprocess.run") as run, \
                 self.assertRaisesRegex(SystemExit, "run_script_queue.py.*--prep-only"):
                self.invoke(module)
            run.assert_not_called()
            self.assertFalse((self.root / output).exists())

    def test_missing_script_inputs_reports_the_exact_film(self):
        self.inputs(prepared=False)
        for module in (shot_luma, verify_matches):
            with self.subTest(module=module.__name__), mock.patch("subprocess.run") as run, \
                 self.assertRaisesRegex(SystemExit, "--film synthetic --prep-only"):
                self.invoke(module)
            run.assert_not_called()

    def test_luma_all_does_not_silently_skip_a_film_missing_preparation(self):
        self.inputs(slug="prepared")
        self.inputs(slug="missing", prepared=False)
        with mock.patch("subprocess.run") as run, self.assertRaisesRegex(SystemExit, "--film missing --prep-only"):
            self.invoke(shot_luma)
        run.assert_not_called()

    def test_zero_jobs_fail_clearly_without_empty_statistics(self):
        self.inputs(count=0)
        for module, message in ((shot_luma, "No shots to measure"),
                                (verify_matches, "No eligible source matches")):
            with self.subTest(module=module.__name__), mock.patch("subprocess.run") as run, \
                 self.assertRaisesRegex(SystemExit, message):
                self.invoke(module)
            run.assert_not_called()

    def test_no_matched_source_times_is_not_a_successful_comparison(self):
        self.inputs(matched=False)
        with mock.patch("subprocess.run") as run, self.assertRaisesRegex(SystemExit, "No eligible source matches"):
            self.invoke(verify_matches)
        run.assert_not_called()
        self.assertFalse((self.root / "match_confidence.json").exists())

    def test_all_decoder_failures_preserve_nulls_and_exit_nonzero(self):
        self.inputs()
        for module, output, message in (
            (shot_luma, "shot_luma.json", "No measurable shot frames"),
            (verify_matches, "match_confidence.json", "No measurable comparisons"),
        ):
            with self.subTest(module=module.__name__), \
                 mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, b"", b"decode failed")), \
                 self.assertRaisesRegex(SystemExit, message) as raised:
                self.invoke(module)
            self.assertNotEqual(raised.exception.code, 0)
            self.assertEqual(json.loads((self.root / output).read_text()), {"synthetic/s0": None})

    def test_flat_video_is_valid_luma_but_not_measurable_correlation(self):
        self.inputs()
        with mock.patch("subprocess.run", side_effect=lambda cmd, **kw: self.frame(cmd, flat=True)):
            self.invoke(shot_luma)
            with self.assertRaisesRegex(SystemExit, "No measurable comparisons.*too flat"):
                self.invoke(verify_matches)
        self.assertEqual(json.loads((self.root / "shot_luma.json").read_text()), {"synthetic/s0": [0.0, 0.0]})
        self.assertEqual(json.loads((self.root / "match_confidence.json").read_text()), {"synthetic/s0": None})

    def test_valid_comparison_and_flat_missing_value_remain_distinct(self):
        self.inputs(count=2)
        def decode(command, **kwargs):
            return self.frame(command, flat=float(command[command.index("-ss") + 1]) < 10)
        with mock.patch("subprocess.run", side_effect=decode):
            self.invoke(verify_matches)
        self.assertEqual(json.loads((self.root / "match_confidence.json").read_text()),
                         {"synthetic/s0": None, "synthetic/s1": 1.0})


if __name__ == "__main__":
    unittest.main()
