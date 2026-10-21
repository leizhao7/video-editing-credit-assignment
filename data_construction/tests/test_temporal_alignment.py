"""Temporal alignment regressions on synthetic descriptors and private temp media."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_s23_tasks as builder
import classify_cards
import p2_match
import p2_refine
import release_config
from temporal_alignment import (SCORE_KIND, aligned_source_start, colour_verdict,
                                paired_frame_scores, peak_interval)


class PairedScoring(unittest.TestCase):
    def test_pairs_recover_known_offset_instead_of_static_average_distractor(self):
        query = np.eye(3)
        # The static average looks more similar under the old all-pairs formula
        # than the exact ordered sequence. Only corresponding frames should count.
        source = np.vstack([np.full((3, 3), 1 / np.sqrt(3)),
                            np.zeros((3, 3)), query])
        similarity = source @ query.T
        old = [similarity[o:o + 3, np.arange(3)].mean()
               for o in range(len(source) - 2)]
        scores = paired_frame_scores(source, query)
        self.assertEqual(int(np.argmax(old)), 0)
        self.assertEqual(int(np.argmax(scores)), 6)
        self.assertAlmostEqual(scores[6], 1)
        self.assertLess(paired_frame_scores(query[::-1], query)[0], 1)

    def test_scores_equal_explicit_paired_sum_for_every_valid_offset(self):
        rng = np.random.default_rng(7)
        source, query = rng.normal(size=(8, 5)), rng.normal(size=(3, 5))
        expected = [sum(np.dot(source[start + i], query[i]) for i in range(3)) / 3
                    for start in range(6)]
        np.testing.assert_allclose(paired_frame_scores(source, query), expected)

    def test_sparse_query_includes_final_valid_start_and_exact_length_window(self):
        source = np.eye(8)
        query = source[[5, 7]]
        scores = paired_frame_scores(source, query, [0, 2])
        self.assertEqual(len(scores), 6)
        self.assertEqual(int(np.argmax(scores)), 5)
        self.assertEqual(paired_frame_scores(query, query).tolist(), [1.0])
        self.assertEqual(len(paired_frame_scores(source[:1], query)), 0)
        self.assertEqual(len(paired_frame_scores(source, np.empty((0, 8)))), 0)

    def test_bad_dimensions_offsets_and_nonfinite_scores_are_rejected(self):
        for source, query, offsets in (
            (np.eye(2), np.eye(3), None),
            (np.eye(2), np.eye(2), [1, 2]),
            (np.eye(2), np.eye(2), [0, 0]),
            (np.eye(2), np.eye(2), [0, 1.5]),
            (np.array([[np.nan, 0]]), np.ones((1, 2)), None),
        ):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                paired_frame_scores(source, query, offsets)

    def test_peak_width_is_contiguous_and_disjoint_candidates_remain_visible(self):
        p = peak_interval([.99, .4, .989, 1, .995, .6])
        self.assertEqual(p, {"selected_offset": 3, "peak_left_offset": 2,
            "peak_right_offset": 4, "plateau_frames": 3,
            "near_peak_count": 4, "disjoint_near_peaks": 1})
        flat = peak_interval(np.ones(8))
        self.assertEqual(flat["plateau_frames"], 8)
        self.assertEqual(flat["disjoint_near_peaks"], 0)

    def test_offsets_use_actual_query_and_clamped_decoder_origins(self):
        # Source search started at zero; the nominal st-win would be negative.
        self.assertAlmostEqual(aligned_source_start(0, 2, 10, 2, 2), .2)
        # A middle-shot query must subtract its actual shift from shot start.
        self.assertAlmostEqual(aligned_source_start(10, 24, 24, 3.5, 3), 10.5)
        self.assertAlmostEqual(aligned_source_start(0, 24, 24000 / 1001, 0, 0), 1.001)
        for fps in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                aligned_source_start(0, 0, fps, 0, 0)

    def test_colour_acceptance_requires_explicit_new_score_threshold(self):
        self.assertEqual(colour_verdict(.99), "REVIEW(colour)")
        self.assertEqual(colour_verdict(.8, .8), "MATCH(colour)")
        self.assertEqual(colour_verdict(.79, .8), "NOT-IN-SOURCE")
        for threshold in (float("nan"), float("inf"), -1.1, 1.1):
            with self.assertRaises(ValueError):
                colour_verdict(.9, threshold)


class AlignmentIntegration(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="alignment regression ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.exports = self.root / "exports/synthetic"
        self.exports.mkdir(parents=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(release_config, "DATA_ROOT", str(self.root)))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))

    def csv(self, name, rows):
        path = self.exports / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def refine_input(self, **overrides):
        return self.csv("p2_final.csv", [{"golden_id": "s1", "g_start_s": 2,
            "g_end_s": 2.3, "matched_src_s": .2, "verdict": "MATCH", **overrides}])

    def refine(self, extra=()):
        with mock.patch.object(sys, "argv", ["p2_refine", "--film", "synthetic",
                "--source", "source.mkv", "--golden", "golden.mkv",
                "--fps", "10", "--win", ".5", *extra]):
            try:
                p2_refine.main()
            except SystemExit as exc:
                if exc.code:
                    raise
        with (self.exports / "p2_refined.csv").open() as stream:
            return list(csv.DictReader(stream))[0]

    def test_refine_wiring_updates_fallback_query_origin_and_clamped_search(self):
        before = self.refine_input().read_bytes()
        frames = np.random.default_rng(9).integers(0, 256, (6, 58, 128), dtype=np.uint8)
        calls = []
        def decoder(command, **kwargs):
            calls.append(command)
            if command[command.index("-i") + 1] == "source.mkv":
                data = frames
            elif float(command[command.index("-ss") + 1]) > 2:
                data = frames[2:4]  # Too short: retry using the whole reference shot.
            else:
                data = frames[2:5]
            return subprocess.CompletedProcess(command, 0, data.tobytes(), b"")
        with mock.patch("subprocess.run", side_effect=decoder):
            row = self.refine()
        self.assertEqual(float(row["refined_src_s"]), .2)
        self.assertEqual(row["selected_offset"], "2")
        self.assertEqual(row["alignment_score_kind"], SCORE_KIND)
        self.assertEqual(row["plateau_kind"], "contiguous-peak-band-width-v1")
        self.assertEqual(float(row["alignment_fps"]), 10)
        self.assertEqual(calls[-1][calls[-1].index("-ss") + 1], "0.0000")
        self.assertEqual((self.exports / "p2_final.csv").read_bytes(), before)

    def test_review_colour_passes_through_without_refined_source_time(self):
        self.refine_input(verdict="REVIEW(colour)", matched_src_s="",
                          candidate_src_s="12.5", review_reason="uncalibrated_colour_fallback")
        with mock.patch("subprocess.run") as run:
            row = self.refine()
        run.assert_not_called()
        self.assertEqual(row["refined_src_s"], "")
        self.assertEqual(row["candidate_src_s"], "12.5")
        self.assertEqual(row["review_reason"], "uncalibrated_colour_fallback")

    def test_insufficient_decode_is_review_not_source_match_with_empty_time(self):
        self.refine_input()
        frame = bytes(58 * 128)
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, frame, b"")):
            row = self.refine()
        self.assertEqual(row["verdict"], "REVIEW(alignment)")
        self.assertEqual(row["refined_src_s"], "")
        self.assertEqual(row["alignment_review_reason"], "insufficient_decoded_frames")

    def test_failed_decoder_does_not_publish_refinement(self):
        self.refine_input()
        with mock.patch("subprocess.run", return_value=subprocess.CompletedProcess([], 1, b"", b"broken input")), \
             self.assertRaisesRegex(RuntimeError, "broken input"):
            self.refine()
        self.assertFalse((self.exports / "p2_refined.csv").exists())

    def match(self, threshold=None, orb_support=False, decode_error=False):
        self.csv("golden_clips/manifest.csv", [{"id": "s1", "start_s": 0, "end_s": 1}])
        self.csv("source_manifest.csv", [{"id": "clip1", "start_s": 0}])
        rng = np.random.default_rng(12)
        gray = rng.integers(20, 220, (10, 44, 96), dtype=np.uint8)
        np.save(self.exports / "src_gray96_4.npy", gray)
        colour = rng.integers(0, 256, (6, 28, 64, 3), dtype=np.uint8)
        def decoder(command, **kwargs):
            source = command[command.index("-i") + 1] == "source.mkv"
            if command[command.index("-pix_fmt") + 1] == "rgb24":
                data = colour if source else colour[2:6]
                if decode_error:
                    return subprocess.CompletedProcess(command, 1, data.tobytes(), b"partial RGB decode")
            elif "-frames:v" in command:
                data = np.zeros((int(960 / 2.39), 960), dtype=np.uint8)
            else:
                data = gray[2:6]
            return subprocess.CompletedProcess(command, 0, data.tobytes(), b"")
        options = [] if threshold is None else ["--paired-colour-threshold", str(threshold)]
        orb = mock.Mock()
        orb.detectAndCompute.return_value = ([], None)
        stack = ExitStack()
        with stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["p2_match", "--film", "synthetic",
                "--source", "source.mkv", "--golden", "golden.mkv", "--out",
                str(self.exports / "p2_final.csv"), *options]))
            stack.enter_context(mock.patch("subprocess.run", side_effect=decoder))
            stack.enter_context(mock.patch("cv2.ORB_create", return_value=orb))
            if orb_support:
                points = [mock.Mock(pt=(i, i)) for i in range(30)]
                orb.detectAndCompute.return_value = (points, np.zeros((30, 32), dtype=np.uint8))
                matcher = mock.Mock()
                matcher.match.return_value = [mock.Mock(queryIdx=i, trainIdx=i) for i in range(30)]
                stack.enter_context(mock.patch("cv2.BFMatcher", return_value=matcher))
                stack.enter_context(mock.patch("cv2.findHomography", return_value=(None, np.ones((30, 1)))))
            p2_match.main()
        with (self.exports / "p2_final.csv").open() as stream:
            return list(csv.DictReader(stream))[0]

    def test_match_colour_defaults_to_review_with_only_candidate_fields_populated(self):
        row = self.match()
        self.assertEqual(row["verdict"], "REVIEW(colour)")
        self.assertEqual(row["source_clip"], "")
        self.assertEqual(row["matched_src_s"], "")
        self.assertEqual(row["candidate_source_clip"], "clip1")
        self.assertTrue(row["candidate_src_s"])
        self.assertEqual(row["colour_score_kind"], SCORE_KIND)
        self.assertEqual(row["review_reason"], "uncalibrated_colour_fallback")

    def test_explicit_colour_policy_and_orb_support_still_allow_source_matches(self):
        row = self.match(.9)
        self.assertEqual(row["verdict"], "MATCH(colour)")
        self.assertEqual(row["colour_threshold"], "0.9")
        self.assertTrue(row["matched_src_s"])
        row = self.match(orb_support=True)
        self.assertEqual(row["verdict"], "MATCH")
        self.assertEqual(row["colour_score_kind"], "")

    def test_failed_colour_decode_cannot_accept_partial_frames(self):
        with self.assertRaisesRegex(RuntimeError, "partial RGB decode"):
            self.match(.9, decode_error=True)
        self.assertFalse((self.exports / "p2_final.csv").exists())

    def test_review_candidates_still_receive_card_filtering(self):
        rows = [{"golden_id": f"s{i}", "verdict": v} for i, v in enumerate(
            ["MATCH", "REVIEW(colour)", "NOT-IN-SOURCE", "BLACK", "REVIEW(alignment)"])]
        p2 = self.csv("p2_refined.csv", rows)
        self.csv("golden_clips/manifest.csv", [{"id": r["golden_id"], "start_s": i,
            "dur_s": 1} for i, r in enumerate(rows)])
        (self.exports / "probe.json").write_text(json.dumps({"dir": "synthetic",
            "golden": "golden.mkv", "crop": 0}))
        before = p2.read_bytes()
        with mock.patch.object(classify_cards, "ROOT", str(self.root)), \
             mock.patch.object(sys, "argv", ["classify_cards", "--film", "synthetic"]), \
             mock.patch.object(classify_cards, "feats", return_value={"edge": .02, "sharp": .5}) as feats:
            classify_cards.main()
        result = json.loads((self.exports / "card_class.json").read_text())
        self.assertEqual(set(result), {"s1", "s2", "s4"})
        self.assertTrue(all(r["kind"] == "card" for r in result.values()))
        self.assertEqual(feats.call_count, 3)
        self.assertEqual(p2.read_bytes(), before)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for synthetic media regression")
    def test_real_lossless_sequence_recovers_known_offset_near_source_start(self):
        frames = np.random.default_rng(101).integers(0, 256, (40, 64, 96), dtype=np.uint8)
        for name, data in (("source", frames), ("golden", frames[7:27])):
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                "-s", "96x64", "-r", "10", "-i", "-", "-c:v", "ffv1",
                str(self.root / f"{name}.mkv")], input=data.tobytes(), check=True, capture_output=True)
        input_path = self.refine_input(g_start_s=0, g_end_s=2, matched_src_s=.8)
        before = input_path.read_bytes()
        row = self.refine(["--source", str(self.root / "source.mkv"), "--golden",
                           str(self.root / "golden.mkv"), "--win", "1.5"])
        self.assertAlmostEqual(float(row["refined_src_s"]), .7, places=4)
        self.assertAlmostEqual(float(row["peak"]), 1, places=5)
        self.assertEqual(row["plateau_frames"], "1")
        self.assertEqual(input_path.read_bytes(), before)


class AlignmentTolerance(unittest.TestCase):
    def test_alignment_samples_convert_at_measured_fps_not_source_fps(self):
        row = builder.inpoint_tolerance({"plateau": 48, "alignment_fps": 24,
            "alignment_score_kind": SCORE_KIND, "plateau_kind": "contiguous-peak-band-width-v1"}, 60, 30)
        self.assertEqual(row["in_tol_s"], 1)
        self.assertEqual(row["in_tol_frames"], 60)
        self.assertEqual(row["alignment_fps"], 24)

    def test_existing_six_source_frame_minimum_is_preserved_in_seconds(self):
        row = builder.inpoint_tolerance({"plateau": 1, "alignment_fps": 24}, 60, 24)
        self.assertEqual(row["in_tol_s"], .1)
        self.assertEqual(row["in_tol_frames"], 6)

    def test_invalid_recorded_fps_does_not_silently_fall_back(self):
        for fps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                builder.inpoint_tolerance({"plateau": 1, "alignment_fps": fps}, 60, 24)

    def test_legacy_csv_uses_known_pipeline_reference_fps_and_keeps_legacy_label(self):
        row = builder.inpoint_tolerance({"plateau": 24}, 60, 24)
        self.assertEqual(row["in_tol_s"], .5)
        self.assertEqual(row["in_tol_frames"], 30)
        self.assertEqual(row["plateau_kind"], "legacy-near-peak-count")

    def test_collected_alignment_metadata_reaches_generated_rc_answer(self):
        pb = {"src": {"fps": 60, "dur": 100}, "gold": {"fps": 24}}
        inputs = {"shots": [{"id": "s1", "kind": "source", "dur_s": 1, "start_s": 0}]}
        match = {"refined_src_s": "2", "plateau_frames": "48", "alignment_fps": "24",
                 "alignment_score_kind": SCORE_KIND, "plateau_kind": "contiguous-peak-band-width-v1",
                 "near_peak_count": "50", "disjoint_near_peaks": "1",
                 "alignment_review_reason": "disjoint_near_peaks"}
        ch = {"id": "c1", "shots": ["s1", "s1"], "what": "synthetic sequence"}
        with mock.patch.object(builder, "cover", return_value=(0, 10)):
            seq, cards = builder.collect(ch, inputs, {"s1": match}, [(0, 10)], pb,
                                         "source.mkv", "golden.mkv", 0, 0)
        with tempfile.TemporaryDirectory(prefix="alignment rc key ") as directory, \
             mock.patch.object(builder, "cut", return_value=True), \
             mock.patch.object(builder, "run_legacy_rubric_hook"), \
             mock.patch.object(builder, "build_id", return_value="synthetic"):
            builder.build_s3("synthetic", "Synthetic", ch, seq, cards, pb, directory)
            key = json.loads((Path(directory) / "verifier/answer.json").read_text())["cuts"][0]
        self.assertEqual(key["in_tol_s"], 1)
        self.assertEqual(key["in_tol_frames"], 60)
        self.assertEqual(key["alignment_fps"], 24)
        self.assertEqual(key["near_peak_count"], 50)
        self.assertEqual(key["alignment_review_reason"], "disjoint_near_peaks")


if __name__ == "__main__":
    unittest.main()
