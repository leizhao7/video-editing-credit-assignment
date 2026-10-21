"""Shot-aware visual preparation without model calls or real film assets."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import annotation_visuals as visuals


class AnnotationVisualTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="script visuals ")
        self.addCleanup(temporary.cleanup)
        self.task = Path(temporary.name).resolve()
        (self.task / "stages").mkdir()
        (self.task / "reference").mkdir()
        self.input_path = self.task / "stages/script_inputs.json"
        self.video = self.task / "reference/golden.mp4"
        self.video.write_bytes(b"synthetic video metadata; frame extraction is mocked")
        self.shots = [{"id": f"s{i:04d}", "start_s": i * 0.043,
                       "dur_s": 0.001 if i % 2 else 0.04} for i in range(13)]
        self.write_inputs()

    def write_inputs(self):
        self.input_path.write_text(json.dumps({"film": "synthetic", "shots": self.shots}))

    def fake_extract(self, command, **kwargs):
        import cv2
        import numpy as np
        frame = np.full((90, 160, 3), (35, 120, 210), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(command[-1], frame))
        return subprocess.CompletedProcess(command, 0, "", "")

    def prepare_fake(self):
        with mock.patch.object(visuals.subprocess, "run", side_effect=self.fake_extract) as run:
            result = visuals.prepare_script_visuals(self.task)
        return result, run

    def test_each_short_shot_has_its_own_midpoint_and_labeled_sheet_cell(self):
        import cv2
        original_input, original_video = self.input_path.read_bytes(), self.video.read_bytes()
        result, run = self.prepare_fake()
        manifest = json.loads(result["manifest_path"].read_text())
        self.assertEqual(run.call_count, 13)
        self.assertEqual(len(result["sheet_paths"]), 2)
        self.assertFalse(result["draft_path"].exists())
        self.assertEqual([s["id"] for s in manifest["shots"]], [s["id"] for s in self.shots])
        for index, (shot, record, call) in enumerate(zip(self.shots, manifest["shots"], run.call_args_list)):
            self.assertEqual(record["requested_midpoint_s"], shot["start_s"] + shot["dur_s"] / 2)
            self.assertEqual((record["row"], record["column"]), (index % 12 // 4, index % 4))
            command = call.args[0]
            self.assertEqual(float(command[command.index("-ss") + 1]), record["frame_window_start_s"])
            self.assertAlmostEqual(float(command[command.index("-t") + 1]),
                                   record["frame_window_end_s"] - record["frame_window_start_s"])
            self.assertTrue((self.task / record["frame_path"]).is_file())
            self.assertEqual(self.task / record["sheet_path"], result["sheet_paths"][index // 12])
        first = cv2.imread(str(result["sheet_paths"][0]))
        self.assertEqual(first.shape, (780, 1536, 3))
        self.assertGreater(int(first[216:260, :384].max()), 200)
        self.assertEqual(self.input_path.read_bytes(), original_input)
        self.assertEqual(self.video.read_bytes(), original_video)

    def test_valid_cache_is_reused_but_input_or_video_metadata_changes_invalidate(self):
        first, _ = self.prepare_fake()
        with mock.patch.object(visuals.subprocess, "run") as run:
            self.assertEqual(visuals.prepare_script_visuals(self.task), first)
            run.assert_not_called()
        self.shots[0]["dur_s"] += 0.02
        self.write_inputs()
        second, run = self.prepare_fake()
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(run.call_count, len(self.shots))
        stat = self.video.stat()
        os.utime(self.video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        third, _ = self.prepare_fake()
        self.assertNotEqual(second["fingerprint"], third["fingerprint"])

    def test_corrupt_cached_image_requires_rebuild_and_failed_rebuild_unpublishes(self):
        result, _ = self.prepare_fake()
        manifest = json.loads(result["manifest_path"].read_text())
        for image in (result["sheet_paths"][0], self.task / manifest["shots"][0]["frame_path"]):
            with self.subTest(image=image.name):
                image.write_bytes(b"corrupt image")
                with mock.patch.object(visuals.subprocess, "run",
                                       return_value=subprocess.CompletedProcess([], 1, "", "decode failed")):
                    with self.assertRaisesRegex(RuntimeError, "Frame extraction failed.*decode failed"):
                        visuals.prepare_script_visuals(self.task)
                self.assertFalse(result["manifest_path"].exists())
                self.prepare_fake()

    def test_extraction_failure_never_publishes_a_success_manifest(self):
        def failure(command, **kwargs):
            Path(command[-1]).write_bytes(b"partial frame")
            return subprocess.CompletedProcess(command, 1, "", "synthetic decode error")
        with mock.patch.object(visuals.subprocess, "run", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "s0000.*synthetic decode error"):
                visuals.prepare_script_visuals(self.task)
        self.assertEqual(list((self.task / "work").rglob("manifest.json")), [])
        self.assertEqual(list((self.task / "work").rglob(".frame-*")), [])

    def test_invalid_ids_and_times_are_rejected_before_extraction(self):
        variants = [
            [{"id": "../escape", "start_s": 0, "dur_s": 1}],
            [{"id": "s1", "start_s": 0, "dur_s": 1}] * 2,
            [{"id": "s1", "start_s": -0.1, "dur_s": 1}],
            [{"id": "s1", "start_s": 0, "dur_s": 0}],
            [{"id": "s1", "start_s": float("nan"), "dur_s": 1}],
            [{"id": "s1", "start_s": 0, "dur_s": float("inf")}],
            [{"id": "s1", "start_s": True, "dur_s": 1}],
            [],
        ]
        for shots in variants:
            with self.subTest(shots=shots), mock.patch.object(visuals.subprocess, "run") as run:
                self.shots = shots
                self.write_inputs()
                with self.assertRaises(ValueError):
                    visuals.prepare_script_visuals(self.task)
                run.assert_not_called()
        self.assertFalse((self.task / "work").exists())

    def test_missing_reference_video_is_an_explicit_error(self):
        self.video.unlink()
        with self.assertRaises(FileNotFoundError):
            visuals.prepare_script_visuals(self.task)

    def test_only_millisecond_starts_receive_a_bounded_rounding_tolerance(self):
        self.shots = [{"id": "rounded", "start_s": 0.167, "dur_s": 0.042},
                      {"id": "precise", "start_s": 0.16683333333333333, "dur_s": 0.042}]
        self.write_inputs()
        result, _ = self.prepare_fake()
        rounded, precise = json.loads(result["manifest_path"].read_text())["shots"]
        self.assertEqual(rounded["start_rounding_tolerance_s"], 0.0005)
        self.assertEqual(rounded["frame_window_start_s"], 0.1665)
        self.assertEqual(rounded["requested_midpoint_s"], 0.188)
        self.assertEqual(rounded["frame_window_end_s"], 0.188 + 1e-9)
        self.assertEqual(precise["start_rounding_tolerance_s"], 0)
        self.assertEqual(precise["frame_window_start_s"], self.shots[1]["start_s"])

    def test_short_shots_require_individual_frame_review_at_strict_threshold(self):
        self.shots = [{"id": "short", "start_s": 0, "dur_s": 0.749},
                      {"id": "threshold", "start_s": 0.749, "dur_s": 0.75},
                      {"id": "long", "start_s": 1.499, "dur_s": 1.0}]
        self.write_inputs()
        result, _ = self.prepare_fake()
        manifest = json.loads(result["manifest_path"].read_text())
        self.assertEqual([shot["requires_individual_frame_read"] for shot in manifest["shots"]],
                         [True, False, False])
        self.assertEqual(manifest["individual_frames"]["max_long_edge"], 1024)
        self.assertFalse(manifest["individual_frames"]["upscale"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "real frame extraction requires FFmpeg")
    def test_real_single_frame_shot_does_not_take_the_next_shots_frame(self):
        import cv2
        ffmpeg = shutil.which("ffmpeg")
        command = [ffmpeg, "-v", "error"]
        for color, duration in (("red", "0.08"), ("green", "0.04"), ("blue", "0.08")):
            command.extend(["-f", "lavfi", "-i", f"color={color}:s=160x90:r=25:d={duration}"])
        command.extend(["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(self.video)])
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        self.shots = [{"id": "red", "start_s": 0, "dur_s": 0.08},
                      {"id": "green_one_frame", "start_s": 0.08, "dur_s": 0.04},
                      {"id": "blue", "start_s": 0.12, "dur_s": 0.08}]
        self.write_inputs()
        result = visuals.prepare_script_visuals(self.task, ffmpeg=ffmpeg)
        manifest = json.loads(result["manifest_path"].read_text())
        for record, expected_channel in zip(manifest["shots"], (2, 1, 0)):
            frame = cv2.imread(str(self.task / record["frame_path"]))
            pixel = frame[frame.shape[0] // 2, frame.shape[1] // 2]
            self.assertEqual(int(pixel.argmax()), expected_channel)
            self.assertGreater(int(pixel[expected_channel]), 100)
        self.assertEqual(manifest["shots"][1]["requested_midpoint_s"], 0.1)
        # No timestamp falls in this narrow window; never borrow the next shot.
        self.shots = [{"id": "empty_window", "start_s": 0.09, "dur_s": 0.01}]
        self.write_inputs()
        with self.assertRaisesRegex(RuntimeError, "did not produce a readable frame"):
            visuals.prepare_script_visuals(self.task, ffmpeg=ffmpeg)

    @unittest.skipUnless(shutil.which("ffmpeg"), "real frame extraction requires FFmpeg")
    def test_real_23976fps_single_frame_survives_millisecond_start_rounding(self):
        import cv2
        ffmpeg = shutil.which("ffmpeg")
        command = [ffmpeg, "-v", "error"]
        for color in ("red", "green", "blue"):
            command.extend(["-f", "lavfi", "-i", f"color={color}:s=160x90:r=24000/1001:d=1"])
        command.extend(["-filter_complex",
                        "[0:v]trim=end_frame=4,setpts=PTS-STARTPTS[r];"
                        "[1:v]trim=end_frame=1,setpts=PTS-STARTPTS[g];"
                        "[2:v]trim=end_frame=4,setpts=PTS-STARTPTS[b];"
                        "[r][g][b]concat=n=3:v=1:a=0[v]", "-map", "[v]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(self.video)])
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        self.shots = [{"id": "green_one_frame", "start_s": round(4 * 1001 / 24000, 3),
                       "dur_s": round(1001 / 24000, 3)}]
        self.write_inputs()
        result = visuals.prepare_script_visuals(self.task, ffmpeg=ffmpeg)
        manifest = json.loads(result["manifest_path"].read_text())
        record = manifest["shots"][0]
        self.assertEqual(record["start_s"], 0.167)
        frame = cv2.imread(str(self.task / record["frame_path"]))
        pixel = frame[frame.shape[0] // 2, frame.shape[1] // 2]
        self.assertEqual(int(pixel.argmax()), 1)
        self.assertGreater(int(pixel[1]), 100)

    @unittest.skipUnless(shutil.which("ffmpeg"), "real frame extraction requires FFmpeg")
    def test_individual_frames_retain_detail_while_sheets_keep_their_layout(self):
        import cv2
        ffmpeg = shutil.which("ffmpeg")
        for width, height, expected in ((1920, 1080, (576, 1024)),
                                        (1080, 1920, (1024, 576)),
                                        (160, 90, (90, 160))):
            with self.subTest(size=(width, height)):
                subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                                f"testsrc2=s={width}x{height}:r=25:d=0.08",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(self.video)],
                               check=True, capture_output=True, text=True, timeout=30)
                self.shots = [{"id": "detail", "start_s": 0, "dur_s": 0.08}]
                self.write_inputs()
                result = visuals.prepare_script_visuals(self.task, ffmpeg=ffmpeg)
                record = json.loads(result["manifest_path"].read_text())["shots"][0]
                frame = cv2.imread(str(self.task / record["frame_path"]))
                self.assertEqual(frame.shape[:2], expected)
                sheet = cv2.imread(str(result["sheet_paths"][0]))
                self.assertEqual(sheet.shape[:2], (780, 1536))


if __name__ == "__main__":
    unittest.main()
