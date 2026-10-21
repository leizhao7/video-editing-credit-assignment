"""Boundary picture contracts, cache invalidation and real decoded-frame regressions."""
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
import annotation_cut_visuals as visuals


class CutVisualsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cut visuals ")
        self.addCleanup(self.temp.cleanup)
        self.task = Path(self.temp.name)
        (self.task / "stages").mkdir()
        (self.task / "reference").mkdir()
        self.video = self.task / "reference/golden.mp4"
        self.video.write_bytes(b"synthetic video metadata placeholder")
        self.write_inputs([1.0, 2.0], [1.0, 1.0, 1.0])

    def write_inputs(self, times, durations):
        cuts = [{"idx": i + 1, "after": f"s{i:04d}", "before": f"s{i+1:04d}",
                 "timeline_s": value, "assembled_s": 10000 + i} for i, value in enumerate(times)]
        docs = {"cut_list.json": {"cuts": cuts},
                "s3_units.json": {"units": [{"id": f"s{i:04d}", "dur": value} for i, value in enumerate(durations)]},
                "audio_measured.json": {"cuts": [{"idx": c["idx"], "speech": {"type": "hard"}, "music": []} for c in cuts]},
                "picture_measured.json": {"cuts": [{"idx": c["idx"], "continuous": False} for c in cuts]}}
        for name, doc in docs.items():
            (self.task / "stages" / name).write_text(json.dumps(doc))

    def snapshots(self):
        return {name: (self.task / "stages" / name).read_bytes() for name in visuals._INPUT_NAMES}

    def fake_decode(self, cv2, video, cut, directory, ffmpeg):
        import numpy as np
        result = []
        for side, value in (("before", cut["boundary_split_s"] - .01), ("after", cut["boundary_split_s"] + .01)):
            ticks = round(value * 1000000)
            result.append({"pts_s": ticks / 1000000, "pts_ticks": ticks, "time_base": [1, 1000000],
                           "side": side, "image": np.full((90, 160, 3), (0, 0, 220) if side == "before" else (0, 220, 0), dtype=np.uint8)})
        return result

    def test_every_boundary_uses_reference_time_two_sheets_per_twelve_and_inputs_unchanged(self):
        self.write_inputs(list(range(1, 14)), [1] * 14)
        original = self.snapshots()
        with mock.patch.object(visuals, "_decode_window", side_effect=self.fake_decode) as decode:
            result = visuals.prepare_cut_visuals(self.task)
        self.assertEqual(decode.call_count, 13)
        manifest = json.loads(result["manifest_path"].read_text())
        self.assertEqual([len(s["cut_indices"]) for s in manifest["sheets"]], [6, 6, 1])
        self.assertEqual([c["timeline_s"] for c in manifest["cuts"]], list(range(1, 14)))
        self.assertTrue(all(c["requires_individual_strip_read"] for c in manifest["cuts"]))
        self.assertTrue(all(len(c["frame_paths"]) == 2 for c in manifest["cuts"]))
        self.assertFalse(result["draft_path"].exists())
        self.assertEqual(self.snapshots(), original)
        import cv2
        self.assertEqual(cv2.imread(str(result["sheet_paths"][0])).shape[:2], (783, 1152))

    def test_cache_reuses_valid_images_and_invalidates_each_input_or_video_metadata(self):
        with mock.patch.object(visuals, "_decode_window", side_effect=self.fake_decode) as decode:
            first = visuals.prepare_cut_visuals(self.task)
            self.assertEqual(visuals.prepare_cut_visuals(self.task), first)
            self.assertEqual(decode.call_count, 2)
            previous = first["fingerprint"]
            for name in visuals._INPUT_NAMES:
                path = self.task / "stages" / name
                path.write_bytes(path.read_bytes() + b"\n")
                current = visuals.prepare_cut_visuals(self.task)
                self.assertNotEqual(current["fingerprint"], previous)
                previous = current["fingerprint"]
            stat = self.video.stat()
            os.utime(self.video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            self.assertNotEqual(visuals.prepare_cut_visuals(self.task)["fingerprint"], previous)

    def test_bad_cache_rebuild_and_failed_rebuild_unpublishes_manifest(self):
        with mock.patch.object(visuals, "_decode_window", side_effect=self.fake_decode):
            result = visuals.prepare_cut_visuals(self.task)
        manifest = json.loads(result["manifest_path"].read_text())
        (self.task / manifest["cuts"][0]["frame_paths"][0]).write_bytes(b"broken")
        with mock.patch.object(visuals, "_decode_window", side_effect=RuntimeError("synthetic extraction failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic extraction failure"):
                visuals.prepare_cut_visuals(self.task)
        self.assertFalse(result["manifest_path"].exists())
        self.assertEqual(list(result["manifest_path"].parent.rglob(".decode-*")), [])
        with mock.patch.object(visuals, "_decode_window", side_effect=self.fake_decode):
            visuals.prepare_cut_visuals(self.task)
        data = json.loads(result["manifest_path"].read_text())
        data["cuts"][0]["frames"] = []
        result["manifest_path"].write_text(json.dumps(data))
        with mock.patch.object(visuals, "_decode_window", side_effect=self.fake_decode) as decode:
            visuals.prepare_cut_visuals(self.task)
        self.assertEqual(decode.call_count, 2)

    def test_invalid_times_adjacency_or_measured_indices_fail_before_extraction(self):
        original = self.snapshots()
        for change in (lambda d: d["cuts"][0].update(timeline_s=float("nan")),
                       lambda d: d["cuts"][0].update(timeline_s=True),
                       lambda d: d["cuts"][0].update(timeline_s=0),
                       lambda d: d["cuts"][1].update(timeline_s=0.9),
                       lambda d: d["cuts"][0].update(after="wrong"),
                       lambda d: d["cuts"][1].update(idx=1)):
            snapshots = dict(original)
            doc = json.loads(snapshots["cut_list.json"])
            change(doc)
            snapshots["cut_list.json"] = json.dumps(doc).encode()
            with self.assertRaises(ValueError):
                visuals._read_cuts(snapshots)
        bad = dict(original)
        bad["picture_measured.json"] = b'{"cuts":[]}'
        with self.assertRaises(ValueError):
            visuals._read_cuts(bad)
        records = visuals._read_cuts(original)
        self.assertAlmostEqual(records[0]["boundary_split_s"], .9995)
        high = dict(original)
        doc = json.loads(high["cut_list.json"])
        doc["cuts"][0]["timeline_s"] = 1.000123
        high["cut_list.json"] = json.dumps(doc).encode()
        self.assertEqual(visuals._read_cuts(high)[0]["boundary_split_s"], 1.000123)

    def test_changed_inputs_during_decode_do_not_publish(self):
        def change(*args):
            path = self.task / "stages/audio_measured.json"
            path.write_bytes(path.read_bytes() + b"\n")
            return self.fake_decode(*args)
        with mock.patch.object(visuals, "_decode_window", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                visuals.prepare_cut_visuals(self.task)
        self.assertEqual(list((self.task / "work").rglob("manifest.json")), [])

    def test_missing_frame_slots_are_gray_labeled_placeholders_and_sides_are_nominal(self):
        labels = []
        real_label = visuals._label
        def record_label(cv2, canvas, text, *args, **kwargs):
            labels.append(text)
            return real_label(cv2, canvas, text, *args, **kwargs)
        with mock.patch.object(visuals, "_decode_window", side_effect=self.fake_decode), \
             mock.patch.object(visuals, "_label", side_effect=record_label):
            result = visuals.prepare_cut_visuals(self.task)
        data = json.loads(result["manifest_path"].read_text())
        import cv2
        strip = cv2.imread(str(self.task / data["cuts"][0]["strip_path"]))
        # Second BEFORE tile has no sample; a corner remains visibly gray.
        pixel = strip[visuals._HEADER_H + 8, visuals._CELL_W + 8]
        self.assertTrue(all(60 < int(value) < 85 for value in pixel), pixel)
        empty = strip[visuals._HEADER_H:visuals._HEADER_H + visuals._CELL_H,
                      visuals._CELL_W:visuals._CELL_W * 2]
        self.assertGreater(int(empty.max()), 180)  # Visible text, not a blank tile.
        self.assertIn("NO FRAME", labels)
        self.assertTrue(any(label.startswith("nominal AFTER PTS=") for label in labels))
        self.assertTrue(any("not a certified visual edit" in note for note in data["notes"]))

    def ffmpeg(self):
        executable = shutil.which("ffmpeg")
        if not executable:
            self.skipTest("FFmpeg not installed")
        return executable

    def make_video(self, fps):
        import numpy as np
        frames = [np.full((90, 160, 3), rgb, dtype=np.uint8) for rgb in
                  [(255, 0, 0)] * 4 + [(0, 255, 0)] + [(0, 0, 255)] * 4]
        subprocess.run([self.ffmpeg(), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                        "-video_size", "160x90", "-framerate", fps, "-i", "pipe:0", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", "0", "-pix_fmt", "yuv444p", "-y", str(self.video)],
                       input=b"".join(frame.tobytes() for frame in frames), capture_output=True, check=True)

    def assert_color(self, path, channel):
        import cv2
        mean = cv2.imread(str(path)).mean(axis=(0, 1))
        self.assertGreater(mean[channel], 200)
        self.assertTrue(all(v < 30 for i, v in enumerate(mean) if i != channel), mean)

    def test_real_short_single_frame_units_and_rounded_23976_boundaries(self):
        for fps_text, fps in (("25", 25), ("24000/1001", 24000 / 1001)):
            with self.subTest(fps=fps):
                self.make_video(fps_text)
                self.write_inputs([round(4 / fps, 3), round(5 / fps, 3)],
                                  [round(4 / fps, 3), round(1 / fps, 3), round(4 / fps, 3)])
                original = self.snapshots()
                result = visuals.prepare_cut_visuals(self.task, self.ffmpeg())
                data = json.loads(result["manifest_path"].read_text())
                first, second = data["cuts"]
                self.assertEqual(first["frame_counts"], {"before": 3, "after": 1})
                self.assertEqual(second["frame_counts"], {"before": 1, "after": 3})
                self.assert_color(self.task / first["frames"][-1]["path"], 1)
                self.assert_color(self.task / second["frames"][0]["path"], 1)
                self.assert_color(self.task / first["frames"][0]["path"], 2)
                self.assert_color(self.task / second["frames"][-1]["path"], 0)
                self.assertAlmostEqual(first["frames"][-1]["pts_s"], 4 / fps, places=8)
                self.assertAlmostEqual(second["frames"][1]["pts_s"], 5 / fps, places=8)
                import cv2
                self.assertEqual(cv2.imread(str(self.task / first["frame_paths"][0])).shape[:2], (90, 160))
                self.assertEqual(self.snapshots(), original)

    def test_real_empty_short_unit_cannot_borrow_next_frame(self):
        self.make_video("25")
        # The middle unit occupies [.171234, .179876), with no decoded frame.
        self.write_inputs([.171234, .179876], [.171234, .008642, .18])
        with self.assertRaisesRegex(RuntimeError, "no decodable frame on after side"):
            visuals.prepare_cut_visuals(self.task, self.ffmpeg())
        self.assertEqual(list((self.task / "work").rglob("manifest.json")), [])

    def test_real_ffmpeg_failure_has_no_success_manifest(self):
        self.ffmpeg()
        with self.assertRaisesRegex(RuntimeError, "frame extraction failed"):
            visuals.prepare_cut_visuals(self.task, self.ffmpeg())
        self.assertEqual(list((self.task / "work").rglob("manifest.json")), [])


if __name__ == "__main__":
    unittest.main()
