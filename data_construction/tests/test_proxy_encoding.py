"""Proxy encoding checks with temporary metadata and an optional real FFmpeg clip."""
from contextlib import ExitStack
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
import pipeline


class ProxyEncoding(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="proxy encoding ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(pipeline, "ROOT", str(self.root)))
        self.stack.enter_context(mock.patch.object(pipeline, "SRC", str(self.root / "sources")))
        self.stack.enter_context(mock.patch.dict(os.environ, VIDEO_EDITING_PROXY_ENCODER=""))

    def film(self, name="Synthetic film", duration=10):
        film = pipeline.Film(name)
        Path(film.dir).mkdir(parents=True)
        metadata = {"source": "source with spaces.mp4", "src": {"vdur": duration},
                    "src_crop_expr": "crop=160:90:16:8"}
        (Path(film.E) / "probe.json").write_text(json.dumps(metadata), encoding="utf-8")
        return film

    @staticmethod
    def encoded(command):
        Path(command[-1]).write_bytes(b"temporary encoded output")
        return subprocess.CompletedProcess(command, 0, "", "")

    def test_platform_default_preserves_crop_scale_bitrate_and_paths(self):
        for platform, encoder in (("darwin", "h264_videotoolbox"), ("linux", "libx264")):
            with self.subTest(platform=platform):
                film = self.film("Synthetic " + platform)
                with mock.patch.object(sys, "platform", platform), \
                     mock.patch.object(pipeline, "sh", side_effect=self.encoded) as execute, \
                     mock.patch.object(pipeline, "_dur", return_value=10):
                    film._proxy()
                command = execute.call_args.args[0]
                self.assertEqual(command[command.index("-c:v") + 1], encoder)
                self.assertEqual("-hwaccel" in command, platform == "darwin")
                self.assertEqual(command[command.index("-i") + 1], str(Path(film.dir) / "source with spaces.mp4"))
                self.assertEqual(command[command.index("-vf") + 1], "crop=160:90:16:8,scale=960:-2")
                self.assertEqual(command[command.index("-b:v") + 1], "1800k")
                self.assertIn("-an", command)
                self.assertTrue(film.done("proxy"))
                self.assertFalse((Path(film.dir) / "_proxy.part.mp4").exists())

    def test_invalid_encoder_is_rejected_before_running_ffmpeg(self):
        film = self.film()
        for encoder in ("libx264 -crf 0", "h264_nvenc", "$(touch unwanted)"):
            with self.subTest(encoder=encoder), \
                 mock.patch.dict(os.environ, VIDEO_EDITING_PROXY_ENCODER=encoder), \
                 mock.patch.object(pipeline, "sh") as execute, \
                 self.assertRaisesRegex(ValueError, "VIDEO_EDITING_PROXY_ENCODER"):
                film._proxy()
            execute.assert_not_called()

    def test_videotoolbox_override_requires_macos(self):
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.dict(os.environ, VIDEO_EDITING_PROXY_ENCODER="h264_videotoolbox"), \
             self.assertRaisesRegex(ValueError, "requires macOS"):
            self.film()._proxy()

    def test_hardware_failure_does_not_fallback_or_publish_partial_video(self):
        film = self.film()
        def fail(command):
            self.encoded(command)
            return subprocess.CompletedProcess(command, 1, "", "hardware encoder unavailable")
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch.object(pipeline, "sh", side_effect=fail) as execute, \
             self.assertRaisesRegex(RuntimeError, "hardware encoder unavailable"):
            film._proxy()
        self.assertEqual(execute.call_count, 1)
        self.assertFalse(film.done("proxy"))
        self.assertFalse((Path(film.dir) / "_proxy.mp4").exists())
        self.assertFalse((Path(film.dir) / "_proxy.part.mp4").exists())

    def test_duration_mismatch_does_not_publish_software_output(self):
        film = self.film(duration=10)
        with mock.patch.dict(os.environ, VIDEO_EDITING_PROXY_ENCODER="libx264"), \
             mock.patch.object(pipeline, "sh", side_effect=self.encoded), \
             mock.patch.object(pipeline, "_dur", return_value=1), \
             self.assertRaises(RuntimeError):
            film._proxy()
        self.assertFalse(film.done("proxy"))
        self.assertFalse((Path(film.dir) / "_proxy.mp4").exists())
        self.assertFalse((Path(film.dir) / "_proxy.part.mp4").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg and ffprobe required")
    def test_real_software_encode_of_short_generated_video(self):
        encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                  capture_output=True, text=True, check=True, timeout=10)
        if "libx264" not in encoders.stdout:
            self.skipTest("FFmpeg build has no libx264 encoder")
        film = self.film(duration=1.25)
        source = Path(film.dir) / "source with spaces.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=192x108:rate=12", "-t", "1.25",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                       capture_output=True, check=True, timeout=30)
        with mock.patch.dict(os.environ, VIDEO_EDITING_PROXY_ENCODER=" libx264 "):
            film._proxy()
        proxy = Path(film.dir) / "_proxy.mp4"
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "stream=codec_name,width,height,avg_frame_rate:format=duration",
                                 "-of", "json", str(proxy)],
                                capture_output=True, text=True, check=True, timeout=10)
        metadata = json.loads(result.stdout)
        self.assertEqual(metadata["streams"], [{"codec_name": "h264", "width": 960,
                                               "height": 540, "avg_frame_rate": "12/1"}])
        self.assertAlmostEqual(float(metadata["format"]["duration"]), 1.25, delta=1 / 12)
        self.assertTrue(film.done("proxy"))
        self.assertTrue(film.state["proxy"]["cropped"])
        self.assertFalse((Path(film.dir) / "_proxy.part.mp4").exists())


if __name__ == "__main__":
    unittest.main()
