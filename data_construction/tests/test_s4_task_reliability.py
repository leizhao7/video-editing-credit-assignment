"""S4 intermediate construction failures with synthetic on-disk inputs."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_s1_tasks
import build_s4_tasks as builder
import release_config


class S4TaskReliabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="s4 intermediate tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.slug, self.chunk = "synthetic_film", "c1"
        self.exports = self.root / "exports" / self.slug
        self.source = self.root / "training_task" / self.slug / "s_3" / self.chunk
        self.destination = self.root / "training_task" / self.slug / "s_4" / self.chunk
        self.fail_music = self.fail_cut = False
        self.music_commands = []
        self.write_json(self.exports / "probe.json", {
            "slug": self.slug, "dir": str(self.root / "media"), "source": "source.mp4",
            "golden": "trailer.mp4", "crop": None, "src": {"dur": 30}})
        edits = self.root / "edit" / ("task_" + self.slug)
        units = [{"id": f"s{i:04d}", "dur": 1.0, "kind": "source"} for i in range(1, 4)]
        self.write_json(edits / "s3_units.json", {"units": units})
        self.write_json(edits / "stages/cut_list.json", {"cuts": [
            {"idx": i, "after": units[i-1]["id"], "before": units[i]["id"],
             "after_kind": "source", "before_kind": "source"} for i in (1, 2)]})
        self.write_json(edits / "stages/s4_cuts.json", {"cuts": [
            {"idx": i, "transition": {"type": "cut"}} for i in (1, 2)]})
        self.write_json(edits / "stages/audio_measured.json", {"cuts": [
            {"idx": i, "speech": {"type": "hard"}, "music": []} for i in (1, 2)]})
        self.write_json(self.source / "verifier/answer.json", {"fps": 24, "cuts": [
            {"trailer_shot": unit["id"], "origin": "film", "clip": f"p{i:02d}", "in_s": 0.0, "dur_s": 1.0}
            for i, unit in enumerate(units, 1)]})
        for i in range(1, 4):
            path = self.source / "initial_state/clips" / f"p{i:02d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic clip; external decoding is stubbed")
        (self.source / "task_prompt.md").write_text("This stretch covers:\n\n> Synthetic context.\n")
        self.music = self.exports / "stems/music.wav"
        self.music.parent.mkdir()
        self.music.write_bytes(b"synthetic audio; external decoding is stubbed")

    @staticmethod
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def media_command(self, argv, **kwargs):
        if argv[0] == "ffprobe":
            path = Path(argv[-1])
            if not path.is_file() or path.read_bytes().startswith(b"corrupt"):
                return subprocess.CompletedProcess(argv, 1, "", "synthetic invalid media")
            kind = "audio" if path.suffix in (".wav", ".m4a") else "video"
            return subprocess.CompletedProcess(argv, 0, json.dumps({"format": {"duration": "30"},
                                                "streams": [{"codec_type": kind}]}), "")
        self.assertEqual(argv[0], "ffmpeg")
        self.music_commands.append(argv)
        if self.fail_music:
            Path(argv[-1]).write_bytes(b"corrupt partial output")
            return subprocess.CompletedProcess(argv, 7, "", "synthetic conversion failure")
        shutil.copy(self.music, argv[-1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    def cut(self, source, start, duration, crop, output, **kwargs):
        if self.fail_cut:
            Path(output).write_bytes(b"corrupt partial shot")
            return False
        shutil.copy(source, output)
        return True

    def build(self):
        with mock.patch.object(release_config, "DATA_ROOT", str(self.root)), \
                mock.patch.object(sys, "argv", ["build_s4_tasks.py", "--film", self.slug, "--keep-plain"]), \
                mock.patch.object(build_s1_tasks, "load_matches", return_value={}), \
                mock.patch.object(build_s1_tasks, "build_id", return_value="synthetic-build"), \
                mock.patch.object(build_s1_tasks, "cut", side_effect=self.cut), \
                mock.patch("subprocess.run", side_effect=self.media_command), redirect_stdout(io.StringIO()):
            builder.main()

    def assert_unpublished(self):
        self.assertFalse((self.destination / "task.json").exists())
        self.assertFalse((self.destination / "initial_state/manifest.json").exists())

    def test_success_preserves_music_window_command_and_task_contract(self):
        self.build()
        task = json.loads((self.destination / "task.json").read_text())
        manifest = json.loads((self.destination / "initial_state/manifest.json").read_text())
        answer = json.loads((self.destination / "verifier/answer.json").read_text())
        self.assertEqual(task["id"], manifest["taskId"])
        self.assertEqual(task["id"], answer["task"])
        self.assertEqual(task["deliverables"], ["output/treatments.json"])
        self.assertEqual(len(manifest["shots"]), 3)
        command = self.music_commands[0]
        self.assertEqual(command[command.index("-ss") + 1], "0.000")
        self.assertEqual(command[command.index("-t") + 1], "7.000")

    def test_missing_music_fails_without_success_metadata(self):
        self.music.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unreadable audio"):
            self.build()
        self.assert_unpublished()

    def test_failed_music_conversion_cleans_partial_file_and_can_retry(self):
        self.fail_music = True
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self.build()
        self.assert_unpublished()
        self.assertFalse((self.destination / "initial_state/music.wav").exists())
        self.fail_music = False
        self.build()
        self.assertTrue((self.destination / "task.json").exists())

    def test_corrupt_cached_music_is_regenerated(self):
        self.build()
        cached = self.destination / "initial_state/music.wav"
        cached.write_bytes(b"corrupt cached output")
        self.build()
        self.assertEqual(cached.read_bytes(), self.music.read_bytes())

    def test_failed_shot_conversion_is_an_error_not_a_silent_skip(self):
        self.fail_cut = True
        with self.assertRaisesRegex(RuntimeError, "shot conversion failed"):
            self.build()
        self.assert_unpublished()
        self.assertFalse((self.destination / "initial_state/shots/p01.mp4").exists())

    def test_verifier_write_failure_invalidates_old_publication_and_can_retry(self):
        for relative in ("verifier/answer.json", "verifier/build.json"):
            with self.subTest(relative=relative):
                self.build()
                submission = self.destination / "output/treatments.json"
                submission.write_text('{"user_submission": true}')
                target = self.destination / relative
                dump = json.dump

                def fail_verifier(data, stream, *args, **kwargs):
                    if Path(stream.name) == target:
                        self.assert_unpublished()
                        stream.write('{"partial":')
                        raise OSError("synthetic verifier write failure")
                    return dump(data, stream, *args, **kwargs)

                with mock.patch("json.dump", side_effect=fail_verifier):
                    with self.assertRaisesRegex(OSError, "verifier write failure"):
                        self.build()
                self.assert_unpublished()
                self.assertEqual(submission.read_text(), '{"user_submission": true}')
                self.build()
                self.assertTrue((self.destination / "task.json").is_file())
                json.loads(target.read_text())

    def test_partial_publication_failure_removes_both_markers_and_can_retry(self):
        for relative in ("initial_state/manifest.json", "task.json"):
            with self.subTest(relative=relative):
                self.build()
                target = self.destination / relative
                dump = json.dump

                def fail_publication(data, stream, *args, **kwargs):
                    if Path(stream.name) == target:
                        if relative == "task.json":
                            manifest = self.destination / "initial_state/manifest.json"
                            self.assertEqual(json.loads(manifest.read_text())["n_shots"], 3)
                        stream.write('{"partial":')
                        raise OSError("synthetic publication write failure")
                    return dump(data, stream, *args, **kwargs)

                with mock.patch("json.dump", side_effect=fail_publication):
                    with self.assertRaisesRegex(OSError, "publication write failure"):
                        self.build()
                self.assert_unpublished()
                self.build()
                task = json.loads((self.destination / "task.json").read_text())
                manifest = json.loads((self.destination / "initial_state/manifest.json").read_text())
                self.assertEqual(task["id"], manifest["taskId"])


if __name__ == "__main__":
    unittest.main()
