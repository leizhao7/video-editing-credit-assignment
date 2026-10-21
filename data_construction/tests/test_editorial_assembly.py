"""Exercise FC editorial assembly with synthetic on-disk task inputs.

No real media, model, or cloud service is used. Only external FFmpeg/FFprobe
execution is stubbed; the builder writes and connects all task metadata itself.
"""
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
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_s4_editorial as editorial
import release_config


class EditorialAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="editorial assembly ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fail_music_conversion = False
        self.music_exit = 7
        self.slug, self.chunk = "synthetic_film", "c1"
        self.exports = self.root / "exports" / self.slug
        self.source = self.root / "training_task" / self.slug / "s_4" / self.chunk
        self.exports.mkdir(parents=True)
        self.write_json(self.exports / "probe.json", {"fps": 24})
        self.write_json(self.root / "edit" / ("task_" + self.slug) / "s3_units.json",
                        {"units": [{"id": f"s{i:04d}", "dur": 1.0} for i in range(1, 4)]})
        shots = []
        for i in range(1, 4):
            pid = f"p{i:02d}"
            path = self.source / "initial_state/shots" / (pid + ".mp4")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque synthetic clip; not decoded in this test")
            shots.append({"id": pid, "unit": f"s{i:04d}", "dur_s": 1.0, "origin": "film"})
        self.write_json(self.source / "initial_state/manifest.json", {"shots": shots})
        self.write_json(self.source / "verifier/answer.json", {"fps": 24,
                        "cuts": [{"cut": i, "picture": "cut", "audio": "hard"} for i in (1, 2)]})
        (self.source / "task_prompt.md").write_text("This stretch covers:\n\n> A synthetic narrative beat.\n")
        self.music = self.exports / "stems/music.wav"
        self.music.parent.mkdir()
        with wave.open(str(self.music), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\x00\x00" * 8000 * 30)
        take = self.exports / "takes/dialogue.m4a"
        take.parent.mkdir()
        take.write_bytes(b"opaque synthetic dialogue; not decoded in this test")
        self.write_json(self.exports / "takes.json", {"film": self.slug, "n_takes": 1, "takes": [{
            "file": "takes/dialogue.m4a", "trailer_s": 0.5, "trailer_e": 1.0,
            "words_start_at_s": 0.1, "words_dur_s": 0.5, "take_dur_s": 1.0,
            "film_text": "Synthetic line.", "speech_spans_s": [[0.1, 0.6]]}]})

    @staticmethod
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def media_command(self, argv, **kwargs):
        if argv[0] == "ffprobe":
            path = Path(argv[-1])
            if not path.exists() or path.read_bytes().startswith(b"corrupt"):
                return subprocess.CompletedProcess(argv, 1, "", "synthetic corrupt media")
            kind = "audio" if path.suffix in (".wav", ".m4a") else "video"
            return subprocess.CompletedProcess(argv, 0, json.dumps({"format": {"duration": "30"},
                                                "streams": [{"codec_type": kind}]}), "")
        self.assertEqual(argv[0], "ffmpeg")
        self.assertEqual(Path(argv[argv.index("-i") + 1]), self.music)
        if self.fail_music_conversion:
            Path(argv[-1]).write_bytes(b"corrupt partial conversion")
            return subprocess.CompletedProcess(argv, self.music_exit, "", "synthetic encoder failure")
        shutil.copyfile(self.music, argv[-1])
        return subprocess.CompletedProcess(argv, 0, "", "")

    def build(self, *options):
        output = io.StringIO()
        with mock.patch.object(release_config, "DATA_ROOT", str(self.root)), \
                mock.patch.object(sys, "argv", ["build_s4_editorial.py", "--film", self.slug, *options]), \
                mock.patch("subprocess.run", side_effect=self.media_command), redirect_stdout(output):
            editorial.main()
        self.last_stdout = output.getvalue()
        stage = "s_4x" if "--explore" in options else "s_4e"
        return self.root / "training_task" / self.slug / stage / self.chunk

    def assert_complete_task(self, directory, has_dialogue=True):
        def read(relative):
            return json.loads((directory / relative).read_text())
        task = read("task.json")
        manifest = read("initial_state/manifest.json")
        answer = read("verifier/answer.json")
        seed = read("verifier/_rubric_seed.json")
        metadata = read("verifier/task_meta.json")
        expected_id = f"{self.slug}_s4e_{self.chunk}"
        self.assertEqual([task["id"], manifest["taskId"], answer["task"], seed["task"]], [expected_id] * 4)
        self.assertEqual(task["stage"], metadata["stage"])
        self.assertEqual(task["deliverables"], ["output/edit.json", "output/preview.mp4"])
        self.assertEqual(manifest["n_joins"], len(answer["golden_joins"]))
        self.assertEqual([join["join"] for join in manifest["joins"]], [1, 2])
        self.assertEqual(seed["reference"]["golden_dialogue_ids"], ["d01"] if has_dialogue else [])
        prompt = (directory / "task_prompt.md").read_text()
        self.assertIn("A synthetic narrative beat.", prompt)
        self.assertIn("output/edit.json", prompt)
        self.assertIn("output/preview.mp4", prompt)
        for shot in manifest["shots"]:
            self.assertTrue((directory / "initial_state/shots" / (shot["id"] + ".mp4")).is_file())
        for line in manifest["dialogue"]:
            self.assertTrue((directory / "initial_state" / line["file"]).is_file())
        self.assertTrue((directory / "initial_state" / manifest["music"]["file"]).is_file())
        self.assertTrue((directory / "output").is_dir())

    def test_actual_assembly_writes_prompt_task_and_consistent_metadata(self):
        self.assert_complete_task(self.build())

    def test_explore_assembly_keeps_private_dialogue_metadata_out_of_public_manifest(self):
        directory = self.build("--explore")
        self.assert_complete_task(directory)
        manifest = json.loads((directory / "initial_state/manifest.json").read_text())
        self.assertEqual(set(manifest["dialogue"][0]), {"id", "file", "take_dur_s"})
        private = json.loads((directory / "verifier/takes_content.json").read_text())
        self.assertEqual(private["takes"][0]["text"], "Synthetic line.")

    def test_force_rebuild_regenerates_prompt_without_changing_submission(self):
        directory = self.build()
        submission = directory / "output/edit.json"
        submission.write_bytes(b'{"synthetic_submission":true}\n')
        (directory / "task_prompt.md").unlink()
        self.build("--force")
        self.assert_complete_task(directory)
        self.assertEqual(submission.read_bytes(), b'{"synthetic_submission":true}\n')

    def assert_unpublished(self):
        directory = self.root / "training_task" / self.slug / "s_4e" / self.chunk
        self.assertFalse((directory / "task.json").exists())
        self.assertFalse((directory / "initial_state/manifest.json").exists())
        return directory

    def test_missing_music_fails_without_publishing_task(self):
        self.music.unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unreadable audio"):
            self.build()
        self.assert_unpublished()

    def test_failed_music_conversion_is_removed_and_partial_directory_retries(self):
        self.fail_music_conversion = True
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            self.build()
        directory = self.assert_unpublished()
        self.assertFalse((directory / "initial_state/music.wav").exists())
        (directory / "output/edit.json").write_text('{"keep":"submission"}')
        self.fail_music_conversion = False
        self.assert_complete_task(self.build())
        self.assertEqual((directory / "output/edit.json").read_text(), '{"keep":"submission"}')

    def test_incomplete_prompt_or_corrupt_music_rebuilds_without_force(self):
        directory = self.build()
        (directory / "task_prompt.md").unlink()
        self.assert_complete_task(self.build())
        (directory / "initial_state/music.wav").write_bytes(b"corrupt cached file")
        self.assert_complete_task(self.build())
        self.assertFalse((directory / "initial_state/music.wav").read_bytes().startswith(b"corrupt"))

    def test_zero_exit_with_unreadable_generated_music_still_fails(self):
        self.fail_music_conversion = True
        self.music_exit = 0
        with self.assertRaisesRegex(RuntimeError, "Missing or unreadable audio"):
            self.build()
        directory = self.assert_unpublished()
        self.assertFalse((directory / "initial_state/music.wav").exists())

    def test_complete_task_is_skipped_without_changing_metadata(self):
        directory = self.build()
        before = (directory / "task.json").stat().st_mtime_ns
        self.build()
        self.assertEqual((directory / "task.json").stat().st_mtime_ns, before)
        self.assertIn("0 creative editing tasks", self.last_stdout)

    def test_missing_takes_fails_with_preparation_instructions(self):
        (self.exports / "takes.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "source transcription, locate_lines.py.*cut_takes.py"):
            self.build()
        self.assert_unpublished()

    def test_valid_empty_takes_pool_is_preserved_with_warning(self):
        self.write_json(self.exports / "takes.json", {"film": self.slug, "n_takes": 0, "takes": []})
        directory = self.build()
        self.assert_complete_task(directory, has_dialogue=False)
        self.assertIn("zero takes", self.last_stdout)
        manifest = json.loads((directory / "initial_state/manifest.json").read_text())
        self.assertEqual(manifest["dialogue"], [])

    def test_missing_referenced_take_is_not_silently_dropped(self):
        (self.exports / "takes/dialogue.m4a").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unreadable audio"):
            self.build()
        self.assert_unpublished()

    def test_missing_shot_fails_before_task_is_published(self):
        (self.source / "initial_state/shots/p01.mp4").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing or unreadable video"):
            self.build()
        self.assert_unpublished()

    def test_failed_forced_rebuild_invalidates_metadata_but_preserves_submission(self):
        directory = self.build()
        submission = directory / "output/edit.json"
        submission.write_text('{"keep":true}')
        self.music.unlink()
        with self.assertRaises(RuntimeError):
            self.build("--force")
        self.assert_unpublished()
        self.assertEqual(submission.read_text(), '{"keep":true}')


if __name__ == "__main__":
    unittest.main()
