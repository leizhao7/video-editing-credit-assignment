"""Translation lifecycle tests with a local fake CLI and synthetic JSON only."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import translate_script as translation


FAKE_CLI = r'''
import json, os, sys, time
from pathlib import Path

backend = os.environ["FAKE_TRANSLATION_BACKEND"]
mode = os.environ.get("FAKE_TRANSLATION_MODE", "success")
document = json.loads(Path(os.environ["FAKE_TRANSLATION_DOCUMENT"]).read_text())
source = Path("stages/script.json")
working = max(Path("work").glob("translation-*"), key=lambda p: p.stat().st_mtime_ns)
output = working / "fields.json"
Path("fake-argv.json").write_text(json.dumps(sys.argv[1:]))
if mode == "changed_source":
    changed = json.loads(source.read_text())
    changed["shots"][0]["desc"] = "unauthorized change"
    source.write_text(json.dumps(changed))
if mode == "deleted_source":
    source.unlink()
if mode == "missing_field":
    document.pop("logline_en")
if mode == "wrong_id":
    document["beats"][0]["id"] = "another"
if mode == "extra_field":
    document["shots"] = []
if mode == "extra_entry_field":
    document["beats"][0]["name"] = "unauthorized source field"
output.write_text(json.dumps(document))
if mode == "deleted_output":
    output.unlink()
if mode == "malformed_output":
    output.write_text("{")

def emit(event):
    print(json.dumps(event), flush=True)

tool = {
    "claude": {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "stages/script.json"}}]}},
    "codex": {"type": "item.started", "item": {"id": "t1", "type": "command_execution", "command": "read synthetic fixture"}},
    "opencode": {"type": "tool_use", "part": {"id": "t1", "tool": "read", "state": {"input": {"filePath": "stages/script.json"}}}},
}[backend]
completion = {
    "claude": {"type": "result", "is_error": False},
    "codex": {"type": "turn.completed"},
    "opencode": {"type": "step_finish", "part": {"reason": "stop"}},
}[backend]
emit(tool)
print("synthetic stderr marker", file=sys.stderr, flush=True)
if mode == "provider_error":
    emit({"type": "result", "is_error": True, "errors": ["synthetic provider rejection"]})
elif mode == "no_completion":
    pass
elif mode.startswith("timeout"):
    if mode == "timeout_after_completion":
        emit(completion)
    emit({"type": "assistant", "message": {"content": [
        {"type": "thinking", "thinking": "PRIVATE_SYNTHETIC_CONTENT_NOT_FOR_PROGRESS"}]}})
    # Bounded fallback keeps a failing deadline regression from hanging the suite.
    for i in range(150):
        emit({"type": "system", "subtype": "thinking_tokens", "estimated_tokens": i + 1})
        time.sleep(0.03)
else:
    emit(completion)
if mode == "exit_failure":
    sys.exit(9)
'''


class TranslationProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="translation lifecycle ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "edit/task_sample"
        (self.task / "stages").mkdir(parents=True)
        self.path = self.task / "stages/script.json"
        original = {"film": "sample", "logline": "A shape appears.",
                    "shots": [{"id": "s1", "desc": "A red shape."}],
                    "beats": [{"id": "b1", "name": "Opening", "what": "A red shape appears."}],
                    "chunks": [{"id": "c1", "what": "Show the shape."}]}
        self.original = (json.dumps(original, indent=3) + "\n\n").encode()
        self.path.write_bytes(self.original)
        self.translated = deepcopy(original)
        self.translated["logline_en"] = original["logline"]
        self.translated["beats"][0].update(name_en="Opening", what_en="A red shape appears.")
        self.translated["chunks"][0]["what_en"] = "Show the shape."
        self.patch = {"logline_en": "A shape appears.",
                      "beats": [{"id": "b1", "name_en": "Opening", "what_en": "A red shape appears."}],
                      "chunks": [{"id": "c1", "what_en": "Show the shape."}]}
        self.fixture = self.root / "translated.json"
        self.fixture.write_text(json.dumps(self.patch))
        self.cli = self.root / "fake annotation cli"
        self.cli.write_text(f"#!{sys.executable}\n" + FAKE_CLI)
        self.cli.chmod(0o755)

    def invoke(self, mode="success", backend="claude", timeout="4"):
        output = io.StringIO()
        with mock.patch.object(translation, "ROOT", str(self.root)), \
             mock.patch.dict(os.environ, VIDEO_EDITING_ANNOTATION_CLI=str(self.cli),
                             VIDEO_EDITING_ANNOTATION_BACKEND=backend,
                             VIDEO_EDITING_ANNOTATION_MODEL="fixture-model",
                             VIDEO_EDITING_ANNOTATION_EFFORT="high",
                             VIDEO_EDITING_ANNOTATION_TIMEOUT_S=timeout,
                             FAKE_TRANSLATION_MODE=mode, FAKE_TRANSLATION_BACKEND=backend,
                             FAKE_TRANSLATION_DOCUMENT=str(self.fixture)), redirect_stdout(output):
            result = translation.run("sample", "Sample")
        return result, output.getvalue()

    def logs(self):
        return sorted((self.root / "edit/runs/sample").glob("*_translation-*"))

    def test_success_for_every_backend_keeps_adapter_settings_and_logs(self):
        for backend in ("claude", "codex", "opencode"):
            with self.subTest(backend=backend):
                self.path.write_bytes(self.original)
                ok, _ = self.invoke(backend=backend)
                self.assertTrue(ok)
                self.assertEqual(json.loads(self.path.read_text()), self.translated)
                argv = json.loads((self.task / "fake-argv.json").read_text())
                self.assertEqual(argv[argv.index("--model") + 1], "fixture-model")
                if backend == "claude":
                    self.assertEqual(argv[argv.index("--effort") + 1], "high")
                    self.assertEqual(argv[argv.index("--permission-mode") + 1], "acceptEdits")
                elif backend == "codex":
                    self.assertIn('model_reasoning_effort="high"', argv)
                    self.assertEqual(argv[argv.index("--sandbox") + 1], "workspace-write")
                else:
                    self.assertEqual(argv[argv.index("--variant") + 1], "high")
        self.assertEqual(len(self.logs()), 3)
        working = list((self.task / "work").glob("translation-*"))
        self.assertEqual(len(working), 3)
        for directory in working:
            self.assertTrue(directory.resolve().is_relative_to(self.task.resolve()))
            self.assertEqual(json.loads((directory / "fields.json").read_text()), self.patch)
        for directory in self.logs():
            self.assertEqual({p.name for p in directory.iterdir()},
                             {"PROMPT.md", "session.jsonl", "progress.log", "stderr.log"})
            self.assertIn("validated translation", (directory / "progress.log").read_text())
            self.assertIn("原文可能已经是英文", (directory / "PROMPT.md").read_text())
            self.assertIn("synthetic stderr marker", (directory / "stderr.log").read_text())

    def test_valid_output_without_completion_is_rejected_for_every_backend(self):
        for backend in ("claude", "codex", "opencode"):
            with self.subTest(backend=backend):
                self.path.write_bytes(self.original)
                ok, output = self.invoke("no_completion", backend)
                self.assertFalse(ok)
                self.assertEqual(self.path.read_bytes(), self.original)
                self.assertIn("did not emit a successful completion", output)

    def test_process_and_validation_failures_restore_exact_original_bytes(self):
        for mode in ("provider_error", "exit_failure", "changed_source", "deleted_source",
                     "missing_field", "deleted_output", "malformed_output", "wrong_id",
                     "extra_field", "extra_entry_field"):
            with self.subTest(mode=mode):
                self.path.write_bytes(self.original)
                ok, output = self.invoke(mode)
                self.assertFalse(ok)
                self.assertEqual(self.path.read_bytes(), self.original)
                self.assertIn("original script restored", output)

    def test_continuous_events_and_even_early_completion_cannot_bypass_deadline(self):
        for mode in ("timeout", "timeout_after_completion"):
            with self.subTest(mode=mode):
                self.path.write_bytes(self.original)
                started = time.monotonic()
                ok, output = self.invoke(mode, timeout="0.3")
                self.assertFalse(ok)
                self.assertLess(time.monotonic() - started, 3)
                self.assertEqual(self.path.read_bytes(), self.original)
                self.assertIn("total runtime limit", output)
                directory = max(self.logs(), key=lambda p: p.stat().st_mtime_ns)
                lines = (directory / "session.jsonl").read_text().splitlines()
                self.assertGreater(len(lines), 3)
                self.assertIn("tool: Read", (directory / "progress.log").read_text())
                self.assertNotIn("PRIVATE_SYNTHETIC_CONTENT", output)
                self.assertNotIn("PRIVATE_SYNTHETIC_CONTENT", (directory / "progress.log").read_text())

    def test_session_and_progress_are_flushed_before_the_cli_exits(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.invoke, "timeout", "claude", "0.8")
            until = time.monotonic() + 2
            visible = False
            while time.monotonic() < until and not future.done():
                directories = self.logs()
                if directories:
                    directory = directories[0]
                    progress, session = directory / "progress.log", directory / "session.jsonl"
                    if progress.exists() and session.exists():
                        visible = "tool: Read" in progress.read_text() and bool(session.read_text().strip())
                        if visible:
                            break
                time.sleep(0.01)
            self.assertTrue(visible, "Logs must be readable while translation is still running")
            self.assertFalse(future.done())
            self.assertFalse(future.result(timeout=5)[0])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_already_translated_document_skips_the_cli(self):
        self.path.write_text(json.dumps(self.translated))
        with mock.patch.object(translation, "ROOT", str(self.root)), \
             mock.patch.object(translation, "annotation_process") as process, redirect_stdout(io.StringIO()):
            self.assertTrue(translation.run("sample", "Sample"))
        process.assert_not_called()
        self.assertFalse((self.root / "edit/runs").exists())

    def test_previous_attempt_patch_cannot_be_accepted_when_new_output_is_missing(self):
        self.assertTrue(self.invoke()[0])
        self.path.write_bytes(self.original)
        ok, _ = self.invoke("deleted_output")
        self.assertFalse(ok)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(len(list((self.task / "work").glob("translation-*"))), 2)

    def test_duplicate_or_invalid_source_ids_are_rejected_before_starting_cli(self):
        for kind in ("beats", "chunks"):
            for identifier in (None, " ", []):
                with self.subTest(kind=kind, identifier=identifier):
                    doc = json.loads(self.original)
                    doc[kind][0]["id"] = identifier
                    self.path.write_text(json.dumps(doc))
                    before = self.path.read_bytes()
                    ok, _ = self.invoke()
                    self.assertFalse(ok)
                    self.assertEqual(self.path.read_bytes(), before)
            doc = json.loads(self.original)
            doc[kind].append(deepcopy(doc[kind][0]))
            self.path.write_text(json.dumps(doc))
            self.assertFalse(self.invoke()[0])
        self.assertFalse((self.task / "fake-argv.json").exists())

    def test_patch_rejects_missing_reordered_duplicate_extra_and_blank_fields(self):
        source = json.loads(self.original)
        source["beats"].append({"id": "b2", "name": "Ending", "what": "The shape vanishes."})
        source["chunks"].append({"id": "c2", "what": "Show its departure."})
        good = deepcopy(self.patch)
        good["beats"].append({"id": "b2", "name_en": "Ending", "what_en": "The shape vanishes."})
        good["chunks"].append({"id": "c2", "what_en": "Show its departure."})
        for kind in ("beats", "chunks"):
            variants = [good[kind][:1], good[kind][::-1], [good[kind][0]] * 2,
                        good[kind] + [{**good[kind][0], "id": "extra"}]]
            for entries in variants:
                with self.subTest(kind=kind, entries=entries), self.assertRaises(ValueError):
                    translation.merge_translation_patch(source, {**good, kind: entries})
        for field, value in (("logline_en", " "), ("logline_en", None), ("beats", None)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                translation.merge_translation_patch(source, {**good, field: value})
        for kind, field in (("beats", "name_en"), ("beats", "what_en"), ("chunks", "what_en")):
            for missing in (False, True):
                bad = deepcopy(good)
                if missing:
                    bad[kind][0].pop(field)
                else:
                    bad[kind][0][field] = " "
                with self.subTest(kind=kind, field=field, missing=missing), self.assertRaises(ValueError):
                    translation.merge_translation_patch(source, bad)
        before = deepcopy((source, good))
        merged = translation.merge_translation_patch(source, good)
        self.assertEqual(translation.source_fields(merged), translation.source_fields(source))
        merged["beats"][0]["name"] = "mutated copy"
        self.assertEqual((source, good), before)


if __name__ == "__main__":
    unittest.main()
