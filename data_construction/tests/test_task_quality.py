"""Portable audit contracts, with synthetic media and fake provider responses."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import task_quality as quality


def decision(value, action="keep_candidate"):
    return {"task": value["task"], "action": action, "reason": "The public task is supported.",
            "media_observations": [{"path": a["path"], "modality": a["mime"], "observed": "A visible colored frame.",
                                    "limitations": "Sampled evidence.", "inspected": True} for a in value["assets"]],
            "issues": [], "repair_plan": [], "additional_evidence_needed": [], "scope_change_needed": False}


class Client:
    def __init__(self, result, finish="STOP", mutate=None):
        self.result, self.finish, self.mutate = result, finish, mutate
        self.requests = 0

    def request(self, payload):
        self.requests += 1
        if self.mutate:
            self.mutate()
        return {"modelVersion": quality.MODEL, "candidates": [{"finishReason": self.finish,
                 "content": {"parts": [{"text": json.dumps(self.result)}]}}]}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = self.root / "tasks/sample/s_1/c1"
        (self.task / "initial_state/clips").mkdir(parents=True)
        (self.task / "verifier").mkdir()
        (self.task / "task.json").write_text('{}')
        (self.task / "task_prompt.md").write_text('Select the relevant colored clip.')
        (self.task / "initial_state/clips/c01.mp4").write_bytes(b'synthetic fake transport')
        (self.task / "initial_state/manifest.json").write_text(json.dumps({"candidates": [{"id": "c01", "file": "clips/c01.mp4"}]}))
        (self.task / "verifier/answer.json").write_text('{"answer":["c01"]}')
        self.value = quality.prepare(self.task)

    def test_preparation_binds_public_private_runtime_and_sources(self):
        self.assertIn("verifier/answer.json", self.value["private_fallible_grading_files"])
        self.assertNotIn("verifier/answer.json", self.value["public_files"])
        self.assertEqual(len(self.value["assets"]), 1)
        self.assertTrue(self.value["current_local_RL_tools"])
        self.assertIn("not a live deployment snapshot", self.value["runtime_contract_scope"])
        result = quality.audit(self.task, self.root / "prepared", prepare_only=True)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["model_requests"], 0)

    def test_superseded_backups_excluded_actual_grading_sources_bound(self):
        backup = self.task / "verifier/.evaluation-backups"; backup.mkdir()
        (backup / "rubric.json.oldhash").write_text("superseded")
        value = quality.prepare(self.task)
        self.assertFalse(any(".evaluation-backups" in key for key in value["source_files"]))
        self.assertIn("released_grading_sources/evaluation/task_pipeline.py", value["private_fallible_grading_files"])
        quality.verify_sources(value)
        for stage, names in [("s_3", ["grade_s3.py", "s3_score.py", "rubrics_video.py", "s3_judge_ref.py"]),
                             ("s_4e", ["s4e_score.py", "s4e_judge_ref.py", "rubrics_video.py"])]:
            sources = quality.grading_sources(stage)
            self.assertTrue(all("released_grading_sources/evaluation/" + name in sources for name in names))

    def test_happy_provider_binds_complete_decision_without_editing(self):
        before = {p: p.read_bytes() for p in quality.source_files(self.task)}
        client = Client(decision(self.value))
        result = quality.audit(self.task, self.root / "complete", client=client)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["action"], "keep_candidate")
        self.assertTrue(result["source_integrity_verified"])
        self.assertTrue(result["module_did_not_modify_sources"])
        self.assertNotIn("original_files_modified", result)
        self.assertEqual(client.requests, 1)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertTrue((self.root / "complete/request.json").is_file())

    def test_changed_harness_source_invalidates_bound_audit(self):
        harness = Path(quality.__file__).resolve().parents[1] / "evaluation/s12_task.py"
        original_sha = quality.sha
        with mock.patch.object(quality, "sha", side_effect=lambda path: (
                "changed-harness" if Path(path) == harness else original_sha(path))):
            with self.assertRaisesRegex(ValueError, "harness source changed"):
                quality.verify_sources(self.value)

    def test_changed_bound_or_released_tools_invalidate_audit(self):
        changed = deepcopy(self.value)
        changed["current_local_RL_tools"][0]["description"] = "Changed bound tool"
        with self.assertRaisesRegex(ValueError, "Bound tool contract changed"):
            quality.verify_sources(changed)
        with mock.patch.object(quality, "FUNCTIONS", [{"name": "changed_release_tool"}]):
            with self.assertRaisesRegex(ValueError, "Released tool contract changed"):
                quality.verify_sources(self.value)
        changed = deepcopy(self.value)
        changed["tool_contract_source"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown tool contract source"):
            quality.verify_sources(changed)

    def test_operator_tools_remain_bound_without_requiring_released_schema(self):
        value = quality.prepare(self.task, tools=[{"name": "custom_tool", "description": "Custom runtime"}])
        self.assertEqual(value["tool_contract_source"], "operator_supplied")
        with mock.patch.object(quality, "FUNCTIONS", [{"name": "different_release_tool"}]):
            quality.verify_sources(value)
        value["current_local_RL_tools"][0]["description"] = "Changed custom runtime"
        with self.assertRaisesRegex(ValueError, "Bound tool contract changed"):
            quality.verify_sources(value)

    def test_missing_duplicate_unknown_or_uninspected_media_rejected(self):
        good = decision(self.value)
        bads = []
        for observations in [[], good["media_observations"] * 2,
                             [dict(good["media_observations"][0], path="unknown.mp4")],
                             [dict(good["media_observations"][0], inspected=False)]]:
            bad = deepcopy(good); bad["media_observations"] = observations; bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):
                quality.validate_result(bad, self.value)

    def test_filter_requires_major_evidence_repair_is_not_acceptance(self):
        result = decision(self.value, "filter_recommended")
        with self.assertRaises(ValueError):
            quality.validate_result(result, self.value)
        result["issues"] = [{"severity": "major", "category": "contradiction", "explanation": "Unsupported public requirement.",
                             "evidence_paths": ["task_prompt.md"], "media_time_windows": []}]
        quality.validate_result(result, self.value)
        result["issues"][0]["evidence_paths"] = ["not_supplied"]
        with self.assertRaises(ValueError):
            quality.validate_result(result, self.value)
        result = decision(self.value, "repair_needed")
        with self.assertRaises(ValueError):
            quality.validate_result(result, self.value)
        result["repair_plan"] = ["Clarify the public requirement."]
        quality.validate_result(result, self.value)
        result["action"] = "keep_candidate"
        with self.assertRaises(ValueError):
            quality.validate_result(result, self.value)

    def test_nonstop_and_provider_failure_are_held_not_filter_and_not_retried(self):
        for finish in ["MAX_TOKENS", "OTHER"]:
            client = Client(decision(self.value), finish)
            result = quality.audit(self.task, self.root / finish, client=client)
            self.assertEqual(result["status"], "held")
            self.assertIsNone(result["action"])
            self.assertEqual(client.requests, 1)
        client = mock.Mock(); client.request.side_effect = TimeoutError("no response")
        result = quality.audit(self.task, self.root / "timeout", client=client)
        self.assertEqual(result["status"], "held")
        self.assertEqual(client.request.call_count, 1)

    def test_bad_json_and_wrong_model_keep_raw_response_but_publish_no_result(self):
        for name, response in [("bad-json", {"modelVersion": quality.MODEL, "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{"}]}}]}),
                               ("wrong-model", {"modelVersion": "another-model", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(decision(self.value))}]}}]})]:
            client = mock.Mock(); client.request.return_value = response
            result = quality.audit(self.task, self.root / name, client=client)
            self.assertEqual(result["status"], "held")
            self.assertEqual(client.request.call_count, 1)
            self.assertTrue((self.root / name / "response.json").exists())
            self.assertFalse((self.root / name / "result.json").exists())

    def test_source_mutation_cannot_publish_decision(self):
        client = Client(decision(self.value), mutate=lambda: (self.task / "task_prompt.md").write_text("changed"))
        result = quality.audit(self.task, self.root / "changed", client=client)
        self.assertEqual(result["status"], "held")
        self.assertIn("changed", result["error"])
        self.assertFalse(result["source_integrity_verified"])
        self.assertTrue(result["module_did_not_modify_sources"])
        self.assertFalse((self.root / "changed/result.json").exists())

    def test_rejects_extra_fields_types_duplicate_json_and_reused_output(self):
        result = decision(self.value); result["extra"] = "ignored?"
        with self.assertRaises(ValueError): quality.validate_result(result, self.value)
        result = decision(self.value); result["scope_change_needed"] = 0
        with self.assertRaises(ValueError): quality.validate_result(result, self.value)
        with self.assertRaises(ValueError): quality.load_json('{"a":1,"a":2}')
        quality.audit(self.task, self.root / "unique", prepare_only=True)
        with self.assertRaises(FileExistsError): quality.audit(self.task, self.root / "unique", prepare_only=True)

    def test_all_tasks_summary_only_lists_completed_keeps(self):
        tasks = self.root / "tasks.json"; tasks.write_text(json.dumps([str(self.task)]))
        splits = self.root / "splits.json"; splits.write_text('{"sample":"train"}')
        with mock.patch.object(quality, "Gemini", return_value=Client(decision(self.value))):
            code = quality.main(["--tasks", str(tasks), "--output", str(self.root / "batch"), "--film-splits", str(splits)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((self.root / "batch/accepted_tasks.json").read_text()), [str(self.task.resolve())])
        index = json.loads((self.root / "batch/accepted_index.json").read_text())
        self.assertEqual(index["tasks"][0]["original_split"], "train")
        self.assertFalse(index["split_assignment_required"])

    def test_files_upload_accepts_empty_start_body_and_validates_host(self):
        media = self.task / "initial_state/clips/c01.mp4"
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {"X-Goog-Upload-URL": quality.HOST + "/upload/session"}
        response.read.side_effect = AssertionError("Start body must not be parsed")
        with mock.patch.dict("os.environ", {"GOOGLE_AI_STUDIO_API_KEY": "test-key"}):
            client = quality.Gemini()
        with mock.patch.object(quality.urllib.request, "urlopen", return_value=response), mock.patch.object(
                client, "json_request", return_value=({"file": {"state": "ACTIVE", "uri": quality.HOST + "/v1beta/files/sample"}}, {})):
            self.assertIn("fileData", client.upload(media, "video/mp4"))
        response.headers = {"X-Goog-Upload-URL": "https://unrelated.invalid/upload"}
        with mock.patch.object(quality.urllib.request, "urlopen", return_value=response):
            with self.assertRaises(ValueError): client.upload(media, "video/mp4")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_reel_keeps_all_video_timelines_and_embedded_sound(self):
        source = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=96x64:r=24:d=0.25",
                        "-f", "lavfi", "-i", "sine=frequency=500:duration=0.25", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(source)], check=True, capture_output=True)
        assets = []
        for index in range(11):
            path = self.task / f"initial_state/clips/c{index:02d}.mp4"
            shutil.copyfile(source, path)
            assets.append({"path": str(path.relative_to(self.task)), "mime": "video/mp4", "sha256": quality.sha(path), "visibility": "public"})
        value = dict(self.value, assets=assets)
        output = self.root / "reel"; output.mkdir()
        supplied, receipt = quality.transport(value, output)
        self.assertEqual(len(receipt["timeline"]), 11)
        self.assertEqual(len(receipt["embedded_audio"]), 11)
        self.assertEqual(sum(a["mime"].startswith("video/") for a, _ in supplied), 1)
        self.assertEqual(sum(a["mime"].startswith("audio/") for a, _ in supplied), 11)
        self.assertTrue(all(t["original_start_s"] == 0 and t["slowdown"] >= 1 for t in receipt["timeline"]))
        self.assertTrue(all(path.is_file() for _, path in supplied))


if __name__ == "__main__":
    unittest.main()
