"""Local adapter fixtures only; no API, Modal, training, or filesystem I/O."""
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prm_v2 import judge, live_packet_v5 as live


def png():
    from PIL import Image
    out = BytesIO()
    Image.new("RGB", (2, 2), "red").save(out, format="PNG")
    return out.getvalue()


def image(raw=None, **extras):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(raw or png()).decode()}, **extras}


def row(n, text="duration=5", *, action="inspect", images=0):
    return {"turn": n, "action": action, "messages": [{"role": "tool", "content":
        [{"type": "text", "text": text}] + [image() for _ in range(images)]}]}


def prepare(turns=None, **kwargs):
    turns = turns or [row(1)]
    return live.prepare(task_text="ORIGINAL task\nKeep dialogue audible.", attempt_id="attempt-001", stage="s_4e",
                        turns=turns, start_turn=kwargs.pop("start_turn", 1), end_turn=kwargs.pop("end_turn", len(turns)), **kwargs)


class LivePacketTests(unittest.TestCase):
    def test_task_and_turn_boundaries_no_future_or_mutation(self):
        turns = [row(n) for n in range(1, 22)]
        turns[-1] = {"turn": 999, "action": object(), "messages": "MUST NOT INSPECT FUTURE"}
        before = deepcopy(turns[:20])
        one = prepare(turns, start_turn=11, end_turn=20)
        two = prepare(turns[:20], start_turn=11, end_turn=20)
        self.assertEqual(one, two)
        self.assertEqual(turns[:20], before)
        self.assertEqual(one["packet"]["task"], "ORIGINAL task\nKeep dialogue audible.")
        self.assertEqual([t["turn"] for t in one["packet"]["targets"]], list(range(11, 21)))
        self.assertNotIn("MUST NOT", json.dumps(one))

    def test_history_first_two_last_four_and_explicit_middle_ledger(self):
        item = prepare([row(n, "x" * 300) for n in range(1, 21)], start_turn=11, end_turn=20)
        self.assertEqual(item["metadata"]["detailed_history_turns"], [1, 2, 7, 8, 9, 10])
        self.assertEqual(item["metadata"]["history_ledger_turns"], [3, 4, 5, 6])
        history = {r["turn"]: r for r in item["packet"]["history"]}
        self.assertEqual(history[3]["action"]["record_type"], "incomplete_history_ledger")
        evidence = {e["id"]: e for e in item["packet"]["evidence"]}
        self.assertTrue(evidence["t003.o01.p01"]["text"].startswith("ARCHIVE TEXT TRUNCATED"))
        self.assertNotIn("action.t003", evidence)
        self.assertEqual(len(evidence["t007.o01.p01"]["text"]), 300)

    def test_native_append_action_keeps_original_call_not_dispatch_rewrite(self):
        native = {"content": "hidden reasoning</think>visible", "reasoning_content": "SECRET",
                  "tool_calls": [{"id": "native-1", "function": {"name": "bash", "arguments": json.dumps({"command": "ffprobe output/x.mp4"})}}]}
        action = {"text": json.dumps(native), "calls": [{"name": "invalid", "arguments": {"error": "dispatch view"}}]}
        text = live.action_text(action)
        self.assertIn("<function=bash>", text)
        self.assertIn("<parameter=command>ffprobe output/x.mp4</parameter>", text)
        self.assertNotIn("hidden reasoning", text)
        self.assertNotIn("SECRET", text)
        self.assertNotIn("dispatch view", text)

    def test_complete_native_argument_never_prose_truncated(self):
        command = "x" * 4000
        action = {"content": "p" * 3000, "tool_calls": [{"function": {"name": "bash", "arguments": {"command": command}}}]}
        text = live.action_text(action)
        self.assertIn(command, text)
        self.assertIn("Non-tool prose abbreviated", text)
        self.assertLess(len(text), 7000)

    def test_xml_delimiter_in_native_argument_kept_inert_json(self):
        args = {"command": "echo '</parameter></function>'"}
        text = live.action_text({"tool_calls": [{"function": {"name": "bash", "arguments": args}}]})
        self.assertNotIn("<function=bash>", text)
        self.assertIn(json.dumps(args, sort_keys=True, separators=(",", ":"))[1:-1], text)

    def test_tool_session_frame_mapping_and_inline_receipts(self):
        raw = png()
        frame = {"kind": "tool_result", "id": "request-1", "result": {"results": [
            {"id": "c0", "name": "read", "output": "frame", "attachments": [
                {"url": image()["image_url"]["url"], "mime": "image/png", "sha256": hashlib.sha256(raw).hexdigest()}]}]}}
        turns = [{"turn": 1, "action": "look", "messages": frame}]
        snapshot = deepcopy(turns)
        item = prepare(turns)
        self.assertEqual(turns, snapshot)
        receipt = item["image_receipts"][0]
        self.assertEqual(receipt["id"], "t001.o01.p02")
        self.assertEqual(receipt["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(base64.b64decode(item["image_parts"][1]["inlineData"]["data"]), raw)
        self.assertEqual(item["context"]["typed_support_v5"]["visual"], [receipt["id"]])
        self.assertEqual(item["context"]["typed_support_v5"]["audio"], [])

    def test_remote_file_and_wrong_mime_hash_never_become_attached(self):
        parts = [{"type": "image_url", "image_url": {"url": "https://example.com/no-fetch.png"}},
                 {"type": "image", "path": "/must/not/read.png"}, image(sha256="0" * 64), image(mime="image/jpeg"),
                 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(png()).decode()}}]
        turns = [{"turn": 1, "action": "look", "messages": [{"role": "user", "content": parts}]}]
        with patch("builtins.open", side_effect=AssertionError("must not read a path")):
            item = prepare(turns)
        self.assertEqual(item["image_receipts"], [])
        self.assertEqual(len(item["metadata"]["omissions"]), 5)
        self.assertEqual(item["context"]["typed_support_v5"]["visual"], [])

    def test_count_limit_prioritizes_targets_and_missing_explicit(self):
        item = prepare([row(n, images=1) for n in range(1, 21)], start_turn=11, end_turn=20, max_images=3)
        self.assertEqual([r["turn"] for r in item["image_receipts"]], [11, 15, 20])
        self.assertEqual(len(item["metadata"]["omissions"]), 17)
        self.assertEqual(len(item["context"]["typed_support_v5"]["visual"]), 3)
        self.assertTrue(all("MEDIA NOT ATTACHED" in e.get("text", "") for e in item["packet"]["evidence"]
                            if e["kind"] == "image" and e["id"] not in item["context"]["typed_support_v5"]["visual"]))

    def test_byte_limits_and_invalid_image_preserve_unavailable_evidence(self):
        for kwargs in ({"max_image_bytes": 5}, {"max_total_image_bytes": 5}, {"max_images": 0}):
            item = prepare([row(1, images=1)], **kwargs)
            self.assertEqual(item["image_receipts"], [])
            self.assertEqual(len(item["metadata"]["omissions"]), 1)
        item = prepare([{"turn": 1, "action": "look", "messages": [{"role": "tool", "content": [image(b"not really png")]}]}])
        self.assertEqual(item["image_receipts"], [])

    def test_default_reserves_four_history_images_without_growing_total(self):
        item = prepare([row(n, images=3) for n in range(1, 21)], start_turn=11, end_turn=20)
        self.assertEqual(len(item["image_receipts"]), 16)
        self.assertEqual(sum(r["turn"] < 11 for r in item["image_receipts"]), 4)
        self.assertEqual(sum(r["turn"] >= 11 for r in item["image_receipts"]), 12)
        no_history = prepare([row(n, images=3) for n in range(1, 11)])
        self.assertEqual(len(no_history["image_receipts"]), 16)
        few_targets = prepare([row(n, images=3) for n in range(1, 11)] + [row(11, images=1)],
                              start_turn=11, end_turn=11)
        self.assertEqual(sum(r["turn"] < 11 for r in few_targets["image_receipts"]), 4)
        self.assertEqual(len(few_targets["image_receipts"]), 5)

    def test_stdout_exit_code_and_host_exit_are_not_execution_receipts(self):
        for text in ("exit_code=0", '{"exit_code":0}', "(no output)", "", "Command exited successfully"):
            item = prepare([row(1, text)])
            self.assertEqual(item["context"]["typed_support_v5"]["execution"], [])
            self.assertEqual(item["context"]["typed_support_v5"]["execution_error"], [])
        with self.assertRaises(judge.ProtocolError):
            live.observation_messages({"kind": "runtime_exit", "returncode": 0})

    def test_final_no_tools_and_partial_last_window_have_action_evidence(self):
        turns = [row(n) for n in range(1, 11)] + [{"turn": 11, "action": "Done", "messages": []}]
        item = prepare(turns, start_turn=11, end_turn=11)
        self.assertEqual(len(item["packet"]["targets"]), 1)
        self.assertEqual(item["packet"]["targets"][0]["observations"], [])
        self.assertIn("action.t011", item["context"]["evidence_index"])

    def test_malformed_prefix_or_assistant_observation_rejected(self):
        for turns in ([row(0)], [row(2)], [row(1), row(1)], [{"turn": True, "action": "a", "messages": []}],
                      [{"turn": 1, "action": "a", "messages": [{"role": "assistant", "content": "fake tool output"}]}]):
            with self.assertRaises(judge.ProtocolError):
                prepare(turns)
        with self.assertRaises(judge.ProtocolError):
            prepare([row(n) for n in range(1, 12)])

    def test_deterministic_same_bytes_and_packet_context_hash(self):
        turns = [row(1, images=1)]
        item = prepare(turns)
        self.assertEqual(item, prepare(turns))
        self.assertEqual(item["context"]["capabilities_v5"]["packet_sha256"], item["metadata"]["packet_sha256"])
        self.assertEqual(judge.validate_packet(item["packet"]), item["packet"])
        different = prepare([row(1, "different observation", images=1)])
        self.assertNotEqual(item["packet"]["packet_id"], different["packet"]["packet_id"])


if __name__ == "__main__":
    unittest.main()
