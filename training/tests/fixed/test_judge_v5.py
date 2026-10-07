from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prm_v2 import evidence_context, judge, judge_v5 as v5, provenance_v5


def fixture():
    p = {"packet_id": "fixture-v5", "task_id": "t", "stage": "s_3", "task": "Fix output duration.",
         "history": [{"turn": 1, "action": "inspect total", "observations": "duration=2000"}],
         "targets": [{"turn": 2, "action": "inspect streams", "observations": "stream times", "evidence_ids": ["before", "now"]}],
         "evidence": [{"id": "before", "turn": 1, "kind": "tool_text", "text": "duration=2000"},
                      {"id": "now", "turn": 2, "kind": "tool_text", "text": "video=10; data=2000"}]}
    item = evidence_context.prepare(p, [])
    item["context"].update(typed_support_v5={"visual": [], "audio": [], "execution": []}, fact_index_v5={})
    item["context"]["capabilities_v5"] = provenance_v5.prepare(item["packet"], item["context"])
    return item["packet"], item["context"]


def output(event="new_information", scope="metadata"):
    score = v5.EVENTS[event][0]
    return {"assessments": [{"turn": 2, "event": event, "scope": scope, "score": score,
        "status": "insufficient_evidence" if score is None else "scored",
        "prior_evidence_ids": ["before"], "current_evidence_ids": ["now", "action.t002"],
        "support_evidence_ids": ["before", "now"], "fact_refs": [], "limitations": []}]}


class JudgeV5Tests(unittest.TestCase):
    def test_diagnosis_and_canonical_protocol(self):
        p, ctx = fixture(); raw = output()
        accepted = v5.parse_assessment(raw, p, ctx)
        canonical = v5.canonical_result(accepted, ctx)
        judge.parse_assessment(canonical, p)
        self.assertEqual(accepted, raw)
        self.assertEqual(canonical["assessments"][0]["score"], 1)
        self.assertNotIn("2000", canonical["assessments"][0]["rationale"])

    def test_free_text_and_invented_arithmetic_not_accepted(self):
        p, ctx = fixture()
        for field in ("rationale", "prior_fact", "observed_delta", "value_decimal"):
            raw = output(); raw["assessments"][0][field] = "8.3 seconds; all candidates screened"
            with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_task_or_action_not_observation(self):
        p, ctx = fixture()
        raw = output(); r = raw["assessments"][0]
        r.update(prior_evidence_ids=["task.original"], current_evidence_ids=["action.t002"],
                 support_evidence_ids=["task.original", "action.t002"])
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_no_output_not_success(self):
        p, ctx = fixture(); p["evidence"][1]["text"] = "(no output)"
        raw = output(); raw["assessments"][0].update(prior_evidence_ids=[], support_evidence_ids=["now"])
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_modality_and_execution_boundaries(self):
        p, ctx = fixture()
        for scope in ("visual", "audio", "execution"):
            with self.assertRaises(ValueError): v5.parse_assessment(output(scope=scope), p, ctx)
        for scope in ("visual", "audio", "execution"):
            with self.assertRaises(ValueError): v5.parse_assessment(output("observable_error", scope), p, ctx)

    def test_future_or_cross_side_evidence(self):
        p, ctx = fixture()
        for patch in ({"prior_evidence_ids": ["now"]}, {"current_evidence_ids": ["before"]},
                      {"support_evidence_ids": ["task.original"]}):
            raw = output(); raw["assessments"][0].update(patch)
            with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_abstention_is_not_zero(self):
        p, ctx = fixture(); raw = output("insufficient_evidence", "unknown")
        self.assertIsNone(v5.parse_assessment(raw, p, ctx)["assessments"][0]["score"])
        raw["assessments"][0]["score"] = 0
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_terminal_neutral_still_identifies_its_action_not_observation(self):
        p, ctx = fixture()
        raw = output('honest_status', 'reasoning')
        row = raw['assessments'][0]
        row.update(current_evidence_ids=[], support_evidence_ids=['before'])
        with self.assertRaisesRegex(judge.ProtocolError, 'current action or observation must be identified'):
            v5.parse_assessment(raw, p, ctx)
        row['current_evidence_ids'] = ['action.t002']
        self.assertEqual(v5.parse_assessment(raw, p, ctx)['assessments'][0]['score'], 0)
        self.assertIn('NEVER be empty, including a final answer', v5.SYSTEM_PROMPT)
        # Action identity remains insufficient to substantiate a positive fact.
        row.update(event='new_information', scope='metadata', score=1,
                   prior_evidence_ids=[], support_evidence_ids=['action.t002'])
        with self.assertRaisesRegex(judge.ProtocolError, 'actual observational support'):
            v5.parse_assessment(raw, p, ctx)

    def test_unverified_world_versus_invalid_proof_scope(self):
        p, ctx = fixture(); raw = output("unsupported_verification", "reasoning")
        v5.parse_assessment(raw, p, ctx)
        raw["assessments"][0]["scope"] = "audio"
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_new_inference_from_old_observations(self):
        p, ctx = fixture(); raw = output("new_inference", "reasoning")
        raw["assessments"][0].update(current_evidence_ids=["action.t002"], support_evidence_ids=["before"])
        v5.parse_assessment(raw, p, ctx)

    def test_correction_requires_before_and_after(self):
        p, ctx = fixture(); raw = output("verified_correction", "artifact")
        v5.parse_assessment(raw, p, ctx)
        raw["assessments"][0]["support_evidence_ids"] = ["now"]
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_routine_io_zero_single_repetition_not_negative(self):
        p, ctx = fixture()
        v5.parse_assessment(output("routine_io", "io"), p, ctx)
        with self.assertRaises(ValueError): v5.parse_assessment(output("persistent_repetition", "io"), p, ctx)
        with self.assertRaises(ValueError): v5.parse_assessment(output("new_information", "io"), p, ctx)

    def test_fact_reference_requires_exact_known_source(self):
        p, ctx = fixture(); raw = output()
        raw["assessments"][0]["fact_refs"] = ["sum-1"]
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)
        ctx["fact_index_v5"]["sum-1"] = {"turn": 2, "source_evidence_id": "now", "derived_decimal": "8.4", "basis": "tool_observation"}
        v5.parse_assessment(raw, p, ctx)
        ctx["fact_index_v5"]["sum-1"]["turn"] = 3
        with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_strict_json_and_array_envelope(self):
        p, ctx = fixture(); raw = output()
        value, normalization = v5.parse_wire(json.dumps(raw["assessments"]), p, ctx)
        self.assertEqual(raw, value)
        self.assertEqual(normalization, "bare_assessment_array_wrapped")
        for text in ('{"assessments":[],"assessments":[]}', '{"assessments":NaN}'):
            with self.assertRaises(ValueError): v5.parse_assessment(text, p, ctx)

    def test_event_and_reference_bounds(self):
        p, ctx = fixture()
        for patch in ({"score": True}, {"event": "made_up"}, {"scope": []},
                      {"turn": 3}, {"limitations": ["invented"]}, {"current_evidence_ids": []}):
            raw = output(); raw["assessments"][0].update(patch)
            with self.assertRaises(ValueError): v5.parse_assessment(raw, p, ctx)

    def test_missing_turns_do_not_prove_continuous_waste(self):
        from prm_v2 import quality_cases, context_v5
        q = quality_cases
        command = q._bash("cat output/order.json")
        case = q._case("gapped-repeat", "test", "s_2", "Review order.",
            [q._row(1, command, '{"order":["a"]}'), q._row(3, command, '{"order":["a"]}')],
            q._row(5, command, '{"order":["a"]}'), [0], ["scored"], "Fixture only.")
        item = context_v5.prepare(case["packet"], [])
        p, ctx = item["packet"], item["context"]
        obs = [ev["id"] for ev in p["evidence"] if ev["kind"] == "tool_text"]
        known = {ev["id"]: ev for ev in p["evidence"]}
        row = {"turn": 5, "event": "persistent_repetition", "scope": "io", "score": -1,
            "status": "scored", "prior_evidence_ids": [i for i in obs if known[i]["turn"] < 5],
            "current_evidence_ids": [i for i in obs if known[i]["turn"] == 5] + ["action.t005"],
            "support_evidence_ids": obs, "fact_refs": [], "limitations": ["incomplete_history"]}
        with self.assertRaises(ValueError): v5.parse_assessment({"assessments": [row]}, p, ctx)

    def test_repetition_prompt_exposes_existing_literal_protocol(self):
        for phrase in ("N-2, N-1, N", "one single", "same command string",
                       "non-bash tools", "adjacent timestamps/frame indices",
                       "nonconsecutive rechecks", "necessary, NOT sufficient"):
            self.assertIn(phrase, v5.SYSTEM_PROMPT)
        self.assertIn('not observable_error/-1', v5.SYSTEM_PROMPT)
        self.assertIn('Do not use unchanged_check for matching quota-error text', v5.SYSTEM_PROMPT)

    def test_media_quota_errors_do_not_alias_distinct_actions(self):
        from prm_v2 import quality_cases as q, context_v5
        for actions in (
            [f'<function=inspect_frames><parameter=frame_indices>[{n}]</parameter>'
             '<parameter=path>clip.mp4</parameter></function>' for n in (13, 14, 15)],
            [f'<function=inspect_audio><parameter=path>{name}.m4a</parameter></function>'
             for name in ('a', 'b', 'c')],
            [q._bash('cat /work/order.json'), q._bash('cat /work/other.json'),
             q._bash('cat /work/order.json')],
        ):
            rows = [q._row(n, action, 'Audio inspection quota exceeded')
                    for n, action in enumerate(actions, 1)]
            case = q._case('distinct-actions-same-error', 'test', 's_3',
                'Inspect the supplied media.', rows[:2], rows[2], [0], ['scored'],
                'Only a protocol fixture; not a semantic zero label.')
            item = context_v5.prepare(case['packet'], [])
            ids = [e['id'] for e in item['packet']['evidence'] if e['kind'] == 'tool_text']
            known = {e['id']: e for e in item['packet']['evidence']}
            row = dict(turn=3, event='persistent_repetition', scope='reasoning',
                score=-1, status='scored',
                prior_evidence_ids=[i for i in ids if known[i]['turn'] < 3],
                current_evidence_ids=[i for i in ids if known[i]['turn'] == 3] + ['action.t003'],
                support_evidence_ids=ids, fact_refs=[], limitations=[])
            with self.subTest(actions=actions), self.assertRaisesRegex(
                    judge.ProtocolError, 'consecutive same literal action'):
                v5.parse_assessment({'assessments': [row]}, item['packet'], item['context'])


if __name__ == "__main__":
    unittest.main()
