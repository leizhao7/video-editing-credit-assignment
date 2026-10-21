"""Offline compatibility checks for current and historical judge error strings."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'training/src/opencode_rl'
ERRORS = (
    '空回复 finish=PROHIBITED_CONTENT',
    'Empty response finish=PROHIBITED_CONTENT',
    '无 candidates: {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}',
    'No candidates: {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}',
)


def load_module(name, path, **kwargs):
    spec = importlib.util.spec_from_file_location(name, path, **kwargs)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RefusalCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.prefix = '_release_refusal_compatibility'

        def unload():
            for name in list(sys.modules):
                if name.startswith(cls.prefix):
                    sys.modules.pop(name, None)
        cls.addClassCleanup(unload)
        cls.clients = []
        for variant, source in (
                ('fixed', BASE / 'grading_client.py'),
                ('decay', ROOT / 'training/overlays/decay/opencode_rl/grading_client.py')):
            package = cls.prefix + '_' + variant
            load_module(package, BASE / '__init__.py', submodule_search_locations=[str(BASE)])
            cls.clients.append(load_module(package + '.grading_client', source))
        cls.opsd = load_module(cls.prefix + '_opsd',
                              ROOT / 'opsd/adapters/frozen_v15/append_refusal_boundary.py')

    def case(self, client, error, stage='s_3', policy='judge_only', score=0.5):
        payload = {'task': 'synthetic-task', 'stage': stage, 'source_root': '_tt_train',
                   'deliverable_json': '{"synthetic": true}'}
        sources, graders = {'task.json': 'synthetic-task-hash'}, {'grader.py': 'synthetic-grader-hash'}
        prefix = {'s_3': 's3', 's_4e': 's4e'}[stage]
        result = {
            'format': 's12-remote-grade-result-v1', 'task': payload['task'], 'stage': stage,
            'input_contract': {**client.input_identity(payload), 'source_sha256': sources},
            'service_contract': {'grader_sha256': graders,
                                 'consistency_contract': client.DIRECT_JUDGE if policy == 'judge_only' else client.CONSISTENCY},
            'score': score,
            'judge_files': {f'{prefix}_score.json': {'rubrics': [{'rubric': 'synthetic', 'err': error}]}},
        }
        return payload, sources, graders, result

    def test_both_languages_exclude_refusal_before_partial_or_missing_score(self):
        for client in self.clients:
            for stage in ('s_3', 's_4e'):
                for policy in ('reviewed', 'judge_only'):
                    for error in ERRORS:
                        for score in (None, 0.5):
                            with self.subTest(client=client.__name__, stage=stage, policy=policy,
                                              error=error, score=score):
                                payload, sources, graders, result = self.case(client, error, stage, policy, score)
                                original = deepcopy(result)
                                with self.assertRaises(client.GradeProhibitedContent) as raised:
                                    client.validate_result(result, payload, graders, sources, policy)
                                self.assertEqual(raised.exception.evidence[0]['provider_error'], error)
                                self.assertEqual(result, original)

    def test_only_trusted_error_fields_and_exact_prefixes_are_recognized(self):
        unrelated = (
            'PROHIBITED_CONTENT',
            'Empty response finish=PROHIBITED_CONTENT extra text',
            'No candidates: {"message": "PROHIBITED_CONTENT"}',
            'No candidates: {"promptFeedback": {"blockReason": "SAFETY"}}',
            'Reported by user: ' + ERRORS[3],
        )
        for client in self.clients:
            for error in unrelated:
                with self.subTest(client=client.__name__, error=error):
                    payload, sources, graders, result = self.case(client, error)
                    result['judge_files']['s3_score.json']['rubrics'][0]['explanation'] = ERRORS[1]
                    self.assertEqual(client.prohibited_content_evidence(result, 's_3'), [])
                    self.assertEqual(client.validate_result(result, payload, graders, sources, 'judge_only'), 0.5)
                    self.assertEqual(self.opsd.candidate_evidence(result, 's_3'), [])

    def test_identity_checks_still_precede_refusal(self):
        for client in self.clients:
            payload, sources, graders, result = self.case(client, ERRORS[1])
            result['input_contract']['source_sha256'] = {'task.json': 'different-task-hash'}
            with self.assertRaisesRegex(client.GradeUnavailable, 'different delivery or corpus bytes'):
                client.validate_result(result, payload, graders, sources, 'judge_only')

    async def test_terminal_refusal_is_durable_and_never_retried(self):
        with tempfile.TemporaryDirectory(prefix='refusal-compatibility-') as temporary:
            for client_index, client in enumerate(self.clients):
                for error_index, error in enumerate(ERRORS):
                    with self.subTest(client=client.__name__, error=error):
                        payload, sources, graders, result = self.case(client, error)
                        calls = []

                        async def call(*args):
                            calls.append(args)
                            return result

                        async def commit():
                            pass

                        grader = client.OutcomeGrader(call, grader_hashes=graders, retries=3,
                                                      review_policy='judge_only')
                        directory = Path(temporary) / f'{client_index}-{error_index}'
                        for _ in range(2):
                            with self.assertRaises(client.GradeProhibitedContent):
                                await grader.grade(payload, directory, sources, commit)
                        self.assertEqual(len(calls), 1)
                        self.assertFalse((directory / 'canonical.json').exists())
                        discard = json.loads((directory / 'discarded.json').read_bytes())
                        self.assertFalse(discard['retriable'])
                        self.assertIsNone(discard['policy_reward'])
                        self.assertEqual(discard['evidence'][0]['provider_error'], error)

    async def test_nonrefusal_empty_response_still_retries(self):
        with tempfile.TemporaryDirectory(prefix='retry-compatibility-') as temporary:
            for client_index, client in enumerate(self.clients):
                for error_index, error in enumerate(('空回复 finish=MAX_TOKENS', 'Empty response finish=MAX_TOKENS')):
                    with self.subTest(client=client.__name__, error=error):
                        payload, sources, graders, failed = self.case(client, error, score=None)
                        _, _, _, success = self.case(client, '', score=0.5)
                        calls = []

                        async def call(*args):
                            calls.append(args)
                            return failed if len(calls) == 1 else success

                        async def commit():
                            pass

                        grader = client.OutcomeGrader(call, grader_hashes=graders, retries=2,
                                                      review_policy='judge_only')
                        directory = Path(temporary) / f'{client_index}-{error_index}'
                        self.assertEqual(await grader.grade(payload, directory, sources, commit), 0.5)
                        self.assertEqual(len(calls), 2)
                        self.assertFalse((directory / 'discarded.json').exists())

    def test_opsd_extension_accepts_bilingual_candidates_and_english_prompt_blocks(self):
        client = self.clients[0]
        for stage in ('s_3', 's_4e'):
            for error in (ERRORS[0], ERRORS[1], ERRORS[3], ERRORS[3][:-2]):
                with self.subTest(stage=stage, error=error):
                    _, _, _, result = self.case(client, error, stage)
                    self.assertEqual(self.opsd.candidate_evidence(result, stage), [{
                        'file': ('s3' if stage == 's_3' else 's4e') + '_score.json',
                        'rubric': 'synthetic', 'reason': 'PROHIBITED_CONTENT', 'provider_error': error,
                    }])
        # The old prompt-block form remains the frozen core's responsibility.
        _, _, _, result = self.case(client, ERRORS[2])
        self.assertEqual(self.opsd.candidate_evidence(result, 's_3'), [])
        # Compatibility must never relax the pinned-source qualification gate.
        with self.assertRaisesRegex(client.GradeUnavailable, 'qualified frozen grading source'):
            self.opsd.qualified_grader_type(client)


if __name__ == '__main__':
    unittest.main()
