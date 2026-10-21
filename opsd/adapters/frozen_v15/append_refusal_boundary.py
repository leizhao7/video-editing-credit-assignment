"""Explicit caller-side refusal extension around the unchanged frozen grader.

Only this instance's validator/evidence dependency is rebound. Original module
globals, exception classes, identity checks, durable journals, budgets and the
agent loop remain untouched. No provider result is rewritten or synthesized.
"""
import functools
import hashlib
from pathlib import Path
import re
import types

POLICY = 'trusted-candidate-finish-refusal-v1'
FROZEN_SOURCE_SHA = '818a4e84d9625c7064b2c32bcd674c334019b2446a7e9609dcf6702bf94f726e'


def candidate_evidence(result, stage):
    prefix = {'s_3': 's3', 's_4e': 's4e'}.get(stage)
    files = result.get('judge_files') if isinstance(result, dict) else None
    if prefix is None or not isinstance(files, dict):
        return []
    evidence = []
    for name in (f'{prefix}_score.json', f'{prefix}_score_ref.json'):
        value = files.get(name)
        rows = value.get('rubrics', []) if isinstance(value, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            error = row.get('err') if isinstance(row, dict) else None
            # The frozen core already recognizes the historical Chinese
            # prompt-block prefix. Add its English equivalent here, alongside
            # both spellings of the candidate-finish extension.
            if (isinstance(error, str) and (
                    error in ('空回复 finish=PROHIBITED_CONTENT',
                              'Empty response finish=PROHIBITED_CONTENT') or
                    (error.startswith('No candidates: ') and
                     re.search(r'"promptFeedback"\s*:\s*\{\s*"blockReason"\s*:\s*"PROHIBITED_CONTENT"', error)))):
                evidence.append({'file': name, 'rubric': row.get('rubric'),
                    'reason': 'PROHIBITED_CONTENT', 'provider_error': error})
    return evidence


def _bind(function, namespace):
    bound = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return functools.update_wrapper(bound, function)


def qualified_grader_type(core=None):
    if core is None:
        from opencode_rl import grading_client as core
    if hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() != FROZEN_SOURCE_SHA:
        raise core.GradeUnavailable('Refusal extension requires the qualified frozen grading source')
    namespace = dict(vars(core))

    def evidence(result, stage):
        return core.prohibited_content_evidence(result, stage) + candidate_evidence(result, stage)

    namespace['prohibited_content_evidence'] = evidence
    validator = _bind(core.validate_result, namespace)
    namespace['validate_result'] = validator

    class CandidateRefusalGrader(core.OutcomeGrader):
        boundary_policy = POLICY
        frozen_grading_source_sha256 = FROZEN_SOURCE_SHA
        # This is the original function's bytecode, not a second grade loop.
        grade = _bind(core.OutcomeGrader.grade, namespace)

    return CandidateRefusalGrader
