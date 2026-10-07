"""Pinned, durable outcome boundary; an unavailable judge is NEVER a policy zero.

The callable may be a local eval service or a detached Modal FunctionCall. All
results (including failures) are retained before validation. Retrying this layer
never regenerates a policy answer. A successful canonical record is immutable.
"""
import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path
import uuid

from .contract import ParityError, canonical, digest
from .store import atomic_json
from .capacity import durable_io

CONSISTENCY = 's34-joint-consistency-v3'
DIRECT_JUDGE = 's34-direct-judge-v1'
ARTIFACTS = {f'{prefix}_{suffix}.json' for prefix in ('s3', 's4e')
             for suffix in ('evidence', 'score', 'score_ref', 'final', 'consistency')}


class GradeUnavailable(ParityError):
    pass


class GradeRetryable(GradeUnavailable):
    """A trusted grading attempt completed without a usable score."""


class GradeRetriesExhausted(GradeUnavailable):
    """Discard this GRPO group, never fabricate reward or hide parity errors."""


class GradePending(GradeRetriesExhausted):
    """Remote completion is uncertain: do not submit another grading call."""


class GradeProhibitedContent(GradeRetriesExhausted):
    """Terminal provider refusal: retain evidence and exclude, never retry/zero."""
    reason_code = 'PROHIBITED_CONTENT'

    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('Gemini PROHIBITED_CONTENT; exclude trajectory and its GRPO group')


def prohibited_content_evidence(result, stage):
    # Inspect only trusted provider-error fields, not task text, model answers,
    # rubric explanations, or a bare occurrence of the marker. The existing
    # Gemini adapter truncates its JSON error, so tolerate a truncated suffix.
    prefix = {'s_3': 's3', 's_4e': 's4e'}.get(stage)
    files = result.get('judge_files')
    if prefix is None or not isinstance(files, dict):
        return []
    evidence = []
    for name in (f'{prefix}_score.json', f'{prefix}_score_ref.json'):
        value = files.get(name)
        rows = value.get('rubrics', []) if isinstance(value, dict) else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            error = row.get('err')
            # Gemini can refuse at prompt level (no candidates) or at the
            # candidate finish boundary. Match only the adapter's exact trusted
            # error format; an explanation mentioning the marker is not enough.
            if (isinstance(error, str) and (
                    error == '空回复 finish=PROHIBITED_CONTENT' or
                    (error.startswith('无 candidates: ') and
                     re.search(r'"promptFeedback"\s*:\s*\{\s*"blockReason"\s*:\s*"PROHIBITED_CONTENT"', error)))):
                evidence.append({'file': name, 'rubric': row.get('rubric'),
                                 'reason': 'PROHIBITED_CONTENT', 'provider_error': error})
    return evidence


def input_identity(payload):
    raw = str(payload.get('deliverable_json') or '').encode()
    return {**{k: payload[k] for k in ('task', 'stage', 'source_root')},
            'deliverable_sha256': hashlib.sha256(raw).hexdigest(),
            **({'preview_sha256': hashlib.sha256(payload['preview_bytes']).hexdigest()}
               if isinstance(payload.get('preview_bytes'), bytes) else {})}


def validate_result(result, payload, grader_hashes, source_hashes, review_policy='reviewed',
                    submission_validation=None):
    if review_policy not in ('reviewed', 'judge_only'):
        raise GradeUnavailable('Unknown grading review policy')
    if not isinstance(result, dict) or result.get('format') != 's12-remote-grade-result-v1':
        raise GradeUnavailable('Invalid grader response format')
    identity = input_identity(payload)
    if any(result.get(k) != identity[k] for k in ('task', 'stage')):
        raise GradeUnavailable('Cross-task/stage grade response')
    receipt = result.get('input_contract', {})
    if any(receipt.get(k) != v for k, v in identity.items()) or receipt.get('source_sha256') != source_hashes:
        raise GradeUnavailable('Grader used different delivery or corpus bytes')
    service = result.get('service_contract', {})
    if (not grader_hashes or service.get('grader_sha256') != grader_hashes or
            service.get('consistency_contract') != (DIRECT_JUDGE if review_policy == 'judge_only' else CONSISTENCY)):
        raise GradeUnavailable('Stale/unpinned grader deployment')
    score = result.get('score')
    if submission_validation is not None and service.get('submission_validation') != submission_validation:
        raise GradeUnavailable('Submission-validation policy differs from the launch plan')
    refusal = prohibited_content_evidence(result, payload['stage'])
    if refusal:
        # Identity/source/service checks above still fail closed. Even a partial
        # numeric aggregate cannot turn a provider refusal into a policy score.
        raise GradeProhibitedContent(refusal)
    prefix = {'s_3': 's3', 's_4e': 's4e'}.get(payload['stage'])
    files = result.get('judge_files', {})
    if prefix and isinstance(files, dict):
        for name in (f'{prefix}_score.json', f'{prefix}_score_ref.json'):
            value = files.get(name, {})
            rows = value.get('rubrics', []) if isinstance(value, dict) else []
            if isinstance(rows, list) and any(
                    isinstance(row, dict) and
                    str(row.get('err', '')).startswith('GRADING_TRANSPORT_FAILURE:')
                    for row in rows):
                raise GradeRetryable('Trusted judge transport failed; retry delivery, not partial aggregate')
    if score is None:
        raise GradeRetryable('Trusted grader produced no outcome; retry the same delivery')
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise GradeUnavailable('No finite outcome grade; exclude from policy training')
    if result.get('invalid_submission') is True and score != 0:
        raise GradeUnavailable('Invalid model submission must have outcome zero')
    if review_policy == 'judge_only':
        # The trusted judge owns the reward. Retain its artifacts for analysis,
        # but do not veto a numerical score using schema/evidence/review checks.
        return float(score)
    files = result.get('judge_files')
    if not isinstance(files, dict) or set(files) - ARTIFACTS:
        raise GradeUnavailable('Unexpected/missing grader artifacts')
    if payload['stage'] in ('s_1', 's_2'):
        if result.get('invalid_submission') is True and score != 0:
            raise GradeUnavailable('Invalid deterministic submission is not zero')
        return float(score)
    prefix = {'s_3': 's3', 's_4e': 's4e'}[payload['stage']]
    final = files.get(f'{prefix}_final.json', {})
    invalid = result.get('invalid_submission') is True or final.get('invalid_submission') is True
    if invalid:
        if score != 0:
            raise GradeUnavailable('Invalid model submission must have outcome zero')
    else:
        review = files.get(f'{prefix}_consistency.json', {})
        if (review.get('status') != 'approved' or review.get('contract') != CONSISTENCY or
                final.get('consistency_contract') != CONSISTENCY or final.get('final') != score):
            raise GradeUnavailable('Unapproved/mismatched joint review')
        if review.get('bindings', {}).get('output/cuts.json' if prefix == 's3' else 'output/edit.json') != identity['deliverable_sha256']:
            raise GradeUnavailable('Review not bound to the policy answer')
    return float(score)


class OutcomeGrader:
    def __init__(self, call, *, grader_hashes, concurrency=16, retries=3, retry_delay=0,
                 review_policy='reviewed', submission_validation=None, timeout_seconds=600):
        if concurrency < 1 or retries < 1:
            raise ValueError('Positive grader concurrency/retries required')
        self.call, self.grader_hashes = call, dict(grader_hashes)
        if review_policy not in ('reviewed', 'judge_only'):
            raise ValueError('Unknown grading review policy')
        self.review_policy = review_policy
        if submission_validation not in (None, 'enabled', 'disabled'):
            raise ValueError('Unknown submission-validation policy')
        self.submission_validation = submission_validation
        self.gate = asyncio.Semaphore(concurrency)
        self.retries, self.retry_delay = retries, retry_delay
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 600:
            raise ValueError('Grading total budget must be positive and at most 600 seconds')
        self.timeout_seconds = timeout_seconds

    async def grade(self, payload, directory, source_hashes, commit):
        root = Path(directory)
        await durable_io(root.mkdir, parents=True, exist_ok=True)
        binding = await durable_io(lambda: digest({'input': input_identity(payload), 'sources': source_hashes,
                          'grader': self.grader_hashes,
                          'contract': DIRECT_JUDGE if self.review_policy == 'judge_only' else CONSISTENCY,
                          'submission_validation': self.submission_validation}))
        success = root / 'canonical.json'
        discarded = root / 'discarded.json'
        prior_discard = await durable_io(lambda: json.loads(discarded.read_bytes())
                                        if discarded.exists() else None)
        if prior_discard is not None:
            if (prior_discard.get('binding') != binding or
                    prior_discard.get('reason_code') != 'PROHIBITED_CONTENT' or
                    prior_discard.get('policy_reward') is not None or
                    not isinstance(prior_discard.get('evidence'), list) or
                    not prior_discard['evidence']):
                raise GradeUnavailable('Refusing to reuse a stale/corrupt grade exclusion')
            raise GradeProhibitedContent(prior_discard['evidence'])
        has_saved, saved = await durable_io(lambda: (True, json.loads(success.read_bytes()))
                                            if success.exists() else (False, None))
        if has_saved:
            if saved['binding'] != binding:
                raise GradeUnavailable('Refusing to reuse a stale grade')
            return await durable_io(validate_result, saved['result'], payload, self.grader_hashes, source_hashes,
                                   self.review_policy, self.submission_validation)
        budget_file = root / 'budget.json'
        prior_budget = await durable_io(lambda: json.loads(budget_file.read_bytes())
                                        if budget_file.exists() else None)
        if prior_budget is None:
            prior_budget = {'binding': binding, 'started_at': time.time(),
                            'deadline_unix': time.time() + self.timeout_seconds}
            await durable_io(atomic_json, budget_file, prior_budget)
        if (prior_budget.get('binding') != binding or
                type(prior_budget.get('deadline_unix')) not in (float, int) or
                not math.isfinite(prior_budget['deadline_unix'])):
            raise GradeUnavailable('Refusing stale/corrupt grading deadline')
        deadline = prior_budget['deadline_unix']
        bounded_payload = {**payload, 'grading_deadline_unix': deadline}
        for index in range(self.retries):
            attempt_dir = root / 'attempts' / uuid.uuid4().hex
            await durable_io(attempt_dir.mkdir, parents=True, exist_ok=False)
            await durable_io(lambda: atomic_json(attempt_dir / 'input.json',
                                                {'binding': binding, 'input': input_identity(payload)}))
            try:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise GradePending('Total grading deadline exhausted; exclude without resubmission')
                timer = asyncio.timeout(remaining)
                try:
                    async with timer:
                        async with self.gate:
                            # A wait timeout leaves the exact call journal intact.
                            # Never resubmit an uncertain remote request.
                            result = await self.call(bounded_payload, attempt_dir, commit)
                except TimeoutError as exc:
                    if timer.expired():
                        raise GradePending('Total grading deadline exhausted; remote call retained, no blind retry') from exc
                    raise
                await durable_io(atomic_json, attempt_dir / 'result.json', result)
                score = await durable_io(validate_result, result, payload, self.grader_hashes, source_hashes,
                                        self.review_policy, self.submission_validation)
                await durable_io(atomic_json, success, {'binding': binding, 'result': result})
                # Raw result and canonical decision become durable together.
                # The failure path below also flushes raw result + error; no
                # result is discarded or converted to a synthetic reward.
                await commit()
                return score
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await durable_io(atomic_json, attempt_dir / 'failure.json', {'type': type(exc).__name__, 'error': str(exc)[:1500]})
                if isinstance(exc, GradeProhibitedContent):
                    await durable_io(atomic_json, discarded, {'binding': binding,
                        'status': 'discarded_prohibited_content', 'reason_code': exc.reason_code,
                        'policy_reward': None, 'retriable': False,
                        'attempt': attempt_dir.name, 'evidence': exc.evidence})
                await commit()
                if not isinstance(exc, GradeRetryable):
                    # Corrupt journals, wrong identities/hashes, unknown code
                    # exceptions and uncertain calls are never blind retries.
                    raise
                if index + 1 < self.retries:
                    await asyncio.sleep(min(self.retry_delay * 2 ** index, max(0, deadline - time.time())))
        raise GradeRetriesExhausted(f'Grading failed after {self.retries} attempts; delivery retained at {root}')


def modal_caller(function):
    async def call(payload, directory, commit):
        import modal
        receipt = Path(directory) / 'call.json'
        has_saved_call, saved_call = await durable_io(lambda: (True, json.loads(receipt.read_bytes()))
                                                      if receipt.exists() else (False, None))
        if has_saved_call:
            handle = modal.FunctionCall.from_id(saved_call['call_id'])
        else:
            async def launch_and_record():
                handle = await function.spawn.aio(payload)
                await durable_io(atomic_json, receipt, {'call_id': handle.object_id})
                await commit()
                return handle
            launch = asyncio.create_task(launch_and_record())
            try:
                handle = await asyncio.shield(launch)
            except asyncio.CancelledError:
                # Cancellation must never orphan a detached call without its
                # durable handle (in particular during parallel preparation).
                await launch
                raise
        # No local deadline triggers a duplicate remote judge. The remote
        # function owns its finite deadline; detach survives local disconnect.
        from modal.exception import ConnectionError as ModalConnectionError, FunctionTimeoutError, InternalFailure
        from grpclib.exceptions import GRPCError
        from grpclib.const import Status
        for index in range(3):
            try:
                return await handle.get.aio()
            except asyncio.CancelledError:
                raise
            except (FunctionTimeoutError, InternalFailure) as exc:
                # SDK only raises these after a terminal remote result. A new
                # attempt is safe; connection failures below reuse this ID.
                raise GradeRetryable(f'Terminal remote grade failure: {handle.object_id}: {exc}') from exc
            except (ModalConnectionError, ConnectionError, TimeoutError, GRPCError) as exc:
                if isinstance(exc, GRPCError) and exc.status not in (
                        Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED, Status.RESOURCE_EXHAUSTED,
                        Status.INTERNAL, Status.UNKNOWN):
                    raise
                await durable_io(atomic_json, Path(directory) / f'result_wait_{index}.json', {
                    'call_id': handle.object_id, 'type': type(exc).__name__, 'error': str(exc)[:1500]})
                await commit()
                if index == 2:
                    raise GradePending(f'Result still uncertain; saved call {handle.object_id}; do not resubmit') from exc
                await asyncio.sleep(0)  # Immediately re-read the SAME call, never duplicate it.
    return call
