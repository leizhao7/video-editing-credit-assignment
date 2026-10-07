"""Explicit, hash-bound preparation extension around the frozen v5 agent loop.

No sampling/tokenization/renderer/loss changes. The old loader validates every
sealed request; repeated exact typed image references need only one proof in
that invocation. Actual training pixels are still loaded by the OLD builder.
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time

from append_agent_rl.generator import AppendGenerator
from opencode_rl.capacity import CoalescedCommit, durable_io
from opencode_rl.contract import ParityError, digest
from opencode_rl.grading_client import GradeRetriesExhausted
from opencode_rl.store import TraceStore, atomic_json
from opencode_rl.training import prepare_trajectory
from append_resume_delivery import DELIVERY_POLICY, rollout_task as grade_first_rollout

POLICY = 'typed-proof-scored-first-v1'


class _ValidationView:
    """Private validation-only view; NEVER pass this to a tensor consumer.

The pinned old turn_from_events ignores load_tensor's return value. Only its
successful validation matters here. Nothing retaining this view escapes
prepare_deduplicated, and no mutable tensor or cross-attempt proof is cached.
"""
    def __init__(self, store):
        self._store, self.root, self._seen = store, store.root, set()

    def attempt_requests(self, identity):
        return self._store.attempt_requests(identity)

    def read_request(self, identity):
        return self._store.read_request(identity)

    def load_tensor(self, ref):
        key = digest(ref)  # Includes shape, dtype, storage, byte count and hash.
        if key not in self._seen:
            self._store.load_tensor(ref)
            self._seen.add(key)


def prepare_deduplicated(store, attempt_id, contract):
    return prepare_trajectory(_ValidationView(store), attempt_id, contract)


class PreparedAppendGenerator(AppendGenerator):
    """Same pinned controller/output builder; explicit parent delivery extension."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # One shared owner for both _one (training) and evaluation_one.
        # The extension is hashed in the resume adapter, not a global monkey patch.
        self.rollout = grade_first_rollout

    @asynccontextmanager
    async def _completed_rollout(self, task, repetition, group, dataset):
        row = await asyncio.to_thread(dataset.verify_task, task)
        directory = group/f'r{repetition:03d}'
        started = time.monotonic()
        engine = self.engines[self.next_engine % len(self.engines)]
        self.next_engine += 1
        store = await durable_io(TraceStore, engine['trace_root'], read_cache_bytes=self.read_cache_bytes)
        grade_job = preparation_job = attempt_dir = None
        timing = {'policy': POLICY}
        try:
            async with self.gate:
                for retry in range(self.retries):
                    attempt_dir = directory/f'attempt_{retry:02d}'

                    async def delivery_ready(status):
                        nonlocal grade_job
                        if grade_job is not None or status.get('status') not in ('grading_ready', 'rollout_complete'):
                            raise ParityError('Duplicate or premature delivery callback')
                        grade_job = asyncio.create_task(self.grader(task, row, attempt_dir))

                    status = await self.rollout(app=self.app, image_id=self.image_id,
                        bundle=dataset.bundles/task, destination=attempt_dir,
                        config=self.config, contract=self.contract, backend_url=engine['url'],
                        commit=self.commit, request_gate=engine.get('request_gate'), on_delivery=delivery_ready)
                    if status.get('status') == 'rollout_complete':
                        break
                    if grade_job is not None:
                        if not grade_job.done(): grade_job.cancel()
                        outcome = (await asyncio.gather(grade_job, return_exceptions=True))[0]
                        grade_job = None
                        if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                            raise outcome
                else:
                    raise ParityError(f'No complete native rollout for {task}/{repetition}; attempts retained')
            timing['rollout_returned_at'] = time.time()
            if grade_job is None:
                grade_job = asyncio.create_task(self.grader(task, row, attempt_dir))
            # Neither the rollout slots nor the eight materialization slots
            # are held while waiting for the early, already-running grader.
            score = await grade_job
            timing['grade_returned_at'] = time.time()
            timing['preparation_queued_at'] = time.time()
            async with self.preparation_gate:
                timing['preparation_started_at'] = time.time()
                preparation_job = asyncio.create_task(durable_io(prepare_deduplicated,
                    store, status['attempt_id'], self.contract))
                prepared = await preparation_job
                timing['preparation_and_grade_ready_at'] = time.time()
                status['preparation_timing'] = timing
                # The frozen _one/evaluation_one consumers and output builder
                # receive the REAL store, never the private validation view.
                timing['build_output_started_at'] = time.time()
                yield row, directory, started, engine, store, status, prepared, score
                timing['build_output_completed_at'] = time.time()
        finally:
            jobs = [job for job in (preparation_job, grade_job) if job is not None]
            for job in jobs:
                if not job.done(): job.cancel()
            outcomes = await asyncio.gather(*jobs, return_exceptions=True)
            store.clear_read_cache()
            if attempt_dir is not None and timing.get('preparation_queued_at') is not None:
                timing['preparation_owner_finished_at'] = time.time()
                await durable_io(atomic_json, attempt_dir/'preparation_timing.json', timing)
            fatal = next((x for x in outcomes if isinstance(x, BaseException)
                and not isinstance(x, (asyncio.CancelledError, GradeRetriesExhausted))), None)
            if fatal is not None:
                raise fatal


def make_generator(experiment, cfg):
    """Frozen get_generator wiring, with this named preparation extension only."""
    import modal
    from opencode_rl.grading_client import OutcomeGrader, modal_caller
    from opencode_rl.train_entrypoint import grading_payload
    from opencode_rl.train_capacity import training_capacity
    p = experiment.plan
    if p['resume'].get('preparation') != POLICY:
        raise ParityError('Explicit preparation extension identity required')
    if p['resume'].get('delivery') != DELIVERY_POLICY:
        raise ParityError('Explicit delivery extension identity required')
    boundary = p['resume'].get('grading_boundary')
    if boundary is not None:
        from append_refusal_boundary import POLICY as REFUSAL_POLICY, qualified_grader_type
        if boundary != REFUSAL_POLICY:
            raise ParityError('Unknown grading boundary extension')
        OutcomeGrader = qualified_grader_type()
    volume = modal.Volume.from_name(p['logs_volume'])
    commit = CoalescedCommit(volume.commit.aio)
    function = modal.Function.from_name(p['grader_app'], p['grader_function'])
    capacity = training_capacity()
    if p.get('capacity') != capacity:
        raise ParityError('Launcher and training capacity differ')
    outcome = OutcomeGrader(modal_caller(function), grader_hashes=p['grader_hashes'],
        concurrency=capacity['judge_concurrency'], review_policy=p.get('grading_review_policy', 'reviewed'),
        submission_validation=p.get('submission_validation'))

    async def grade(task, row, attempt):
        payload = await durable_io(grading_payload, task, row, attempt)
        return await outcome.grade(payload, attempt/'grading', row['source_hashes'], commit)

    return PreparedAppendGenerator(dataset=experiment.train_dataset, contract=experiment.contract,
        config=p['config'], app=modal.App.lookup(p['sandbox_app'], create_if_missing=True),
        image_id=p['sandbox_image'],
        engines=[{**e, 'request_gate': asyncio.Semaphore(capacity['streams_per_engine'])} for e in p['engines']],
        root=p['run_root'], grader=grade, commit=commit, k=cfg.generator.n_samples_per_prompt,
        concurrency=p['rollout_concurrency'], preparation_concurrency=capacity['preparation_concurrency'],
        read_cache_bytes=capacity['read_cache_bytes'])
