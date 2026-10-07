"""Real OpenCode -> immutable outcome -> native recorded turns -> SkyRL.

This is a GeneratorInterface-compatible async implementation, not a second agent
loop. The exact task_rollout.rollout_task used in validation owns every sandbox.
No policy text is re-tokenized here. Entire GRPO sibling groups succeed together;
judge outages cannot silently shrink a group or invent a negative policy reward.
"""
import asyncio
import json
import time
from pathlib import Path
import uuid
from contextlib import asynccontextmanager

from .contract import ParityError, canonical, digest
from .records import load_turn
from .store import TraceStore, atomic_json
from .training import AttemptResult, build_generator_output, original_request_rows, prepare_trajectory
from .task_rollout import rollout_task
from .context_recovery import primary_inventory
from .capacity import durable_io
from .grading_client import GradeRetriesExhausted
from .rollout_residency import ResidencyPool


def primary_requests(store, attempt_id, contract):
    rows, _ = primary_inventory(store, attempt_id, contract)
    for row in rows:
        load_turn(store, row['request_id'], contract)
    return rows


class OpenCodeGenerator:
    def __init__(self, *, dataset, contract, config, app, image_id, engines, root,
                 grader, commit, k=8, concurrency=128, rollout_retries=2,
                 rollout=rollout_task, concatenate=None, build_output=build_generator_output,
                 preparation_concurrency=8, read_cache_bytes=64 * 1024 * 1024,
                 prepare=None, preparation_policy='overlap', process_judge=None, residency_limits=None):
        if digest(config) != contract.config_sha256 or not engines or k < 2:
            raise ParityError('Invalid generator runtime/GRPO group')
        self.process_judge = process_judge
        self.dataset, self.contract, self.config = dataset, contract, config
        self.app, self.image_id, self.engines = app, image_id, engines
        self.root, self.grader, self.commit = Path(root), grader, commit
        self.k, self.retries = k, rollout_retries
        if not isinstance(residency_limits, (list, tuple)) or len(residency_limits) != len(engines):
            raise ParityError('Explicit per-engine whole-rollout residency limits required')
        self.residency = ResidencyPool(residency_limits)
        self.gate = asyncio.Semaphore(concurrency)
        self.rollout, self.concatenate, self.build_output = rollout, concatenate, build_output
        if type(preparation_concurrency) is not int or preparation_concurrency < 1:
            raise ValueError('Positive preparation concurrency required')
        self.preparation_gate = asyncio.Semaphore(preparation_concurrency)
        self.read_cache_bytes = read_cache_bytes
        if preparation_policy not in ('overlap', 'scored_first'):
            raise ValueError('Unknown trajectory preparation scheduling policy')
        self.prepare = prepare if prepare is not None else prepare_trajectory
        self.preparation_policy = preparation_policy

    @asynccontextmanager
    async def _completed_rollout(self, task, repetition, group, dataset, *, process_enabled=False):
        """Shared train/validation rollout, evidence checks and terminal grader."""
        row = await asyncio.to_thread(dataset.verify_task, task)
        directory = group / f'r{repetition:03d}'
        started = time.monotonic()
        engine = store = None
        residency_timing = {}
        residency_attempts = []
        grade_job = preparation_job = process_job = None
        process_attempts = []
        try:
            async with self.gate:
                for retry in range(self.retries):
                    async with self.residency.lease() as lease:
                        engine = self.engines[lease.engine_index]
                        store = await durable_io(TraceStore, engine['trace_root'], read_cache_bytes=self.read_cache_bytes)
                        try:
                            attempt_dir = directory / f'attempt_{retry:02d}'

                            async def delivery_ready(status):
                                nonlocal grade_job
                                if grade_job is not None or status.get('status') not in ('grading_ready', 'rollout_complete'):
                                    raise ParityError('Duplicate or premature delivery callback')
                                grade_job = asyncio.create_task(self.grader(task, row, attempt_dir))

                            extra = {'process_judge': self.process_judge} if process_enabled and self.process_judge is not None else {}
                            status = await self.rollout(**extra, app=self.app, image_id=self.image_id,
                                bundle=dataset.bundles / task, destination=attempt_dir,
                                config=self.config, contract=self.contract, backend_url=engine['url'],
                                commit=self.commit, request_gate=engine.get('request_gate'),
                                on_delivery=delivery_ready)
                            if extra and status.get('attempt_id'):
                                process_attempts.append(status['attempt_id'])
                            if status.get('status') == 'rollout_complete':
                                break
                            # A score cannot rescue an unarchived/infrastructure-failed
                            # rollout. Keep the old attempt's grade/handle and drain it
                            # before retrying; never bind its score to the next attempt.
                            if grade_job is not None:
                                if not grade_job.done():
                                    grade_job.cancel()
                                outcome = (await asyncio.gather(grade_job, return_exceptions=True))[0]
                                grade_job = None
                                if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                                    raise outcome
                        finally:
                            residency_timing = {'engine_index': lease.engine_index,
                                'queue_seconds': max(0., lease.acquired_at-lease.queued_at),
                                'held_seconds': max(0., time.monotonic()-lease.acquired_at),
                                'per_engine_limit': self.residency.limits[lease.engine_index]}
                            residency_attempts.append({'attempt_index': retry, **residency_timing})
                    # Only a failed attempt reaches here. Its journal stays
                    # on the original engine; the next attempt gets a new
                    # lease/store. Never move an in-flight context.
                    store.clear_read_cache()
                    store = None
                else:
                    raise ParityError(f'No complete native rollout for {task}/{repetition}; failed attempts retained')
            # Real train/validation starts scoring as soon as delivery is
            # frozen, before archival. Fallback supports explicit test runners
            # without the callback; both paths launch at most one grade.
            if grade_job is None:
                grade_job = asyncio.create_task(self.grader(task, row, attempt_dir))
            if process_enabled and self.process_judge is not None:
                process_job = asyncio.create_task(self.process_judge.result(status['attempt_id']))
            timing = {'policy': self.preparation_policy, 'rollout_returned_at': time.time(),
                      'residency': residency_timing, 'residency_attempts': residency_attempts}
            if process_job is not None:
                # Outside rollout admission and before tensor materialization;
                # outcome was already launched at delivery, independently.
                _, process = await asyncio.gather(grade_job, process_job)
                status['process_reward'] = process
                timing['process_returned_at'] = time.time()
            if self.preparation_policy == 'scored_first':
                # Grading already began at delivery_ready. Do not let a slow
                # judge occupy one of the bounded materialization slots (and
                # hold a hot snapshot) while scored trajectories queue behind
                # it. A grader failure still raises, never invents reward zero.
                score = await grade_job
                timing['grade_returned_at'] = time.time()
            # Bound both materialization concurrency and hot validated snapshots
            # while a score is outstanding. No unbounded tensor prefetch queue.
            timing['preparation_queued_at'] = time.time()
            async with self.preparation_gate:
                timing['preparation_started_at'] = time.time()
                preparation_job = asyncio.create_task(durable_io(
                    self.prepare, store, status['attempt_id'], self.contract))
                prepared, score = await asyncio.gather(preparation_job, grade_job, return_exceptions=True)
                for value in (prepared, score):
                    if isinstance(value, BaseException):
                        raise value
                timing['preparation_and_grade_ready_at'] = time.time()
                status['preparation_timing'] = timing
                yield row, directory, started, engine, store, status, prepared, score
        finally:
            jobs = [job for job in (preparation_job, grade_job, process_job) if job is not None]
            for job in jobs:
                if not job.done():
                    job.cancel()
            outcomes = await asyncio.gather(*jobs, return_exceptions=True)
            for attempt_id in process_attempts:
                await self.process_judge.release(attempt_id)
            if store is not None:
                store.clear_read_cache()
            fatal = next((x for x in outcomes if isinstance(x, BaseException)
                          and not isinstance(x, (asyncio.CancelledError, GradeRetriesExhausted))), None)
            if fatal is not None:
                raise fatal

    async def _one(self, task, repetition, group, step):
        async with self._completed_rollout(task, repetition, group, self.dataset, process_enabled=True) as completed:
            row, directory, started, engine, store, status, prepared, score = completed
            requests = prepared.snapshot['recorded']
            process = status.get('process_reward')
            request_ids = tuple(r['request_id'] for r in requests)
            if self.process_judge is not None and (process is None or tuple(process['request_ids']) != request_ids):
                raise ParityError('PRM scores do not bind the exact accepted engine requests')
            attempt = AttemptResult(task_id=task, repetition_id=repetition,
                trajectory_id=f'{group.name}-r{repetition:03d}', attempt_id=status['attempt_id'],
                session_id=requests[0]['metadata']['session_id'], request_ids=tuple(r['request_id'] for r in requests),
                outcome_reward=score, process_scores=tuple(process['scores']) if process else (0.,) * len(requests),
                process_valid=tuple(process['valid']) if process else (False,) * len(requests),
                runtime_completed=True)
            status['preparation_timing']['build_output_started_at'] = time.time()
            output = await durable_io(self.build_output, store, [attempt], self.contract, prepared=prepared)
            status['preparation_timing']['build_output_completed_at'] = time.time()
        del prepared, completed
        for request_row in output['ocrl_rows']:
            request_row['stage'] = row['stage']
        for request_row in original_request_rows(output['ocrl_rows']):
            request_row['stage'] = row['stage']
        if process is not None:
            from prm_v2.live_metrics_v5 import summarize
            summary = summarize(process)
            for request_row in output['ocrl_rows']:
                request_row['process_judge_summary'] = summary
        # SkyRL's concatenator expects a timing for EVERY flattened row before
        # is_last_step filtering, not just one value per original trajectory.
        output['trajectory_generation_times'] = [time.monotonic() - started] * len(output['response_ids'])
        await durable_io(atomic_json, directory / 'admitted.json', {'step_when_scheduled': step,
            'task': task, 'repetition': repetition, 'attempt': attempt.__dict__,
            'contract_sha256': self.contract.sha256, 'engine': engine['url'],
            'preparation_timing': status['preparation_timing'],
            'process_reward': process,
            'requests': original_request_rows(output['ocrl_rows']),
            'training_packets': [r for r in output['ocrl_rows'] if 'causal_spans' in r]})
        await self.commit()
        return output

    async def evaluation_one(self, task, group, dataset, expected_version):
        # Evaluation never constructs optimizer tensors/advantages or enters
        # the training generation queue. The raw native journals remain intact.
        async with self._completed_rollout(task, 0, group, dataset) as completed:
            row, directory, started, engine, store, status, prepared, score = completed
            def evidence():
                snapshot = prepared.checked(store, status['attempt_id'], self.contract)
                result = []
                for turn in snapshot['turns']:
                    if turn['input']['weight_version'] != expected_version:
                        raise ParityError('Validation used mixed or stale policy weights')
                    result.append({'request_id': turn['request_id'],
                        'input_sha256': turn['input']['input_sha256'],
                        'weight_version': turn['input']['weight_version'],
                        'response_length': len(turn['response_ids'])})
                return result
            native_evidence = await durable_io(evidence)
        from math import isfinite
        if type(score) not in (float, int) or not isfinite(score) or not 0 <= score <= 1:
            raise ParityError('Invalid validation outcome; infrastructure cannot become zero')
        result = {'task': task, 'stage': row['stage'], 'score': score,
            'attempt_id': status['attempt_id'], 'contract_sha256': self.contract.sha256,
            'engine': engine['url'], 'requests': native_evidence,
            'seconds': time.monotonic() - started}
        await durable_io(atomic_json, directory / 'evaluation.json', result)
        await self.commit()
        return result

    async def generate(self, input_batch):
        ids = input_batch.get('trajectory_ids') or []
        extras = input_batch.get('env_extras') or []
        batch = input_batch.get('batch_metadata')
        if batch is None or batch.training_phase != 'train':
            raise ParityError('Use the shared validation runner for evaluation')
        if len(ids) != self.k or len(extras) != self.k or len({x.instance_id for x in ids}) != 1:
            raise ParityError('Expected one complete original-task GRPO group')
        task = extras[0]['task']
        if any(x['task'] != task or tid.instance_id != task for tid, x in zip(ids, extras)):
            raise ParityError('Dataset/task UID mismatch')
        if sorted(x.repetition_id for x in ids) != list(range(self.k)):
            raise ParityError('Missing/duplicated sibling repetition')
        group = self.root / 'groups' / uuid.uuid4().hex
        await durable_io(group.mkdir, parents=True, exist_ok=False)
        await durable_io(atomic_json, group / 'input.json', {'task': task, 'step_when_scheduled': batch.global_step,
            'contract_sha256': self.contract.sha256, 'k': self.k})
        jobs = [asyncio.create_task(self._one(task, tid.repetition_id, group, batch.global_step)) for tid in ids]
        committed = False
        discarded = None
        try:
            outputs = await asyncio.gather(*jobs)
            if self.concatenate is None:
                from skyrl.train.generators.utils import concatenate_generator_outputs
                self.concatenate = concatenate_generator_outputs
            result = await durable_io(self.concatenate, outputs, step_wise=True)
            await durable_io(atomic_json, group / 'complete.json', {'task': task, 'trajectories': self.k,
                'rows': len(result['response_ids']), 'step_when_scheduled': batch.global_step})
            await self.commit()
            committed = True
            return result
        except GradeRetriesExhausted as exc:
            reason_code = getattr(exc, 'reason_code', 'grading_unavailable')
            discarded = {'task': task, 'group': group.name, 'k': self.k,
                'step_when_scheduled': batch.global_step,
                'status': ('discarded_prohibited_content' if reason_code == 'PROHIBITED_CONTENT'
                           else 'discarded_grading_unavailable'), 'reason_code': reason_code,
                'type': type(exc).__name__, 'reason': str(exc), 'policy_reward': None,
                'contract_sha256': self.contract.sha256}
            raise
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            results = await asyncio.gather(*jobs, return_exceptions=True)
            if not committed:
                fatal = next((r for r in results if isinstance(r, BaseException)
                              and not isinstance(r, (GradeRetriesExhausted, asyncio.CancelledError))), None)
                if discarded is not None:
                    await durable_io(atomic_json, group / 'discarded.json', {**discarded,
                        'siblings_drained': True, 'fatal_sibling_error': str(fatal) if fatal else None})
                await self.commit()
                if fatal is not None:
                    raise fatal
