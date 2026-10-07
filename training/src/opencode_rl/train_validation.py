"""Validation at the actual FullyAsync post-sync/pre-next-optimizer boundary.

Uses the generator's identical native OpenCode rollout + terminal grader. Only
small score/evidence records are aggregated; never materialize all validation
image tensors as a training batch. The fixed validation split is not sampled by
the training dataloader. No extra GPU pool or policy weight update is launched.
"""
import asyncio
import time
import uuid
from pathlib import Path

from .contract import ParityError, canonical
from .store import atomic_bytes
from .grading_client import GradeRetriesExhausted


async def policy_snapshot(client):
    replies = await client._call_all_servers('/ocrl/status', method='GET')
    if len(replies) != 4:
        raise ParityError('Validation requires all four inference engines')
    states = [r['body'] for r in replies.values()]
    if any(r['status'] != 200 for r in replies.values()) or any(s['phase'] != 'open' for s in states):
        raise ParityError('Validation cannot start across an incomplete weight transaction')
    versions = {s['version'] for s in states}
    if len(versions) != 1 or not next(iter(versions)):
        raise ParityError('Four-engine validation weight version mismatch')
    return next(iter(versions))


async def evaluate(generator, dataset, client, step):
    if dataset is None or dataset.split != 'val' or len(dataset) != 81:
        raise ParityError('Validation must cover the complete frozen 81-task split')
    if set(dataset.by_task) & set(generator.dataset.by_task):
        raise ParityError('Train/validation overlap')
    root = generator.root / 'validation' / f'step_{step:06d}' / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    started, jobs = time.monotonic(), []
    try:
        version = await policy_snapshot(client)
        identity = {'step': step, 'weight_version': version, 'tasks': len(dataset),
            'prepared_sha256': dataset.sha256, 'contract_sha256': generator.contract.sha256}
        atomic_bytes(root / 'started.json', canonical(identity))
        await generator.commit()
        # Uses the SAME global rollout semaphore and per-engine admission as
        # training, so eval cannot multiply the sandbox/sequence token budget.
        jobs = [asyncio.create_task(generator.evaluation_one(row['task'], root / row['task'], dataset, version))
                for row in dataset.rows]
        outcomes = await asyncio.gather(*jobs, return_exceptions=True)
        results, excluded = [], []
        for row, result in zip(dataset.rows, outcomes):
            if isinstance(result, GradeRetriesExhausted):
                failed = {'task': row['task'], 'stage': row['stage'],
                          'type': type(result).__name__, 'reason': str(result), 'score': None}
                excluded.append(failed)
                atomic_bytes(root / row['task'] / 'grading_excluded.json', canonical(failed))
            elif isinstance(result, BaseException):
                raise result
            else:
                results.append(result)
        if await policy_snapshot(client) != version:
            raise ParityError('Policy changed during validation')
        if (len(results) + len(excluded) != len(dataset) or
                {r['task'] for r in results + excluded} != set(dataset.by_task)):
            raise ParityError('Incomplete or duplicated validation results')
        # Never publish an incomplete subset as the comparable 81-task score.
        # API failures may not turn into policy zeros or stop the next step.
        prefix = 'eval_observed' if excluded else 'eval'
        metrics = {'eval/complete': int(not excluded), 'eval/grading_excluded': len(excluded),
                   'eval/all/tasks': len(results), 'timing/validation': time.monotonic() - started}
        if results:
            metrics[f'{prefix}/all/avg_score'] = sum(r['score'] for r in results) / len(results)
        for stage in sorted({r['stage'] for r in results}):
            subset = [r for r in results if r['stage'] == stage]
            metrics[f'{prefix}/{stage}/avg_score'] = sum(r['score'] for r in subset) / len(subset)
            metrics[f'eval/{stage}/tasks'] = len(subset)
        atomic_bytes(root / ('partial.json' if excluded else 'complete.json'),
                     canonical({**identity, 'metrics': metrics, 'results': results, 'excluded': excluded}))
        await generator.commit()
        return metrics
    except BaseException as exc:
        atomic_bytes(root / 'failure.json', canonical({'type': type(exc).__name__, 'message': str(exc)[:1500]}))
        raise
    finally:
        for job in jobs:
            if not job.done():
                job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        await generator.commit()


class ValidationTrainerMixin:
    async def eval(self, vllm_metrics_scraper=None):
        # FullyAsync normally emits all_metrics only AFTER eval. A terminal
        # judge failure must not leave an already-finished optimizer/sync step
        # visible only in stdout. Persist without advancing W&B's step cursor
        # or pretending the checkpoint/validation has already completed.
        if self.global_step > 0:
            await asyncio.to_thread(self._ocrl_persistence.snapshot_before_validation,
                int(self.global_step), self.all_metrics, self.all_timings)
        # Upstream saves after eval. Persist the just-trained policy first so
        # a slow/offline judge cannot leave step 10/20/... without its checkpoint.
        # Skip only the immediately following same-step periodic duplicate;
        # the final native safety save is still allowed to save sampler state.
        if self.global_step > 0 and self.global_step % self.cfg.trainer.ckpt_interval == 0:
            path = await asyncio.to_thread(super().save_checkpoints)
            self._ocrl_val_checkpoint = (self.global_step, path)
        return await evaluate(self.generator, self.eval_dataset, self.inference_engine_client, self.global_step)

    def save_checkpoints(self):
        saved = getattr(self, '_ocrl_val_checkpoint', None)
        if saved and saved[0] == self.global_step:
            self._ocrl_val_checkpoint = None
            return saved[1]
        return super().save_checkpoints()
