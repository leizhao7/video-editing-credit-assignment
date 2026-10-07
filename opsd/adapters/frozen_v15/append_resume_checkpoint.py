"""Bounded publication of already-frozen rolling checkpoint files.

Native model/optimizer/dataloader/sampler saves and finalize_pending_saves stay
synchronous. Only the unchanged RunPersistence.save publication runs off-path.
Never publish a recovery receipt until the original volume commit succeeds.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import asyncio
from pathlib import Path
import threading
import time
import uuid

from opencode_rl.contract import ParityError
from opencode_rl.store import atomic_json
from opencode_rl.train_persistence import RunPersistence, validate_checkpoint_layout

PUBLICATION_POLICY = 'bounded-rolling-checkpoint-publication-v1'


@dataclass(frozen=True)
class SavedStep:
    global_step: int


class BackgroundCheckpointPersistence(RunPersistence):
    def __init__(self, plan, logs, checkpoints):
        if plan.get('resume', {}).get('checkpoint_publication') != PUBLICATION_POLICY:
            raise ParityError('Explicit checkpoint publication identity required')
        super().__init__(plan, logs, checkpoints)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint-publication')
        self._pending = None
        self._lock = threading.RLock()
        self._closed = False

    def check_publication(self):
        """Surface completed I/O failures at existing trainer logging boundaries."""
        with self._lock:
            pending = self._pending
        if pending is not None and pending.done():
            pending.result()  # Never turn failed persistence into a success.

    def wait_for_publication(self):
        with self._lock:
            pending = self._pending
        if pending is not None:
            pending.result()
            with self._lock:
                if self._pending is pending:
                    self._pending = None

    def _publish(self, saved, path):
        started = time.monotonic()
        record = {'step': saved.global_step, 'checkpoint': str(path),
            'policy': PUBLICATION_POLICY, 'started_at': time.time(), 'state': 'publishing'}
        event = self.root/'checkpoint_publication'/f'{saved.global_step:06d}-{uuid.uuid4().hex}.json'
        atomic_json(event, record)
        try:
            # Original implementation owns permanent copies, volume publication,
            # source identity, and ordering of the durable recovery receipt.
            super().save(saved, path)
        except BaseException as exc:
            record.update(state='failed', finished_at=time.time(),
                seconds=time.monotonic()-started, error_type=type(exc).__name__, error=str(exc)[:1500])
            try:
                atomic_json(event, record)
            except Exception as diagnostic_error:
                exc.add_note('Publication diagnostic also failed: '+str(diagnostic_error)[:300])
            raise
        record.update(state='published', finished_at=time.time(), seconds=time.monotonic()-started)
        atomic_json(event, record)
        # Existing tracker/periodic commits retain this telemetry. close() also
        # flushes it. Do not log to W&B from a thread with an already-advanced step.

    def save(self, trainer, path):
        self.wait_for_publication()
        # Capture only immutable step/path. The live trainer is never retained
        # by the publication job and may advance to another optimizer step.
        saved, path = SavedStep(int(trainer.global_step)), Path(path)
        expected = Path(self.plan['checkpoint_root'])/'rolling'/f'global_step_{saved.global_step}'
        if path.resolve() != expected.resolve() or not 1 <= saved.global_step <= self.plan['requested_steps']:
            raise ParityError('Checkpoint escaped source run or completed-step horizon')
        validate_checkpoint_layout(path)
        with self._lock:
            if self._closed:
                raise ParityError('Checkpoint publisher is closed')
            if self._pending is not None:
                raise ParityError('Concurrent checkpoint producers are not allowed')
            # Decades remain fully durable BEFORE validation. The final safety
            # save and odd end-of-epoch saves remain synchronous as well.
            synchronous = (saved.global_step % 10 == 0 or saved.global_step % 2 != 0
                           or saved.global_step == self.plan['requested_steps'])
            if not synchronous:
                self._pending = self._executor.submit(self._publish, saved, path)
                return
            self._publish(saved, path)

    def log(self, data, step, commit=False):
        self.check_publication()
        super().log(data, step, commit)
        self.check_publication()

    def snapshot_before_validation(self, step, metrics, timings):
        self.check_publication()
        return super().snapshot_before_validation(step, metrics, timings)

    def close(self):
        with self._lock:
            self._closed = True
        failure = None
        try:
            self.wait_for_publication()
        except BaseException as exc:
            failure = exc
        finally:
            # Do not cancel an accepted writer; its exact snapshot must finish
            # even after the training coroutine fails or is cancelled.
            self._executor.shutdown(wait=True, cancel_futures=False)
        try:
            self.logs.commit()
        except BaseException as exc:
            if failure is None:
                raise
            failure.add_note('Final publication log commit also failed: '+str(exc)[:300])
        if failure is not None:
            raise failure


class CheckpointPublicationMixin:
    """Place AFTER ValidationTrainerMixin and BEFORE PersistedAsyncTrainerMixin."""
    def save_checkpoints(self):
        # Must precede the NATIVE save, which also performs rolling cleanup.
        # Waiting only in RunPersistence.save would be too late for deletion.
        self._ocrl_persistence.wait_for_publication()
        return super().save_checkpoints()

    async def train(self):
        try:
            return await super().train()
        finally:
            # Repeated cancellation must not abandon an accepted publisher.
            # A platform hard preemption is different: recovery then uses the
            # last independently committed receipt, never this pending job.
            closing = asyncio.create_task(asyncio.to_thread(self._ocrl_persistence.close))
            cancelled = None
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError as exc:
                    cancelled = exc
            closing.result()
            if cancelled is not None:
                raise cancelled


def run_background_persistence(plan):
    import modal
    return BackgroundCheckpointPersistence(plan, modal.Volume.from_name(plan['logs_volume']),
        modal.Volume.from_name(plan['checkpoint_volume']))
