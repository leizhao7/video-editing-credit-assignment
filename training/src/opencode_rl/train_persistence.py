"""Step metrics, full provenance, immutable decade checkpoints; fail on I/O errors."""
import json
import math
import numbers
from pathlib import Path
import shutil
import uuid

from .contract import canonical, digest, ParityError
from .store import atomic_bytes
from .checkpoint_digest import checkpoint_sha_file


def metric_json(value):
    """Retain scalar diagnostics, including nonfinite values, as valid JSON."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else {'nonfinite': str(value)}
    if isinstance(value, dict):
        return {str(k): metric_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [metric_json(v) for v in value]
    if hasattr(value, 'item'):
        return metric_json(value.item())
    raise TypeError(f'Unsupported persisted metric type: {type(value).__name__}')


def checkpoint_manifest(path):
    validate_checkpoint_layout(path)
    path = Path(path)

    files = [p for p in path.rglob('*') if p.is_file()]
    return {str(p.relative_to(path)): {'bytes': p.stat().st_size, 'sha256': checkpoint_sha_file(p)} for p in files}


def validate_checkpoint_layout(path):
    path = Path(path)
    required = ('data.pt', 'trainer_state.pt', 'fully_async_state.pt')
    if (not all((path / n).is_file() and (path / n).stat().st_size for n in required)
            or not any(p.is_file() and p.stat().st_size for p in (path / 'policy').rglob('*'))):
        raise ParityError('Incomplete checkpoint: model/optimizer/sampler state must all be saved')
    if any(p.is_symlink() for p in path.rglob('*')):
        raise ParityError('Checkpoint contains unexpected links')


def protect_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    # Copy, never hardlink: a final save may overwrite the periodic checkpoint
    # at the same step. A permanent copy must not change under that writer.
    manifest = checkpoint_manifest(source)
    marker = destination.parent / (destination.name + '.receipt.json')
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_bytes())['files'] != manifest:
            raise ParityError('Refusing to overwrite a permanent checkpoint')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial-' + uuid.uuid4().hex)
    shutil.copytree(source, temporary)
    if checkpoint_manifest(temporary) != manifest:
        raise ParityError('Permanent checkpoint copy failed integrity check; partial retained')
    temporary.rename(destination)
    atomic_bytes(marker, canonical({'files': manifest, 'sha256': digest(manifest)}))


class RunPersistence:
    def __init__(self, plan, logs, checkpoints):
        self.plan, self.logs, self.checkpoints = plan, logs, checkpoints
        self.root = Path(plan['run_root'])

    def log(self, data, step, commit=False):
        atomic_bytes(self.root / 'metrics' / f'{step:06d}-{uuid.uuid4().hex}.json',
            canonical({'step': step, 'commit': commit, 'metrics': metric_json(data),
                       'contract_sha256': digest(self.plan['contract'])}))
        self.logs.commit()

    def snapshot_before_validation(self, step, metrics, timings):
        """Keep completed training evidence even if validation never returns.

        This is not a completed-step/checkpoint receipt or a synthetic W&B log.
        Native tracking still owns the final combined metrics and timing commit.
        """
        if type(step) is not int or step < 1:
            raise ParityError('Pre-validation snapshot needs a completed training step')
        record = {'step': step, 'phase': 'post_optimizer_sync_pre_validation',
            'metrics': metric_json(metrics), 'timings_so_far': metric_json(timings),
            'validation_complete': False, 'checkpoint_committed': False,
            'contract_sha256': digest(self.plan['contract'])}
        path = self.root / 'training_steps' / f'{step:06d}.json'
        body = canonical(record)
        if path.exists():
            if path.read_bytes() != body:
                raise ParityError('Refusing to overwrite different pre-validation training evidence')
        else:
            atomic_bytes(path, body)
        self.logs.commit()

    def save(self, trainer, path):
        plan, step, path = self.plan, int(trainer.global_step), Path(path)
        expected = Path(plan['checkpoint_root']) / 'rolling' / f'global_step_{step}'
        if path.resolve() != expected.resolve() or step < 1:
            raise ParityError('Checkpoint escaped this run or has no completed optimizer step')
        # Called only AFTER the fully-async sampler state and pending model /
        # optimizer writes have finished. Never publish a half checkpoint.
        validate_checkpoint_layout(path)
        receipt = {'step': step, 'checkpoint': str(path),
            'contract_sha256': digest(plan['contract']), 'prepared_sha256': plan['prepared_sha256'],
            'grader_hashes': plan['grader_hashes'], 'mode': plan['mode']}
        atomic_bytes(path / 'ocrl_resume.json', canonical(receipt))
        if step % 10 == 0:
            permanent = Path(plan['checkpoint_root']) / 'permanent' / path.name
            if permanent.exists():
                # The final safety save can repeat the periodic step. Preserve
                # the first copy even if torch serialization changes its bytes.
                saved = json.loads((permanent / 'ocrl_resume.json').read_bytes())
                marker = permanent.parent / (permanent.name + '.receipt.json')
                if saved != receipt or checkpoint_manifest(permanent) != json.loads(marker.read_bytes())['files']:
                    raise ParityError('Permanent checkpoint identity/integrity mismatch')
            else:
                protect_checkpoint(path, permanent)
        self.checkpoints.commit()
        atomic_bytes(self.root / 'checkpoints' / f'{step:06d}.json', canonical(receipt))
        self.logs.commit()


class DurableTracking:
    """Wrap the actual tracker.log calls; FullyAsync rejects TrainingCallback."""
    def __init__(self, tracker, persistence):
        self._tracker, self._persistence = tracker, persistence

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    def log(self, data, step, commit=False):
        self._persistence.log(data, step, commit=commit)
        return self._tracker.log(data, step=step, commit=commit)


class PersistedAsyncTrainerMixin:
    def __init__(self, *args, persistence, **kwargs):
        if kwargs.get('callbacks'):
            raise ParityError('FullyAsync persistence must not use unsupported callbacks')
        self._ocrl_persistence = persistence
        kwargs['tracker'] = DurableTracking(kwargs['tracker'], persistence)
        super().__init__(*args, **kwargs)

    def save_checkpoints(self):
        path = super().save_checkpoints()
        self.dispatch.finalize_pending_saves('policy')
        if self.has_critic:
            self.dispatch.finalize_pending_saves('critic')
        self._ocrl_persistence.save(self, path)
        return path


def run_persistence(plan):
    import modal
    return RunPersistence(plan, modal.Volume.from_name(plan['logs_volume']),
                          modal.Volume.from_name(plan['checkpoint_volume']))
