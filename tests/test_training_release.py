"""Private release preparation, real public-bundle staging and launch boundaries."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'training/src'), str(ROOT / 'training/environment')]
from prepare import create_release, prepare_inputs
from cloud import validate_release
from configuration import load_config, validate_config
from launcher import bounded_run, base_command
from reconstruct import apply_source_patch
from opencode_rl.contract import ParityError
from opencode_rl.task_bundle import release_tasks
from opencode_rl.readiness import REQUIRED, TRAINING_REQUIRED, require_training_ready


class TaskReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = []
        for film in ('train-film', 'held-out-film'):
            folder = self.root / 'reviewed' / film / 'selection-01'
            (folder / 'initial_state').mkdir(parents=True)
            (folder / 'verifier').mkdir()
            (folder / 'task.json').write_text(json.dumps({'stage': 's1', 'taskId': film}))
            (folder / 'task_prompt.md').write_text('Choose a shot.')
            (folder / 'probe.json').write_text('{}')
            (folder / 'initial_state/manifest.json').write_text(json.dumps({
                'taskId': film, 'candidates': [{'id': 'c1', 'file': 'clip.mp4', 'dur_s': 1.0}]}))
            (folder / 'verifier/answer.json').write_text('{"private_answer": "c1"}')
            (folder / 'verifier/grade.py').write_text('# trusted verifier code\n')
            self.tasks.append(folder)
        self.index = self.root / 'accepted_tasks.json'
        self.index.write_text(json.dumps([str(p) for p in self.tasks]))
        self.output = self.root / 'private-release'

    def release(self):
        return create_release(self.index, self.output, validation_films=['held-out-film'])

    def test_film_holdout_and_sha_binding(self):
        receipt = self.release()
        path = self.output / 'release.json'
        train = release_tasks(path, 'train', expected_sha256=receipt['release_sha256'])
        val = release_tasks(path, 'val', expected_sha256=receipt['release_sha256'])
        self.assertEqual([r['probe']['film'] for r in train], ['train-film'])
        self.assertEqual([r['probe']['film'] for r in val], ['held-out-film'])
        with self.assertRaises(ParityError):
            release_tasks(path, 'train') # Historic default must reject a new collection.
        path.write_text(path.read_text() + ' ')
        with self.assertRaises(ParityError):
            release_tasks(path, 'train', expected_sha256=receipt['release_sha256'])

    def test_rejects_symlinks_and_existing_destination(self):
        (self.tasks[0] / 'initial_state/leak').symlink_to(self.tasks[0] / 'verifier/answer.json')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.release()
        self.assertFalse(self.output.exists())
        (self.tasks[0] / 'initial_state/leak').unlink()
        self.release()
        with self.assertRaisesRegex(ValueError, 'immutable'):
            self.release()

    def test_rejects_unknown_holdout_and_no_training_tasks(self):
        with self.assertRaisesRegex(ValueError, 'do not occur'):
            create_release(self.index, self.output, validation_films=['unknown-film'])
        with self.assertRaisesRegex(ValueError, 'training task'):
            create_release(self.index, self.output, validation_films=['train-film', 'held-out-film'])
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe') and
                         importlib.util.find_spec('aiohttp'), 'FFmpeg/FFprobe and aiohttp required')
    def test_real_preparation_separates_private_material_and_detects_drift(self):
        for folder in self.tasks:
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:r=5',
                            '-t', '1', '-c:v', 'libx264', str(folder / 'initial_state/clip.mp4')], check=True)
        self.release()
        result = asyncio.run(prepare_inputs(self.output))
        self.assertEqual(result['prepared']['train']['task_count'], 1)
        self.assertEqual(result['prepared']['val']['task_count'], 1)
        _, validated = validate_release(self.output)
        self.assertEqual(validated['release_sha256'], result['release_sha256'])
        for archive in (self.output / 'prepared').rglob('input.tar'):
            with tarfile.open(archive) as stream:
                names = stream.getnames()
                self.assertTrue(any(n.startswith('frames/') for n in names))
                self.assertFalse(any('verifier' in n or 'answer' in n or n == 'probe.json' for n in names))
        # Prepared inputs are immutable and resumable; a byte change is fatal.
        again = asyncio.run(prepare_inputs(self.output))
        self.assertEqual(result, again)
        next((self.output / 'prepared').rglob('task.txt')).write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'changed'):
            asyncio.run(prepare_inputs(self.output))


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'training/cloud.example.toml')

    def test_configuration_and_bounded_run_reject_ambiguous_inputs(self):
        for name, mode, steps in (('../old', 'smoke', 2), ('new', 'smoke', 3), ('new', 'train', True),
                                  ('new', 'unknown', 2), ('new', 'train', 51)):
            with self.subTest(name=name, mode=mode, steps=steps), self.assertRaises(ValueError):
                bounded_run(name, mode, steps)
        for key, value in (('model_volume', self.config['release_volume']), ('residency_limits', [1, 2]),
                            ('profile', 'guess'), ('unexpected', 'value')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**self.config, key: value})

    def test_command_preserves_math_and_has_no_historic_paths(self):
        plan = {'model_path': '/models/test', 'run_root': '/logs/runs/new', 'mode': 'smoke'}
        values = dict(x.split('=', 1) for x in base_command(plan)[3:])
        self.assertEqual(values['trainer.algorithm.loss_reduction'], 'token_mean')
        self.assertEqual(values['trainer.algorithm.advantage_batch_normalize'], 'false')
        self.assertEqual(values['trainer.train_batch_size'], '2')
        self.assertEqual(values['generator.n_samples_per_prompt'], '2')
        self.assertFalse(any('resume_path' in k or 'v15' in v for k, v in values.items()))
        plan['opsd'] = {}
        values = dict(x.split('=', 1) for x in base_command(plan)[3:])
        self.assertEqual(values['trainer.train_batch_size'], '16')

    def test_new_release_readiness_cannot_reuse_historical_evidence(self):
        contract = SimpleNamespace(sha256='a' * 64)
        report = {'contract_sha256': contract.sha256,
                  'checks': {k: True for k in (*REQUIRED, *TRAINING_REQUIRED)},
                  'token_mismatches': 0, 'image_mismatches': 0, 'evidence_paths': ['private/evidence']}
        require_training_ready(report, contract)
        with self.assertRaises(ParityError):
            require_training_ready(report, contract, release_sha256='b' * 64)
        report.update(release_sha256='b' * 64)
        report['checks']['dataset_release'] = True
        require_training_ready(report, contract, release_sha256='b' * 64)
        report['checks']['actual_optimizer_step'] = False
        with self.assertRaises(ParityError):
            require_training_ready(report, contract, release_sha256='b' * 64)


class SourcePatchTests(unittest.TestCase):
    def test_patch_accepts_only_complete_preimage_or_postimage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.py').write_text('before\n')
            patch = root / 'change.patch'
            patch.write_text('diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-before\n+after\n')
            digest = lambda raw: hashlib.sha256(raw).hexdigest()
            entries = {'a.py': {'before_sha256': digest(b'before\n'), 'after_sha256': digest(b'after\n')}}
            identity = digest(patch.read_bytes())
            self.assertTrue(apply_source_patch(root, patch, entries, identity)['changed'])
            self.assertFalse(apply_source_patch(root, patch, entries, identity)['changed'])
            (root / 'a.py').write_text('unreviewed\n')
            with self.assertRaisesRegex(ValueError, 'neither'):
                apply_source_patch(root, patch, entries, identity)


if __name__ == '__main__':
    unittest.main()
