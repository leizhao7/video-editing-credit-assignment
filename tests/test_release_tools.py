"""Dependency-free regressions for source assembly and release validation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'training/scripts')]
import check_release
import materialize
import source_integrity
import verify_skyrl_patch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(root, name, data):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


class TemporaryTree(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)


class MaterializeTests(TemporaryTree):
    def setUp(self):
        super().setUp()
        self.source = self.root / 'source'
        self.output = self.root / 'runtime'
        self.rows = []
        patcher = patch.object(materialize, 'ROOT', self.source)
        patcher.start()
        self.addCleanup(patcher.stop)

    def add(self, name, content):
        put(self.source, name, content)
        self.rows.append({'path': name, 'public_sha256': digest(content)})

    def manifest(self):
        put(self.source, 'provenance.json', json.dumps({'files': self.rows}).encode())

    def test_decay_override_is_independent_of_manifest_order(self):
        self.add('overlays/decay/module.py', b'decay\n')
        self.add('src/module.py', b'fixed\n')
        self.add('src/nested/run.py', b'run\n')
        (self.source / 'src/nested/run.py').chmod(0o755)
        self.manifest()
        materialize.materialize('decay', self.output)
        self.assertEqual((self.output / 'module.py').read_bytes(), b'decay\n')
        self.assertEqual((self.output / 'nested/run.py').stat().st_mode & 0o777, 0o755)
        fixed = self.root / 'fixed'
        materialize.materialize('fixed', fixed)
        self.assertEqual((fixed / 'module.py').read_bytes(), b'fixed\n')

    def test_hash_failure_does_not_create_destination(self):
        self.add('src/module.py', b'original')
        self.manifest()
        (self.source / 'src/module.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            materialize.materialize('fixed', self.output)
        self.assertFalse(self.output.exists())

    def test_copies_the_verified_bytes_if_source_changes(self):
        self.add('src/module.py', b'verified')
        self.manifest()
        mkdir = Path.mkdir

        def change_on_create(path, *args, **kwargs):
            if path == self.output:
                (self.source / 'src/module.py').write_bytes(b'changed after verification')
            return mkdir(path, *args, **kwargs)

        with patch.object(Path, 'mkdir', change_on_create):
            materialize.materialize('fixed', self.output)
        self.assertEqual((self.output / 'module.py').read_bytes(), b'verified')

    def test_failed_write_removes_only_its_new_destination(self):
        self.add('src/module.py', b'content')
        self.manifest()
        with patch.object(Path, 'write_bytes', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                materialize.materialize('fixed', self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual((self.source / 'src/module.py').read_bytes(), b'content')

    def test_existing_destination_is_preserved(self):
        put(self.output, 'keep.py', b'keep')
        with self.assertRaises(FileExistsError):
            materialize.materialize('fixed', self.output)
        self.assertEqual((self.output / 'keep.py').read_bytes(), b'keep')

    def test_dangling_destination_symlink_is_preserved(self):
        self.output.symlink_to(self.root / 'absent')
        with self.assertRaises(FileExistsError):
            materialize.materialize('fixed', self.output)
        self.assertTrue(self.output.is_symlink())

    def test_symlinked_source_directory_is_rejected(self):
        put(self.root, 'external/module.py', b'content')
        self.source.mkdir()
        (self.source / 'src').symlink_to(self.root / 'external', target_is_directory=True)
        self.rows = [{'path': 'src/module.py', 'public_sha256': digest(b'content')}]
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'symlink'):
            materialize.materialize('fixed', self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_or_missing_layers_fail_before_writing(self):
        self.add('src/module.py', b'content')
        self.rows.append(self.rows[0].copy())
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            materialize.materialize('fixed', self.output)
        self.rows.pop()
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'no files for layer'):
            materialize.materialize('decay', self.output)
        self.assertFalse(self.output.exists())

    def test_symlinked_manifest_is_rejected(self):
        self.add('src/module.py', b'content')
        self.manifest()
        manifest = self.source / 'provenance.json'
        external = self.root / 'external.json'
        manifest.rename(external)
        manifest.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            materialize.materialize('fixed', self.output)
        self.assertFalse(self.output.exists())

    def test_noncanonical_paths_are_rejected(self):
        for name in ('../outside', '/absolute', './src/a.py', 'src/../a.py',
                     'src//a.py', '.', '', None, 'src\\a.py'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                source_integrity.relative_path(name)


class PatchVerificationTests(TemporaryTree):
    def setUp(self):
        super().setUp()
        self.source = self.root / 'release'
        self.skyrl = self.root / 'skyrl'
        put(self.source, 'patches/skyrl/native-startup.patch', b'patch')
        self.manifest = {'patch_sha256': digest(b'patch'), 'files': [
            {'path': 'trainer.py', 'before_sha256': digest(b'before'), 'after_sha256': digest(b'after')}
        ]}
        self.save_manifest()
        patcher = patch.object(verify_skyrl_patch, 'ROOT', self.source)
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_manifest(self):
        put(self.source, 'patches/skyrl/manifest.json', json.dumps(self.manifest).encode())

    def test_before_and_after_are_distinct_and_verified(self):
        put(self.skyrl, 'trainer.py', b'before')
        self.assertEqual(verify_skyrl_patch.verify_patch('before', self.skyrl), 1)
        with self.assertRaisesRegex(ValueError, 'Not the qualified after'):
            verify_skyrl_patch.verify_patch('after', self.skyrl)
        put(self.skyrl, 'trainer.py', b'after')
        self.assertEqual(verify_skyrl_patch.verify_patch('after', self.skyrl), 1)

    def test_patch_drift_and_malformed_manifest_are_rejected(self):
        put(self.source, 'patches/skyrl/native-startup.patch', b'changed')
        with self.assertRaisesRegex(ValueError, 'Patch hash mismatch'):
            verify_skyrl_patch.verify_patch('before', self.skyrl)
        self.manifest['files'] = [None]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'entries must be objects'):
            verify_skyrl_patch.verify_patch('before', self.skyrl)

    def test_symlinked_manifest_is_rejected(self):
        put(self.skyrl, 'trainer.py', b'before')
        manifest = self.source / 'patches/skyrl/manifest.json'
        external = self.root / 'external.json'
        manifest.rename(external)
        manifest.symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verify_skyrl_patch.verify_patch('before', self.skyrl)


class ReleaseCheckTests(TemporaryTree):
    def setUp(self):
        super().setUp()
        put(self.root, 'sample.py', b'answer = 42\n')
        self.rows = [{'path': 'sample.py', 'sha256': digest(b'answer = 42\n'), 'bytes': 12}]
        self.save_manifest()

    def save_manifest(self):
        put(self.root, 'configs/source_manifest.json', json.dumps({'files': self.rows}).encode())

    def test_clean_source_passes_and_hash_drift_fails(self):
        result = check_release.check_tree(self.root)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.verified, 1)
        put(self.root, 'sample.py', b'answer = 43\n')
        self.assertTrue(any('hash differs' in e for e in check_release.check_tree(self.root).errors))

    def test_malformed_manifests_report_errors_without_crashing(self):
        for value in ('{', '[]', '{}', '{"files": null}', '{"files": [null]}',
                      '{"files": [{"path": []}]}', '{"files": [{"path": "../outside"}]}'):
            with self.subTest(value=value):
                put(self.root, 'configs/source_manifest.json', value.encode())
                self.assertTrue(check_release.check_tree(self.root).errors)

    def test_empty_release_is_not_reported_as_verified(self):
        (self.root / 'sample.py').unlink()
        self.rows = []
        self.save_manifest()
        result = check_release.check_tree(self.root)
        self.assertTrue(result.errors)
        self.assertEqual(result.verified, 0)

    def test_duplicate_invalid_hash_and_wrong_size_are_errors(self):
        for rows in ([self.rows[0], self.rows[0]],
                     [{**self.rows[0], 'sha256': None}],
                     [{**self.rows[0], 'bytes': 1}],
                     [{**self.rows[0], 'bytes': True}]):
            with self.subTest(rows=rows):
                self.rows = rows
                self.save_manifest()
                self.assertTrue(check_release.check_tree(self.root).errors)

    def test_unlisted_data_and_unapproved_json_are_rejected(self):
        put(self.root, 'extra.py', b'pass\n')
        put(self.root, 'sample.csv', b'fake,fixture\n')
        put(self.root, 'unreviewed.json', b'{}')
        put(self.root, 'unreviewed-upper.JSON', b'{}')
        errors = check_release.check_tree(self.root).errors
        self.assertTrue(any('extra.py: not listed' in e for e in errors))
        self.assertTrue(any('sample.csv: generated/private' in e for e in errors))
        self.assertTrue(any('unreviewed.json: JSON is not' in e for e in errors))
        self.assertTrue(any('unreviewed-upper.JSON: JSON is not' in e for e in errors))

    def test_symlinked_manifest_and_source_are_rejected(self):
        original = self.root / 'sample.py'
        link = self.root / 'linked.py'
        link.symlink_to(original)
        self.rows.append({**self.rows[0], 'path': 'linked.py'})
        self.save_manifest()
        result = check_release.check_tree(self.root)
        self.assertTrue(any('linked.py: symlink' in e for e in result.errors))
        self.assertEqual(result.verified, 1)
        manifest = self.root / 'configs/source_manifest.json'
        manifest.unlink()
        manifest.symlink_to(original)
        self.assertTrue(check_release.check_tree(self.root).errors)


class DocumentationImageTests(TemporaryTree):
    def setUp(self):
        super().setUp()
        put(self.root, 'sample.py', b'answer = 42\n')
        self.rows = [{'path': 'sample.py', 'sha256': digest(b'answer = 42\n'), 'bytes': 12}]
        self.save_manifest()
        self.name = 'docs/assets/paper-teaser.png'
        self.image = b'\x89PNG\r\n\x1a\nsynthetic fixture'
        patcher = patch.object(check_release, 'APPROVED_DOCUMENTATION_IMAGES',
                               {self.name: digest(self.image)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def save_manifest(self):
        put(self.root, 'configs/source_manifest.json', json.dumps({'files': self.rows}).encode())

    def add_image(self, name=None, data=None):
        name = self.name if name is None else name
        data = self.image if data is None else data
        put(self.root, name, data)
        self.rows.append({'path': name, 'sha256': digest(data), 'bytes': len(data)})
        self.save_manifest()

    def test_reviewed_image_passes_and_replacement_fails_with_updated_manifest(self):
        self.add_image()
        result = check_release.check_tree(self.root)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.verified, 2)
        changed = self.image + b'changed'
        put(self.root, self.name, changed)
        self.rows[-1].update(sha256=digest(changed), bytes=len(changed))
        self.save_manifest()
        self.assertTrue(any('differs from the reviewed figure' in e
                            for e in check_release.check_tree(self.root).errors))

    def test_same_image_at_another_path_remains_forbidden(self):
        self.add_image('docs/assets/another.png')
        self.assertTrue(any('another.png: generated/private artifact type' in e
                            for e in check_release.check_tree(self.root).errors))

    def test_only_the_reviewed_gif_is_allowed(self):
        self.name = 'docs/assets/demos/reviewed.gif'
        self.image = b'GIF89a synthetic reviewed preview'
        with patch.dict(check_release.APPROVED_DOCUMENTATION_IMAGES,
                        {self.name: digest(self.image)}, clear=True):
            self.add_image()
            self.assertEqual(check_release.check_tree(self.root).errors, [])
            self.add_image('docs/assets/demos/unreviewed.gif', self.image)
            self.assertTrue(any('unreviewed.gif: generated/private artifact type' in e
                                for e in check_release.check_tree(self.root).errors))

    def test_reviewed_image_still_requires_manifest_coverage(self):
        put(self.root, self.name, self.image)
        self.assertTrue(any(self.name + ': not listed' in e
                            for e in check_release.check_tree(self.root).errors))
        (self.root / 'configs/source_manifest.json').unlink()
        self.assertTrue(any('source manifest missing' in e
                            for e in check_release.check_tree(self.root).errors))

    def test_reviewed_path_cannot_be_a_symlink(self):
        self.add_image('fixture.txt', self.image)
        path = self.root / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(self.root / 'fixture.txt')
        self.rows.append({'path': self.name, 'sha256': digest(self.image), 'bytes': len(self.image)})
        self.save_manifest()
        self.assertTrue(any(self.name + ': symlink' in e
                            for e in check_release.check_tree(self.root).errors))


if __name__ == '__main__':
    unittest.main()
