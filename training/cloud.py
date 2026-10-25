"""Validate cloud settings or upload a prepared private task release.

These commands never allocate GPUs. Deployment and training are separate,
explicit Modal commands documented in docs/training.md.
"""
from pathlib import Path
import argparse
import hashlib
import json

from configuration import load_config

REPOSITORY = Path(__file__).resolve().parents[1]


def validate_release(directory):
    directory = Path(directory).expanduser().resolve(strict=True)
    if directory == REPOSITORY or directory.is_relative_to(REPOSITORY):
        raise ValueError('Private releases must be outside the repository')
    if any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('Private releases may not contain symlinks')
    receipt = json.loads((directory / 'preparation_receipt.json').read_bytes())
    raw = (directory / 'release.json').read_bytes()
    if (receipt.get('public_inputs_prepared') is not True or
            hashlib.sha256(raw).hexdigest() != receipt['release_sha256']):
        raise ValueError('Complete input preparation and verify the release before uploading')
    import sys
    sys.path.insert(0, str(REPOSITORY / 'training/src'))
    from opencode_rl.train_dataset import PreparedDataset
    from opencode_rl.task_bundle import release_tasks, validate_source
    for split in ('train', 'val'):
        rows = release_tasks(directory / 'release.json', split, expected_sha256=receipt['release_sha256'])
        if not rows:
            continue
        meta = receipt['prepared'][split]
        folder = directory / 'prepared' / split
        dataset = PreparedDataset(folder / 'manifest.json', folder / 'tasks', directory / 'release.json',
                                  split=split, expected_release_sha256=receipt['release_sha256'])
        if dataset.sha256 != meta['sha256'] or len(dataset) != meta['task_count']:
            raise ValueError('Prepared input identity differs from receipt')
        for row in rows:
            validate_source(directory / 'corpus' / ('_tt_' + split) / row['task'], row)
            dataset.verify_task(row['task'])
    return directory, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('validate', 'upload'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--release-directory')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = {'config_valid': True, 'profile': config['profile'], 'app': config['app'], 'gpu_allocated': False}
    if args.action == 'upload':
        if not args.release_directory:
            parser.error('upload requires --release-directory')
        directory, receipt = validate_release(args.release_directory)
        import modal
        volume = modal.Volume.from_name(config['release_volume'], create_if_missing=True)
        # A release volume is immutable after first upload. Never merge a second
        # collection into the same remote root or overwrite a training corpus.
        with modal.App('editing-release-upload').run():
            if list(volume.iterdir('/')):
                raise ValueError('Release volume is not empty; use a new volume name')
            with volume.batch_upload(force=False) as batch:
                for name in ('release.json', 'release_receipt.json', 'preparation_receipt.json'):
                    batch.put_file(directory / name, '/' + name)
                for name in ('corpus', 'prepared', 'private'):
                    if (directory / name).exists():
                        batch.put_directory(directory / name, '/' + name)
        result.update(uploaded=True, release_sha256=receipt['release_sha256'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
