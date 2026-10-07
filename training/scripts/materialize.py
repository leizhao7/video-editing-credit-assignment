"""Assemble and verify a local CPU-test runtime; never launch training or cloud work."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from source_integrity import file_rows, source_file

ROOT = Path(__file__).resolve().parents[1]


def materialize(variant, destination):
    if variant not in ('fixed', 'decay'):
        raise ValueError('Unknown variant')
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Destination must not exist: ' + str(destination))
    manifest = json.loads(source_file(ROOT, 'provenance.json').read_text(encoding='utf-8'))
    rows = file_rows(manifest, ('public_sha256',))
    selected = {}
    # Apply layers explicitly: manifest ordering must never change the runtime.
    layers = ('src/',) if variant == 'fixed' else ('src/', 'overlays/decay/')
    for prefix in layers:
        layer = {row['path'][len(prefix):]: row for row in rows if row['path'].startswith(prefix)}
        if not layer:
            raise ValueError('Source manifest contains no files for layer: ' + prefix)
        selected.update(layer)
    # Check every input before creating the destination; never silently ignore drift.
    verified = {}
    for relative, row in selected.items():
        source = source_file(ROOT, row['path'])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['public_sha256']:
            raise ValueError('Public-source hash mismatch: ' + row['path'])
        verified[relative] = (data, source.stat().st_mode & 0o777)
    destination.mkdir(parents=True)
    try:
        for relative, (data, mode) in verified.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Copy exactly the bytes verified above, without rereading mutable inputs.
            target.write_bytes(data)
            target.chmod(mode)
    except BaseException:
        # Only remove the destination created by this invocation, never an existing tree.
        shutil.rmtree(destination)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('fixed', 'decay'), required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        materialize(args.variant, args.destination)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')
    print('Materialized', args.variant, 'CPU-test runtime:', args.destination)
    print('No upstream dependencies, cloud resources or training qualification were created.')


if __name__ == '__main__':
    main()
