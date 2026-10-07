"""Verify the qualified parent or patched SkyRL files, without modifying them."""
from pathlib import Path
import argparse
import hashlib
import json
from source_integrity import file_rows, sha256_value, source_file

ROOT = Path(__file__).resolve().parents[1]


def verify_patch(state, skyrl_root):
    if state not in ('before', 'after'):
        raise ValueError('State must be before or after')
    manifest = json.loads(source_file(ROOT, 'patches/skyrl/manifest.json').read_text(encoding='utf-8'))
    rows = file_rows(manifest, ('before_sha256', 'after_sha256'))
    expected_patch = sha256_value(manifest.get('patch_sha256'))
    patch = source_file(ROOT, 'patches/skyrl/native-startup.patch')
    if hashlib.sha256(patch.read_bytes()).hexdigest() != expected_patch:
        raise ValueError('Patch hash mismatch')
    for row in rows:
        target = source_file(skyrl_root, row['path'])
        if hashlib.sha256(target.read_bytes()).hexdigest() != row[state + '_sha256']:
            raise ValueError('Not the qualified ' + state + ' file: ' + row['path'])
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', choices=('before', 'after'))
    parser.add_argument('skyrl_root', type=Path)
    args = parser.parse_args(argv)
    try:
        count = verify_patch(args.state, args.skyrl_root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{parser.prog}: {exc}\n')
    print(count, 'SkyRL files match the', args.state, 'hashes.')


if __name__ == '__main__':
    main()
