"""Read-only legacy source verification; emit a small new audit artifact."""
import argparse
import hashlib
import json
from pathlib import Path

from .contract import canonical
from .store import atomic_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_bytes())
    changed = []
    for entry in manifest['files']:
        path = Path(entry['source_path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            changed.append(str(path))
    archive = Path(manifest['archive'])
    result = {'legacy_source_count': len(manifest['files']), 'changed': changed,
        'archive_sha256_matches': hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256'],
        'archive': str(archive), 'training_started': False}
    output = Path(args.output)
    if output.exists():
        raise RuntimeError('Refusing to overwrite prior source audit')
    atomic_bytes(output, canonical(result))
    print(json.dumps(result))
    if changed or not result['archive_sha256_matches']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
