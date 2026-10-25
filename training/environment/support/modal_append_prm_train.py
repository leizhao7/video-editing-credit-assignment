"""Bind installed public runtime bytes to the retained PRM source guard.

This replaces the private launcher in SUPPORT_FILES with an inspectable public
build helper. It cannot submit a training run or invent a qualification report.
"""
from pathlib import Path
import argparse
import json


def build_manifest(provenance, root='/opt'):
    import prm_training_source as source
    root = Path(root)
    rows = {row['path']: row for row in json.loads(Path(provenance).read_bytes())['files']}
    files, packages = {}, {}
    for package in ('opencode_rl', 'append_agent_rl'):
        hashes = {name: source._hash(path) for name, path in
                  source._package_sources(root / package, package).items()}
        packages[package] = source.digest(hashes)
        files.update({package + '/' + name: sha for name, sha in hashes.items()})
    for name in source.RUNTIME_FILES:
        files['prm_v2/' + name] = source._hash(root / 'prm_v2' / name)
    for name in (*source.SUPPORT_FILES, *source.AUXILIARY_FILES):
        files[name] = source._hash(root / name)
    result = {'version': source.VERSION, 'files': dict(sorted(files.items())),
        'package_hashes': packages, 'overlay_files': list(source.OVERLAY_FILES),
        'runtime_files': list(source.RUNTIME_FILES), 'support_files': list(source.SUPPORT_FILES),
        'auxiliary_files': source.AUXILIARY_FILES,
        'upstream_hashes': {name: rows['src/' + name]['source_sha256'] for name in source.OVERLAY_FILES},
        'scope': 'Public assembled source identity; original source hashes are provenance, not GPU acceptance.'}
    source.verify_manifest(result, root=root)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', required=True)
    args = parser.parse_args()
    import prm_training_source as source
    source.install_manifest(build_manifest(args.provenance))
