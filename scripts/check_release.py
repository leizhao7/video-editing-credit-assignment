#!/usr/bin/env python3
"""Check the assembled source tree without importing it or using the network."""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {'.git'}
FORBIDDEN_DIRS = {'data', 'datasets', 'checkpoints', 'outputs', 'runs', 'logs', 'wandb',
                  '.venv', '__pycache__', '.pytest_cache', 'node_modules'}
FORBIDDEN_SUFFIXES = {'.mp4', '.mov', '.mkv', '.wav', '.mp3', '.flac', '.png', '.jpg', '.gif',
                      '.jpeg', '.safetensors', '.pt', '.pth', '.ckpt', '.parquet',
                      '.arrow', '.jsonl', '.zip', '.gz', '.tar', '.pem', '.key', '.pyc',
                      '.csv', '.tsv', '.npy', '.npz', '.pkl', '.pickle', '.h5', '.hdf5',
                      '.sqlite', '.db', '.srt', '.vtt', '.otio', '.xml', '.log', '.pdf'}
PATTERNS = {
    'personal workstation path': re.compile(r'/Users/[A-Za-z0-9._-]+/'),
    'fixed private cloud object': re.compile(r'\b(?:im|ap|fc|ta)-[A-Za-z0-9]{16,}\b'),
    'Google API key': re.compile(r'\bAIza[0-9A-Za-z_-]{30,}\b'),
    'GitHub access token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'private key block': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}
APPROVED_METADATA_PATHS = {
    'configs/dependency_pins.json',
    'configs/evaluation_sources.json',
    'configs/grpo.json',
    'configs/opsd.json',
    'configs/prm_decay.json',
    'configs/prm_fixed.json',
    'configs/source_manifest.json',
    'opsd/source_map.json',
    'training/patches/skyrl/manifest.json',
    'training/provenance.json',
    'training/third_party/OpenCode/package.json',
    'training/third_party/OpenCode/packages/opencode/package.json',
}
# Reviewed documentation artwork only; other images remain excluded from the release.
APPROVED_DOCUMENTATION_IMAGES = {
    'docs/assets/paper-teaser.png': '9377e1b3bb536b1bdf251bc77e6907ab8eeece37cac70c47c14e5b763f6cd680',
    'docs/assets/film-poster-collage.png': '242d40baf383e8b07f2ccc5408e9bbd38d0c25f209cdc315491ceb7da5dc492b',
}


@dataclass
class CheckResult:
    errors: list[str]
    checked: int
    python_files: int
    verified: int


def check_manifest(root, checked_paths):
    """Report malformed manifests and drift as failures, without trusting their paths."""
    errors = []
    manifest_name = 'configs/source_manifest.json'
    if manifest_name not in checked_paths:
        return [manifest_name + ': source manifest missing or not a regular file'], 0
    try:
        manifest = json.loads((root / manifest_name).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError):
        return [manifest_name + ': cannot read a valid source manifest'], 0
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), list):
        return [manifest_name + ': expected an object with a files list'], 0
    if not manifest['files']:
        return [manifest_name + ': source manifest files list is empty'], 0
    recorded = set()
    verified = 0
    for index, row in enumerate(manifest['files']):
        if not isinstance(row, dict):
            errors.append(f'{manifest_name}: entry {index} must be an object')
            continue
        name = row.get('path')
        if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
            errors.append(f'{manifest_name}: entry {index} has an invalid path')
            continue
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or rel.as_posix() != name or name == '.':
            errors.append(f'{manifest_name}: entry {index} has an unsafe or noncanonical path')
            continue
        if name in recorded or name == manifest_name:
            errors.append(f'{name}: duplicate or self-referencing manifest entry')
            continue
        recorded.add(name)
        digest = row.get('sha256')
        size = row.get('bytes')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            errors.append(f'{name}: invalid SHA-256 in source manifest')
            continue
        if type(size) is not int or size < 0:
            errors.append(f'{name}: invalid byte count in source manifest')
            continue
        # Only open regular files found by the scan; never follow a manifest-only path.
        if name not in checked_paths:
            errors.append(f'{name}: manifest file missing or not a regular file')
            continue
        try:
            data = (root / name).read_bytes()
        except OSError:
            errors.append(f'{name}: cannot read manifest file')
            continue
        if hashlib.sha256(data).hexdigest() != digest:
            errors.append(f'{name}: source hash differs from the assembled release')
        elif len(data) != size:
            errors.append(f'{name}: byte count differs from the assembled release')
        else:
            verified += 1
    for extra in sorted(checked_paths - recorded - {manifest_name}):
        errors.append(f'{extra}: not listed in the assembled source manifest')
    return errors, verified


def check_tree(root):
    root = Path(root)
    errors = []
    checked_paths = set()
    checked = 0
    python_files = 0
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if set(rel.parts) & IGNORED_DIRS:
            continue
        if path.is_symlink():
            errors.append(f'{rel}: symlink is not allowed in the source release')
            continue
        if not path.is_file():
            if not path.is_dir():
                errors.append(f'{rel}: not a regular file or directory')
            continue
        checked += 1
        checked_paths.add(rel.as_posix())
        if set(rel.parts) & FORBIDDEN_DIRS:
            errors.append(f'{rel}: generated/private artifact type')
            continue
        if rel.as_posix() in APPROVED_DOCUMENTATION_IMAGES:
            try:
                data = path.read_bytes()
            except OSError:
                errors.append(f'{rel}: cannot read documentation image')
                continue
            if (not data.startswith(b'\x89PNG\r\n\x1a\n') or
                    hashlib.sha256(data).hexdigest() != APPROVED_DOCUMENTATION_IMAGES[rel.as_posix()]):
                errors.append(f'{rel}: documentation image differs from the reviewed figure')
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f'{rel}: generated/private artifact type')
            continue
        if path.name.startswith('.env') and path.name != '.env.example':
            errors.append(f'{rel}: environment file')
            continue
        if path.suffix.lower() == '.json' and rel.as_posix() not in APPROVED_METADATA_PATHS:
            errors.append(f'{rel}: JSON is not an explicitly reviewed code configuration or source-hash manifest')
        try:
            source = path.read_text(encoding='utf-8')
        except UnicodeError:
            errors.append(f'{rel}: unexpected binary file')
            continue
        except OSError:
            errors.append(f'{rel}: cannot read source file')
            continue
        if path.suffix.lower() == '.py':
            python_files += 1
            try:
                ast.parse(source, filename=str(rel))
            except SyntaxError as exc:
                errors.append(f'{rel}:{exc.lineno}: {exc.msg}')
        if path.suffix.lower() == '.json':
            try:
                json.loads(source)
            except ValueError:
                errors.append(f'{rel}: invalid JSON')
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(source):
                # Report locations, never possible credential values.
                line = source.count('\n', 0, match.start()) + 1
                errors.append(f'{rel}:{line}: {label}')
    manifest_errors, verified = check_manifest(root, checked_paths)
    return CheckResult(errors + manifest_errors, checked, python_files, verified)


def main():
    result = check_tree(ROOT)
    for error in result.errors:
        print(error)
    print(f'Checked {result.checked} files, parsed {result.python_files} Python files, verified {result.verified} source hashes.')
    print(f'{len(result.errors)} issue(s). This is a static check, not end-to-end reproduction or a complete secret audit.')
    return bool(result.errors)


if __name__ == '__main__':
    sys.exit(main())
