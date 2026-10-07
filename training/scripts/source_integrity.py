"""Validate source-manifest entries without following links outside a source tree."""
from pathlib import Path, PurePosixPath
import re


def relative_path(value):
    """Require the canonical POSIX relative paths used by release manifests."""
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('Manifest path must be a nonempty POSIX relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value or value == '.':
        raise ValueError('Unsafe or noncanonical manifest path: ' + value)
    return path


def sha256_value(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('Manifest SHA-256 must contain 64 lowercase hexadecimal digits')
    return value


def source_file(root, name):
    """Resolve a regular source file, rejecting symlinks in any relative component."""
    path = Path(root).resolve()
    for component in relative_path(name).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError('Source path contains a symlink: ' + name)
    if not path.is_file():
        raise ValueError('Source file missing or not a regular file: ' + name)
    return path


def file_rows(manifest, hash_keys):
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), list):
        raise ValueError('Source manifest must contain a files list')
    if not manifest['files']:
        raise ValueError('Source manifest files list is empty')
    seen = set()
    for row in manifest['files']:
        if not isinstance(row, dict):
            raise ValueError('Source manifest entries must be objects')
        name = relative_path(row.get('path')).as_posix()
        if name in seen:
            raise ValueError('Duplicate source manifest path: ' + name)
        seen.add(name)
        for key in hash_keys:
            sha256_value(row.get(key))
    return manifest['files']
