"""Image-only W&B metadata optimization; no logger/validation/math changes.

On the observed Python3.12 image, Pydantic's attribute-docstring extraction walks
frames and repeatedly scans sys.modules via inspect.getmodule during W&B import.
Disable only automatic field-description extraction on the two W&B base configs.
Aliases, value validation, serialization and all logging stay enabled.
"""
import hashlib
import json
import os
from pathlib import Path
import sysconfig

from .contract import ParityError, canonical
from .store import atomic_bytes

PINS = {
    '_pydantic/base.py': '978799eac3212de7e13d354d28f603e7e41fb123154fb1b1132b35660d76cae2',
    'sdk/wandb_settings.py': '1c65a037ab2c585c6ead64b0dac989b4a9ce3188ebffe74a8633b99d46c8245e',
}
MANIFEST = Path('/opt/ocrl_startup_fixes.json')


def transform(name, raw):
    if name not in PINS or hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise ParityError('Unreviewed W&B startup source')
    if raw.count(b'use_attribute_docstrings=True') != 1:
        raise ParityError('Unreviewed W&B docstring config')
    return raw.replace(b'use_attribute_docstrings=True', b'use_attribute_docstrings=False')


def verify_installed():
    entries = json.loads(MANIFEST.read_bytes())
    if set(entries) != set(PINS):
        raise ParityError('Incomplete W&B startup fix manifest')
    for name, entry in entries.items():
        if (entry['before_sha256'] != PINS[name] or
                hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['after_sha256']):
            raise ParityError('W&B startup source changed')
    return entries


def main():
    if os.environ.get('OCRL_ISOLATED_IMAGE_BUILD') != '1':
        raise ParityError('Only install while building the isolated OpenCode image')
    if MANIFEST.exists():
        verify_installed()
        return
    root = Path(sysconfig.get_paths()['purelib']) / 'wandb'
    edits, entries = [], {}
    for name, before in PINS.items():
        path = root / name
        after = transform(name, path.read_bytes())
        edits.append((path, after))
        entries[name] = {'path': str(path), 'before_sha256': before,
                         'after_sha256': hashlib.sha256(after).hexdigest()}
    for path, after in edits:
        atomic_bytes(path, after)
    atomic_bytes(MANIFEST, canonical(entries))
    print('W&B metadata-only startup fix installed', flush=True)


if __name__ == '__main__':
    main()
