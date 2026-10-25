"""Prepare only the pinned OpenCode checkout for the native tool-host build."""
from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
COMMIT = 'd7a6e1daaf2271bcd0b611fbb27d4956806e475a'


def prepare(root):
    root = Path(root).resolve(strict=True)
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise ValueError('Native host requires the pinned OpenCode checkout')
    lock = (HERE / 'opencode.bun.lock').read_text()
    # Pin the *already resolved* dependency from the recovered build lock.
    # The terminal host must independently prove ghostty is absent from its bundle.
    match = re.search(r'ghostty-web@github:anomalyco/ghostty-web#([a-f0-9]{7,40})', lock)
    if not match:
        raise ValueError('Recovered lock has no resolved ghostty commit')
    old = 'github:anomalyco/ghostty-web#main'
    pinned = 'github:anomalyco/ghostty-web#' + match[1]
    if lock.count('"' + old + '"') != 1:
        raise ValueError('Unreviewed floating dependency shape')
    (root / 'bun.lock').write_text(lock.replace('"' + old + '"', '"' + pinned + '"'))
    package = root / 'packages/app/package.json'
    value = json.loads(package.read_bytes())
    if value['dependencies'].get('ghostty-web') != old:
        raise ValueError('Upstream workspace dependency changed')
    value['dependencies']['ghostty-web'] = pinned
    package.write_text(json.dumps(value, indent=2) + '\n')
    native = HERE.parent / 'src/append_agent_rl/native'
    target = root / 'packages/opencode/src/append_agent_rl'
    target.mkdir(exist_ok=True)
    shutil.copyfile(native / 'tool_host.ts', target / 'tool_host.ts')
    shutil.copyfile(native / 'build.ts', root / 'packages/opencode/script/build_append_host.ts')
    return {'opencode_commit': head, 'ghostty_resolution': match[1], 'native_loop_replaced': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    print(json.dumps(prepare(parser.parse_args().root)))
