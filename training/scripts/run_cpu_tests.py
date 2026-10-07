"""Run synthetic tests in separate, network-disabled fixed/decay CPU processes."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
from materialize import ROOT, materialize

CHILD = r'''
import os, socket, sys
from pathlib import Path
root, runtime, variant = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path[:0] = [str(root / 'tests/support'), str(runtime), str(runtime / 'prm_v2')]
def deny(*args, **kwargs):
    raise RuntimeError('Network is disabled in the CPU test suite')
socket.create_connection = deny
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.socket.sendto = deny
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', str(root / 'tests' / variant)]))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('fixed', 'decay', 'all'), default='all')
    args = parser.parse_args()
    variants = ('fixed', 'decay') if args.variant == 'all' else (args.variant,)
    results = []
    with tempfile.TemporaryDirectory(prefix='editing-training-cpu-') as temp:
        for variant in variants:
            runtime = materialize(variant, Path(temp) / variant)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
                   'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1'}
            result = subprocess.run([sys.executable, '-c', CHILD, str(ROOT), str(runtime), variant], env=env)
            results.append({'variant': variant, 'exit_code': result.returncode})
    print(json.dumps(results, indent=2))
    return 0 if all(row['exit_code'] == 0 for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
