"""Independent launcher, retaining the inherited native hard file-size limit."""
import os
from pathlib import Path
import sys

from opencode_rl.policy_limits import apply_file_limit


def main():
    if os.geteuid() == 0 or not Path('/opt/OCRL_SANDBOX_IMAGE').is_file():
        raise RuntimeError('Only the unprivileged isolated sandbox may launch the tool host')
    if len(sys.argv) != 2 or not sys.argv[1].startswith('/var/lib/ocrl/'):
        raise ValueError('Expected the root-staged public task file')
    apply_file_limit()
    os.execv('/usr/local/bin/opencode-tools', ['/usr/local/bin/opencode-tools', sys.argv[1]])


if __name__ == '__main__':
    main()
