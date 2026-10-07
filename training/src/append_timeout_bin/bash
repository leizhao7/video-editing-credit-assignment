#!/usr/local/bin/python -I
"""Candidate native-Bash timeout cleanup plus per-process address-space guard.

Each invocation owns a Linux subreaper. It forwards unmodified argv/env/stdin
and keeps the native exit code/output. Only on SIGTERM/SIGINT/SIGHUP does it
terminate its own descendants, including children that changed process groups.
The persistent tool host, MCP services, and sibling commands are never targets.
The owner and its children inherit a 6 GiB hard virtual-address-space bound;
this is NOT an aggregate process-tree/cgroup limit. The persistent Bun host
is never address-limited. This candidate is not installed in live training.
"""
import ctypes
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import time


POLICY_ADDRESS_BYTES = 6 * 1024**3


def apply_address_limit(limit=POLICY_ADDRESS_BYTES):
    if type(limit) is not int or limit <= 0:
        raise ValueError('Positive address-space byte limit required')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    hard = limit if hard == resource.RLIM_INFINITY else min(hard, limit)
    soft = hard if soft == resource.RLIM_INFINITY else min(soft, hard)
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    return soft, hard


def descendants(owner):
    parents = {}
    for root in Path("/proc").iterdir():
        if not root.name.isdecimal():
            continue
        try:
            stat = (root / "stat").read_text()
            fields = stat[stat.rfind(")")+2:].split()
            parents[int(root.name)] = (int(fields[1]), fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
    found = {owner}
    while True:
        new = {pid for pid, (parent, _) in parents.items() if parent in found}
        if new <= found:
            break
        found |= new
    return {pid: parents[pid][1] for pid in found if pid != owner}


def signal_owned(owner, signum):
    # Pin each target, then recheck it is still our descendant with the same
    # birth identity. Never send a signal to a recycled PID or an arbitrary PGID.
    for pid, birth in descendants(owner).items():
        fd = None
        try:
            fd = os.pidfd_open(pid, 0)
            if descendants(owner).get(pid) == birth:
                signal.pidfd_send_signal(fd, signum)
        except ProcessLookupError:
            pass
        finally:
            if fd is not None:
                os.close(fd)


def main():
    if (sys.platform != "linux" or os.geteuid() == 0 or
            not Path("/opt/OCRL_SANDBOX_IMAGE").is_file() or not sys.argv[1:]):
        raise RuntimeError("Unprivileged Linux rollout command owner required")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Unable to own orphan descendants")
    fd = os.pidfd_open(os.getpid(), 0)
    os.close(fd)  # Fail before launching a command if pidfds are unavailable.
    stopped = []
    def stop(signum, _):
        if not stopped:
            stopped.append(signum)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    # Apply in this fresh, single-threaded command owner, never in the Bun
    # parent or a threaded preexec_fn. The hard limit is inherited by exec.
    apply_address_limit()
    child = subprocess.Popen(["/bin/bash", *sys.argv[1:]])
    while child.poll() is None and not stopped:
        time.sleep(.02)
    if not stopped:
        return child.returncode if child.returncode >= 0 else 128-child.returncode
    # Native OpenCode force-kills the outer owner after three seconds. Finish
    # both cleanup phases before that deadline; subreaping catches double forks.
    deadline = time.monotonic() + .5
    while descendants(os.getpid()) and time.monotonic() < deadline:
        signal_owned(os.getpid(), signal.SIGTERM)
        child.poll()
        time.sleep(.02)
    deadline = time.monotonic() + 1
    while descendants(os.getpid()) and time.monotonic() < deadline:
        signal_owned(os.getpid(), signal.SIGKILL)
        child.poll()
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        time.sleep(.02)
    if descendants(os.getpid()):
        raise RuntimeError("Owned command cleanup did not finish")
    return 128 + stopped[0]


if __name__ == "__main__":
    sys.exit(main())
