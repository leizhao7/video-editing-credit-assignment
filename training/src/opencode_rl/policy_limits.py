"""Inherited OS limit for policy-created files; root archives stay unlimited.

Infinite lavfi sources can write tens of GB before the Bash timeout. Apply a
hard file-size limit after runuser and before OpenCode, so native Bash, Python,
FFmpeg and MCP descendants all inherit it and cannot raise it. Do not use a
preexec_fn in the already-threaded root relay.
"""
import os
from pathlib import Path
import resource
import sys

POLICY_FILE_BYTES = 1024 ** 3


def apply_file_limit(limit=POLICY_FILE_BYTES):
    if type(limit) is not int or limit <= 0:
        raise ValueError('Positive per-file byte limit required')
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    hard = limit if hard == resource.RLIM_INFINITY else min(hard, limit)
    soft = hard if soft == resource.RLIM_INFINITY else min(soft, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
    return soft, hard


def resource_prompt(original):
    # Same shared launch boundary as submission_prompt; frozen bundles remain
    # unchanged. The policy sees the actual constraint before calling tools.
    return original + '\n\n【沙箱文件资源限制】\n' + (
        '单个新生成文件最多 1 GiB（1073741824 字节）。这是文件写入上限，'
        '不限制读取已提供的源 clip。超限的工具会返回文件过大/写入失败错误，'
        '可缩短输出时长、压缩编码或分段处理后重试。\n'
        'FFmpeg 的 anullsrc、color 等生成源和循环输入可能是无限流；'
        '生成静音、黑场或循环音乐时，务必按任务需要用 -t/-to 或有界输入配合 '
        '-shortest 明确限定输出时长，不要先生成无限长 WAV/视频再裁切。\n')


def main():
    if os.geteuid() == 0 or not Path('/opt/OCRL_SANDBOX_IMAGE').exists():
        raise RuntimeError('Policy limit wrapper is only for the unprivileged sandbox runtime')
    argv = sys.argv[1:]
    if not argv or argv[0] != '/usr/local/bin/opencode':
        raise ValueError('Only the pinned native OpenCode entrypoint may be launched')
    apply_file_limit()
    os.execv(argv[0], argv)


if __name__ == '__main__':
    main()
