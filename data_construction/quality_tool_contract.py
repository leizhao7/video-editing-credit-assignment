"""Canonical S12 tool schemas shared by RL and teacher rollouts.

The wire envelope differs between vLLM chat templates and the Responses API,
but the names, descriptions, and JSON schemas must not.  Keeping the function
definitions here makes drift a source-level error instead of a review-time
guess.
"""

from __future__ import annotations

from copy import deepcopy


FUNCTIONS = [
    {
        "name": "bash",
        "description": "在任务目录下执行一条 shell 命令，返回 stdout+stderr。",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "要执行的命令，例如 ls frames",
                },
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "view_image",
        "description": (
            "查看一张或多张图片。传帧文件的相对路径，图片会作为观测返回。"
            "所有 stage 都可查看预抽帧及工作区自制图片。预抽帧一次最多 32 张，自制图片一次最多 8 张。"
            "模型上下文最多 192 张图片；S1/S2 无额外新增图片配额，S3/S4E 新增最多 96 张。重复查看也计入上下文。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "图片路径列表，例如 [\"frames/k01_1.jpg\", "
                        "\"frames/k01_2.jpg\"]"
                    ),
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "inspect_video",
        "description": "所有 stage：自由查看完整源 clip 或自制视频的任意时间点/区间。只传 path 默认给全片 8 张均匀采样图；可指定 timestamps_s，或 start_s/end_s/num_frames。可反复、分段查看。返回采样帧，不是连续播放。每次 1–8 帧，长边最多 448 像素；上下文最多 192 图，S1/S2 无额外新增图片配额，S3/S4E 新增最多 96 图。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "工作区视频路径，例如 initial_state/clips/k01.mp4 或 output/preview.mp4"},
                "timestamps_s": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 1, "maxItems": 8,
                                 "description": "文件内秒数，如 [1.8, 2.0, 2.2]；不能与 start_s/end_s/num_frames 同传"},
                "start_s": {"type": "number", "minimum": 0, "description": "区间起点，默认 0；不与 timestamps_s 同传"},
                "end_s": {"type": "number", "minimum": 0, "description": "区间终点，默认实际视频末尾；不与 timestamps_s 同传"},
                "num_frames": {"type": "integer", "minimum": 1, "maximum": 8, "description": "区间内均匀抽样数量，默认 8；不与 timestamps_s 同传"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "inspect_audio",
        "description": "所有 stage：检查源 clip/音轨/成品一个窗口的实际时长、音量、静音及可选英文对白转录（CPU 离线 small.en）。转录可能漏字/误识别，无转录不等于无对白；不返回评分。每次最多 30 秒，可分段查看更长音频。S1/S2 无累计音频配额；S3/S4E 每轨迹最多请求 180 秒/12 次。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "工作区视频或音频路径，包括 output/preview.mp4"},
                "start_s": {"type": "number", "minimum": 0, "description": "文件内开始秒数，默认 0"},
                "duration_s": {"type": "number", "minimum": 0.05, "maximum": 30, "description": "检查窗口秒数，默认 20"},
                "transcribe": {"type": "boolean", "description": "默认 true；只检查音量/时长可设 false，加快执行"},
            },
            "required": ["path"],
        },
    },
]


def chat_tools() -> list:
    """OpenAI chat/vLLM chat-template envelope."""
    return [{"type": "function", "function": deepcopy(f)} for f in FUNCTIONS]


def responses_tools() -> list:
    """OpenAI Responses API envelope."""
    return [{"type": "function", **deepcopy(f)} for f in FUNCTIONS]


CHAT_TOOLS = chat_tools()
RESPONSES_TOOLS = responses_tools()


TOOLS_FALLBACK_SYSTEM = """你可以使用以下工具。每次要用工具时，输出一个工具调用块：

<tool_call>{"name": "bash", "arguments": {"cmd": "ls frames"}}</tool_call>

可用工具：

1. bash —— 在任务目录下执行一条 shell 命令，返回 stdout 和 stderr。
   参数：cmd（字符串）
   例：<tool_call>{"name": "bash", "arguments": {"cmd": "ls frames"}}</tool_call>

2. view_image —— 查看一张或多张图片。图片会作为观测返回给你，你能真正看到画面。
   参数：paths（字符串数组）
   例：<tool_call>{"name": "view_image", "arguments": {"paths": ["frames/k01_1.jpg", "frames/k01_2.jpg", "frames/k01_3.jpg"]}}</tool_call>

3. inspect_video —— 所有 stage 自由查看源视频或自制视频。只传 path 可概览全片；可选 timestamps_s（秒数数组），或 start_s/end_s/num_frames（区间均匀采样 1–8 帧），两种方式不混用。
   例：<tool_call>{"name": "inspect_video", "arguments": {"path": "output/preview.mp4", "timestamps_s": [1.8, 2.0, 2.2]}}</tool_call>

4. inspect_audio —— 所有 stage 查看源 clip 或音轨的时长、音量和离线英文对白转录。参数 path、start_s（默认 0）、duration_s（默认 20，最多 30）、transcribe（默认 true）。
   只测音量可设 transcribe=false。无转录不证明无对白；ASR 不是评分器。

完整源素材在 initial_state/；可用 bash/FFmpeg/ffprobe/Python 自行分析和抽帧，用 view_image 看自制图片。
模型上下文最多容纳 192 张图。S1/S2 无额外新增图片或累计音频配额；S3/S4E 新增最多 96 图、音频最多 180 秒/12 次。
预抽帧一次最多 32 张，自制图片一次最多 8 张，音频一次最多 30 秒。可自行选择并反复检查任意片段。上下文/轮次上限仍然适用。
媒体解码在隔离 CPU 沙箱里执行，无网络。

**帧是图像，需要调 view_image 或 inspect_video 才看得见。** 不要凭任务描述猜画面内容 —— 那不是这道
题要考的东西。要看某个片段，就把它的几帧一起传给 view_image。

当你不再需要工具、答案已经写好时，直接输出最终回答，不要再带工具调用块。"""
