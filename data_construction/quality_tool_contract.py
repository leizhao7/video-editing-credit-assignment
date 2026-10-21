"""English release translation of the recovered fixed S12 tool schema.

Tool names, parameter schemas, defaults and inspection limits are preserved;
descriptions and fallback instructions are translated, not historical bytes.
The chat/vLLM and Responses API envelopes share these same definitions.
"""

from __future__ import annotations

from copy import deepcopy


FUNCTIONS = [
    {
        "name": "bash",
        "description": "Run one shell command in the task directory and return stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The command to run, for example ls frames.",
                },
            },
            "required": ["cmd"],
        },
    },
    {
        "name": "view_image",
        "description": (
            "View one or more images by supplying relative frame-file paths; the images are returned as observations. "
            "All stages can inspect pre-extracted frames and images created in the workspace. "
            "Each call allows at most 32 pre-extracted frames or 8 workspace-generated images. "
            "The model context holds at most 192 images. S1/S2 have no separate quota for newly inspected images; "
            "S3/S4E allow at most 96 additional images. Repeated views also count toward the context limit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of image paths, for example [\"frames/k01_1.jpg\", "
                        "\"frames/k01_2.jpg\"]"
                    ),
                },
            },
            "required": ["paths"],
        },
    },
    {
        "name": "inspect_video",
        "description": (
            "All stages: inspect any timestamps or intervals in full source clips or workspace-generated videos. "
            "Supplying only path returns 8 evenly spaced frames across the video. Alternatively, supply timestamps_s "
            "or start_s/end_s/num_frames. You may inspect intervals repeatedly. The tool returns sampled frames, "
            "not continuous playback. Each call returns 1–8 frames, with a longest edge of at most 448 pixels. "
            "The context holds at most 192 images. S1/S2 have no separate quota for newly inspected images; "
            "S3/S4E allow at most 96 additional images."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace video path, for example initial_state/clips/k01.mp4 or output/preview.mp4."},
                "timestamps_s": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 1, "maxItems": 8,
                                 "description": "Timestamps in seconds from the start of the file, such as [1.8, 2.0, 2.2]. Cannot be combined with start_s/end_s/num_frames."},
                "start_s": {"type": "number", "minimum": 0, "description": "Interval start in seconds; defaults to 0. Cannot be combined with timestamps_s."},
                "end_s": {"type": "number", "minimum": 0, "description": "Interval end in seconds; defaults to the actual end of the video. Cannot be combined with timestamps_s."},
                "num_frames": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of evenly spaced frames in the interval; defaults to 8. Cannot be combined with timestamps_s."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "inspect_audio",
        "description": (
            "All stages: inspect a window of a source clip, audio track or edited output for actual duration, "
            "volume and silence, with optional English dialogue transcription using offline small.en on CPU. "
            "Transcription may omit or misrecognize words; an empty transcript does not establish that no dialogue "
            "is present. This tool returns no score. Each request covers at most 30 seconds; inspect longer audio "
            "in separate windows. S1/S2 have no cumulative audio quota. S3/S4E allow at most 180 requested seconds "
            "and 12 audio requests per trajectory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace video or audio path, including output/preview.mp4."},
                "start_s": {"type": "number", "minimum": 0, "description": "Start time in seconds from the beginning of the file; defaults to 0."},
                "duration_s": {"type": "number", "minimum": 0.05, "maximum": 30, "description": "Inspection window in seconds; defaults to 20."},
                "transcribe": {"type": "boolean", "description": "Defaults to true. Set false for faster checks when only volume and duration are needed."},
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


TOOLS_FALLBACK_SYSTEM = """You can use the following tools. To use a tool, emit a tool-call block:

<tool_call>{"name": "bash", "arguments": {"cmd": "ls frames"}}</tool_call>

Available tools:

1. bash — Run one shell command in the task directory and return stdout and stderr.
   Parameter: cmd (string).
   Example: <tool_call>{"name": "bash", "arguments": {"cmd": "ls frames"}}</tool_call>

2. view_image — View one or more images. The images are returned as visual observations.
   Parameter: paths (array of strings).
   Example: <tool_call>{"name": "view_image", "arguments": {"paths": ["frames/k01_1.jpg", "frames/k01_2.jpg", "frames/k01_3.jpg"]}}</tool_call>

3. inspect_video — Inspect source videos or workspace-generated videos in any stage. Supplying only path gives an overview of the whole video. Optionally supply timestamps_s (an array of times in seconds), or start_s/end_s/num_frames (1–8 evenly spaced frames within an interval). Do not combine these two modes.
   Example: <tool_call>{"name": "inspect_video", "arguments": {"path": "output/preview.mp4", "timestamps_s": [1.8, 2.0, 2.2]}}</tool_call>

4. inspect_audio — Inspect the duration, volume and offline English dialogue transcript of a source clip or audio track in any stage. Parameters: path, start_s (default 0), duration_s (default 20, maximum 30), transcribe (default true).
   Set transcribe=false when only volume measurements are needed. An empty transcript does not prove that no dialogue is present; ASR does not grade the submission.

Full source assets are in initial_state/. Use bash/FFmpeg/ffprobe/Python for your own analysis and frame extraction, then view_image to inspect the images you create.
The model context holds at most 192 images. S1/S2 have no separate quota for newly inspected images or cumulative audio use. S3/S4E allow at most 96 additional images, 180 requested seconds of audio and 12 audio requests.
Each call allows at most 32 pre-extracted frames, 8 workspace-generated images or 30 seconds of audio. You may choose and repeatedly inspect any segment. Context and turn limits still apply.
Media decoding runs in an isolated CPU sandbox without network access.

**Frames are images: call view_image or inspect_video to see them.** Do not infer visual content from the task description alone; this task requires inspecting the footage. To inspect a clip, pass several of its frames together to view_image.

Once your answer has been written and no more tools are needed, provide your final response without a tool-call block."""
