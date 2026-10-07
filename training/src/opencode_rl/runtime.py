"""One hermetic OpenCode configuration shared verbatim by train and eval."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path

from .contract import ID, ParityError, canonical, digest
from .store import atomic_bytes


def configuration(*, context, output, temperature, top_p, thinking, max_turns=100, music=False,
                  model_name="qwen35-9b-policy", reasoning_effort=None):
    from .music_observation import policy_sha256
    if not isinstance(model_name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", model_name):
        raise ParityError("Invalid served model identity")
    if reasoning_effort is not None and (model_name != 'gpt-5.5' or reasoning_effort != 'xhigh'):
        raise ParityError('Unreviewed explicit reasoning effort')
    return {
        "$schema": "https://opencode.ai/config.json", "autoupdate": False,
        "share": "disabled", "enabled_providers": ["ocrl"],
        "model": f"ocrl/{model_name}", "small_model": f"ocrl/{model_name}", "default_agent": "build",
        "plugin": ["file:///opt/opencode_rl/provenance_plugin.mjs", "file:///opt/opencode_rl/media_gate_plugin.mjs"],
        "provider": {"ocrl": {"npm": "@ai-sdk/openai-compatible",
            # Native OpenCode's timeout is a TOTAL request deadline. Long
            # streaming reasoning must not be cancelled while data arrives.
            # chunkTimeout detects stalled streams; the rollout still has its
            # independent whole-trajectory deadline in task_rollout.py.
            "options": {"baseURL": "http://127.0.0.1:18080/v1", "apiKey": "local-relay-only",
                        "timeout": False, "chunkTimeout": 600000},
            "models": {model_name: {"attachment": True, "tool_call": True, "reasoning": thinking,
                **({'options': {'reasoningEffort': reasoning_effort}} if reasoning_effort else {}),
                "temperature": True, "interleaved": {"field": "reasoning_content"},
                "modalities": {"input": ["text", "image"], "output": ["text"]},
                "limit": {"context": context, "output": output}}}}},
        "agent": {agent: {"temperature": temperature, "top_p": top_p,
                          **({"steps": max_turns} if agent == "build" else {})}
                  for agent in ("build", "compaction", "summary", "title")},
        "permission": {"*": "allow", "webfetch": "deny", "websearch": "deny", "task": "deny",
                       "question": "deny", "external_directory": "deny", "repo_clone": "deny"},
        "tools": {"plan_enter": False, "plan_exit": False},
        "mcp": {"s12": {"type": "local", "enabled": True, "timeout": 150000,
            "command": ["/usr/local/bin/python", "-m", "opencode_rl.mcp_tools"],
            "environment": {"PYTHONPATH": "/opt", **({"OCRL_MUSIC_POLICY_SHA256": policy_sha256()} if music else {})}}},
        "compaction": {"auto": True, "prune": True, "tail_turns": 2},
    }


def verify_binary(path, contract):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != contract.binary_sha256:
        raise ParityError("OpenCode binary changed")


def prepare_runtime(directory, config, contract, *, attempt_id, stage="s_1", music_catalog=None):
    """Only used inside a fresh isolated Sandbox, never on the user's host."""
    root = Path(directory)
    if not ID.fullmatch(attempt_id) or stage not in ("s_1", "s_2", "s_3", "s_4e"):
        raise ParityError("Invalid attempt/stage")
    if root.exists():
        raise ParityError("Runtime directory must be new for each attempt")
    if digest(config) != contract.config_sha256:
        raise ParityError("Runtime config differs from training contract")
    music_env = {}
    music_policy = config.get("mcp", {}).get("s12", {}).get("environment", {}).get("OCRL_MUSIC_POLICY_SHA256")
    if music_policy:
        from .music_observation import policy_sha256, validate_catalog
        if music_policy != policy_sha256() or music_catalog is None:
            raise ParityError("Pinned music tool requires a matching prepared catalogue")
        music_catalog = validate_catalog(music_catalog)
        music_env = {"OCRL_MUSIC_POLICY_SHA256": music_policy,
                     "OCRL_MUSIC_CATALOG": str(root / "music_catalog.json"),
                     "OCRL_MUSIC_CATALOG_SHA256": digest(music_catalog)}
    elif music_catalog is not None:
        raise ParityError("Cannot enable music outside the pinned config")
    root.mkdir(parents=True, mode=0o700)
    for name in ("home", "config", "data", "cache", "state"):
        (root / name).mkdir(mode=0o700)
    (root / "config" / "opencode").mkdir(mode=0o555)
    config_path = root / "opencode.json"
    atomic_bytes(config_path, canonical(config))
    if music_catalog is not None:
        catalog_path = root / "music_catalog.json"
        atomic_bytes(catalog_path, canonical(music_catalog))
        catalog_path.chmod(0o444)
    # Deliberately build a new environment; do not mutate process-global env.
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(root / "home"), "USER": "rollout",
        "XDG_CONFIG_HOME": str(root / "config"), "XDG_DATA_HOME": str(root / "data"),
        "XDG_CACHE_HOME": str(root / "cache"), "OPENCODE_CONFIG": str(config_path),
        "XDG_STATE_HOME": str(root / "state"),
        "OCRL_CONTRACT_SHA256": contract.sha256, "OCRL_ATTEMPT_ID": attempt_id,
        "S12_STAGE": stage,
        **music_env,
        "OPENCODE_DISABLE_PROJECT_CONFIG": "true", "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_AUTOUPDATE": "true", "OPENCODE_DISABLE_TERMINAL_TITLE": "true",
        "OPENCODE_EXPERIMENTAL_NATIVE_LLM": "false", "OPENCODE_EXPERIMENTAL": "false",
        "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    }


def task_prompt(original):
    """Identical offline/media instructions for both train and evaluation."""
    return original + "\n\n" + (
        "本任务运行在 OpenCode 隔离环境中。任务要求和输出文件格式不变。\n"
        "环境不能联网。禁止访问互联网、在线搜索、下载素材或模型、安装在线依赖；"
        "不要尝试通过浏览工具、curl/wget、pip/npm、git 或其他程序联网。"
        "仅使用已提供的本地素材、已安装工具、离线模型和预先生成的音乐听感标签完成任务。\n"
        "原说明的 view_image、inspect_video、inspect_audio 分别对应实际工具 "
        "s12_view_image、s12_inspect_video、s12_inspect_audio。\n"
        "s12_view_image 可查看预抽帧及自制图片；s12_inspect_video 可查看源 clip 或成品的"
        "指定时间点/区间，返回抽样帧而不是连续播放；s12_inspect_audio 可检查源 clip、音乐、"
        "音轨或成品的时长、音量、静音情况及可选离线英文对白转录。"
        "音频工具返回分析文本，不提供直接听音能力；转录可能漏字，无转录不等于无对白或无音乐。"
        "各工具的单次和累计限制以工具说明为准。\n"
        "新增 s12_contact_sheet：把最多 16 张图片或不同 clip 的指定帧拼成一张带编号/时间的"
        "联系表并直接返回图片；先概览，再按单格来源放大。s12_inspect_frames：按帧号、时间，"
        "或 center_s 加 before/after 查看切点前后连续帧；layout=individual 可高清查看，"
        "crop=[x,y,w,h] 是源图上的归一化裁剪框。返回实际解码时间与 PTS，不是连续播放。\n"
        "原生 read 看图和媒体工具的图片都经过同一尺寸检查；所有 stage 均不设累计新增图片配额，"
        "仍记录视觉用量。至少每 448×448 像素算一个单位，官方拼图还按源帧数量计；单图最长边 1792。"
        "拼成一张不等于只消耗一张小图的视觉 token，密集拼图应分组或放大细看。"
        "单次模型请求含保留的历史图片最多 999 张；无独立总像素额度，仍受模型总 token 上下文和传输大小上限约束；"
        "图片及保留的历史内容都占用 token，大量高清图片会增加推理时间与显存需求。\n"
        "s12_analyze_audio 提供离线响度、真实峰值、逐声道音量/静音和节拍候选，支持选择音轨；"
        "候选节拍不是确定节拍，不提供语义代答。s12_validate_media 检查自己的 JSON 引用/时间范围、"
        "成品技术参数和解码，并报告黑屏/冻结/静音候选。候选可能是有意的剪辑，不等于错误或评分；"
        "它不读取隐藏答案，也不证明渲染和 JSON 的内容完全一致。\n"
        "若启用了 s12_listen_music，可先用空参数列出本题已登记的原始音乐，再按 music_id 和 "
        "window_index 查询固定 15 秒窗口的乐器、音乐情绪、节奏与能量标签。"
        "它只查预先生成的只读标签库，不在线调用 Gemini；不能提交问题、音频、成品、"
        "歌词或转录请求，不提供剪辑建议、答案或评分。标签是估计听感，不是精确声学测量。\n"
        "bash 使用 OpenCode 原生 bash 工具及其参数；可用 FFmpeg、ffprobe 和 Python "
        "进行本地剪切、拼接、变速、缩放、音轨提取、混音、音量调整及淡入淡出，"
        "并检查媒体编码、尺寸、帧率、时长和音轨信息。\n"
        "完整源素材在 /work/initial_state，预抽帧在 /work/frames，答案写到题目指定位置。"
    )
