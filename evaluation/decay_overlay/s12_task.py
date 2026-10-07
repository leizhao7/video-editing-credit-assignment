"""The S1/S2 harness, in one place.

The teacher run and the student eval have to present a task identically or the
comparison measures the prompt rather than the model. Frame extraction, the
appended instructions and the grader call therefore live here, and both callers
import them instead of carrying their own copy.

Nothing here is Modal-specific — it is plain filesystem work, so it can be
exercised locally before anything is launched.
"""
import fcntl
import threading
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 每个 clip 抽几帧。s_2 的候选少但要判顺序，多给一帧看动作的接续。
# s_3/s_4e 的产出名取自各任务 task.json 的 deliverables，这里是兜底默认值。
FRAMES_PER_CLIP = {"s_1": 2, "s_2": 3, "s_3": 3, "s_4e": 3}
DELIVERABLE = {"s_1": "selection.json", "s_2": "order.json",
               "s_3": "cuts.json", "s_4e": "edit.json"}
# 判分方式：s_1/s_2 有 verifier/grade.py 直接算分；s_3/s_4e 没有 grade.py，
# 要先跑 evidence 脚本（ffmpeg 渲染 + 联络表）再让配置的 LLM 按 rubric 打分。
# 当前 RL/SFT launcher 显式选择 Gemini 3.1 Pro；Azure gpt-5.5 只是兼容后端。
# 完整评分链是**四步**，不是两步：
#   evidence  → 机械证据
#   S3: 0.2×blind/non-reference + 0.4×referenced
#       + 0.4×task-verifier deterministic golden distance
#   S4E: 保留 0.6×blind + 0.4×referenced
#
# Aggregate all required components; a blind score alone is not the final reward.
# The referenced judgement uses the reference edit as supporting evidence.
JUDGED = {"s_3":  ("s3_evidence.py",  "s3_judge.py",  "s3_judge_ref.py",
                   "s3_score.py",  "s3_evidence.json",  "s3_final.json"),
          "s_4e": ("s4e_evidence.py", "s4e_judge.py", "s4e_judge_ref.py",
                   "s4e_score.py", "s4e_evidence.json", "s4e_final.json")}
FRAME_WIDTH = 448


def out_name(stage: str) -> str:
    return DELIVERABLE[stage]


def validate_deliverable_data(root, stage: str) -> None:
    """Validate a parsed policy deliverable before a trusted grader reads it.

    This contract intentionally lives in the shared task harness.  Local RL,
    validation, teacher rollouts, and the detached remote grader must reject the
    same malformed/resource-bomb payloads instead of each transport carrying a
    subtly different copy of the schema.
    """
    nodes = 0

    def walk(value, depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > 20_000:
            raise ValueError("JSON 节点超过 20000")
        if depth > 32:
            raise ValueError("JSON 嵌套超过 32 层")
        if isinstance(value, dict):
            if len(value) > 2_000:
                raise ValueError("单个 JSON object 字段过多")
            for key, child in value.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise ValueError("JSON key 非字符串或超过 256 字符")
                walk(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 2_000:
                raise ValueError("单个 JSON array 项过多")
            for child in value:
                walk(child, depth + 1)
        elif isinstance(value, str):
            if len(value) > 65_536:
                raise ValueError("JSON 字符串超过 65536 字符")
        elif isinstance(value, float):
            if not math.isfinite(value) or abs(value) > 1e6:
                raise ValueError(f"JSON 数值越界: {value!r}")
        elif isinstance(value, int) and not isinstance(value, bool):
            if abs(value) > 1_000_000:
                raise ValueError(f"JSON 整数越界: {value!r}")
        elif value is not None and not isinstance(value, bool):
            raise ValueError(f"JSON 包含不支持的类型: {type(value).__name__}")

    walk(root)
    if not isinstance(root, dict):
        raise ValueError("交付物顶层必须是 JSON object")

    if stage == "s_1":
        selected = root.get("selected")
        if not isinstance(selected, list) or not all(
                isinstance(item, str) and 0 < len(item) <= 128 for item in selected):
            raise ValueError("s_1 交付物必须包含字符串数组 selected")
    elif stage == "s_2":
        order = root.get("order")
        if not isinstance(order, list) or not all(
                isinstance(item, str) and 0 < len(item) <= 128 for item in order):
            raise ValueError("s_2 交付物必须包含字符串数组 order")
    elif stage == "s_3":
        cuts = root.get("cuts")
        if not isinstance(cuts, list) or len(cuts) > 512:
            raise ValueError("s_3 交付物必须包含至多 512 项的 cuts")
        for cut in cuts:
            if not isinstance(cut, dict) or not isinstance(cut.get("clip"), str):
                raise ValueError("s_3 的每项 cut 必须包含字符串 clip")
            start, end = cut.get("in_s"), cut.get("out_s")
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float))
                    or not isinstance(end, (int, float))
                    or not math.isfinite(float(start))
                    or not math.isfinite(float(end))
                    or start < 0 or end <= start or end > 1_000_000):
                raise ValueError("s_3 的 in_s/out_s 必须是有限且递增的非负秒数")
    elif stage == "s_4e":
        for key in ("dialogue", "joins", "production_sound", "cards"):
            if key in root and (not isinstance(root[key], list) or len(root[key]) > 512):
                raise ValueError(f"s_4e 的 {key} 必须是数组")
        music = root.get("music")
        if music is not None and not isinstance(music, dict):
            raise ValueError("s_4e 的 music 必须是 object")

        def number(row, key, *, minimum=None, required=True):
            value = row.get(key)
            if value is None and not required:
                return None
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise ValueError(f"s_4e 的 {key} 必须是有限数")
            value = float(value)
            if minimum is not None and value < minimum:
                raise ValueError(f"s_4e 的 {key} 必须 >= {minimum}")
            return value

        if music is not None and music.get("starts_at_s") is not None:
            number(music, "starts_at_s", minimum=0)
        for line in root.get("dialogue") or []:
            if not isinstance(line, dict) or not isinstance(line.get("take"), str):
                raise ValueError("s_4e dialogue 每项必须含字符串 take")
            start = number(line, "in_s", minimum=0)
            end = number(line, "out_s", minimum=0)
            number(line, "at_s", minimum=-60)
            if end <= start:
                raise ValueError("s_4e dialogue 的 out_s 必须大于 in_s")
            if line.get("gain_db") is not None:
                number(line, "gain_db", minimum=-120)
        seen_joins = set()
        for join in root.get("joins") or []:
            if not isinstance(join, dict):
                raise ValueError("s_4e joins 每项必须是 object")
            index = join.get("join")
            if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                raise ValueError("s_4e join 必须是正整数")
            if index in seen_joins:
                raise ValueError(f"s_4e join 重复: {index}")
            seen_joins.add(index)
            if join.get("picture") not in ("cut", "dip_to_black", "dissolve", "other"):
                raise ValueError(f"s_4e join {index} 的 picture 无效")
            if join.get("sound") not in ("hard", "carries_over", "prelap", "under"):
                raise ValueError(f"s_4e join {index} 的 sound 无效")
            for field in ("black_hold_s", "fade_in_s", "fade_out_s", "dur_s"):
                if join.get(field) is not None:
                    number(join, field, minimum=0)
        for sound in root.get("production_sound") or []:
            if not isinstance(sound, dict) or not isinstance(sound.get("shot"), str):
                raise ValueError("s_4e production_sound 每项必须含字符串 shot")
            timed = [sound.get(k) is not None for k in ("in_s", "out_s", "at_s")]
            if any(timed) and not all(timed):
                raise ValueError("s_4e production_sound 的 in_s/out_s/at_s 必须一起出现")
            if all(timed):
                start = number(sound, "in_s", minimum=0)
                end = number(sound, "out_s", minimum=0)
                number(sound, "at_s", minimum=-60)
                if end <= start:
                    raise ValueError("s_4e production_sound 的 out_s 必须大于 in_s")
            if sound.get("gain_db") is not None:
                number(sound, "gain_db", minimum=-120)
        for card in root.get("cards") or []:
            if not isinstance(card, dict) or not isinstance(card.get("shot"), str):
                raise ValueError("s_4e cards 每项必须含字符串 shot")
    else:
        raise ValueError(f"不支持的 stage: {stage!r}")


def load_and_validate_deliverable(path: str, stage: str):
    """Parse strict JSON and apply :func:`validate_deliverable_data`."""
    def reject_constant(value: str):
        raise ValueError(f"非有限 JSON 数值 {value}")

    with open(path, encoding="utf-8") as fh:
        root = json.load(fh, parse_constant=reject_constant)
    validate_deliverable_data(root, stage)
    return root


def n_frames(stage: str) -> int:
    return FRAMES_PER_CLIP.get(stage, 2)


def gold_count(task_dir: str) -> int:
    """s_1 的答案数。取不到就返回 0，调用方据此决定告不告知。"""
    try:
        with open(f"{task_dir}/verifier/answer.json", encoding="utf-8") as fh:
            return len(json.load(fh)["answer"])
    except Exception:
        return 0


def visual_items(task_dir: str, stage: str, manifest: dict) -> list:
    """Return the video-backed items whose frames the policy may inspect.

    S1 stores its media under ``candidates`` and S2/S3 under ``clips``.  S4E is
    structurally different: its rough-cut picture is described by ``shots`` and
    the entries intentionally omit ``file`` because the stable convention is
    ``shots/<shot_id>.mp4``.  Card shots marked ``build_it`` have no source file
    by design and must not be mistaken for missing media.

    Keeping this mapping in the shared task module is important: RL, validation
    and GPT teacher rollouts all call :func:`prepare`, so a transport-specific
    workaround would make their agent loops incomparable again.
    """
    if stage == "s_1":
        raw = manifest.get("candidates")
    elif stage in ("s_2", "s_3"):
        raw = manifest.get("clips")
    elif stage == "s_4e":
        raw = manifest.get("shots")
    else:
        raise ValueError(f"不支持的 stage: {stage!r}")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{stage} manifest 缺少非空素材数组")

    out, seen = [], set()
    for index, original in enumerate(raw):
        if not isinstance(original, dict):
            raise ValueError(f"{stage} 第 {index} 个素材不是 object")
        item = dict(original)
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{stage} 第 {index} 个素材缺少 id")
        if item_id in seen:
            raise ValueError(f"{stage} manifest 含重复素材 id: {item_id}")
        seen.add(item_id)

        if stage == "s_4e" and not item.get("file"):
            item["file"] = f"shots/{item_id}.mp4"
        rel = item.get("file")
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"{stage} 素材 {item_id} 缺少 file")
        path = os.path.join(task_dir, "initial_state", rel)
        if os.path.isfile(path):
            out.append(item)
            continue

        # S4E title-card placeholders are authored by the policy and therefore
        # legitimately have no input MP4.  Every other manifest item promises a
        # viewable video; fail closed instead of silently telling the model that
        # zero/fewer frames were prepared.
        planned_card = (
            stage == "s_4e"
            and (item.get("build_it") is True
                 or item.get("kind") == "card"
                 or item.get("origin") == "to_build")
        )
        if not planned_card:
            raise FileNotFoundError(f"{stage} 素材文件不存在: {rel}")
    if not out:
        raise ValueError(f"{stage} 没有任何可看的视频素材")
    return out


def _duration(item: dict, probed: str) -> tuple:
    """Choose a finite duration without confusing S3's ``clip_dur_s`` field."""
    candidates = (probed, item.get("dur_s"), item.get("clip_dur_s"))
    for index, raw in enumerate(candidates):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            return value, ("ffprobe" if index == 0 else "manifest")
    raise ValueError(
        f"素材 {item.get('id')} 没有有限正 duration: "
        f"ffprobe={probed!r} manifest="
        f"{item.get('dur_s', item.get('clip_dur_s'))!r}"
    )


def _frame_ok(path: str) -> bool:
    """Return whether ffmpeg produced a non-empty frame artifact."""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _run_bounded(args: list, *, timeout: float, **kwargs):
    """Run a child process with a synthetic rc=124 timeout result."""
    try:
        completed = subprocess.run(args, timeout=timeout, **kwargs)
        # Modal stops a container with SIGINT.  If ffmpeg is the foreground
        # child it can receive that signal instead of Python; propagating it
        # here prevents the rollout/audit from swallowing cancellation and
        # continuing through thousands of fallback attempts.
        if completed.returncode in (-signal.SIGINT, -signal.SIGTERM):
            raise KeyboardInterrupt(
                f"media subprocess interrupted by signal {-completed.returncode}"
            )
        return completed
    except subprocess.TimeoutExpired as exc:
        textual = bool(kwargs.get("text") or kwargs.get("universal_newlines"))
        empty = "" if textual else b""
        suffix = f"media subprocess timed out after {timeout:.0f}s"
        if textual:
            stdout = exc.stdout or empty
            stderr = exc.stderr or empty
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            stderr = stderr + "\n" + suffix
        else:
            stdout = exc.stdout or empty
            stderr = exc.stderr or empty
            if isinstance(stdout, str):
                stdout = stdout.encode()
            if isinstance(stderr, str):
                stderr = stderr.encode()
            stderr = stderr + b"\n" + suffix.encode()
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=stdout,
            stderr=stderr,
        )


def _decoded_frame_count(src: str) -> int:
    """Count decoded video frames for the rare timestamp-seek fallback.

    Container duration can extend past the presentation timestamp of the last
    frame.  On very short two-frame clips, seeking to an otherwise valid
    midpoint then exits successfully without producing an image.  Counting is
    deliberately lazy: normal clips retain the fast timestamp path, while a
    boundary clip pays for one full decode and is sampled by frame ordinal.
    """
    counted = _run_bounded(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", src],
        timeout=120, capture_output=True, text=True,
    )
    raw = counted.stdout.strip().splitlines()
    try:
        value = int(raw[0]) if raw else 0
    except (TypeError, ValueError):
        value = 0
    if counted.returncode != 0 or value <= 0:
        detail = (counted.stderr or counted.stdout or "")[-300:]
        raise RuntimeError(
            f"视频没有可解码画面: file={os.path.basename(src)!r} "
            f"rc={counted.returncode} frames={raw!r}: {detail}"
        )
    return value


def _frame_error_detail(rendered) -> str:
    detail = (rendered.stderr or rendered.stdout or b"")[-300:]
    if isinstance(detail, bytes):
        detail = detail.decode("utf-8", "replace")
    return detail


def extract_frames(task_dir: str, items: list, nf: int) -> list:
    """容器统一抽帧，不让 agent 自己抽。

    前两次都死在同一个地方：agent 给每个 clip 抽了十几帧，然后一句
    "let me view the detailed frames for each clip" 把七八十张图一起塞进
    上下文，请求在 90k–120k token 处断流。用提示词劝它克制没有用 ——
    把能力拿掉才有用。
    """
    fdir = f"{task_dir}/frames"
    shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir, exist_ok=True)
    made = []
    for it in items:
        src = f'{task_dir}/initial_state/{it["file"]}'
        if not os.path.isfile(src):
            raise FileNotFoundError(f"素材文件不存在: {it['file']}")
        probe_result = _run_bounded(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", src],
            timeout=30, capture_output=True, text=True,
        )
        probe = probe_result.stdout.strip()
        dur, duration_source = _duration(it, probe)
        if duration_source != "ffprobe":
            print(
                f"[DURATION_FALLBACK] clip={it.get('id')} ffprobe={probe!r} "
                f"manifest={it.get('dur_s', it.get('clip_dur_s'))!r} "
                f"selected={dur} source={duration_source}",
                flush=True,
            )
        decoded_frames = None
        for k in range(nf):
            t = dur * (k + 0.5) / nf
            out = f'{fdir}/{it["id"]}_{k + 1}.jpg'
            # Never let a partial file from a failed encoder attempt satisfy
            # the artifact check below.
            try:
                os.unlink(out)
            except FileNotFoundError:
                pass
            rendered = _run_bounded(
                ["ffmpeg", "-v", "error", "-threads", "1",
                 "-filter_threads", "1", "-ss", f"{t:.3f}", "-i", src,
                 "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "5",
                 "-y", out], timeout=60, capture_output=True)
            primary_detail = _frame_error_detail(rendered)
            if rendered.returncode != 0 or not _frame_ok(out):
                try:
                    os.unlink(out)
                except FileNotFoundError:
                    pass
                if decoded_frames is None:
                    decoded_frames = _decoded_frame_count(src)
                # Sample the same midpoint fraction in decoded-frame space.
                # Repeating a frame is intentional when nf exceeds the number
                # of frames (for example 3 requested views of a 2-frame shot).
                frame_index = min(
                    decoded_frames - 1,
                    int(decoded_frames * (k + 0.5) / nf),
                )
                fallback = _run_bounded(
                    ["ffmpeg", "-v", "error", "-threads", "1",
                     "-filter_threads", "1", "-i", src,
                     "-vf", (f"select=eq(n\\,{frame_index}),"
                             f"scale={FRAME_WIDTH}:-2"),
                     "-frames:v", "1", "-q:v", "5", "-y", out],
                    timeout=120, capture_output=True,
                )
                if fallback.returncode == 0 and _frame_ok(out):
                    print(
                        f"[FRAME_INDEX_FALLBACK] clip={it.get('id')} "
                        f"t={t:.3f}s index={frame_index}/"
                        f"{decoded_frames} primary_rc={rendered.returncode}",
                        flush=True,
                    )
                else:
                    fallback_detail = _frame_error_detail(fallback)
                    raise RuntimeError(
                        f"抽帧失败: stage item={it.get('id')} t={t:.3f}s "
                        f"primary_rc={rendered.returncode} "
                        f"primary={primary_detail!r}; "
                        f"fallback_index={frame_index}/{decoded_frames} "
                        f"fallback_rc={fallback.returncode}: {fallback_detail}"
                    )
            if not _frame_ok(out):
                raise RuntimeError(
                    f"抽帧失败: stage item={it.get('id')} t={t:.3f}s "
                    f"生成了空文件"
                )
            made.append(os.path.basename(out))
    return made


def build_prompt(task_dir: str, stage: str, nf: int, n_made: int,
                 n_gold: int = 0) -> str:
    """task_prompt.md 加上这套 harness 的具体交代。

    告知数量时，原文里那条"数量不给"的要求必须先拿掉 —— 留着就是两句
    互相打架的指令，模型要么忽略其中一句，要么在答案里纠结数量该不该自己定。
    """
    with open(f"{task_dir}/task_prompt.md", encoding="utf-8") as fh:
        base = fh.read()
    if n_gold:
        base = re.sub(
            r"^[-*]\s*The count is not given\..*?\n(?=[-*]|\n|\Z)",
            "", base, flags=re.S | re.M).rstrip() + "\n"

    item_word = "shot" if stage == "s_4e" else "clip"
    frame_names = (f"`<{item_word}_id>_1.jpg` through `_{nf}.jpg`" if nf > 1
                   else f"`<{item_word}_id>_1.jpg`")
    stage_note = (
        "For S4E, every supplied rough-cut shot has frames here. Shots marked "
        "as cards to build have no source MP4 and therefore no input frame. "
        "Inspect the supplied shot frames before deciding joins and sound.\n\n"
        if stage == "s_4e" else ""
    )
    media_note = (
        "The source shots and audio under `initial_state/` are available because "
        "you need them to build `output/preview.mp4`. Use `inspect_video` on "
        "source shots or your rendered preview to check joins, cards and visual edits; "
        "`view_image` can also show images you create inside the workspace. "
        "Use `inspect_audio` on dialogue takes and the rendered preview to check "
        "actual sound levels, silence and approximate English ASR. ASR timestamps "
        "are relative to the inspected file; missing ASR is not proof of no speech. "
        "Manifest speech spans are hints: inspect real file duration before using them. "
        "The final preview duration must remain within ±2 seconds of the supplied "
        "rough-cut stretch duration, including any black/card time you add.\n\n"
        if stage == "s_4e" else
        "Source clips under `initial_state/` are available. The prepared frames "
        "are an overview, not sufficient to locate precise cut points. Use "
        "`inspect_video(path, timestamps_s)` around candidate in/out points, for "
        "example [1.8, 2.0, 2.2]. Times are seconds within that clip's file. "
        "Start coarse, then refine only promising boundaries; never request every frame. "
        "`inspect_audio` is available if audio helps locate a cue.\n\n"
        if stage == "s_3" else
        "Full original candidate clip files listed in `initial_state/manifest.json` "
        "are available under `initial_state/`, unchanged and untrimmed. The prepared "
        "frames are optional thumbnails, NOT your complete visual interface. You "
        "may inspect any clip, any time range, and revisit moments as needed. "
        "`inspect_video` with only `path` gives 8 samples across the whole clip; "
        "use `timestamps_s` for chosen moments, OR `start_s`, `end_s`, `num_frames` "
        "to inspect a chosen range at your chosen sampling density. Repeat for "
        "more detail. The observation is sampled frames, not continuous playback. "
        "Use `bash` with ffprobe/FFmpeg/Python for metadata, custom frame extraction, "
        "contact sheets, crops and waveforms; show your generated images with "
        "`view_image`. `inspect_audio` checks source audio windows and optional "
        "English ASR (when the file has audio). Choose your own inspection order "
        "and how much evidence each clip needs before selecting/ordering.\n\n"
    )
    return base + (
        f"\n\n---\n\nWork in `{task_dir}`. "
        f"`initial_state/manifest.json` lists every {item_word}'s id and timing.\n\n"
        f"**The frames are already extracted.** `frames/` holds {nf} evenly spaced "
        f"frames per supplied {item_word}, named {frame_names} "
        f"({n_made} prepared in total). Read them with your vision "
        f"directly.\n\n"
        + stage_note
        + ("Choose which clips and moments to inspect yourself. You are not required "
           "to follow a fixed frame-viewing order or rely on the thumbnails.\n\n"
           if stage in ("s_1", "s_2") else
           f"**Look at one {item_word}'s {nf} frames at a time.** Write your read of that {item_word} "
        f"in a sentence or two before moving to the next one. Do not pull several "
        f"{item_word}s' frames into one look — the context blows up and the request dies "
        f"mid-stream.\n\n")
        + media_note
        + ("Inspection budgets per trajectory: 192 images total, including at most "
           "96 newly inspected/generated images (repeated views count); each video "
           "or generated-image request takes at most 8 images resized to a 448-pixel "
           "long edge. Audio: at most 30 seconds per request, 180 requested seconds "
           "and 12 requests total, including failed requests. For volume-only checks "
           "set `transcribe=false`. Run video editing with `bash`/FFmpeg in the sandbox; "
           "no network is available. Long shell output is summarized without stopping "
           "the command; use the reported `tool_logs/*.log` path to read selected lines.\n\n"
           if stage in ("s_3", "s_4e") else
           "There is NO separate cumulative quota for additional frames or audio "
           "inspection in S1/S2. The current model context can hold 192 images in "
           "total (repeat views count); normal context-token and turn limits still "
           "apply. Per-call limits: inspect_video/generated images 8 images, "
           "prepared images 32, image long edge 448 pixels for new observations, "
           "audio 30 seconds. You can inspect longer audio with consecutive windows. "
           "For volume-only checks set `transcribe=false`. All media inspection "
           "runs in the isolated CPU sandbox with no network. Long shell output "
           "is summarized without stopping the command; use the returned log path "
           "to read additional lines.\n\n")
        + (f"**This stretch uses exactly {n_gold} shots. Pick {n_gold}.**\n\n"
           if n_gold else "")
        + f"Write your answer to `output/{out_name(stage)}`.")


# judge 脚本所在目录。镜像里放在 /opt/s12tools，本地开发时退回仓库根目录。
TOOLS = os.environ.get("S12_TOOLS_DIR", os.path.dirname(os.path.abspath(__file__)))

# 同时进行判分的 rollout 上限。每条 s_3/s_4e judge chain 的 blind/ref
# 脚本还会各自并发 3~4 条 rubric 请求，因此这里限制的是 chain，不是单请求。
# 远端 API 请求集中在生成器主进程内，线程闸门即可限制下游请求雪崩；本地
# evidence 则会跨 Ray 进程运行，下面另用真正的跨进程闸门。
_JUDGE_SLOTS = int(os.environ.get("S12_JUDGE_CONCURRENCY", "8"))
_JUDGE_GATE = threading.BoundedSemaphore(_JUDGE_SLOTS)
# Evidence generation and first-time frame extraction are CPU/ffmpeg-heavy.
# S12Env deliberately shares this host-wide gate for uncached frame extraction;
# cache hits do not acquire it.
_EVIDENCE_SLOTS = int(os.environ.get("S12_EVIDENCE_CONCURRENCY", "8"))


class _ProcessSharedSemaphore:
    """A small host-local semaphore shared by all Ray worker processes.

    ``threading.BoundedSemaphore`` only coordinates callers which imported
    this module in the same process.  The rollout generator runs in several
    Ray processes, so the old evidence gate admitted ``slots`` callers *per
    process*.  FFmpeg then auto-created decoder/filter threads for every one
    of those callers and the 64-vCPU node entered a positive-feedback loop:
    slower evidence retained its CPUs longer, which made still more evidence
    hit the 900-second timeout.

    Each slot is an advisory ``flock`` on a file in the container-local /tmp.
    Locks are released by the kernel if a worker dies, and stale slot files do
    not retain state.  The acquired slot number is exposed so the evidence
    child can bind itself (and therefore its FFmpeg descendants) to a small,
    disjoint CPU set.
    """

    def __init__(self, value: int, directory: str):
        if int(value) < 1:
            raise ValueError("process-shared semaphore value must be positive")
        self._value = int(value)
        self._directory = directory
        os.makedirs(self._directory, mode=0o700, exist_ok=True)
        self._held = threading.local()

    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        if not blocking and timeout is not None:
            raise ValueError("can't specify a timeout for a non-blocking acquire")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        # Different processes/threads begin at different slots to avoid making
        # slot 0 a lock-contention hotspot.
        start = (os.getpid() + threading.get_ident()) % self._value
        while True:
            for offset in range(self._value):
                slot = (start + offset) % self._value
                path = os.path.join(self._directory, f"slot-{slot:03d}.lock")
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                stack = getattr(self._held, "stack", None)
                if stack is None:
                    stack = self._held.stack = []
                stack.append((slot, fd))
                return True
            if not blocking:
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def release(self) -> None:
        stack = getattr(self._held, "stack", None)
        if not stack:
            raise ValueError("process-shared semaphore released too many times")
        _, fd = stack.pop()
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def current_slot(self):
        stack = getattr(self._held, "stack", None)
        return int(stack[-1][0]) if stack else None


_EVIDENCE_GATE = _ProcessSharedSemaphore(
    _EVIDENCE_SLOTS,
    os.environ.get("S12_EVIDENCE_GATE_DIR", "/tmp/s12-evidence-gate-v2"),
)


def _grade_judged(task_dir: str, stage: str) -> tuple:
    """s_3/s_4e：没有 grade.py，靠 rubric + LLM judge 打分。

    四步，缺一不可（见 JUDGED 上面那段）：
      1) evidence  读 agent 的产出，跑 ffmpeg 渲染成片/联络表/黑段检测
      2) judge     blind：按 rubric 各打一次，没看过成片
      3) judge_ref referenced：同样的问题，但成片在手边
      4) score     S3 再加入 task verifier 的确定性 golden 距离分；
                   S4E 保持原有 0.6×blind + 0.4×ref

    每条 rollout 都要走一遍，所以这是**判分侧的主要开销**（ffmpeg + 两轮各
    数次带图的 API 调用）。evidence 必须现场生成 —— 它读的是这一条 rollout 刚
    写出来的 output/，不是预置的。
    """
    return _run_judge_chain(task_dir, *JUDGED[stage])


def _run_judge_phase(cmd: list[str], *, phase: str):
    """Run one grading phase behind the resource-appropriate gate.

    Timeouts deliberately become ordinary rc=124 failures.  ``S12Env`` can
    then use its existing grade retry/fail path and save a reward-zero row,
    instead of an uncaught ``TimeoutExpired`` becoming ``trajectory_error``.
    """
    evidence = phase == "evidence"
    gate = _EVIDENCE_GATE if evidence else _JUDGE_GATE
    slots = _EVIDENCE_SLOTS if evidence else _JUDGE_SLOTS
    timeout = float(os.environ.get(
        "S12_EVIDENCE_TIMEOUT_S" if evidence else "S12_JUDGE_TIMEOUT_S",
        "900" if evidence else "1800",
    ))
    wait_started = time.monotonic()
    deadline = os.environ.get('S12_GRADING_DEADLINE_UNIX')
    remaining = float(deadline) - time.time() if deadline else None
    if remaining is not None and (remaining <= 0 or not gate.acquire(timeout=remaining)):
        return subprocess.CompletedProcess(cmd, 124, '', 'Total grading deadline exhausted at phase gate')
    if remaining is None:
        gate.acquire()
    acquired = time.monotonic()
    result = None
    try:
        deadline = os.environ.get('S12_GRADING_DEADLINE_UNIX')
        if deadline:
            remaining = float(deadline) - time.time()
            if remaining <= 0:
                return subprocess.CompletedProcess(cmd, 124, '', 'Total grading deadline exhausted')
            timeout = min(timeout, remaining)
        child_env = None
        if evidence:
            child_env = os.environ.copy()
            # NumPy/librosa must not silently create a BLAS thread pool per
            # evidence worker.  The child also binds itself and its FFmpeg
            # descendants using this globally unique slot id.
            child_env.update({
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "S12_EVIDENCE_SLOT": str(gate.current_slot()),
            })
        result = _run_bounded(
            cmd, capture_output=True, text=True, timeout=timeout,
            **({"env": child_env} if child_env is not None else {}),
        )
        return result
    finally:
        finished = time.monotonic()
        provider = os.environ.get("S12_JUDGE_PROVIDER", "azure").lower()
        model = (os.environ.get("S12_GEMINI_MODEL", "gemini-3.1-pro-preview")
                 if provider == "gemini" else "azure-gpt-5.5")
        print(
            f"[JUDGE_GATE] phase={phase} provider={provider} model={model} "
            f"slots={slots} wait={acquired - wait_started:.2f}s "
            f"held={finished - acquired:.2f}s "
            f"rc={getattr(result, 'returncode', '?')}",
            flush=True,
        )
        gate.release()


def _run_parallel_judge_child(cmd, phase, cancelled, task_dir):
    """Own one new process group; sibling failure cannot leave writers alive."""
    timeout = float(os.environ.get('S12_JUDGE_TIMEOUT_S', '1800'))
    absolute = os.environ.get('S12_GRADING_DEADLINE_UNIX')
    if absolute:
        timeout = min(timeout, float(absolute) - time.time())
    if timeout <= 0 or cancelled.is_set():
        return subprocess.CompletedProcess(cmd, 124 if timeout <= 0 else 125, '',
            'Total grading deadline exhausted' if timeout <= 0 else 'Sibling judge failed')
    started, wall = time.monotonic(), time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    finished = False
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            stopped = cancelled.is_set()
            if stopped or remaining <= 0:
                # Only this newly created, tracked child group, never a trainer
                # or an App. Also release inherited pipes held by grandchildren.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate(timeout=5)
                result = subprocess.CompletedProcess(cmd, 125 if stopped else 124, stdout,
                    stderr + ('\nSibling judge failed; child group stopped' if stopped
                              else '\nTotal/phase grading deadline exhausted'))
                break
            try:
                stdout, stderr = proc.communicate(timeout=min(.1, remaining))
                if proc.returncode in (-signal.SIGINT, -signal.SIGTERM):
                    raise KeyboardInterrupt(f'Judge child interrupted by signal {-proc.returncode}')
                if proc.returncode:
                    # The leader can fail after a descendant closes its inherited
                    # pipes. communicate() then returns even though that process
                    # still runs; clean our owned group before retry/cleanup.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
                break
            except subprocess.TimeoutExpired:
                continue
        finished = True
        # Small auditable timings; exact judge request/response evidence remains
        # in its existing per-rubric files, unchanged by phase scheduling.
        with open(os.path.join(task_dir, 'evidence', phase + '.phase.json'), 'w') as handle:
            schedule = ('s4e-independent-judges-parallel-v1' if phase.startswith('s4e_')
                        else 's3-independent-judges-parallel-v1')
            json.dump({'phase': phase, 'schedule': schedule,
                'started_at': wall, 'seconds': time.monotonic() - started,
                'returncode': result.returncode}, handle)
        return result
    finally:
        if not finished:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate(timeout=5)


def _run_s3_judge_pair(task_dir, grader_python, judge_py, ref_py):
    if (judge_py, ref_py) != ('s3_judge.py', 's3_judge_ref.py'):
        raise ValueError('Only the audited S3 judge pair may run here')
    return _run_independent_judge_pair(task_dir, grader_python, judge_py, ref_py)


def _run_s4e_judge_pair(task_dir, grader_python, judge_py, ref_py):
    if (judge_py, ref_py) != ('s4e_judge.py', 's4e_judge_ref.py'):
        raise ValueError('Only the audited S4E judge pair may run here')
    return _run_independent_judge_pair(task_dir, grader_python, judge_py, ref_py)


def _run_independent_judge_pair(task_dir, grader_python, judge_py, ref_py):
    """One admitted chain, two independent phases, then a mandatory full join.

The pair owns one existing judge-chain slot. Acquiring that same one-slot gate
again in each child would silently serialize the supposedly parallel work.
S4E reads the completed evidence/video/audio only; its four blind and three
reference rubric IDs, audit paths and final score filenames are disjoint.
"""
    if (judge_py, ref_py) not in (('s3_judge.py', 's3_judge_ref.py'),
                                ('s4e_judge.py', 's4e_judge_ref.py')):
        raise ValueError('Only audited same-stage judge pairs may run in parallel')
    deadline = os.environ.get('S12_GRADING_DEADLINE_UNIX')
    remaining = float(deadline) - time.time() if deadline else None
    if remaining is not None and (remaining <= 0 or not _JUDGE_GATE.acquire(timeout=remaining)):
        return judge_py, subprocess.CompletedProcess([], 124, '', 'Total grading deadline exhausted at pair gate')
    if remaining is None:
        _JUDGE_GATE.acquire()
    cancelled, results = threading.Event(), {}
    failure = None
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs = {executor.submit(_run_parallel_judge_child,
                [grader_python, f'{TOOLS}/{script}', '--task', task_dir, '--quiet'],
                script.removesuffix('.py'), cancelled, task_dir): script
                for script in (judge_py, ref_py)}
            try:
                for future in as_completed(jobs):
                    script, result = jobs[future], future.result()
                    results[script] = result
                    if result.returncode and failure is None:
                        failure = (script, result)
                        cancelled.set()
            finally:
                # Executor exit joins both children before task cleanup or a
                # retry may reuse scratch. Exceptions cannot strand a sibling.
                cancelled.set()
        return failure or (ref_py, results[ref_py])
    finally:
        _JUDGE_GATE.release()


def _run_judge_chain(task_dir, ev_py, judge_py, ref_py, score_py,
                     ev_name, final_name, *, reuse_evidence=False) -> tuple:
    from grading_contract import validate_task, InvalidSubmission, VERSION, judge_only
    try:
        stage = "s_3" if ev_py == "s3_evidence.py" else "s_4e"
        validate_task(task_dir, stage)
    except InvalidSubmission as exc:
        detail = {"final": 0.0, "invalid_submission": True,
                  "reason": str(exc), "contract": VERSION}
        os.makedirs(f"{task_dir}/evidence", exist_ok=True)
        with open(f"{task_dir}/evidence/{final_name}", "w") as handle:
            json.dump(detail, handle)
        return 0.0, json.dumps(detail)
    ev_dir = f"{task_dir}/evidence"
    os.makedirs(ev_dir, exist_ok=True)
    from grading_response_reuse import evidence_reusable, mark_evidence_ready
    reuse_evidence = reuse_evidence and evidence_reusable(task_dir, ev_name)
    # Only the same private invocation may reuse byte-verified prepared evidence.
    _blind = final_name.replace("_final", "_score")
    _ref = final_name.replace("_final", "_score_ref")
    _review = final_name.replace("_final", "_consistency")
    stale = (final_name, _blind, _ref, _review)
    if not reuse_evidence:
        stale = (ev_name,) + stale
    for f in stale:
        p = f"{ev_dir}/{f}"
        if os.path.exists(p):
            os.remove(p)
    # Pass only the flags supported by each script. In particular, --quiet
    # belongs to judge/score scripts, not the evidence CLI.
    scripts = (ev_py, judge_py, ref_py, score_py) if judge_only() else (
        ev_py, judge_py, ref_py, "grading_consistency.py", score_py)
    for script in scripts:
        if script == ev_py and reuse_evidence:
            continue
        if script == ref_py:
            continue  # Already completed and joined with the blind judge.
        # The Modal function bootstrap interpreter is not necessarily the
        # project venv.  Let the deployment name the exact grader interpreter;
        # falling back to this process is still correct for local execution.
        grader_python = os.environ.get("S12_GRADER_PYTHON", sys.executable)
        if not os.path.isfile(grader_python) or not os.access(grader_python, os.X_OK):
            return None, f"grader interpreter is not executable: {grader_python!r}"
        cmd = [grader_python, f"{TOOLS}/{script}", "--task", task_dir]
        if script != ev_py:
            cmd.append("--quiet")
        phase = "evidence" if script == ev_py else script.removesuffix(".py")
        if script == judge_py:
            pair = _run_s3_judge_pair if stage == 's_3' else _run_s4e_judge_pair
            failed_script, r = pair(task_dir, grader_python, judge_py, ref_py)
        else:
            failed_script, r = script, _run_judge_phase(cmd, phase=phase)
        if r.returncode != 0:
            return None, f"{failed_script} rc={r.returncode}: " + (
                (r.stderr or r.stdout or "")[-400:])
        if script == ev_py and not judge_only():
            health_error = _validate_evidence_health(task_dir, ev_name)
            if health_error:
                return None, health_error
        if script == ev_py:
            mark_evidence_ready(task_dir, ev_name)
    sp = f"{ev_dir}/{final_name}"
    if not os.path.exists(sp):
        return None, f"合成器没产出 {final_name}"
    try:
        with open(sp, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as e:  # noqa: BLE001
        return None, f"{final_name} 解析失败: {e}"
    # 取 final，不是 score —— 合成器缺任一半就把 final 置 None 并写明 missing，
    # 这是它有意为之，不要绕过去用单边的分。
    sc = d.get("final")
    from grading_contract import valid_score
    return (float(sc) if valid_score(sc) else None), json.dumps(
        {k: d.get(k) for k in (
            "final", "blind", "referenced", "deterministic", "weight_sum",
            "spread", "missing", "consistency_contract", "initial_scores", "judgement_mode",
        )},
        ensure_ascii=False)


def _validate_evidence_health(task_dir: str, ev_name: str):
    """Reject missing/partial evidence before any LLM is allowed to score it."""
    path = os.path.join(task_dir, "evidence", ev_name)
    if not os.path.isfile(path):
        return f"evidence script did not produce {ev_name}"
    try:
        with open(path, encoding="utf-8") as handle:
            evidence = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return f"{ev_name} parse failed: {type(exc).__name__}: {exc}"
    health = evidence.get("measurement_health")
    if not isinstance(health, dict) or health.get("format") != "s12-evidence-health-v3":
        return f"{ev_name} missing s12-evidence-health-v3 contract"
    for component in ("asr", "audio_analysis"):
        status = (health.get(component) or {}).get("status")
        if component == "audio_analysis" and ev_name == "s3_evidence.json":
            continue
        if status != "ok":
            return f"{ev_name} {component} status={status!r}"
    asr = health.get("asr") or {}
    expected_asr = ("s12-asr-v5-render-clock" if ev_name == "s4e_evidence.json"
                    else "s12-asr-v3-vad500-noprev-canned-filter")
    if asr.get("contract") != expected_asr:
        return f"{ev_name} ASR contract is stale: {asr.get('contract')!r}"
    if asr.get("vad_filter") is not True:
        return f"{ev_name} ASR VAD is not enabled"
    if ev_name == "s4e_evidence.json":
        if evidence.get("grading_contract") != "s34-grading-v5":
            return "v5 S4E evidence contract missing"
        audio = evidence.get("render_audio") or {}
        if audio.get("status") not in ("ok", "no_audio_stream"):
            return "v5 render audio unavailable"
        preview = health.get("preview") or {}
        if preview.get("exists") is not True or preview.get("video_decoded") is not True:
            return f"{ev_name} preview was not successfully decoded"
    sheets = evidence.get("contact_sheets") or {}
    for sheet in health.get("required_contact_sheets") or []:
        meta = sheets.get(sheet)
        rel = meta.get("file") if isinstance(meta, dict) else meta
        if not isinstance(rel, str):
            return f"{ev_name} missing required contact sheet {sheet}"
        full = os.path.realpath(os.path.join(task_dir, rel))
        root = os.path.realpath(task_dir)
        if os.path.commonpath((root, full)) != root:
            return f"{ev_name} contact sheet escaped task root: {rel!r}"
        if not os.path.isfile(full) or os.path.getsize(full) <= 0:
            return f"{ev_name} required contact sheet is empty: {sheet}"
    return None


def grade(task_dir: str, stage: str) -> tuple:
    """跑任务自带的 grade.py。返回 (score, 原始输出)。

    s_3/s_4e 没有 grade.py，转交 _grade_judged 走 rubric + LLM judge。
    """
    deliverable = f"{task_dir}/output/{out_name(stage)}"
    if not os.path.exists(deliverable):
        return None, ""
    if stage in JUDGED:
        return _grade_judged(task_dir, stage)
    from score_cli import run_verifier
    score, _, detail = run_verifier(f"{task_dir}/verifier/grade.py", task_dir, allow_legacy=True)
    return score, detail


def prepare(task_dir: str, stage: str, tell_n: bool) -> tuple:
    """抽帧 + 组 prompt。返回 (prompt, n_made, n_gold)。"""
    with open(f"{task_dir}/initial_state/manifest.json", encoding="utf-8") as fh:
        man = json.load(fh)
    items = visual_items(task_dir, stage, man)
    nf = n_frames(stage)
    made = extract_frames(task_dir, items, nf)
    n_gold = gold_count(task_dir) if (stage == "s_1" and tell_n) else 0
    return build_prompt(task_dir, stage, nf, len(made), n_gold), len(made), n_gold
