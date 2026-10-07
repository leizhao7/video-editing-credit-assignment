"""Independent Gemini 3.1 Pro continuous-video grader.
Every rubric receives task text, delivered video and its navigational evidence.
No audio/image/text-only downgrade; API failures remain unscored, not policy zero.
Legacy Azure code is retained but the video evidence contract rejects that route.
"""
import argparse, base64, json, os, random, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

if __name__ != "__main__":
    raise ImportError(f"{__file__} 是命令行脚本，不能被 import（会写文件、发请求）")

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
API_VERSION = "preview"

P = argparse.ArgumentParser()
P.add_argument("--task")
P.add_argument("--tasks")
P.add_argument("-j", "--workers", type=int, default=3)
P.add_argument("-m", "--model", default=DEPLOY)
P.add_argument("--max-tokens", type=int, default=4000)
P.add_argument("--timeout", type=int, default=180)
P.add_argument("--quiet", action="store_true")
A = P.parse_args()

PROVIDER = os.environ.get("S12_JUDGE_PROVIDER", "gemini").lower()
KEY = os.environ.get("AZURE_AI_KEY")
if PROVIDER == "azure" and not (KEY and ENDPOINT and A.model):
    sys.exit("Set AZURE_AI_KEY, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT (or --model)")
if PROVIDER == "gemini" and not os.environ.get("GOOGLE_AI_STUDIO_API_KEY"):
    sys.exit("GOOGLE_AI_STUDIO_API_KEY 没设")


def judge_identity():
    from grading_video import VIDEO_CONTRACT
    return {
        "video_contract": VIDEO_CONTRACT,
        "provider": PROVIDER,
        "model": (os.environ.get("S12_GEMINI_MODEL", "gemini-3.1-pro-preview")
                  if PROVIDER == "gemini" else A.model),
        "temperature": (os.environ.get("S12_JUDGE_TEMP", "0")
                        if PROVIDER == "gemini" else "provider-default"),
        "seed": os.environ.get("S12_JUDGE_SEED", "42"),
    }


def parse_band_response(value):
    """Accept only the evidence-bearing one-line contract requested in the prompt."""
    from grading_contract import judge_only, direct_band_response
    if judge_only():
        return direct_band_response(value)
    pattern = re.compile(
        r"^\s*BAND:\s*([0-4])\s+ROWS:\s*(\S(?:.*?\S)?)\s+WHY:\s*(\S(?:.*\S)?)\s*$"
    )
    lines = (value or "").splitlines()
    if len(lines) != 1:
        return None, "", ""
    match = pattern.fullmatch(lines[0])
    if not match:
        return None, "", ""
    return int(match.group(1)), match.group(2), match.group(3)


# ── Gemini（AI Studio）判分后端 ────────────────────────────────────────────
# The Gemini backend uses its own sampling settings and payload format.
#
# 两边消息格式不同，这里做转换：
#   system            -> systemInstruction
#   {"type":"text"}   -> {"text": ...}
#   image_url(data:)  -> {"inlineData": {"mimeType", "data"}}
# Gemini 3 是思考模型，回复里可能混 thought 部件，只取非 thought 的 text。
_GEMINI_MODEL = os.environ.get("S12_GEMINI_MODEL", "gemini-3.1-pro-preview")
_GEMINI_HOST = "https://generativelanguage.googleapis.com/v1beta"


def _to_gemini(messages):
    from grading_judge import gemini_payload
    return gemini_payload(messages)


def chat_gemini(messages, max_tokens):
    from grading_judge import gemini_chat
    return gemini_chat(messages, max_tokens, model=A.model, timeout=A.timeout)

def chat(messages, max_tokens):
    if os.environ.get("S12_JUDGE_PROVIDER", "gemini").lower() == "gemini":
        return chat_gemini(messages, max_tokens)
    # ↓ 以下是 Azure/gpt-5.5 那条路
    # Keep the optional seed explicit when the configured Azure backend supports it.
    _body = {"model": A.model, "messages": messages,
             "max_completion_tokens": max_tokens}
    _seed = os.environ.get("S12_JUDGE_SEED", "42")
    if _seed != "":
        _body["seed"] = int(_seed)
    body = json.dumps(_body).encode()
    url = f"{ENDPOINT}/openai/v1/chat/completions?api-version={API_VERSION}"
    last = ""
    for attempt in range(6):
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("api-key", KEY)
        try:
            with urllib.request.urlopen(req, timeout=A.timeout) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or "", ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt)
                continue
            return "", last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return "", last


# ---------------------------------------------------------------------------
# 时长：不调模型。题面给了目标，答案里带容差，够判了。
# ---------------------------------------------------------------------------
def score_duration(key, sub_cuts):
    """时长对不对，是算术问题，没有可判断的余地。

    按百分比判，不按绝对秒数——19 秒的段落差 0.4 秒和 5 秒的段落差 0.4 秒不是一回事。
    答案里的 `dur_tol_s`（4 帧）只当下限用：短段落上 3% 可能比 4 帧还小，那时候按
    4 帧算，不能比逐帧对齐还苛刻。

    题面写的是 "the lengths should add up to **close to** the target"——是软要求，
    所以 3% 以内就算满分。19 秒的段落上那是 0.57 秒，一两个镜头的取舍空间。
    """
    target = key.get("target_duration_s")
    frame_tol = key.get("dur_tol_s") or 0.1668
    total = round(sum(max(0.0, c.get("out_s", 0) - c.get("in_s", 0))
                      for c in sub_cuts), 4)
    if not target:
        return {"band": None, "why": "题目没写目标时长"}
    delta = total - target
    pct = abs(delta) / target * 100
    # 每档的绝对下限：不小于 4 帧，短段落不至于被百分比卡死
    def lim(p):
        return max(target * p / 100, frame_tol)
    band = (4 if abs(delta) <= lim(3) else
            3 if abs(delta) <= lim(8) else
            2 if abs(delta) <= lim(18) else
            1 if abs(delta) <= lim(35) else 0)
    return {"band": band,
            "target_s": target, "assembled_s": total,
            "delta_s": round(delta, 4), "delta_pct": round(pct, 2),
            "frame_tol_s": frame_tol,
            "why": (f"拼出来 {total:.3f}s，目标 {target:.3f}s，差 {delta:+.3f}s "
                    f"({pct:.1f}%)" + ("，在 3% 以内" if band == 4 else "")),
            "bands_note": "≤3% 4 分；≤8% 3 分；≤18% 2 分；≤35% 1 分；更差 0 分。"
                          "每档另有 4 帧的绝对下限，短段落不会被百分比卡死"}


# ---------------------------------------------------------------------------
from rubrics_video import S3_REFERENCE as RUBRICS


def evidence(task):
    from grading_contract import validate_task
    validate_task(task, "s_3")
    key = json.load(open(f"{task}/verifier/answer.json"))
    sub = json.load(open(f"{task}/output/cuts.json"))
    man = json.load(open(f"{task}/initial_state/manifest.json"))
    fps = key.get("fps") or 24.0
    sc = {c["clip"]: c for c in sub.get("cuts", [])}
    by = {c["id"]: c for c in man["clips"]}
    per, gl, sl = [], [], []
    for g in key["cuts"]:
        s = sc.get(g["clip"], {})
        si, so = s.get("in_s"), s.get("out_s")
        gk = round(g["out_s"] - g["in_s"], 3)
        sk = round((so - si), 3) if None not in (si, so) else None
        gl.append(gk)
        if sk is not None:
            sl.append(sk)
        per.append({
            "clip": g["clip"], "clip_dur_s": by.get(g["clip"], {}).get("clip_dur_s"),
            "golden_in_s": g["in_s"], "sub_in_s": si,
            "in_delta_s": round(si - g["in_s"], 3) if si is not None else None,
            "in_delta_frames": round((si - g["in_s"]) * fps, 1) if si is not None else None,
            "golden_out_s": g["out_s"], "sub_out_s": so,
            "out_delta_s": round(so - g["out_s"], 3) if so is not None else None,
            "out_delta_frames": round((so - g["out_s"]) * fps, 1) if so is not None else None,
            "golden_kept_s": gk, "sub_kept_s": sk,
            "kept_ratio": round(sk / gk, 2) if sk and gk else None})

    def cv(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        v = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return round(v / m, 3) if m else 0.0

    beat = None
    tp = f"{task}/task_prompt.md"
    if os.path.exists(tp):
        m = re.search(r"covers:\s*\n+>\s*(.+?)\n", open(tp).read())
        beat = m.group(1).strip() if m else None
    from grading_video import s3_videos
    observed = json.load(open(f"{task}/evidence/s3_evidence.json"))
    continuous = observed["continuous_video"]
    reference = s3_videos(task, man, key, reference=True)
    continuous["videos"] += reference["videos"]
    return {"continuous_video": continuous, "measurement_health": observed.get('measurement_health', {}),
            "task": task.split("training_task/")[-1], "fps": round(fps, 4), "beat": beat,
            "per_clip": per,
            "shape": {"golden_lengths_s": gl, "sub_lengths_s": sl,
                      "golden_cv": cv(gl), "sub_cv": cv(sl),
                      "interpretation_status": "descriptive_only",
                      "note": "Length variation is not a quality target. Judge observable pacing."},
            "key": key}


def sheet_part(task, r):
    ep = f"{task}/evidence/s3_evidence.json"
    if not os.path.exists(ep):
        return None, ""
    meta = (json.load(open(ep)).get("contact_sheets") or {}).get(r.get("sheet"))
    if not meta:
        return None, ""
    p = os.path.join(task, meta["file"] if isinstance(meta, dict) else meta)
    if not os.path.exists(p):
        return None, ""
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return ({"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            meta.get("note", "") if isinstance(meta, dict) else "")


def build_messages(task, ev, r):
    bands = "\n".join(f"    {k} — {v}" for k, v in sorted(r["bands"].items(), reverse=True))
    img, note = sheet_part(task, r)
    prompt_txt = ""
    tp = f"{task}/task_prompt.md"
    if os.path.exists(tp):
        prompt_txt = open(tp).read()
    sl = {k: v for k, v in ev.items() if k not in ("key", "continuous_video")}
    body = f"""# {r['title']}

{r['rubric']}

# Bands

{bands}

# The task the editor was given

```markdown
{prompt_txt}
```

# The submission, with the trailer's edit alongside

`sub_*` is what was handed in — that is what you are scoring. `golden_*` is what the
finished trailer did with the same clips, for reference. Deltas are submission minus
trailer, in seconds and in frames.

```json
{json.dumps(sl, ensure_ascii=False, indent=1)}
```
"""
    if img:
        body += f"""
# Contact sheet

{note}

The sheet shows the **submission's** frames, not the trailer's. Use it to judge what a
delta actually costs — the numbers say how far off, the frames say whether it matters.
"""
    body += """
# How to answer

Score the cut. The trailer's version is a reference for what this material supports — cite
it where it decided your reading, and say so when the submission's different choice is
just as good. Cite clip ids and the deltas you used.

Reply with exactly one line and nothing else:

BAND: <0-4>  ROWS: <clip ids you used>  WHY: <one sentence>"""
    parts = [{"type": "text", "text": body}]
    if img:
        parts.append(img)
    messages = [{"role": "system",
             "content": "You judge whether a rough cut is well made. The trailer's own edit "
                        "of the same material is provided as a reference point, not as the "
                        "answer key. Answer in the exact shape asked for."},
            {"role": "user", "content": parts}]
    from grading_judge import augment
    return augment(task, ev, r, messages)


def call(task, ev, r):
    from grading_judge import audited_chat
    from grading_contract import judge_only
    request = lambda messages, limit: audited_chat(task, r["id"], chat, messages, limit)
    t0 = time.time()
    expected_sheets = ev.get('measurement_health', {}).get('required_contact_sheets')
    absent_by_submission = (judge_only() and isinstance(expected_sheets, list)
                            and r.get('sheet') not in expected_sheets)
    if r.get("sheet") and sheet_part(task, r)[0] is None and not absent_by_submission:
        return {"rubric": r["id"], "title": r["title"], "band": None,
                "rows": "", "why": "", "retried_format": False,
                "sec": 0.0,
                "err": f"required contact sheet {r['sheet']} is missing"}
    messages = build_messages(task, ev, r)
    out, err = request(messages, A.max_tokens)

    band, rows, why = parse_band_response(out)
    retried_format = False
    if band is None and out:
        retried_format = True
        retry_messages = messages + [{
            "role": "user",
            "content": (
                "Your preceding answer could not be parsed because it did not provide "
                "one complete BAND/ROWS/WHY line. Return exactly one line and nothing "
                "else: BAND: <0-4>  ROWS: <ids used>  WHY: <one sentence>"
            ),
        }]
        retry_out, retry_err = request(retry_messages, A.max_tokens)
        if retry_out:
            out, err = retry_out, retry_err
            band, rows, why = parse_band_response(out)
        elif retry_err:
            err = f"格式重试失败 — {retry_err}"
    if band is None and out:
        err = f"回答不符合完整 BAND/ROWS/WHY 单行格式 — {out[-160:]}"
    return {"rubric": r["id"], "title": r["title"], "band": band,
            "rows": rows[:140], "why": why, "retried_format": retried_format, "saw_video": True,
            "sec": round(time.time() - t0, 1),
            "err": err if band is None else ""}


W_DUR = 0.20   # 时长自成一档：它是硬约束，但不该盖过剪辑判断


def judge(task):
    name = task.split("training_task/")[-1]
    if not os.path.exists(f"{task}/output/cuts.json"):
        print(f"  {name:<30} ★ 没有 output/cuts.json", flush=True)
        return None
    ev = evidence(task)
    dur = score_duration(ev["key"], json.load(open(f"{task}/output/cuts.json"))["cuts"])
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lambda r: call(task, ev, r), RUBRICS))
    bands = {x["rubric"]: x["band"] for x in res if x["band"] is not None}
    if dur["band"] is not None:
        bands["duration"] = dur["band"]
    ws = {r["id"]: r["weight"] * (1 - W_DUR) for r in RUBRICS}
    ws["duration"] = W_DUR
    tot = sum(ws[k] for k in bands)
    score = round(sum(bands[k] * ws[k] for k in bands) / tot / 4.0, 3) if tot else None
    zeros = [k for k, v in bands.items() if v == 0]
    if score is not None and zeros:
        score = round(score * (0.75 ** len(zeros)), 3)
    missing = [r["id"] for r in RUBRICS if r["id"] not in bands]
    if dur["band"] is None:
        missing.append("duration")
    out = {"task": name, "judge": judge_identity(), "mode": "golden-as-reference",
           "score": score, "per_rubric": bands, "weights": ws,
           "duration": dur, "rubrics": res,
           "partial": bool(missing), "missing": missing,
           "weight_answered": round(sum(ws[k] for k in bands), 3),
           "penalty": (f"{len(zeros)} 条拿 0（{','.join(zeros)}），乘 "
                       f"{0.75 ** len(zeros):.2f}") if zeros else None}
    json.dump(out, open(f"{task}/evidence/s3_score_ref.json", "w"),
              ensure_ascii=False, indent=1)
    if not A.quiet:
        print(f"  {name:<30} {score if score is not None else '  —  '}   "
              f"dur {dur['band']}  " +
              "  ".join(f'{x["rubric"][:4]} {x["band"] if x["band"] is not None else "?"}'
                        for x in res), flush=True)
        print(f"      时长: {dur['why']}", flush=True)
        for x in res:
            if x["why"]:
                print(f"      {x['title']}: {x['why'][:104]}", flush=True)
            elif x["err"]:
                print(f"      {x['title']}: ★ {x['err'][-88:]}", flush=True)
        if out["penalty"]:
            print(f"      ⚠ {out['penalty']}", flush=True)
    return None if out["partial"] else score


tasks = json.load(open(A.tasks)) if A.tasks else [A.task]
_jid = judge_identity()
print(f"参照式判分 {len(tasks)} 份 × (3 条 rubric + 时长)   "
      f"判官 {_jid['model']} ({_jid['provider']})\n",
      flush=True)
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    got = [x for x in ex.map(judge, tasks) if x is not None]
if got:
    import statistics as st
    print(f"\n中位 {st.median(got):.3f}   分布 {min(got):.2f}–{max(got):.2f}")
