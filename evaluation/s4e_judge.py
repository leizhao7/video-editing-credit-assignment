"""Independent Gemini 3.1 Pro continuous-video grader.
Every rubric receives task text, delivered video and its navigational evidence.
No audio/image/text-only downgrade; API failures remain unscored, not policy zero.
Legacy Azure code is retained but the video evidence contract rejects that route.
"""
import random
import argparse, base64, json, os, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

if __name__ != "__main__":
    raise ImportError(f"{__file__} 是命令行脚本，不能被 import（会写文件、发请求）")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s4e_rubrics import RUBRICS, aggregate  # noqa: E402

ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEPLOY = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")          # 部署名，不是 "gpt-5.5"
API_VERSION = "preview"

P = argparse.ArgumentParser()
P.add_argument("--task")
P.add_argument("--tasks")
P.add_argument("-j", "--workers", type=int, default=4)
P.add_argument("-m", "--model", default=DEPLOY)
P.add_argument("--max-tokens", type=int, default=4000,
               help="推理模型，可见回答之前还要花一部分在隐藏推理上，给宽一点")
P.add_argument("--timeout", type=int, default=180)
P.add_argument("--no-images", action="store_true",
               help="不传联络表，只看数字——用来对照『看得见画面』值多少")
P.add_argument("--suffix", default="",
               help="写进 s4e_score<suffix>.json，别覆盖上一轮")
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
        "grading_contract": "s34-grading-v5",
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


def chat(messages, max_tokens):
    if os.environ.get("S12_JUDGE_PROVIDER", "gemini").lower() == "gemini":
        return chat_gemini(messages, max_tokens)
    # ↓ 以下是 Azure/gpt-5.5 那条路
    """Azure chat completions。gpt-5.5 是推理模型：用 max_completion_tokens，
    不要动 temperature。429 退避重试。"""
    # Keep the optional seed explicit when the configured Azure backend supports it.
    _body = {"model": A.model, "messages": messages,
             "max_completion_tokens": max_tokens}
    _seed = os.environ.get("S12_JUDGE_SEED", "42")
    if _seed != "":
        _body["seed"] = int(_seed)
    body = json.dumps(_body).encode()
    url = f"{ENDPOINT}/openai/v1/chat/completions?api-version={API_VERSION}"
    # Configurable retries and jitter keep concurrent requests from retrying in lockstep.
    _tries = int(os.environ.get("S12_JUDGE_RETRIES", "10"))
    _cap = float(os.environ.get("S12_JUDGE_BACKOFF_CAP", "60"))
    last = ""
    for attempt in range(_tries):
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("api-key", KEY)
        try:
            with urllib.request.urlopen(req, timeout=A.timeout) as r:
                d = json.loads(r.read())
            ch = d["choices"][0]
            txt = ch["message"]["content"] or ""
            if not txt.strip():
                # 推理模型把预算全花在隐藏推理上，可见回答是空的。这个失败原本无声无息：
                # err 只在 `out` 非空时才写，所以 band=None 配 err="" 看着像没发生过。
                u = d.get("usage", {})
                return "", (f'空回答 finish_reason={ch.get("finish_reason")} '
                            f'推理 {u.get("completion_tokens_details", {}).get("reasoning_tokens", "?")} tok '
                            f'/ 上限 {max_tokens}')
            return txt, ""
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {body_txt}"
            if e.code == 429 or e.code >= 500:
                time.sleep(min(_cap, 2 ** attempt)
                           * (0.5 + random.random()))
                continue
            return "", last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(_cap, 2 ** attempt) * (0.5 + random.random()))
    return "", last


PROSE = ("declared_note", "note", "why", "declared_reason", "music_bridge_because")


def strip(x):
    """把作业自己写的说明全部摘掉，只留下做了什么。

    量出来的：music 那条同一个入点数字，把平铺的 note 换成一段有论证的文字，十份的
    band 从均 1.20 涨到 2.90，无一例外。判官在读文案，不在听音乐。作业的自述不是证据，
    是主张——判官该看的是画面、声音和测量，自己判断剪得好不好。

    机械核对（reconcile）留着：那是拿声明的结构（cut / under / prelap 这些取值）去对
    渲染，对的是事实不是文笔。摘掉的只有散文。
    """
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in PROSE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x



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

def slice_evidence(ev, rid):
    """只给这条 rubric 要的那部分。

    判 bridge 的不需要读音乐包络，判音乐的不需要逐条 take 的文本。输入收窄，回答就不会
    在四件事之间飘。
    """
    from grading_judge import black_context
    base = {"task": ev["task"], "stretch_dur_s": ev["stretch_dur_s"],
            "how_measured": ev.get("how_measured"), **black_context(ev)}
    if rid == "dialogue":
        base |= {"beat": ev.get("beat"),
                 "shots": [{k: v for k, v in s.items()
                            if k in ("id", "start_s", "end_s", "dur_s", "origin",
                                     "on_screen_says")} for s in ev["shots"]],
                 "joins": [{k: v for k, v in j.items()
                            if k in ("join", "t_s", "leaving", "arriving")}
                           for j in ev["joins"]],
                 "dialogue_placed": ev["dialogue_placed"],
                 "dialogue_omitted": ev["dialogue_omitted"],
                 "render": ev.get("render"),
                 "render_transcript": ev.get("render_transcript")}
    elif rid == "music":
        base |= {"beat": ev.get("beat"),
                 "music": ev.get("music"),
                 "joins": ev.get("joins"),
                 "shots": [{k: v for k, v in s.items()
                            if k in ("id", "start_s", "end_s", "dur_s")}
                           for s in ev["shots"]],
                 "dialogue_spans": [{"take": p["take"],
                                     "timeline_span_s": p.get("timeline_span_s"),
                                     "measured_in_render": p.get("measured_in_render")}
                                    for p in ev["dialogue_placed"]]}
    elif rid == "bridges":
        base |= {"reconcile": ev.get("reconcile"),
                 "render_black_runs": ev.get("render_black_runs"),
                 "joins_render_time_s": ev.get("joins_render_time_s"),
                 "bridge_count": ev.get("bridge_count"),
                 "production_sound": ev.get("production_sound"),
                 "bridges": ev["bridges"], "joins": ev["joins"],
                 "render": ev.get("render")}
    else:
        # Unconfirmed crossings are unknown, not plain cuts or missed opportunities.
        base |= {"beat": ev.get("beat"),
                 "bridge_count": ev.get("bridge_count"),
                 "reconcile": ev.get("reconcile"),
                 "render_black_runs": ev.get("render_black_runs"),
                 "bridges": ev["bridges"],
                 "shots": [{k: v for k, v in s.items()
                            if k in ("id", "dur_s", "origin", "on_screen_says")}
                           for s in ev["shots"]],
                 "dialogue_crossings": [{"take": p["take"],
                                         "crosses_joins": p.get("crosses_joins")}
                                        for p in ev["dialogue_placed"]
                                        if p.get("crosses_joins")]}
    from grading_judge import delivery_evidence
    return delivery_evidence(strip(base))


def sheet_part(task, ev, r):
    """把这条 rubric 该看的那张联络表做成 image_url 部件。

    只挂一张。三张全给会让它在不该看的图里找证据——判 bridge 该不该有的那条要看接点
    两侧，给它每镜一帧那张只会引它去谈内容。
    """
    sid = r.get("sheet")
    if sid == "music_timeline":
        # This plot is computed from the SOURCE cue at a declared offset, not
        # the delivered mix. Every rubric still receives actual video and PCM.
        return None, "Source music plot withheld; listen to actual delivered audio."
    meta = (ev.get("contact_sheets") or {}).get(sid)
    if A.no_images or not meta:
        return None, ""
    p = os.path.join(task, meta["file"] if isinstance(meta, dict) else meta)
    if not os.path.exists(p):
        return None, ""
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    return ({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            meta.get("note", "") if isinstance(meta, dict) else "")


def build_messages(task, ev, r, with_image=True):
    sl = slice_evidence(ev, r["id"])
    bands = "\n".join(f"    {k} — {v}" for k, v in sorted(r["bands"].items(), reverse=True))
    img, note = sheet_part(task, ev, r) if with_image else (None, "")
    system = ("You score one aspect of a trailer fine cut, from measurements and frames "
              "someone else prepared. Watch/listen to the attached inline videos/audio — "
              "everything you need is in this message. Answer in the exact shape asked for.")
    pic = (f"""
# The contact sheet

The image attached is the picture evidence for this rubric. {note}

Read it. The numbers say how long and how far apart; only the frames say what is actually on
screen. Where the two disagree, say which one you are trusting and why.
""" if img else f"""
# No contact sheet

{'This rubric has none, and that is deliberate — where a cut falls against a music accent is a fact about time, and a frame cannot show it.' if r['id'] == 'music' else 'There should have been one for this rubric; it did not reach you.'} Work from the numbers. Where a
distinction would need the picture, say the evidence is absent rather than guessing at it,
and score what the numbers actually support.

Absent evidence is not a defence for the submission. Not being able to confirm that a choice
was wrong is not the same as confirming it was right, and a band whose condition is only
*vacuously* met — nothing was done, so nothing was done badly — is the wrong band. Score
what was delivered.
""")
    user = f"""# {r['title']}

{r['question']}

{r['rubric']}

# Bands

{bands}
{pic}
# Evidence

Measured mechanically before you were called. `how_measured` says exactly how, so you can
tell how much weight each number deserves. Where a field is split into what was *declared*
and what was *measured in the render*, the render is the fact and the declaration is the
claim.

```json
{json.dumps(sl, ensure_ascii=False, indent=1)}
```

# How to answer

Pick the band whose condition the evidence meets — not the one that feels about right. If
the evidence for a distinction is absent rather than negative, say so and score what is
actually supported.

Cite the rows. A band with nothing behind it is worth nothing. When the frames decided it,
cite the tile labels you read (`j07-1`, `j07+2`, `p05`).

Reply with exactly one line in this shape and nothing else:

BAND: <0-4>  ROWS: <clip ids / join numbers / take ids / tile labels you used>  WHY: <one sentence>"""
    parts = [{"type": "text", "text": user}]
    if img:
        parts.append(img)
    from grading_judge import augment
    return augment(task, ev, r, [{"role": "system", "content": system}, {"role": "user", "content": parts}])


BLOCKED = ("content_policy_violation", "content management policy",
           "not allowed by our content safety")


def call(task, ev, r):
    """One rubric per call; missing/blocked media stays unscored, never downgraded."""
    from grading_judge import audited_chat
    request = lambda messages, limit: audited_chat(task, r["id"], chat, messages, limit)
    t0 = time.time()
    img_ok = sheet_part(task, ev, r)[0] is not None
    if A.no_images or (r.get("sheet") and not img_ok):
        return {"rubric": r["id"], "title": r["title"], "band": None,
                "rows": "", "why": "", "retried_long": False,
                "retried_format": False, "sheet": r.get("sheet"),
                "saw_image": False, "image_blocked": False, "sec": 0.0,
                "err": f"required contact sheet {r['sheet']} is missing"}
    messages = build_messages(task, ev, r)
    out, err = request(messages, A.max_tokens)
    def is_blocked(e): return any(x in (e or "") for x in BLOCKED)

    blocked = is_blocked(err)
    if blocked:
        return {"rubric": r["id"], "title": r["title"], "band": None, "rows": "", "why": "",
                "saw_image": img_ok, "saw_audio": False, "image_blocked": True,
                "sec": round(time.time()-t0, 1), "err": "required media blocked; v5 forbids text-only downgrade"}
    if not out and err.startswith("空回答"):
        # 加图之后推理变长，4000 不够。翻倍再来一次比丢掉一整条 rubric 强得多。
        messages = build_messages(task, ev, r, with_image=img_ok)
        out, err = request(messages, A.max_tokens * 2)
        retried_long = True
    else:
        retried_long = False

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
            "rows": rows[:120], "why": why, "retried_long": retried_long,
            "retried_format": retried_format,
            "sheet": r.get("sheet"), "saw_image": img_ok,
            "saw_audio": (ev.get("render_audio") or {}).get("status") == "ok",
            "saw_video": True,
            "image_blocked": blocked,
            "sec": round(time.time() - t0, 1), "err": err if band is None else ""}


def judge(task):
    ep = f"{task}/evidence/s4e_evidence.json"
    name = task.split("training_task/")[-1]
    if not os.path.exists(ep):
        print(f"  {name:<30} ★ 没有 evidence，先跑 s4e_evidence.py", flush=True)
        return None
    ev = json.load(open(ep))
    if not A.no_images and not ev.get("contact_sheets"):
        print(f"  {name:<30} ⚠ evidence 里没有联络表，重跑 s4e_evidence.py 才有图",
              flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda r: call(task, ev, r), RUBRICS))
    agg = aggregate({x["rubric"]: x["band"] for x in res})
    # 缺条目的分是用剩下几条重新归一化算的，不能跟完整的分并排比。标出来。
    W = {r["id"]: r["weight"] for r in RUBRICS}
    agg["weight_answered"] = round(sum(W[k] for k in (agg.get("per_rubric") or {})), 3)
    lost_images = [x["rubric"] for x in res if x.get("sheet") and not x.get("saw_image")]
    agg["required_images_missing"] = lost_images
    agg["partial"] = bool(agg.get("missing") or lost_images)
    json.dump({"task": name, "judge": judge_identity(), "images": not A.no_images,
               **agg, "rubrics": res},
              open(f"{task}/evidence/s4e_score{A.suffix}.json", "w"),
              ensure_ascii=False, indent=1)
    if not A.quiet:
        s = agg["score"]
        print(f"  {name:<30} {('%.3f' % s) if s is not None else '  —  '}   " +
              "  ".join(f'{x["rubric"][:4]} {x["band"] if x["band"] is not None else "?"}'
                        + ("" if x["saw_image"] or not x["sheet"] else "(无图)")
                        for x in res), flush=True)
        for x in res:
            if x["why"]:
                print(f"      {x['title']}: {x['why'][:104]}", flush=True)
            elif x["err"]:
                print(f"      {x['title']}: ★ {x['err'][-90:]}", flush=True)
        if agg.get("penalty"):
            print(f"      ⚠ {agg['penalty']}", flush=True)
        if agg["partial"]:
            print(f"      ⚠ 缺 {'、'.join(agg['missing'])}，这个 {agg['score']} 只是"
                  f"剩下 {agg['weight_answered']:.0%} 权重归一化来的，不入统计", flush=True)
    return None if agg["partial"] else agg["score"]


tasks = json.load(open(A.tasks)) if A.tasks else [A.task]
_jid = judge_identity()
print(f"评 {len(tasks)} 份 × 4 条 rubric   判官 {_jid['model']} "
      f"({_jid['provider']})\n", flush=True)
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    got = [x for x in ex.map(judge, tasks) if x is not None]
blk = []
for t in tasks:
    f = f"{t}/evidence/s4e_score{A.suffix}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        blk += [f'{d["task"]}·{x["rubric"]}' for x in d["rubrics"] if x.get("image_blocked")]
if blk:
    print(f"\n图被内容过滤挡下 {len(blk)} 次，已退回纯数字重判：{'、'.join(blk)}", flush=True)
if got:
    import statistics as st
    print(f"\n完整评出 {len(got)}/{len(tasks)}   中位 {st.median(got):.3f}   "
          f"分布 {min(got):.2f}–{max(got):.2f}")
