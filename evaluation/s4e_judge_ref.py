"""Independent Gemini 3.1 Pro continuous-video grader.
Every rubric receives task text, delivered video and its navigational evidence.
No audio/image/text-only downgrade; API failures remain unscored, not policy zero.
Legacy Azure code is retained but the video evidence contract rejects that route.
"""
import argparse, base64, json, os, random, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

if __name__ != "__main__":
    raise ImportError(f"{__file__} 是命令行脚本，不能被 import（会写文件、发请求）")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s4e_ref_rubrics import RUBRICS, aggregate  # noqa: E402

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
    last = ""
    for attempt in range(6):
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
                time.sleep(2 ** attempt)
                continue
            return "", last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
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


def golden_block(task):
    """从 verifier/answer.json 读出 golden 的做法，摆成人能读的样子。

    只取"做了什么"：用了哪几条台词、落在哪、每一刀怎么处理、音乐从文件哪一秒进来。
    不取任何评价性的东西——判官要自己判，不是照着抄。
    """
    p = f"{task}/verifier/answer.json"
    if not os.path.exists(p):
        return None
    a = json.load(open(p))
    tc_p = f"{task}/verifier/trailer_cuts.json"
    tc = json.load(open(tc_p)) if os.path.exists(tc_p) else None
    return {
        "what_this_is": "预告片当初用同一批素材做出来的成品。一个答案，不是唯一答案。",
        "dialogue": [{"take": g["id"], "text": g.get("text"),
                      "starts_at_s": g.get("trailer_s")}
                     for g in a.get("golden_dialogue", [])],
        "dialogue_legend": "starts_at_s 是这句话在本段时间线上开口的时刻；负数表示它在"
                           "本段开始之前就已经在说了。text 是这条 take 的内容",
        "left_out": a.get("distractors"),
        "joins": [{"join": j.get("cut"), "picture": j.get("picture"),
                   "audio": j.get("audio"), "audio_offset_s": j.get("audio_offset_s")}
                  for j in a.get("golden_joins", [])],
        "joins_legend": "picture: cut / dip_to_black / dissolve / other。"
                        "audio: J=prelap（到达镜的声音提前进来）、"
                        "L=carries_over（离开镜的声音延后出去）、hard=声画同断。"
                        "audio_offset_s 是声音跨过画面切点多少秒",
        "music_align_s": a.get("music_align_s"),
        "music_note": "预告片把音乐文件的这一秒对到本段的 0.0",
            "measured_cuts": ({
            "what": "成片自己在这一段每一刀上的实测糙度。同样的两个镜头、同样的接点"
                    "位置，成片做出来是多少",
            "joins": [{k2: v2 for k2, v2 in x.items()
                       if k2 in ("join", "gap_db", "step_db", "spec_dist",
                                 "trailer_audio", "trailer_picture")}
                      for x in tc["joins"]],
            "how_to_read": "gap_db is a mixed-signal level statistic near a reference cut. "
                           "It does not establish a source crossing or an unintended audible defect. "
                           "Compare cautiously with status=measured submission cut_roughness, "
                           "then verify actual submitted video/audio.",
        } if tc else None),
    }


def align_dialogue(ev, g):
    """把成片的台词和作业的台词并排算好，别让判官自己比。

    量出来的：bridges_ref 的引证率 70–80%，因为 join_alignment 把"成片这刀铺了 1.5 秒、
    作业干接"直接算成一行结论；而 dialogue_ref 只有 30–40% 的时候提到成片，music_ref
    在拉平组上是 0/10——参照就摆在证据 JSON 里，判官不去比。
    """
    gd = {d["take"]: d for d in (g.get("dialogue") or [])}
    sd = {}
    for p_ in ev["dialogue_placed"]:
        sp = p_.get("timeline_span_s") or [None, None]
        sd.setdefault(p_["take"], {"take": p_["take"], "at_s": sp[0], "end_s": sp[1],
                                   "heard": (p_.get("heard_here") or "").strip()})
    rows, both, only_g, only_s = [], [], [], []
    for tk in sorted(set(gd) | set(sd)):
        G, S = gd.get(tk), sd.get(tk)
        r = {"take": tk,
             "trailer": ({"starts_at_s": G.get("starts_at_s")} if G else None),
            "submission": ({"declared_rough_starts_at_s": S.get("at_s"),
                             "asr_text_near_declared_span": S.get("heard") or None,
                             "source_identity_confirmation": "unknown; listen to actual render"} if S else None)}
        if G and S:
            a, b = G.get("starts_at_s"), S.get("at_s")
            if a is not None and b is not None:
                r["offset_s"] = round(b - a, 3)
                r["comparison_basis"] = "declared rough timeline, not measured rendered speech timing"
            both.append(tk)
        elif G:
            r["verdict"] = "Reference includes this take; submission does not declare it. This is not proof of absence or a defect."
            only_g.append(tk)
        else:
            r["verdict"] = "Submission declares an alternate take. Verify actual delivered relevance and intelligibility."
            only_s.append(tk)
        rows.append(r)
    return {"takes": rows,
            "both_used": both, "trailer_only": only_g, "submission_only": only_s,
            "trailer_left_out": g.get("left_out"),
            "how_to_read": "Take IDs and submission times are declarations, not confirmed delivery. "
                           "Reference-only takes are options, not mandatory missing lines. "
                           "Use actual rendered video/audio and explicit task requirements; "
                           "ASR text alone does not confirm source identity or intelligibility."}


def align_music(ev, g):
    """成片的音乐入点和作业的并排，换算成本段时间线上的位置。"""
    mus = ev.get("music") or {}
    ga = g.get("music_align_s")
    sa = mus.get("declared_start_s")
    joins = ev.get("joins") or []
    out = {"trailer_starts_at_file_s": ga, "submission_starts_at_file_s": sa,
           "how_to_read": "两个都是**音乐文件里的秒数**，对到本段 0.0 的那一点。差得远"
                          "说明作业选了完全不同的一段乐句；差得近说明它找到了成片同一个"
                          "入点。不一致不直接扣分——判的是作业那个入点在这段画面上成不成立，"
                          "成片只是证明这条素材撑得住某个入法"}
    if ga is not None and sa is not None:
        d = round(sa - ga, 3)
        out["offset_s"] = d
        out["verdict"] = ("和成片几乎同一个入点" if abs(d) < 0.5 else
                          f'比成片{"晚" if d > 0 else "早"} {abs(d):.2f}s 进，'
                          f'是乐曲里另一处')
    acc = (mus.get("accents") or [])
    if acc and joins:
        near = sum(1 for a in acc if a.get("dist_to_join_s") is not None
                   and abs(a["dist_to_join_s"]) <= 0.12)
        out["declared_source_accents_on_rough_cuts"] = {"within_0.12s": near, "accents": len(acc),
                                                      "joins": len(joins), "render_confirmation": "unknown"}
    return out


def align_joins(ev, g):
    """Reference choices compared to positive/negative/unknown render evidence."""
    reference = {j["join"]: j for j in (g.get("joins") or [])}
    rows, unknown, absent, added = [], [], [], []
    for bridge in ev["bridges"]:
        n = bridge["join"]
        trailer = reference.get(n, {})
        kinds = bridge.get("bridge_kinds") or []
        own = any(k.startswith("sound:") for k in kinds)
        any_carry = own or "dialogue_spans_join" in kinds
        no_audio = bridge["measurements"].get("render_audio_status") == "no_audio_stream"
        status = "confirmed" if any_carry else "absent" if no_audio else "unknown"
        rows.append({"join": n, "trailer": trailer,
                     "submission": {"declared_sound": bridge.get("declared_sound"),
                                    "declared_picture": bridge.get("declared_picture"),
                                    "measured_kinds": kinds, "crossing_status": status,
                                    "own_sound_crosses": True if own else False if no_audio else None,
                                    "anything_crosses": True if any_carry else False if no_audio else None}})
        if status == "unknown":
            unknown.append(n)
        if trailer.get("audio") in ("carries_over", "prelap", "J", "L") and no_audio:
            absent.append(n)
        if trailer.get("audio") == "hard" and any_carry:
            added.append(n)
    return {"joins": rows, "crossing_unknown_joins": unknown,
            "reference_carry_confirmed_absent_in_submission": absent,
            "submission_carried_trailer_did_not": added,
            "comparison_status": "contains_unknown" if unknown else "observed_only",
            "how_to_read": "Reference choices are optional unless the task requires them. "
                           "Unknown is not absent. Empty lists are not proof of quality. "
                           "Judge actual submitted continuous video/audio, not bridge counts."}


def slice_evidence(ev, rid, task=None):
    """作业这边只给这条 rubric 要的，再挂上 golden 的对应部分。"""
    from grading_judge import black_context
    base = {"task": ev["task"], "stretch_dur_s": ev["stretch_dur_s"],
            "how_measured": ev.get("how_measured"), **black_context(ev)}
    g = golden_block(task) if task else None
    if rid == "dialogue_ref":
        base |= {"beat": ev.get("beat"),
                 "shots": [{k: v for k, v in s.items()
                            if k in ("id", "start_s", "end_s", "dur_s", "on_screen_says")}
                           for s in ev["shots"]],
                 "joins": [{k: v for k, v in j.items()
                            if k in ("join", "t_s", "leaving", "arriving")}
                           for j in ev["joins"]],
                 "submission_dialogue": ev["dialogue_placed"],
                 "submission_omitted": ev["dialogue_omitted"],
                 "render_transcript": ev.get("render_transcript")}
        if g:
            base["THE_TRAILER"] = {k: g[k] for k in
                                   ("what_this_is", "dialogue", "dialogue_legend",
                                    "left_out")}
            base["dialogue_alignment"] = align_dialogue(ev, g)
    elif rid == "bridges_ref":
        plain = [{"join": b["join"], "leaving": b["leaving"], "arriving": b["arriving"]}
                 for b in ev["bridges"] if not b["bridge_kinds"]]
        base |= {"beat": ev.get("beat"),
                 "submission_cuts": [
                     {"join": b["join"], "declared_sound": b["declared_sound"],
                      "declared_picture": b["declared_picture"],
                      "bridge_kinds": b["bridge_kinds"],
                      **(b["measurements"].get("cut_roughness") or {})}
                     for b in ev["bridges"]],
                 "submission_bridges": ev["bridges"],
                 "unconfirmed_crossing_joins": plain,
                 "bridge_count": ev.get("bridge_count"),
                 "reconcile": ev.get("reconcile"),
                 "render_black_runs": ev.get("render_black_runs"),
                 "shots": [{k: v for k, v in s.items()
                            if k in ("id", "dur_s", "on_screen_says")} for s in ev["shots"]]}
        if g:
            base["THE_TRAILER"] = {k: g[k] for k in
                                   ("what_this_is", "joins", "joins_legend")}
            if g.get("measured_cuts"):
                base["THE_TRAILER"]["measured_cuts"] = g["measured_cuts"]
            base["join_alignment"] = align_joins(ev, g)
    else:
        base |= {"beat": ev.get("beat"),
                 "submission_music": ev.get("music"),
                 "joins": ev.get("joins"),
                 "dialogue_spans": [{"take": p["take"],
                                     "timeline_span_s": p.get("timeline_span_s"),
                                     "measured_in_render": p.get("measured_in_render")}
                                    for p in ev["dialogue_placed"]]}
        if g:
            base["THE_TRAILER"] = {k: g[k] for k in
                                   ("what_this_is", "music_align_s", "music_note")}
            base["music_alignment"] = align_music(ev, g)
    from grading_judge import delivery_evidence
    return delivery_evidence(strip(base))


def sheet_part(task, ev, r):
    """把这条 rubric 该看的那张联络表做成 image_url 部件。

    只挂一张。三张全给会让它在不该看的图里找证据——判 bridge 该不该有的那条要看接点
    两侧，给它每镜一帧那张只会引它去谈内容。
    """
    sid = r.get("sheet")
    if sid == "music_timeline":
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
    sl = slice_evidence(ev, r["id"], task)
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
    if not os.path.exists(f"{task}/verifier/answer.json"):
        print(f"  {task.split('training_task/')[-1]:<30} ★ 没有 answer.json，参照判官需要它",
              flush=True)
        return None
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
              open(f"{task}/evidence/s4e_score_ref{A.suffix}.json", "w"),
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
print(f"参照判 {len(tasks)} 份 × 3 条 rubric   判官 {_jid['model']} "
      f"({_jid['provider']})\n", flush=True)
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    got = [x for x in ex.map(judge, tasks) if x is not None]
blk = []
for t in tasks:
    f = f"{t}/evidence/s4e_score_ref{A.suffix}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        blk += [f'{d["task"]}·{x["rubric"]}' for x in d["rubrics"] if x.get("image_blocked")]
if blk:
    print(f"\n图被内容过滤挡下 {len(blk)} 次，已退回纯数字重判：{'、'.join(blk)}", flush=True)
if got:
    import statistics as st
    print(f"\n完整评出 {len(got)}/{len(tasks)}   中位 {st.median(got):.3f}   "
          f"分布 {min(got):.2f}–{max(got):.2f}")
