"""Independent Gemini 3.1 Pro continuous-video grader.
Every rubric receives task text, delivered video and its navigational evidence.
No audio/image/text-only downgrade; API failures remain unscored, not policy zero.
Legacy Azure code is retained but the video evidence contract rejects that route.
"""
import random
import argparse, base64, json, os, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

if __name__ != "__main__":
    raise ImportError(f"{__file__} is a command-line script and cannot be imported because it writes files and sends requests")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_rubrics import RUBRICS, aggregate  # noqa: E402

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
P.add_argument("--no-images", action="store_true",
               help="Legacy diagnostic flag: omit contact sheets; rubrics requiring them remain unscored")
P.add_argument("--quiet", action="store_true")
A = P.parse_args()

PROVIDER = os.environ.get("S12_JUDGE_PROVIDER", "gemini").lower()
KEY = os.environ.get("AZURE_AI_KEY")
if PROVIDER == "azure" and not (KEY and ENDPOINT and A.model):
    sys.exit("Set AZURE_AI_KEY, AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT (or --model)")
if PROVIDER == "gemini" and not os.environ.get("GOOGLE_AI_STUDIO_API_KEY"):
    sys.exit("GOOGLE_AI_STUDIO_API_KEY is not set")


def judge_identity():
    from grading_video import VIDEO_CONTRACT
    return {
        "video_contract": VIDEO_CONTRACT,
        "provider": PROVIDER,
        "model": (os.environ.get("S12_GEMINI_MODEL", "gemini-3.1-pro-preview")
                  if PROVIDER == "gemini" else A.model),
        "temperature": (os.environ.get("S12_JUDGE_TEMP", "0")
                        if PROVIDER == "gemini" else "provider-default"),
        "seed": (os.environ.get("S12_JUDGE_SEED", "42")
                 if PROVIDER == "gemini" else os.environ.get("S12_JUDGE_SEED", "42")),
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
    # Legacy Azure/gpt-5.5 path below
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
            return d["choices"][0]["message"]["content"] or "", ""
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code == 429 or e.code >= 500:
                time.sleep(min(_cap, 2 ** attempt)
                           * (0.5 + random.random()))
                continue
            return "", last
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(_cap, 2 ** attempt) * (0.5 + random.random()))
    return "", last



# ── Gemini (AI Studio) grading backend ────────────────────────────────────────────
# The Gemini backend uses its own sampling settings and payload format.
#
# Convert between the provider message formats:
#   system            -> systemInstruction
#   {"type":"text"}   -> {"text": ...}
#   image_url(data:)  -> {"inlineData": {"mimeType", "data"}}
# Gemini 3 responses may include thought parts; retain only non-thought text.
_GEMINI_MODEL = os.environ.get("S12_GEMINI_MODEL", "gemini-3.1-pro-preview")
_GEMINI_HOST = "https://generativelanguage.googleapis.com/v1beta"


def _to_gemini(messages):
    from grading_judge import gemini_payload
    return gemini_payload(messages)


def chat_gemini(messages, max_tokens):
    from grading_judge import gemini_chat
    return gemini_chat(messages, max_tokens, model=A.model, timeout=A.timeout)

def slice_evidence(ev, rid):
    """Include only evidence needed by this rubric.

    Jump-cut grading does not require transcripts, and coverage grading does not require motion energy.
    Narrow the inputs to keep each judgment focused."""
    base = {"task": ev["task"], "how_measured": ev.get("how_measured")}
    if rid == "placement":
        base |= {
            "beat": ev.get("beat"),
            "trim_shape": ev.get("trim_shape"),
            "clips": [{k: v for k, v in c.items()
                       if k in ("clip", "clip_dur_s", "in_s", "out_s", "kept_s", "kept_pct",
                                "speech_at_in", "speech_at_out", "motion", "given")}
                      for c in ev["clips"]]}
    elif rid == "coverage":
        base |= {
            "beat": ev.get("beat"),
            "target_duration_s": ev.get("target_duration_s"),
            "assembled_duration_s": ev.get("assembled_duration_s"),
            "duration_delta_s": ev.get("duration_delta_s"),
            "clips": [{k: v for k, v in c.items()
                       if k in ("clip", "clip_dur_s", "kept_s", "kept_pct", "given",
                                "dropped_head_s", "dropped_tail_s",
                                "kept_words", "dropped_words")}
                      for c in ev["clips"]]}
    else:
        base |= {"joins": ev.get("joins")}
    return base


def sheet_part(task, ev, r):
    """Convert this rubric's contact sheet to an image_url part.

    Attach only the relevant sheet. Jump-cut grading concerns the two sides of a join;
    a kept-content sheet could redirect the judgment toward shot content.
    """
    sid = r.get("sheet")
    meta = (ev.get("contact_sheets") or {}).get(sid)
    if A.no_images or not meta:
        return None, None
    p = os.path.join(task, meta["file"] if isinstance(meta, dict) else meta)
    if not os.path.exists(p):
        return None, None
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    note = meta.get("note", "") if isinstance(meta, dict) else ""
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}}, note


def build_messages(task, ev, r):
    sl = slice_evidence(ev, r["id"])
    bands = "\n".join(f"    {k} — {v}" for k, v in sorted(r["bands"].items(), reverse=True))
    img, note = sheet_part(task, ev, r)
    head = f"""# {r['title']}

{r['question']}

{r['rubric']}

# Bands

{bands}
"""
    pic = (f"""
# The contact sheet

The image attached is the picture evidence for this rubric. {note}

Read it. The numbers say how long and how similar; only the frames say what is actually
there. Where the two disagree, say which you are trusting and why.
""" if img else """
# No contact sheet

You are working from numbers alone this time. Where a distinction needs the picture and you
do not have it, say the evidence is absent rather than guessing at what is on screen.
""")
    tail = f"""
# Evidence

Measured mechanically before you were called. `how_measured` says exactly how.

```json
{json.dumps(sl, ensure_ascii=False, indent=1)}
```

# How to answer

Pick the band whose condition the evidence meets — not the one that feels about right.

Cite what you used: clip ids, join numbers, and — when the frames decided it — the tile
labels you read (`p03·2/5`, `j07 pre-1`, `p05 post-out`).

Reply with exactly one line in this shape and nothing else:

BAND: <0-4>  ROWS: <ids you used>  WHY: <one sentence>"""
    parts = [{"type": "text", "text": head + pic + tail}]
    if img:
        parts.append(img)
    messages = [{"role": "system",
             "content": "You score one aspect of a rough cut, from measurements and frames "
                        "someone else prepared. Answer in the exact shape asked for."},
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
    if A.no_images or (r.get("sheet") and sheet_part(task, ev, r)[0] is None and not absent_by_submission):
        return {"rubric": r["id"], "title": r["title"], "band": None,
                "rows": "", "why": "", "retried_format": False,
                "saw_image": False, "sec": 0.0,
                "err": f"required contact sheet {r['sheet']} is missing"}
    messages = build_messages(task, ev, r)
    out, err = request(messages, A.max_tokens)

    band, rows, why = parse_band_response(out)
    retried_format = False
    if band is None and out:
        # A non-empty reasoning answer can still omit the machine-readable
        # final line.  Transport retries repeat that deterministic mistake, so
        # ask once more with the same evidence and an explicit format repair.
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
            err = f"Format retry failed — {retry_err}"
    if band is None and out:
        err = f"Response does not match the complete one-line BAND/ROWS/WHY format — {out[-160:]}"
    return {"rubric": r["id"], "title": r["title"], "band": band,
            "rows": rows[:140], "why": why,
            "retried_format": retried_format, "saw_video": True,
            "saw_image": sheet_part(task, ev, r)[0] is not None,
            "sec": round(time.time() - t0, 1), "err": err if band is None else ""}


def judge(task):
    ep = f"{task}/evidence/s3_evidence.json"
    name = task.split("training_task/")[-1]
    if not os.path.exists(ep):
        print(f"  {name:<30} ★ Missing evidence; run s3_evidence.py first", flush=True)
        return None
    ev = json.load(open(ep))
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lambda r: call(task, ev, r), RUBRICS))
    agg = aggregate({x["rubric"]: x["band"] for x in res})
    weights = {r["id"]: r["weight"] for r in RUBRICS}
    agg["weight_answered"] = round(
        sum(weights[k] for k in (agg.get("per_rubric") or {})), 3
    )
    agg["partial"] = bool(agg.get("missing"))
    out = f"{task}/evidence/s3_score{'_noimg' if A.no_images else ''}.json"
    json.dump({"task": name, "judge": judge_identity(),
               "images": not A.no_images, **agg, "rubrics": res},
              open(out, "w"), ensure_ascii=False, indent=1)
    if not A.quiet:
        s = agg["score"]
        print(f"  {name:<30} {('%.3f' % s) if s is not None else '  —  '}   " +
              "  ".join(f'{x["rubric"][:4]} {x["band"] if x["band"] is not None else "?"}'
                        + ("" if x["saw_image"] else "(no image)") for x in res), flush=True)
        for x in res:
            if x["why"]:
                print(f"      {x['title']}: {x['why'][:110]}", flush=True)
            elif x["err"]:
                print(f"      {x['title']}: ★ {x['err'][-90:]}", flush=True)
        if agg.get("penalty"):
            print(f"      ⚠ {agg['penalty']}", flush=True)
    return None if agg["partial"] else agg["score"]


tasks = json.load(open(A.tasks)) if A.tasks else [A.task]
_jid = judge_identity()
print(f"Grading {len(tasks)} submissions × 3 rubrics; judge {_jid['model']} ({_jid['provider']})"
      f"{'  [contact sheets disabled]' if A.no_images else '  [with contact sheets]'}\n", flush=True)
with ThreadPoolExecutor(max_workers=A.workers) as ex:
    got = [x for x in ex.map(judge, tasks) if x is not None]
if got:
    import statistics as st
    print(f"\nMedian {st.median(got):.3f}; range {min(got):.2f}–{max(got):.2f}")
