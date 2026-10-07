"""Public construction draft: audio_cuts.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/audio_cuts.py [options]\n\n  --film\n  --gap\n  --min-side\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

def main():
    try:
        from .release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)
    except ImportError:  # Direct script invocation.
        from release_config import (CODE_ROOT, DATA_ROOT, require_target,
            require_annotation_model, cloud_volume, run_legacy_rubric_hook)

    import argparse, json, subprocess
    import numpy as np
    import librosa

    # Keep artifact-generating work inside the explicit entry point; importing must not run it.


    P = argparse.ArgumentParser()
    P.add_argument("--film", required=True)
    P.add_argument("--gap", type=float, default=0.35)   # words closer than this = one run
    P.add_argument("--min-side", type=float, default=0.25)  # a straddle needs this much on both sides
    A = P.parse_args()

    E = f"{DATA_ROOT}/exports/{A.film}"
    T = f"{DATA_ROOT}/edit/task_{A.film}"
    SR, HOP = 16000, 160

    cl = json.load(open(f"{T}/stages/cut_list.json"))["cuts"]
    cuts = np.array([c["timeline_s"] for c in cl])

    # ---------------------------------------------------------------- speech runs
    tr = json.load(open(f"{E}/golden_transcript.json"))
    W = sorted([w for s in tr["segments"] for w in s["words"] if w["p"] > 0.4],
               key=lambda w: w["s"])
    runs, cur = [], None
    for w in W:
        if cur and w["s"] - cur[1] <= A.gap:
            cur = (cur[0], max(cur[1], w["e"]), cur[2] + " " + w["w"])
        else:
            if cur: runs.append(cur)
            cur = (w["s"], w["e"], w["w"])
    if cur: runs.append(cur)
    print(f"{len(W)} 词 -> {len(runs)} 个连续发声段  "
          f"（中位 {np.median([r[1]-r[0] for r in runs]):.2f}s，覆盖 "
          f"{100*sum(r[1]-r[0] for r in runs)/cuts.max():.0f}% 片长）")

    # ---------------------------------------------------------------- music stem
    y, _ = librosa.load(f"{E}/stems/music.wav", sr=SR)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    tt = np.arange(len(rms)) * HOP / SR
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13).T
    mfcc /= np.linalg.norm(mfcc, axis=1, keepdims=True) + 1e-9
    ONS = onset.mean() + 1.6 * onset.std()


    def at(t, arr, w):
        a, b = int((t - w) * SR / HOP), int((t + w) * SR / HOP)
        return arr[max(0, a):min(len(arr), b)]


    rows = []
    for c in cl:
        t = c["timeline_s"]
        # --- speech ---
        sp = [r for r in runs if r[0] < t - 0.02 and r[1] > t + 0.02]
        if sp and min(t - sp[0][0], sp[0][1] - t) >= A.min_side:
            r = sp[0]
            before, after = t - r[0], r[1] - t
            speech = {"type": "L" if before > after else "J",
                      "offset": round(min(before, after), 3),
                      "line": r[2][:60]}
        else:
            speech = {"type": "hard"}
        # --- music ---
        pre, post = at(t, rms, 0.6)[: int(0.6 * SR / HOP)], at(t, rms, 0.6)[int(0.6 * SR / HOP):]
        loc = at(t, onset, 0.12)
        m_pre, m_post = at(t, mfcc, 0.7)[: int(0.7 * SR / HOP)], at(t, mfcc, 0.7)[int(0.7 * SR / HOP):]
        tags = []
        if len(loc) and loc.max() > ONS: tags.append("hit")
        if len(pre) and len(post) and post.mean() > pre.mean() * 1.5 + 1e-6: tags.append("swell")
        if len(m_pre) > 3 and len(m_post) > 3:
            sim = float(m_pre.mean(0) @ m_post.mean(0) /
                        (np.linalg.norm(m_pre.mean(0)) * np.linalg.norm(m_post.mean(0)) + 1e-9))
            if sim < 0.80: tags.append("change")
            elif sim > 0.97 and speech["type"] == "hard": tags.append("bridge")
        rows.append({"idx": c["idx"], "t": round(t, 3), "speech": speech,
                     "music": tags, "music_sim": round(sim, 3) if len(m_pre) > 3 else None})

    from collections import Counter
    print(f"\n语音关系 {dict(Counter(r['speech']['type'] for r in rows))}")
    mt = Counter(x for r in rows for x in r["music"])
    print(f"音乐手法 {dict(mt)}   （无标记 {sum(1 for r in rows if not r['music'])} 条）")
    both = [r for r in rows if r["speech"]["type"] != "hard" and r["music"]]
    print(f"语音+音乐同时起作用的切点: {len(both)}")

    out = f"{E}/audio_cuts.json"
    json.dump({"film": A.film, "n": len(rows), "cuts": rows}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print(f"\n-> {out}")
    for r in rows:
        if r["speech"]["type"] != "hard":
            print(f"  #{r['idx']:<3} {r['speech']['type']}  ±{r['speech']['offset']:.2f}s  "
                  f"{'/'.join(r['music']) or '—':<14} 「{r['speech']['line']}」")


if __name__ == "__main__":
    main()
