"""Public construction draft: translate_script.py.

Generic construction code. See docs/data_construction.md for configuration,
entry points, and reconstruction limits.
"""

if __name__ == "__main__":
    import sys as _cli_sys
    if "--help" in _cli_sys.argv[1:] or "-h" in _cli_sys.argv[1:]:
        print('Usage: python data_construction/translate_script.py [options]\n\n  --film\n  --all\n  -j, --workers\n  -h, --help  Show this static help without loading optional dependencies.\n')
        raise SystemExit(0)

try:
    from .release_config import (
        DATA_ROOT, require_target, require_annotation_model, validate_data_slug, export_slug, select_films, annotation_cli
    )
except ImportError:  # Direct script invocation.
    from release_config import (
        DATA_ROOT, require_target, require_annotation_model, validate_data_slug, export_slug, select_films, annotation_cli
    )

import argparse, glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

ROOT = DATA_ROOT
EFFORT = "low"


def need(d):
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    return ([b for b in d.get("beats", []) if not text(b.get("name_en")) or not text(b.get("what_en"))],
            [c for c in d.get("chunks", []) if not text(c.get("what_en"))])


def source_fields(document):
    """Ignore only fields the translation step is authorized to add."""
    result = deepcopy(document)
    result.pop("logline_en", None)
    for beat in result.get("beats", []):
        beat.pop("name_en", None)
        beat.pop("what_en", None)
    for chunk in result.get("chunks", []):
        chunk.pop("what_en", None)
    return result


def run(slug, name):
    validate_data_slug(ROOT, slug)
    p = f"{ROOT}/edit/task_{slug}/stages/script.json"
    original = Path(p).read_bytes()
    d = json.loads(original)
    nb, nc = need(d)
    if not nb and not nc and isinstance(d.get("logline_en"), str) and d["logline_en"].strip():
        print(f"  [{name}] 已有译文，跳过", flush=True); return True
    payload = {"logline": d.get("logline"),
               "beats": [{"id": b["id"], "name": b["name"], "what": b["what"]} for b in d["beats"]],
               "chunks": [{"id": c["id"], "what": c["what"]} for c in d["chunks"]]}
    prompt = f"""这是《{name}》预告片场记表里要放进任务提示词的文字，现在是中文，需要英文版。

```json
{json.dumps(payload, ensure_ascii=False, indent=1)}
```

把它们译成英文，写到 `{p}`：

- 每个 beat 加 `name_en` 和 `what_en`
- 每个 chunk 加 `what_en`
- 顶层加 `logline_en`
- **原有的中文字段一个都不许改、不许删**，只是加新字段
- 其余结构（shots、id、shots 区间等）原样不动

译文要是**剪辑室里会说的英文**，不是逐字直译：具体、干脆、说清画面上发生了什么。
人名地名保留原样。译完把整个 JSON 写回同一个文件。"""
    t0 = time.time()
    r = subprocess.run(
        [annotation_cli(), "-p", prompt, "--model", require_annotation_model(), "--effort", EFFORT,
         "--allowedTools", "Read,Write,Edit,Bash(python3:*)",
         "--disallowedTools", "Agent,Task", "--permission-mode", "acceptEdits",
         "--add-dir", f"{ROOT}/edit/task_{slug}"],
        cwd=f"{ROOT}/edit/task_{slug}", capture_output=True, text=True)
    if r.returncode:
        Path(p).write_bytes(original)
        print(f"  [{name}] ✗ 翻译进程失败 ({r.returncode}): {(r.stderr or '')[-200:]}", flush=True)
        return False
    try:
        d2 = json.loads(Path(p).read_text(encoding="utf-8"))
        nb2, nc2 = need(d2)
        same = source_fields(d2) == source_fields(d)
        logline_ok = isinstance(d2.get("logline_en"), str) and bool(d2["logline_en"].strip())
        ok = not nb2 and not nc2 and logline_ok and same
    except (OSError, ValueError, TypeError, AttributeError) as e:
        Path(p).write_bytes(original)
        print(f"  [{name}] ✗ 翻译结果无效: {e}", flush=True)
        return False
    if not ok:
        Path(p).write_bytes(original)
    print(f"  [{name}] {time.time()-t0:.0f}s  "
          + ("✓" if ok else f"✗ 缺译 beats{len(nb2)}/chunks{len(nc2)}"
             + ("" if same else "  ★原文被改动")), flush=True)
    return ok


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--film"); P.add_argument("--all", action="store_true")
    P.add_argument("-j", "--workers", type=int, default=1)
    A = P.parse_args()
    require_target(P, A)
    films = {}
    for p in sorted(glob.glob(f"{ROOT}/exports/*/probe.json")):
        pb = json.load(open(p))
        slug = export_slug(pb, p)
        st = json.load(open(f"{ROOT}/exports/{slug}/state.json"))
        if st.get("script", {}).get("ok"):
            films[slug] = os.path.basename(pb["dir"])
    try:
        films = select_films(films, A)
    except ValueError as exc:
        P.error(str(exc))
    if not films:
        sys.exit("没有可翻译的片")
    print(f"翻译: {len(films)} 部\n", flush=True)
    if A.workers > 1:
        with ThreadPoolExecutor(max_workers=A.workers) as ex:
            n = sum(bool(x) for x in ex.map(lambda kv: run(*kv), films.items()))
    else:
        n = sum(bool(run(s, v)) for s, v in films.items())
    print(f"\n完成 {n}/{len(films)}")
    if n != len(films):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
