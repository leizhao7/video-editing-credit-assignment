"""One real Gemini audio-rubric call after full FFmpeg/ASR evidence construction.

Synthetic transport smoke, NOT judge quality calibration or a validation score.
"""
import json
import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def command(args):
    proc = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", *args], capture_output=True, timeout=90)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode()[-1000:])


def load_cli(name, root):
    empty = root / "empty_tasks.json"
    empty.write_text("[]")
    before = sys.argv
    try:
        sys.argv = [name, "--tasks", str(empty)]
        return runpy.run_path("/opt/s12tools/" + name, run_name="__main__")
    finally:
        sys.argv = before


def smoke(root):
    root = Path(root)
    source = root/"initial_state"
    (source/"shots").mkdir(parents=True)
    (root/"output").mkdir()
    for sid, color in (("p01", "red"), ("p02", "blue")):
        command(["-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=24:d=2", "-an",
                 "-threads", "1", "-y", str(source/"shots"/(sid+".mp4"))])
    command(["-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-y", str(source/"music.wav")])
    command(["-i", str(source/"shots/p01.mp4"), "-i", str(source/"shots/p02.mp4"),
             "-i", str(source/"music.wav"), "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
             "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-threads", "1",
             "-c:a", "aac", "-y", str(root/"output/preview.mp4")])
    manifest = {"shots": [{"id": sid, "dur_s": 2., "has_sound": False, "origin": "film"}
                           for sid in ("p01", "p02")],
                "joins": [{"join": 1, "t_s": 2., "leaving": "p01", "arriving": "p02"}],
                "dialogue": [], "stretch_dur_s": 4.}
    (source/"manifest.json").write_text(json.dumps(manifest))
    (root/"output/edit.json").write_text(json.dumps({"joins": [{"join": 1, "picture": "cut", "sound": "under"}],
                                                   "music": {"starts_at_s": 0.}}))
    (root/"task_prompt.md").write_text("Transport test: keep red then blue picture, four seconds total. "
        "Use the supplied steady tone as background. There is no dialogue and no musical beat to synchronize.")
    ev = load_cli("s4e_evidence.py", root)["build"](str(root))
    jobs = []
    for name in ("s4e_judge.py", "s4e_judge_ref.py"):
        judge = load_cli(name, root)
        # Synthetic task has no trailer reference; do not invent one.
        if "golden_block" in judge:
            judge["build_messages"].__globals__["golden_block"] = lambda task: None
        jobs.extend((judge["call"], str(root), ev, rubric) for rubric in judge["RUBRICS"])

    # The S3 synthetic fixture deliberately uses equal-length windows: that is
    # explicitly requested and must not be auto-zeroed as 'arithmetic not editing'.
    s3task = root/"s3"
    (s3task/"initial_state").mkdir(parents=True)
    (s3task/"output").mkdir()
    (s3task/"verifier").mkdir()
    import shutil
    for sid in ("p01", "p02"):
        shutil.copy2(source/"shots"/(sid+".mp4"), s3task/"initial_state"/(sid+".mp4"))
    man3 = {"clips": [{"id": sid, "file": sid+".mp4", "clip_dur_s": 2.} for sid in ("p01", "p02")],
            "target_duration_s": 2.}
    cuts = {"cuts": [{"clip": sid, "in_s": 0., "out_s": 1.} for sid in ("p01", "p02")]}
    key = {"task": "transport_s3", "fps": 24., "target_duration_s": 2., "dur_tol_s": .1668,
           "cards_excluded_s": 0., "cuts": [dict(c, scored=True, dur_s=1., in_tol_s=.25, plateau_frames=0)
                                            for c in cuts["cuts"]]}
    (s3task/"initial_state/manifest.json").write_text(json.dumps(man3))
    (s3task/"output/cuts.json").write_text(json.dumps(cuts))
    (s3task/"verifier/answer.json").write_text(json.dumps(key))
    (s3task/"task_prompt.md").write_text("Picture-only transport test: select one second of red then one second of blue. "
        "Equal lengths, static shots and a hard color change are intentional. No finished audio mix is required.")
    s3ev = load_cli("s3_evidence.py", root)["build"](str(s3task))
    for name in ("s3_judge.py", "s3_judge_ref.py"):
        judge = load_cli(name, root)
        evidence = judge["evidence"](str(s3task)) if "evidence" in judge else s3ev
        jobs.extend((judge["call"], str(s3task), evidence, rubric) for rubric in judge["RUBRICS"])
    def grade(job):
        call, task, evidence, rubric = job
        return call(task, evidence, rubric)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(grade, jobs))
    receipts = [json.loads(p.read_text()) for p in root.rglob("*.provider.json")]
    report = {"scope": "13-rubric synthetic continuous video/audio transport; not quality calibration",
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path("/opt/s12tools").glob("*.py")},
              "render_audio": ev["render_audio"], "s4_video": ev["continuous_video"],
              "s3_video": s3ev["continuous_video"], "judge_results": results,
              "provider_receipts": receipts,
              "passed": len(results) == 13 and all(r.get("band") is not None and r.get("saw_video") for r in results)
                        and bool(receipts) and all(r.get("modelVersion") == "gemini-3.1-pro-preview" for r in receipts)}
    (root/"smoke_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
