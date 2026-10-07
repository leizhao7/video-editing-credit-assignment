"""Surgical image-build extension over the existing verified production image.

Never replace a frozen trainer.py/fully_async_trainer.py wholesale: that would
erase its checkpoint, staleness, multimodal and transport fixes. This installer
keeps the previous media manifest and source preimages as provenance, and
extends affected digests only AFTER verifying every existing entry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .answers import canonical, require, sha

SKYRL_COMMIT = "59d4daedee24c7b1a79d857bbf322a6d195c3792"
ROOT = Path("/opt/SkyRL")
RECEIPT = Path("/opt/answer_opsd_source_extension.json")
PATHS = {"config": "skyrl/train/config/config.py", "fully_async": "skyrl/train/fully_async_trainer.py"}
MARKER = "# ANSWER_OPSD_GROUP_ADMISSION_V1\n"


def extension_paths():
    return {**{kind: ROOT / path for kind, path in PATHS.items()},
            "worker": ROOT / "skyrl/backends/skyrl_train/workers/worker.py", "fused": Path("/patch/fused_ce.py")}


def replace_once(source, old, new):
    require(source.count(old) == 1, "Unreviewed async source anchor; do not bypass or overwrite frozen source")
    return source.replace(old, new)


def extend_source(kind, source):
    require(MARKER not in source and 'group_filter' not in source, "Already extended or unknown async admission source")
    if kind == "config":
        source = replace_once(source, '    clear_kv_cache_on_weight_sync: bool = False\n',
            '    group_filter: Literal["reward_variance", "validity"] = "reward_variance"\n'
            '    """OPSD opt-in: validity admission retains zero-outcome K=1 trajectories."""\n'
            '    clear_kv_cache_on_weight_sync: bool = False\n')
    elif kind == "fully_async":
        source = replace_once(source,
            '        if self.sample_full_batch:\n'
            '            assert self.cfg.trainer.algorithm.zero_variance_filter, (\n'
            '                "trainer.fully_async.sample_full_batch=True requires trainer.algorithm.zero_variance_filter=True "\n'
            '                "(it is the async-native equivalent of dynamic_sampling=\'filter\')."\n'
            '            )\n',
            '        if self.sample_full_batch:\n'
            '            group_filter = self.cfg.trainer.fully_async.group_filter\n'
            '            if group_filter == "reward_variance":\n'
            '                assert self.cfg.trainer.algorithm.zero_variance_filter, (\n'
            '                    "sample_full_batch with reward_variance requires zero_variance_filter=True"\n'
            '                )\n'
            '            elif group_filter == "validity":\n'
            '                assert not self.cfg.trainer.algorithm.zero_variance_filter, (\n'
            '                    "validity admission must not subsequently mask singleton/zero-outcome trajectories"\n'
            '                )\n'
            '            else:\n'
            '                raise ValueError(f"Unknown async group_filter: {group_filter}")\n')
        source = replace_once(source, '        rewards = group.generator_output["rewards"]\n',
            '        if self.cfg.trainer.fully_async.group_filter == "validity":\n'
            '            age = self.global_step - group.global_step_when_scheduled\n'
            '            return 0 <= age <= self.max_staleness_steps and any(\n'
            '                any(mask) for mask in group.generator_output["loss_masks"]\n'
            '            )\n'
            '        rewards = group.generator_output["rewards"]\n')
    else:
        raise ValueError("Unknown extension target")
    compile(source, PATHS[kind], "exec")
    return MARKER + source


def verify_extension():
    from .launch import recipe_source_sha256
    record = json.loads(RECEIPT.read_bytes())
    require(record["skyrl_commit"] == SKYRL_COMMIT and record["recipe_sha256"] == recipe_source_sha256(),
            "Installed OPSD source/recipe identity differs")
    paths = extension_paths()
    require(set(record["sources"]) == set(paths), "Incomplete OPSD core extension")
    for kind, entry in record["sources"].items():
        require(entry["path"] == str(paths[kind]), "Core extension path changed")
        require(sha(Path(entry["path"]).read_bytes()) == entry["after_sha256"], "OPSD core source changed")
    return record


def main():
    from opencode_rl.install_training_fixes import verify_installed
    from opencode_rl.store import atomic_bytes
    from .launch import recipe_source_sha256
    from .topk_install import extend_topk_source
    require(os.environ.get("OCRL_ISOLATED_IMAGE_BUILD") == "1", "Only run during an isolated derived image build")
    if RECEIPT.exists():
        verify_installed()
        verify_extension()
        return
    media = verify_installed()
    old_manifest = canonical(media)
    edits, entries = [], {}
    for kind, path in extension_paths().items():
        before = path.read_bytes()
        transform = extend_source if kind in PATHS else extend_topk_source
        after = transform(kind, before.decode()).encode()
        if kind == "fully_async":
            for marker in ("S12_ASYNC_FINAL_CHECKPOINT_STEP", "S12_ASYNC_MAX_STEP_RANGE", "OCRL_EXACT_MEDIA_V7"):
                require(marker in after.decode(), "Required production safety fix is absent")
        entries[kind] = {"path": str(path), "before_sha256": sha(before), "after_sha256": sha(after)}
        edits.append((path, before, after, kind))
    for path, before, after, kind in edits:
        atomic_bytes(Path("/opt/answer_opsd_prior_sources") / (kind + ".py"), before)
        atomic_bytes(path, after)
    atomic_bytes(Path("/opt/answer_opsd_prior_sources/media_manifest.json"), old_manifest)
    for kind in ("fully_async", "worker", "fused"):
        media[kind]["after_sha256"] = entries[kind]["after_sha256"]
    atomic_bytes(Path("/opt/ocrl_training_fixes.json"), canonical(media))
    atomic_bytes(RECEIPT, canonical({"skyrl_commit": SKYRL_COMMIT, "sources": entries,
        "prior_media_manifest_sha256": sha(old_manifest), "recipe_sha256": recipe_source_sha256()}))
    verify_installed()
    verify_extension()


if __name__ == "__main__":
    main()
