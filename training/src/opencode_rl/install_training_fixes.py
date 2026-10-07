"""Source-checked changes ONLY inside the separate OpenCode image.

The legacy checkout and legacy image are never edited. All tensor consumers,
including recomputation, reference forwards and DP padding, carry explicit media
positions. No sampled token is deleted, replaced or loss-masked by this fix.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .contract import ParityError, canonical
from .store import atomic_bytes

MARKER = '# OCRL_EXACT_MEDIA_V7\n'
HF_SHA = '6b5f75e3bd6989cff0dd5470c7abc696f50dfcbad99c1a141c3ae3c6dbf82fe4'


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ParityError(f'Unreviewed source anchor: {old[:100]!r}: {source.count(old)} != {count}')
    return source.replace(old, new)


def transform(kind, source):
    if MARKER in source:
        return source
    if any(f'# OCRL_EXACT_MEDIA_V{version}\n' in source for version in (4, 5, 6)):
        raise ParityError('Build training wiring from the reviewed unpatched base, not an older patched snapshot')
    if kind == 'fully_async':
        source = replace(source, '    async def on_rollout_rejected(self) -> None:',
            '    async def on_rollout_grade_discarded(self) -> None:\n'
            '        async with self._cond:\n'
            '            assert self._stat.running > 0\n'
            '            self._stat.running -= 1\n'
            '            self._stat.filtered += 1\n'
            '            self._cond.notify_all()\n\n'
            '    async def on_rollout_rejected(self) -> None:')
        start = '                if "disable_tqdm" in inspect.signature(self.generator.generate).parameters:\n'
        end = '                group_completion_time_s = time.monotonic() - group_start_time\n'
        if source.count(start) != 1 or source.count(end) != 1:
            raise ParityError('Unreviewed async group generation boundary')
        block = source[source.index(start):source.index(end)]
        source = replace(source, block,
            '                from opencode_rl.grading_client import GradeRetriesExhausted\n'
            '                try:\n' + ''.join('    ' + line if line.strip() else line for line in block.splitlines(True)) +
            '                except GradeRetriesExhausted as exc:\n'
            '                    await self._staleness_manager.on_rollout_grade_discarded()\n'
            '                    slot_acquired = False\n'
            '                    await self.async_train_dataloader.mark_filtered_uids([uids[0]])\n'
            '                    key = "ocrl/grading_discarded_groups_total"\n'
            '                    self._ocrl_grading_discarded_total = getattr(self, "_ocrl_grading_discarded_total", 0) + 1\n'
            '                    self.all_metrics[key] = self._ocrl_grading_discarded_total\n'
            '                    logger.warning(f"Ocrl grading exhausted; archived group {uids[0]}, resampling: {exc}")\n'
            '                    continue\n')
    elif kind == 'trainer':
        source = replace(source, '        fwd_keys = ["sequences", "attention_mask"]',
                         # Reference/old-policy forward must retain real response
                         # lengths when the microbatch removes common left PAD.
                         '        fwd_keys = ["sequences", "attention_mask", "mm_token_type_ids", "response_mask"]')
    elif kind == 'worker':
        # Insert before iterator construction (which synchronizes DP counts).
        anchor = ('        microbatch_iterator = get_microbatch_iterator(\n'
                  '            data,\n'
                  '            micro_batch_size=self.cfg.micro_train_batch_size_per_gpu,\n'
                  '            max_tokens_per_microbatch=self.cfg.max_tokens_per_microbatch,\n'
                  '        )\n'
                  '        all_metrics = defaultdict(list)\n'
                  '        all_loss_fn_outputs = []  # Handle separately from scalar metrics')
        source = replace(source, anchor,
            '        if data.get("ocrl_spans") is not None:\n'
            '            from opencode_rl.prefix_backward import prefix_forward_backward\n'
            '            return prefix_forward_backward(self, data, loss_fn, loss_fn_config, return_per_token_outputs)\n'
            + anchor)
        source = replace(source, '                image_grid_thw=experience.image_grid_thw,',
                         '                image_grid_thw=experience.image_grid_thw,\n'
                         '                mm_token_type_ids=experience.mm_token_type_ids,', 2)
        source = replace(source, '        image_grid_thw = micro_batch.get("image_grid_thw", None)',
                         '        image_grid_thw = micro_batch.get("image_grid_thw", None)\n'
                         '        mm_token_type_ids = micro_batch["mm_token_type_ids"]', 2)
        source = replace(source, '                image_grid_thw=image_grid_thw,',
                         '                image_grid_thw=image_grid_thw,\n'
                         '                mm_token_type_ids=mm_token_type_ids,', 2)
    elif kind == 'experience':
        source = replace(source, '    image_grid_thw: Optional[TensorList] = None',
                         '    image_grid_thw: Optional[TensorList] = None\n'
                         '    mm_token_type_ids: Optional[torch.Tensor] = None\n'
                         '    ocrl_spans: Optional[TensorList] = None\n'
                         '    ocrl_packet_length: Optional[torch.Tensor] = None\n'
                         '    ocrl_original_count: Optional[torch.Tensor] = None')
        source = replace(source, '        self.sequences = to(self.sequences, device)',
                         '        self.sequences = to(self.sequences, device)\n'
                         '        if self.mm_token_type_ids is not None:\n'
                         '            self.mm_token_type_ids = self.mm_token_type_ids.to(device)\n'
                         '        for key in ("ocrl_spans", "ocrl_packet_length", "ocrl_original_count"):\n'
                         '            value = getattr(self, key)\n'
                         '            if value is not None:\n'
                         '                setattr(self, key, value.to(device))')
        source = replace(source, '        self.sequences = pin_memory(self.sequences)',
                         '        self.sequences = pin_memory(self.sequences)\n'
                         '        if self.mm_token_type_ids is not None:\n'
                         '            self.mm_token_type_ids = pin_memory(self.mm_token_type_ids)\n'
                         '        for key in ("ocrl_packet_length", "ocrl_original_count"):\n'
                         '            value = getattr(self, key)\n'
                         '            if value is not None:\n'
                         '                setattr(self, key, pin_memory(value))\n'
                         '        if self.ocrl_spans is not None:\n'
                         '            self.ocrl_spans = TensorList([pin_memory(t) for t in self.ocrl_spans.tensors])')
    elif kind == 'iterator':
        source = replace(source, '            image_grid_thw=batch.get("image_grid_thw"),',
                         '            image_grid_thw=batch.get("image_grid_thw"),\n'
                         '            mm_token_type_ids=batch["mm_token_type_ids"],\n'
                         '            ocrl_spans=batch.get("ocrl_spans"),\n'
                         '            ocrl_packet_length=batch.get("ocrl_packet_length"),\n'
                         '            ocrl_original_count=batch.get("ocrl_original_count"),')
        source = replace(source, '        data.metadata["is_padding_batch"] = True',
                         '        data["mm_token_type_ids"] = torch.zeros_like(data["sequences"])\n'
                         '        if self.data.get("ocrl_spans") is not None:\n'
                         '            from skyrl.backends.skyrl_train.training_batch import TensorList\n'
                         '            data["ocrl_spans"] = TensorList([torch.empty((0, 2), dtype=torch.long, device=device)])\n'
                         '            data["ocrl_packet_length"] = torch.zeros((1, 1), dtype=torch.long, device=device)\n'
                         '            data["ocrl_original_count"] = self.data["ocrl_original_count"][:1].clone()\n'
                         '        data.metadata["is_padding_batch"] = True')
    elif kind == 'training_batch':
        source = replace(source, '            padding = TensorList([tensor[0].clone() for _ in range(pad_size)])',
            '            padding = TensorList([tensor[0].new_empty((0, 2)) if key == "ocrl_spans"\n'
            '                                  else tensor[0].clone() for _ in range(pad_size)])')
    elif kind == 'wrapper':
        source = replace(source, '        self.temperature = temperature',
                         '        from opencode_rl.forward_guard import configure_numerics\n'
                         '        configure_numerics()\n'
                         '        self.temperature = temperature')
        source = replace(source, '        has_image_inputs = pixel_values is not None or image_grid_thw is not None',
                         '        from opencode_rl.forward_guard import check_forward\n'
                         '        check_forward(sequences, attention_mask, mm_token_type_ids, temperature)\n'
                         '        has_image_inputs = pixel_values is not None or image_grid_thw is not None')
    elif kind == 'fused':
        source = replace(source, '    if image_grid_thw is not None and mm_token_type_ids is None:',
                         '    from opencode_rl.forward_guard import check_forward\n'
                         '    check_forward(sequences, attention_mask, mm_token_type_ids, temperature)\n'
                         '    if image_grid_thw is not None and mm_token_type_ids is None:')
        source = replace(source, '    sequences_rolled = torch.roll(sequences, shifts=-1, dims=1)\n', '')
        source = replace(source,
            '    log_probs, entropy = LinearCrossEntropy.apply(\n'
            '        hidden, weight, sequences_rolled, float(temperature), "none", None)\n'
            '    log_probs = log_probs.view(sequences.shape)',
            '    from opencode_rl.response_head import response_suffix_linear_ce\n'
            '    action_log_probs, entropy = response_suffix_linear_ce(\n'
            '        hidden, weight, sequences, num_actions, temperature,\n'
            '        linear_ce=LinearCrossEntropy.apply, compute_entropy=compute_entropy)')
        source = replace(source, '        nan_diagnostic.watch_tensor(log_probs, "fused_log_probs")',
                         '        nan_diagnostic.watch_tensor(action_log_probs, "fused_log_probs")')
        source = replace(source,
            '    if isinstance(num_actions, list):\n'
            '        num_actions = num_actions[0] if len(num_actions) == 1 else __import__("numpy").array(num_actions)\n'
            '    action_log_probs = log_probs[:, -num_actions - 1: -1]\n',
            '    # Response positions were selected before vocabulary projection.\n')
    elif kind == 'hf':
        if hashlib.sha256(source.encode()).hexdigest() != HF_SHA:
            raise ParityError('Unreviewed Qwen3.5 implementation')
        source = replace(source, '        video_features: torch.FloatTensor | None = None,\n',
                         '        video_features: torch.FloatTensor | None = None,\n'
                         '        mm_token_type_ids: torch.Tensor | None = None,\n')
        source = replace(source, '            special_image_mask = input_ids == self.config.image_token_id',
                         '            special_image_mask = ((mm_token_type_ids == 1) if mm_token_type_ids is not None\n'
                         '                                  else (input_ids == self.config.image_token_id))')
        source = replace(source, '            special_video_mask = input_ids == self.config.video_token_id',
                         '            special_video_mask = ((mm_token_type_ids == 2) if mm_token_type_ids is not None\n'
                         '                                  else (input_ids == self.config.video_token_id))')
        for media in ('image', 'video'):
            source = replace(source, f'input_ids, inputs_embeds=inputs_embeds, {media}_features={media}_embeds',
                             f'input_ids, inputs_embeds=inputs_embeds, {media}_features={media}_embeds, mm_token_type_ids=mm_token_type_ids')
    elif kind == 'dispatch':
        source = replace(source, 'import time\n', 'import time\nimport asyncio\n')
        # ray.get inside this function is blocking. Keep relay SSE, admission,
        # grading completion and log commits alive during the NCCL broadcast.
        source = replace(source, '            _broadcast_and_finish()',
                         '            await asyncio.to_thread(_broadcast_and_finish)', 6)
        source = replace(source, '    async def save_weights_for_sampler(self, model_id: Optional[str] = None) -> None:',
                         '    @__import__("opencode_rl.weight_sync", fromlist=["weight_transaction"]).weight_transaction\n'
                         '    async def save_weights_for_sampler(self, model_id: Optional[str] = None) -> None:')
    else:
        raise ParityError(f'Unknown source target {kind}')
    # Put marker at EOF to preserve future-import placement and line references.
    source += '\n' + MARKER
    compile(source, f'<ocrl-{kind}>', 'exec')
    return source


SKYRL_FILES = {
    'fully_async': 'skyrl/train/fully_async_trainer.py',
    'trainer': 'skyrl/train/trainer.py',
    'worker': 'skyrl/backends/skyrl_train/workers/worker.py',
    'experience': 'skyrl/train/dataset/replay_buffer.py',
    'iterator': 'skyrl/backends/skyrl_train/workers/worker_utils.py',
    'training_batch': 'skyrl/backends/skyrl_train/training_batch.py',
    'wrapper': 'skyrl/backends/skyrl_train/workers/model_wrapper.py',
    'dispatch': 'skyrl/backends/skyrl_train/workers/worker_dispatch.py',
}


def verify_installed():
    report = Path('/opt/ocrl_training_fixes.json')
    if not report.exists():
        raise ParityError('OpenCode training requires its separate, patched image')
    entries = json.loads(report.read_bytes())
    if set(entries) != set(SKYRL_FILES) | {'hf', 'fused'}:
        raise ParityError('Incomplete actual training implementation manifest')
    for entry in entries.values():
        actual = Path(entry['path']).read_bytes()
        if MARKER.encode() not in actual or hashlib.sha256(actual).hexdigest() != entry['after_sha256']:
            raise ParityError(f'Actual training implementation was overwritten: {entry["path"]}')
    return entries


def main():
    if os.environ.get('OCRL_ISOLATED_IMAGE_BUILD') != '1':
        raise ParityError('Run only while building the isolated OpenCode image')
    import transformers
    targets = {kind: Path('/opt/SkyRL') / name for kind, name in SKYRL_FILES.items()}
    targets['fused'] = Path('/patch/fused_ce.py')
    targets['hf'] = Path(transformers.__file__).parent / 'models/qwen3_5/modeling_qwen3_5.py'
    edits, report = [], {}
    for kind, path in targets.items():
        before = path.read_text()
        after = transform(kind, before)
        edits.append((path, after))
        report[kind] = {'path': str(path), 'before_sha256': hashlib.sha256(before.encode()).hexdigest(),
                        'after_sha256': hashlib.sha256(after.encode()).hexdigest()}
    for path, after in edits:
        atomic_bytes(path, after.encode())
    atomic_bytes(Path('/opt/ocrl_training_fixes.json'), canonical(report))
    print(canonical(report).decode(), flush=True)


if __name__ == '__main__':
    main()
