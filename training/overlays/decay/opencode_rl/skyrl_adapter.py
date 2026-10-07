"""Opt-in trainer mixin; never monkey-patches the legacy trainer's classes."""
from __future__ import annotations

from pathlib import Path

from .contract import ParityError, canonical
from .store import atomic_bytes
from .training import (attach_modality_ids, compute_turn_advantages, verify_training_batch,
                       original_request_rows, causal_process_signal_groups,
                       packed_process_health_metrics)

OVERLAY_SOURCE_SHA256 = 'd4be27c707b34017dced22ceae9c165ea096b8ff7fdb9c39aae1f808f4dcb27a'


class ExactTurnTrainerMixin:
    def _ocrl_effective_process_weight(self):
        import json
        from prm_v2.live_config_v5 import decay_weight
        if not hasattr(self, '_ocrl_decay_plan'):
            self._ocrl_decay_plan = json.loads(Path(self.cfg.generator.ocrl_plan).read_bytes())
        return decay_weight(self._ocrl_decay_plan.get('process_decay'), self.global_step,
            constant_weight=float(self.cfg.generator.ocrl_process_weight))

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        if getattr(self.cfg.generator, 'ocrl_gradient_probe_interval', 0):
            self._check_ocrl_configuration()
            from prm_v2.live_config_v5 import deterministic_policy_actor
            PolicyWorker = deterministic_policy_actor(PolicyWorker)
        return super().build_models(PolicyWorker, CriticWorker, RefWorker)

    def _ocrl_packed_process_enabled(self):
        enabled = getattr(self.cfg.generator, 'ocrl_packed_prm_enabled', False)
        if type(enabled) is not bool:
            raise ParityError('Packed PRM configuration requires an explicit boolean opt-in')
        return enabled

    def _check_ocrl_configuration(self):
        cfg = self.cfg
        self._ocrl_packed_process_enabled()
        if cfg.trainer.algorithm.temperature != 1.0 or cfg.generator.sampling_params.temperature != 1.0:
            raise ParityError("Both rollout and training require temperature=1")
        if not cfg.generator.step_wise_trajectories or cfg.generator.merge_stepwise_output:
            raise ParityError("OpenCode requires unmerged step-wise multimodal training")
        if cfg.trainer.algorithm.loss_reduction != "token_mean":
            raise ParityError("Other loss reductions require explicit trajectory-weight handling")
        if cfg.trainer.algorithm.advantage_batch_normalize:
            raise ParityError("Extra batch normalization would change PRM/outcome contributions")
        if cfg.trainer.algorithm.advantage_estimator != "grpo":
            raise ParityError("Only the audited GRPO estimator is wired")
        if getattr(cfg.trainer.fully_async, "sample_full_batch", False) and not cfg.trainer.algorithm.zero_variance_filter:
            raise ParityError("Dynamic sampling requires explicit signal filtering")
        interval = getattr(cfg.generator, 'ocrl_gradient_probe_interval', 0)
        if type(interval) is not int or interval not in (0, 10):
            raise ParityError('Gradient probe is opt-in at a qualified ten-step interval')
        if interval and (not self._ocrl_packed_process_enabled()
                or cfg.trainer.algorithm.policy_loss_type != 'rollout_is'
                or cfg.trainer.algorithm.use_entropy_loss
                or cfg.trainer.update_epochs_per_batch != 1
                or cfg.trainer.train_batch_size != cfg.trainer.policy_mini_batch_size):
            raise ParityError('Gradient decomposition requires packed PRM, linear rollout_is, and one update per batch')

    def _normalize_advantages(self, data, mini_batch_boundaries, prompt_boundaries=None):
        """Apply the SAME native minibatch denominator to the two components.

        The normal combined training tensor is always produced by the original
        implementation. Diagnostic tensors never replace its floating-point sum.
        """
        probe = data.metadata.get('prm_gradient_probe')
        if not probe or not probe['enabled']:
            return super()._normalize_advantages(data, mini_batch_boundaries, prompt_boundaries)
        import torch
        self._check_ocrl_configuration()
        from prm_v2.gradient_source_guard_v5 import verify_normalizer
        normalizer_receipt = verify_normalizer(super()._normalize_advantages)
        if probe.get('component_normalized') is not False:
            raise ParityError('Gradient components were already normalized or lack provenance')
        original = data['advantages']
        components = ('ocrl_outcome_advantages', 'ocrl_process_advantages')
        if (any(key not in data or data[key].shape != original.shape for key in components)
                or not torch.allclose(data[components[0]] + data[components[1]], original,
                                      rtol=1e-5, atol=1e-7)):
            raise ParityError('Gradient components no longer reconstruct the training advantage')
        try:
            for key in components:
                data['advantages'] = data[key]
                result = super()._normalize_advantages(data, mini_batch_boundaries, prompt_boundaries)
                if result is not data:
                    raise ParityError('Native advantage normalization changed its in-place contract')
                data[key] = data['advantages']
        finally:
            data['advantages'] = original
        result = super()._normalize_advantages(data, mini_batch_boundaries, prompt_boundaries)
        if result is not data or not torch.allclose(data[components[0]] + data[components[1]],
                                                    data['advantages'], rtol=1e-5, atol=1e-9):
            raise ParityError('Native reduction no longer preserves component linearity')
        probe['component_normalized'] = True
        # Retain evidence without expanding the worker's versioned probe schema.
        data.metadata['prm_gradient_normalizer_receipt'] = normalizer_receipt
        return data

    def _ocrl_signal_groups(self, output):
        """Filter complete task groups by the actual combined GRPO+PRM signal.

        Intermediate zero outcome rewards are NOT independent siblings. PRM-only
        groups survive even when all original trajectory outcomes are zero.
        """
        import torch
        rows = output.get("ocrl_rows")
        if not rows or len(rows) != len(output["loss_masks"]):
            raise ParityError("Missing group provenance for dynamic sampling")
        if self._ocrl_packed_process_enabled():
            return causal_process_signal_groups(rows, output['loss_masks'],
                process_weight=self._ocrl_effective_process_weight(),
                norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
                tolerance=float(self.cfg.trainer.algorithm.zero_variance_filter_tol))
        live = torch.tensor([[bool(any(mask))] for mask in output["loss_masks"]], dtype=torch.float32)
        combined, _, _, _ = compute_turn_advantages(rows, live,
            process_weight=self._ocrl_effective_process_weight(),
            norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std)
        tolerance = float(self.cfg.trainer.algorithm.zero_variance_filter_tol)
        return {row["task_id"] for i, row in enumerate(rows) if float(combined[i].abs().max()) > tolerance}

    def _should_keep_group(self, group):
        self._check_ocrl_configuration()
        # This override must retain the legacy hard-staleness guard. Otherwise
        # our signal filter would bypass the patched base implementation.
        if self.global_step - group.global_step_when_scheduled > self.max_staleness_steps:
            return False
        if {row["task_id"] for row in group.generator_output.get("ocrl_rows", [])} != {group.uid}:
            raise ParityError("Async queue group no longer matches task provenance")
        return group.uid in self._ocrl_signal_groups(group.generator_output)

    def postprocess_generator_output(self, generator_output, uids,
                                     metrics_generator_output=None, metrics_uids=None):
        self._check_ocrl_configuration()
        output = dict(generator_output)
        if self.cfg.trainer.algorithm.zero_variance_filter:
            kept = self._ocrl_signal_groups(output)
            output["loss_masks"] = [mask if uid in kept else [0] * len(mask)
                                    for uid, mask in zip(uids, output["loss_masks"])]
            self.all_metrics["reward/num_zero_variance_filtered"] = len(set(uids) - kept)
        # The pinned base trainer's scalar-reward branch has an outcome-only
        # mask rewrite. Supplying exact token rewards uses its existing no-rewrite
        # branch, retaining its metrics without mutating shared config/classes.
        output["rewards"] = [reward if isinstance(reward, list) else
                             [0.0] * (len(response) - 1) + [float(reward)]
                             for reward, response in zip(output["rewards"], output["response_ids"])]
        return super().postprocess_generator_output(output, uids,
            metrics_generator_output=metrics_generator_output if metrics_generator_output is not None else generator_output,
            metrics_uids=metrics_uids if metrics_uids is not None else uids)

    def convert_to_training_input(self, generator_output, uids):
        self._check_ocrl_configuration()
        rows = generator_output.get("ocrl_rows")
        if not rows or len(rows) != len(generator_output["response_ids"]):
            raise ParityError("Missing turn provenance at trainer boundary")
        if any(row["task_id"] != uid for row, uid in zip(rows, uids)) or len(rows) != len(uids):
            raise ParityError("Task grouping changed before training")
        data = super().convert_to_training_input(generator_output, uids)
        attach_modality_ids(data, generator_output)
        enabled = self._ocrl_packed_process_enabled()
        verify_training_batch(data, generator_output, enable_packed_process=enabled)
        if any('causal_spans' in row for row in rows):
            if self.cfg.trainer.train_batch_size != self.cfg.trainer.policy_mini_batch_size:
                raise ParityError('Causal packets require one full policy minibatch for original KL normalization')
            from .prefix_transport import attach_prefix_transport
            attach_prefix_transport(data, generator_output, dp_size=self.dispatch.get_lcm_dp_size(),
                                    enable_packed_process=enabled)
            originals = original_request_rows(rows)
            data.metadata['avg_response_length'] = sum(r['response_length'] for r in originals)/len(originals)
        data.metadata["ocrl_rows"] = rows
        if data.get('ocrl_spans') is not None:
            from .dp_balance import rebalance_causal_batch
            self.all_metrics.update(rebalance_causal_batch(data,
                dp_size=self.dispatch.get_lcm_dp_size(),
                token_budget=self.cfg.trainer.max_tokens_per_microbatch,
                loss_reduction=self.cfg.trainer.algorithm.loss_reduction))
        return data

    def compute_advantages_and_returns(self, data):
        self._check_ocrl_configuration()
        rows = data.metadata.get("ocrl_rows")
        if not rows:
            raise ParityError("OpenCode request metadata lost in trainer")
        enabled = self._ocrl_packed_process_enabled()
        advantage, outcome, process, details = compute_turn_advantages(
            rows, data["loss_mask"], process_weight=self._ocrl_effective_process_weight(),
            norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            enable_packed_process=enabled,
            response_mask=data['response_mask'] if enabled else None)
        data["advantages"] = advantage
        data["returns"] = advantage.clone()
        interval = getattr(self.cfg.generator, 'ocrl_gradient_probe_interval', 0)
        if interval:
            import torch
            selected = (self.global_step == 1 or self.global_step % interval == 0) and self._ocrl_effective_process_weight() != 0
            data.metadata['prm_gradient_probe'] = {
                'version': 1, 'enabled': selected, 'global_step': self.global_step,
                'interval': interval, 'component_normalized': False,
                'total_loss_tokens': int(data['loss_mask'].sum()),
                'audit_dir': str(Path(self.cfg.generator.ocrl_contribution_dir) / 'gradient_probe')}
            if selected:
                # The row order here is AFTER causal DP balancing, as in the
                # existing training_packets audit saved below. Row indices travel
                # through real tensor selection, never unsliced list metadata.
                data['ocrl_outcome_advantages'] = outcome
                data['ocrl_process_advantages'] = process
                data['ocrl_probe_row_index'] = torch.arange(len(advantage),
                    device=advantage.device, dtype=torch.long).view(-1, 1)
        process_rows = original_request_rows(rows) if enabled else rows
        valid = [row["process"] for row in process_rows if row["process_valid"]]
        total_outcome = sum(row["outcome_adv_abs_sum"] for row in details)
        total_process = sum(row["process_adv_abs_sum"] for row in details)
        metrics = {
            "prm/coefficient": self._ocrl_effective_process_weight(),
            "prm/decay_global_step": self.global_step,
            "prm/outcome_reward_mean": sum(row["outcome_reward"] for row in details) / max(1, len(details)),
            "prm/process_reward_mean": sum(valid) / max(1, len(valid)),
            "prm/enabled": int(self._ocrl_effective_process_weight() != 0),
            "prm/process_contribution_fraction": total_process / max(1e-12, total_process + total_outcome),
            "ocrl/input_mismatch_count": 0,
        }
        if enabled:
            metrics.update({
                'prm/packed_enabled': 1,
                'prm/process_reward_valid_mean': metrics['prm/process_reward_mean'],
                'prm/process_reward_fallback_zero_mean': sum(valid) / max(1, len(process_rows)),
                'prm/process_valid_turns': len(valid),
                'prm/process_fallback_turns': len(process_rows) - len(valid),
                'prm/process_valid_fraction': len(valid) / max(1, len(process_rows)),
                'prm/outcome_contribution_fraction': total_outcome / max(1e-12, total_process + total_outcome),
            })
        # Means use original trajectories, not flattened turns: a longer
        # conversation must not inflate its stage's reported outcome reward.
        identity = {row['trajectory_id']: row for row in rows}
        by_stage = {}
        for detail in details:
            row = identity[detail['trajectory_id']]
            detail.update(task_id=row['task_id'], stage=row.get('stage', 'unknown'))
            by_stage.setdefault(detail['stage'], []).append(detail['outcome_reward'])
        metrics['outcome/mean'] = metrics['prm/outcome_reward_mean']
        for stage, rewards in by_stage.items():
            metrics[f'outcome/{stage}/mean'] = sum(rewards) / len(rewards)
            metrics[f'outcome/{stage}/count'] = len(rewards)
        if enabled:
            from prm_v2.live_metrics_v5 import batch_judge_metrics
            metrics.update(packed_process_health_metrics(rows, details,
                process_weight=self._ocrl_effective_process_weight()))
            metrics.update(batch_judge_metrics(rows,
                active_trajectory_ids={detail['trajectory_id'] for detail in details}))
        self.all_metrics.update(metrics)
        data.metadata.setdefault("metrics", {}).update(metrics)
        directory = Path(self.cfg.generator.ocrl_contribution_dir)
        path = directory / f"step_{self.global_step:06d}.json"
        if path.exists():
            raise ParityError("Refusing to overwrite existing contribution audit")
        atomic_bytes(path, canonical({"global_step": self.global_step, "metrics": metrics, "trajectories": details,
                                      'gradient_probe': data.metadata.get('prm_gradient_probe'),
                                      "requests": original_request_rows(rows),
                                      "dp_layout": data.metadata.get('ocrl_dp_layout'),
                                      "training_packets": [r for r in rows if 'causal_spans' in r]}))
        return data.to("cpu")


def trainer_classes():
    from .install_training_fixes import verify_installed
    verify_installed()
    from skyrl.train.trainer import RayPPOTrainer
    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
    import inspect
    # The inherited final safety save must use the last COMPLETED step. This
    # is baked into the reused legacy image, not a mutation of a live trainer.
    source = Path(inspect.getfile(FullyAsyncRayPPOTrainer)).read_text()
    if any(marker not in source for marker in ('S12_ASYNC_FINAL_CHECKPOINT_STEP', 'S12_ASYNC_MAX_STEP_RANGE')):
        raise ParityError('Async final-step/resume source fixes are missing from this image')

    class OpenCodeTrainer(ExactTurnTrainerMixin, RayPPOTrainer):
        pass

    class OpenCodeAsyncTrainer(ExactTurnTrainerMixin, FullyAsyncRayPPOTrainer):
        pass

    return OpenCodeTrainer, OpenCodeAsyncTrainer
