"""Opt-in SkyRL trainer mixin. No edits to the existing agent loop/GRPO adapter."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import torch

from .answers import canonical, require
from .configuration import check_objective
from .objective import per_turn_audit, sampled_advantages
from .teacher_inputs import encode_hint, make_teacher_batch
from .topk import OBJECTIVES

logger = logging.getLogger(__name__)


class AnswerOpsdTrainerMixin:
    def update_ref_with_policy(self):
        raise ValueError("OPSD teacher is permanently fixed at base weights; policy-to-ref updates are forbidden")

    def _topk_enabled(self):
        return self.opsd_plan["opsd"].get("loss") in OBJECTIVES.values()

    def __init__(self, *args, opsd_plan, private_answers, **kwargs):
        self.opsd_plan, self.private_answers = opsd_plan, private_answers
        check_objective(kwargs["cfg"])
        cfg = opsd_plan["opsd"]
        model = json.loads((Path(cfg["teacher_model_path"]) / "config.json").read_bytes())
        supported = model.get("text_config", model).get("max_position_embeddings", 0)
        require(cfg["teacher_max_context"] <= supported, "Pinned teacher model does not support the reserved context")
        if self._topk_enabled():
            require(cfg["top_k"] < model.get("text_config", model)["vocab_size"], "Teacher top-K must leave a tail bucket")
        tokenizer = kwargs["tokenizer"]
        from .turn_demonstrations import MODE, TurnDemonstrations
        self.turn_demonstrations = None
        if cfg.get("privileged_information", "answers") == MODE:
            require(self._topk_enabled(), "Turn demonstrations require the selected top-K KL objective")
            self._hint_ids = {}
            admission = getattr(kwargs.get("generator"), "demo_admission", None)
            self.turn_demonstrations = admission.demonstrations if admission else TurnDemonstrations(opsd_plan, tokenizer)
        else:
            self._hint_ids = {task: encode_hint(tokenizer, private_answers.hint(task), cfg["max_hint_tokens"])
                              for task in private_answers.tasks}
        super().__init__(*args, **kwargs)

    def _check_ocrl_configuration(self):
        check_objective(self.cfg)

    def _should_keep_group(self, group):
        self._check_ocrl_configuration()
        age = self.global_step - group.global_step_when_scheduled
        require(age >= 0, "Future policy version in rollout queue")
        if age > self.max_staleness_steps:
            return False
        output = group.generator_output
        rows = output.get("ocrl_rows", [])
        require(len(rows) == len(output["loss_masks"]) == 1 and rows[0]["task_id"] == group.uid,
                "Expected one provenance-bound trajectory per task")
        self.private_answers.get(group.uid)
        require(bool(rows[0].get("causal_spans")), "Unpacked trajectory escaped the append loop")
        if self.opsd_plan["opsd"].get("privileged_information") == "gemini_turn_action":
            from .demonstration_gate import FORMAT
            admission = rows[0].get("opsd_demo_admission")
            require(admission and admission["format"] == FORMAT and admission["request_ids"] ==
                    [span["request_id"] for span in rows[0]["causal_spans"]], "Missing exact demonstration admission")
            if not admission["ready"]:
                proof = admission["refusal"]
                require(proof["whole_trajectory_excluded"] and proof["trajectory_id"] == rows[0]["trajectory_id"]
                        and proof["task"] == group.uid and proof["reason"] in
                        ("PROHIBITED_CONTENT", "SAFETY", "BLOCKLIST", "SPII", "invalid_demonstration_output",
                         "demonstration_transport_unavailable"), "Untrusted demo exclusion")
                unavailable_errors = {"invalid_demonstration_output": "DemonstrationOutputError",
                                      "demonstration_transport_unavailable": "DemonstrationTransportError"}
                if proof["reason"] in unavailable_errors:
                    require(proof.get("output_attempts") == self.opsd_plan["opsd"]["demonstrator"]["attempts"] and
                            proof.get("last_error_type") == unavailable_errors[proof["reason"]],
                            "Unbounded/untrusted demonstration exclusion")
                return False  # Native sampler records filtered UID and refills to batch16.
            require(admission["refusal"] is None, "Conflicting demonstration admission")
        # Reward=0 remains trainable. This path never calls the GRPO signal filter.
        return bool(any(output["loss_masks"][0]))

    @torch.no_grad()
    def fwd_logprobs_values_reward(self, data):
        self._check_ocrl_configuration()
        require(not self.has_critic and self.ref_model is not None, "OPSD needs a frozen teacher and no critic")
        rows = data.metadata["ocrl_rows"]
        # ExactTurnTrainerMixin has already verified tokens/media and rebalanced DP.
        # Look up hints in THAT row order, then mirror its synchronization-only padding.
        require(0 < len(rows) <= len(data["sequences"]), "Missing real trajectory rows")
        for i in range(len(rows), len(data["sequences"])):
            require(not bool(data["loss_mask"][i].any()), "Unidentified DP row has live tokens")
        started = time.monotonic()
        teacher_data = None
        try:
            if self.turn_demonstrations is not None:
                from .turn_teacher import teacher_forward
                teacher = teacher_forward(self, data)
            else:
                hints = [self._hint_ids[row["task_id"]] for row in rows]
                hints += [hints[0]] * (len(data["sequences"]) - len(hints))
                teacher_data, metrics = make_teacher_batch(data, hints, self.tokenizer,
                    self.opsd_plan["opsd"]["teacher_max_context"])
                self.all_metrics.update(metrics)
                if self._topk_enabled():
                    from .topk_trainer import teacher_forward
                    teacher = teacher_forward(self, data, teacher_data)
                else:
                    teacher = self._execute_forward_pass("ref", teacher_data, key="logprobs", mini_batch_boundaries=None)
        except BaseException:
            # A rank may already have exited before another enters an FSDP
            # collective. Do not queue cleanup RPCs behind that failed forward:
            # they can stall and replace the original exception with ActorDied.
            logger.exception("OPSD teacher forward failed; preserving the original error")
            raise
        self.dispatch.empty_cache("ref")
        del teacher_data
        self.all_metrics["opsd/teacher_total_seconds"] = time.monotonic() - started
        if self.turn_demonstrations is None:
            self.all_metrics["opsd/teacher_forward_seconds"] = time.monotonic() - started
        keys = [k for k in ("sequences", "attention_mask", "mm_token_type_ids", "response_mask",
                            "pixel_values", "image_grid_thw") if data.get(k) is not None]
        student_data = data.select(keys=keys, metadata_keys=["response_length"])
        started = time.monotonic()
        try:
            if self._topk_enabled():
                from .topk_trainer import student_forward
                student = student_forward(self, data)
            else:
                student = self._execute_forward_pass("policy", student_data, key="logprobs",
                    mini_batch_boundaries=data.metadata.get("policy_mini_batch_boundaries"))
        except BaseException:
            logger.exception("OPSD student forward failed; preserving the original error")
            raise
        self.dispatch.empty_cache()
        self.all_metrics["opsd/student_forward_seconds"] = time.monotonic() - started
        require(teacher.shape == student.shape == data["loss_mask"].shape, "Packed logprobs lost original action alignment")
        data["base_action_log_probs"], data["action_log_probs"], data["values"] = teacher, student, None
        live = data["loss_mask"].bool()
        require(data.get("rollout_logprobs") is not None, "Actual behavior logprobs required for async IS")
        require(bool(torch.isfinite(data["rollout_logprobs"][live]).all()), "Nonfinite actual behavior logprobs")
        diff = (student[live] - data["rollout_logprobs"][live]).abs()
        self.all_metrics.update({"policy/rollout_train_logprobs_abs_diff_mean": diff.mean().item(),
                                 "policy/rollout_train_logprobs_abs_diff_max": diff.max().item()})
        return data

    def apply_reward_kl_penalty(self, data):
        if self._topk_enabled():
            require(data.get("opsd_ids") is not None and data.get("opsd_kl") is not None, "Top-K forward missing")
            data["rewards"] = torch.zeros_like(data["action_log_probs"])
            data.metadata["answer_opsd_guidance_ready"] = True
            return data
        # Reuse SkyRL's OPD hook to REPLACE rewards, never add to outcome labels.
        advantage, metrics = sampled_advantages(data["base_action_log_probs"], data["action_log_probs"],
                                                data["loss_mask"], self.opsd_plan["opsd"]["gap_clip"])
        data["rewards"] = advantage
        data.metadata["answer_opsd_guidance_ready"] = True
        self.all_metrics.update(metrics)
        return data

    def compute_advantages_and_returns(self, data):
        from opencode_rl.store import atomic_bytes
        require(data.metadata.pop("answer_opsd_guidance_ready", False), "Cannot accidentally train on outcome rewards")
        if self._topk_enabled():
            from .topk_trainer import per_turn_audit as distribution_audit
            # This field is only a carrier for SkyRL's global token denominator;
            # the distribution worker multiplies it by actual categorical KL.
            data["advantages"] = data["loss_mask"].float()
            data["returns"] = torch.zeros_like(data["advantages"])
            records = distribution_audit(data, self.private_answers)
        else:
            data["advantages"] = data["rewards"].detach().clone()
            data["returns"] = data["advantages"].clone()
            records = per_turn_audit(data, data["base_action_log_probs"], data["action_log_probs"],
                                     data["advantages"], self.private_answers)
        require(len(records) == 16 and len({r["task"] for r in records}) == 16, "Task batch size/identity changed")
        self.all_metrics["outcome/mean"] = sum(r["outcome_reward"] for r in records) / len(records)
        self.all_metrics["opsd/trajectories"] = len(records)
        for stage in {r["stage"] for r in records}:
            subset = [r for r in records if r["stage"] == stage]
            self.all_metrics[f"outcome/{stage}/mean"] = sum(r["outcome_reward"] for r in subset)/len(subset)
            signal = "topk_kl_sum" if self._topk_enabled() else "guidance_abs_sum"
            self.all_metrics[f"opsd/{stage}/{signal}"] = sum(r[signal] for r in subset)
        path = Path(self.opsd_plan["run_root"]) / "opsd_signals" / f"step_{self.global_step:06d}.json"
        require(not path.exists(), "Refusing to overwrite an earlier OPSD signal audit")
        atomic_bytes(path, canonical({"step": self.global_step, "catalog_sha256": self.private_answers.sha256,
            "objective": self.opsd_plan["opsd"], "trajectories": records,
            "metrics": {k: v for k, v in self.all_metrics.items() if k.startswith(("opsd/", "outcome/"))}}))
        data.metadata.setdefault("metrics", {}).update({k: v for k, v in self.all_metrics.items() if k.startswith("opsd/")})
        return data.to("cpu")

    def _normalize_advantages(self, data, mini_batch_boundaries, prompt_boundaries=None):
        if not self._topk_enabled():
            return super()._normalize_advantages(data, mini_batch_boundaries, prompt_boundaries)
        require(mini_batch_boundaries == [(0, len(data["loss_mask"]))] and
                torch.equal(data["advantages"], data["loss_mask"].float()),
                "Top-K normalization must use the original whole task minibatch once")
        count = data["loss_mask"].sum()
        require(count > 0, "Empty distribution minibatch")
        data["advantages"] = data["loss_mask"].float() / count
        data.metadata["opsd_topk_normalized"] = True
        return data
