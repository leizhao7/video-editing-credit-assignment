"""Fail-closed objective configuration, independent of the old GRPO guard."""
from __future__ import annotations

from pathlib import Path

from .answers import FORMAT, require
from .topk import LOSS_NAME, OBJECTIVES, kl_direction


def check_objective(cfg):
    t, g = cfg.trainer, cfg.generator
    a, f = t.algorithm, t.fully_async
    require(t.train_batch_size == t.policy_mini_batch_size == 16 and g.n_samples_per_prompt == 1,
            "Answer OPSD keeps 16 tasks/update, with exactly one rollout/task")
    require(a.advantage_estimator == "answer_opsd" and not a.advantage_batch_normalize and not a.zero_variance_filter,
            "No group or batch normalization / reward-variance masking in pure OPSD")
    require(a.policy_loss_type in ("rollout_is", LOSS_NAME) and a.use_kl_in_reward and not a.use_kl_loss,
            "Use the explicitly selected OPSD objective; no additional KL loss")
    require(a.kl_loss_coef == 0 and not a.use_entropy_loss and g.ocrl_process_weight == 0,
            "Pure distillation must not silently inherit outcome/PRM/entropy/reference losses")
    require(a.loss_reduction == "token_mean" and a.temperature == g.sampling_params.temperature == 1.0,
            "Keep exact T=1 token-mean training")
    require(g.sampling_params.top_p == 1.0 and g.sampling_params.top_k == -1,
            "Truncated sampling would invalidate the selected-token distillation estimator")
    require(g.step_wise_trajectories and not g.merge_stepwise_output and g.ocrl_causal_prefix,
            "Exact append-only one-packet training is required")
    require(f.enabled and f.sample_full_batch and f.group_filter == "validity" and not f.simulate_training,
            "Use validity-only full-batch async admission")
    require(not t.update_ref_every_epoch, "Teacher is a fixed initial checkpoint, not silently updated/EMA")
    require(Path(t.ref.model.path).resolve() == Path(t.policy.model.path).resolve(),
            "Self-distillation teacher must start from the same pinned model as the student")
    require(not t.policy.language_model_only and not t.ref.language_model_only,
            "Teacher and student must both retain the vision model")


def check_plan(plan, contract, cfg):
    from append_agent_rl.contract import validate_runtime
    from opencode_rl.assets import files_hash
    import opencode_rl

    check_objective(cfg)
    validate_runtime(plan["config"], contract)
    package = Path(opencode_rl.__file__).parent
    require(files_hash(package, [p.name for p in package.iterdir() if p.suffix in (".py", ".mjs")]) == contract.integration_sha256,
            "Student integration differs from inference source")
    require(plan["harness"] == "append-opencode-tools-v1" and plan["causal_prefix"] is True, "Wrong student harness")
    opsd = plan["opsd"]
    t = cfg.trainer
    require(opsd.get("runtime_profile") in ("source", "frozen_v15"), "Unknown harness runtime profile")
    require(opsd["format"] == FORMAT and opsd["teacher_mode"] == "fixed_initial" and
            opsd["loss"] in ("sampled_reverse_kl", *OBJECTIVES.values()) and opsd["outcome_loss_weight"] == 0,
            "Unknown distillation objective")
    if opsd["loss"] in OBJECTIVES.values():
        kl_direction(opsd)
        require(t.strategy == "fsdp", "Top-K distribution head is qualified only for the existing FSDP VLM path")
        require(t.algorithm.policy_loss_type == LOSS_NAME and opsd["gap_clip"] is None and
                type(opsd["top_k"]) is int and 1 < opsd["top_k"] <= 4096 and opsd["topk_tail"] is True and
                type(opsd["projection_chunk_size"]) is int and 0 < opsd["projection_chunk_size"] <= 1024 and
                opsd["action_importance_weighting"] is False, "Invalid top-K distribution objective")
    else:
        require(t.algorithm.policy_loss_type == "rollout_is", "Sampled/Top-K objective wiring differs")
    require(type(opsd["max_hint_tokens"]) is int and opsd["max_hint_tokens"] > 0 and
            type(opsd["teacher_max_context"]) is int and
            opsd["teacher_max_context"] >= contract.max_context + opsd["max_hint_tokens"],
            "Reserve teacher-only answer headroom without shortening the student's context")
    gap_clip = opsd["gap_clip"]
    require(gap_clip is None or (type(gap_clip) in (float, int) and 0 < gap_clip < float("inf")), "Invalid guidance clipping")
    t, g = cfg.trainer, cfg.generator
    require(t.algorithm.max_seq_len == contract.max_context and g.sampling_params.max_generate_length == contract.max_output,
            "Student context/output budget changed")
    require(not t.eval_before_train and t.eval_interval == plan["eval_interval"] == 10 and plan.get("validation"),
            "Keep the existing validation schedule/split")
    require(g.eval_n_samples_per_prompt == 1 and g.eval_sampling_params.temperature == contract.temperature and
            g.eval_sampling_params.top_p == contract.top_p and g.eval_sampling_params.top_k == contract.top_k and
            g.eval_sampling_params.max_generate_length == contract.max_output, "Validation sampling differs")
    require(t.ckpt_interval == 2 and t.max_ckpts_to_keep == 2, "Keep checkpoint interval2/max2; persistence protects every10")
    require(len(plan["engines"]) == len({e["url"] for e in plan["engines"]}) == 4 and
            t.placement.policy_num_nodes == 1 and t.placement.policy_num_gpus_per_node == 4 and
            t.placement.ref_num_nodes == 1 and t.placement.ref_num_gpus_per_node == 4 and
            t.placement.colocate_policy_ref and not t.placement.colocate_all,
            "Expected four inference plus four policy/ref-colocated training GPUs")
    require(g.inference_engine.weight_sync_backend == "nccl" and cfg.data.dataloader.num_workers == 0,
            "Weight-sync / dataloader profile changed")
    require(Path(t.ref.model.path) == Path(opsd["teacher_model_path"]) == Path(plan["model_path"]), "Teacher identity differs")
    require(Path(plan["model_path"]).name == contract.model_revision, "Model weights are not bound to the harness revision")
    private_path = Path(opsd["private_answers_path"])
    require(private_path.is_absolute() and not private_path.is_relative_to(plan["prepared_bundles"]) and
            not private_path.is_relative_to(plan["validation"]["bundles"]), "Never put answers in public task bundles")
    require(plan["mode"] in ("train", "smoke") and t.max_training_steps == plan["requested_steps"], "Unbound training length")
    require(plan["mode"] != "smoke" or 1 <= t.max_training_steps <= 2, "Smoke is limited to two steps")
    if t.resume_mode != "none":
        from .launch import check_resume
        check_resume(t.resume_path, plan)
