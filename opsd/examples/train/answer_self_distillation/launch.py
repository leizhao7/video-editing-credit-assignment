"""Build explicit new-run plans/commands without submitting anything remotely.

Start from the deployed append harness plan and its exact effective base argv.
Do NOT use an old plan to relabel newer loop/tool sources: runtime hash checks
remain mandatory. The Modal image must contain this recipe and the opt-in core
group_filter change in addition to the existing production source fixes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .answers import FORMAT, canonical, require, sha

MODULE = "examples.train.answer_self_distillation.main_answer_opsd"


def recipe_source_sha256():
    root = Path(__file__).parent
    return sha(canonical({p.name: sha(p.read_bytes()) for p in sorted(root.glob("*.py"))}))


def objective_identity(plan):
    return {"opsd": plan["opsd"], "contract_sha256": sha(canonical(plan["contract"])),
            "prepared_sha256": plan["prepared_sha256"], "grader_hashes": plan["grader_hashes"]}


def check_resume(path, plan):
    receipt = Path(path) / "opsd_resume.json"
    require(receipt.is_file(), "Cannot resume an old GRPO optimizer as OPSD; a weights-only warm start is a separate experiment")
    saved = json.loads(receipt.read_bytes())
    require(saved["identity"] == objective_identity(plan),
            "Public draft supports same-identity resume only; historical run-specific migration adapters are not distributed")
    require(0 < saved["step"] < plan["requested_steps"], "Resume step already finished or invalid")


def new_plan(source, *, run_root, checkpoint_root, private_answers_path=None, private_answers_sha256=None,
             mode="smoke", steps=2, max_hint_tokens=2048, gap_clip=None, runtime_profile="frozen_v15",
             top_k=64, projection_chunk_size=128, direction="reverse", privileged_information="answers",
             demonstrator=None):
    from .topk import OBJECTIVES
    require(direction in OBJECTIVES, "Unknown KL direction")
    require(gap_clip is None, "Top-K categorical KL does not use sampled logprob-gap clipping")
    require(type(top_k) is int and 1 < top_k <= 4096, "Expected a configurable teacher top-K in [2,4096]")
    require(type(projection_chunk_size) is int and 0 < projection_chunk_size <= 1024, "Invalid projection chunk size")
    require(mode in ("train", "smoke") and 1 <= steps <= 50 and (mode != "smoke" or steps <= 2), "Invalid bounded run mode")
    require(source["harness"] == "append-opencode-tools-v1" and source["causal_prefix"], "Not an append harness plan")
    require(Path(run_root).is_absolute() and Path(checkpoint_root).is_absolute(), "Absolute distinct run roots required")
    require(run_root != source["run_root"] and checkpoint_root != source["checkpoint_root"], "Old run must remain untouched")
    for new, old_path in ((run_root, source["run_root"]), (checkpoint_root, source["checkpoint_root"])):
        a, b = Path(new).resolve(), Path(old_path).resolve()
        require(not a.is_relative_to(b) and not b.is_relative_to(a), "New run must not overlap an old run tree")
    require(not Path(run_root).exists() and not Path(checkpoint_root).exists(), "New run/checkpoint roots already exist")
    p = copy.deepcopy(source)
    for key in ("resume", "direct_start_authorization", "opsd"):
        p.pop(key, None)
    old = p["run_root"]
    p.update(run_root=run_root, checkpoint_root=checkpoint_root, mode=mode, requested_steps=steps)
    p["engines"] = [{**e, "trace_root": str(Path(run_root) / Path(e["trace_root"]).relative_to(old))} for e in p["engines"]]
    # Each future run gets its own sandbox app; no effect until explicitly launched.
    p["sandbox_app"] = "answer-opsd-" + Path(run_root).name
    p["opsd"] = {"format": FORMAT, "loss": OBJECTIVES[direction], "kl_direction": direction, "teacher_mode": "fixed_initial",
        "teacher_model_path": p["model_path"], "private_answers_path": private_answers_path,
        "private_answers_sha256": private_answers_sha256, "max_hint_tokens": max_hint_tokens,
        "teacher_max_context": p["contract"]["max_context"] + max_hint_tokens,
        "gap_clip": None, "top_k": top_k, "topk_tail": True, "projection_chunk_size": projection_chunk_size,
        "action_importance_weighting": False,
        "outcome_loss_weight": 0.0, "recipe_sha256": recipe_source_sha256(),
        "runtime_profile": runtime_profile}
    from .turn_demonstrations import MODE, defaults, validate_config
    require(privileged_information in ("answers", MODE), "Unknown privileged information")
    if privileged_information == MODE:
        require(private_answers_path is None and private_answers_sha256 is None, "Demo-only mode must not carry private answers")
        p["opsd"].pop("private_answers_path")
        p["opsd"].pop("private_answers_sha256")
        p["opsd"].update(privileged_information=MODE, hint_placement="after_original_turn_context",
                         demonstrator=copy.deepcopy(defaults() if demonstrator is None else demonstrator))
        from .demonstration_gate import FORMAT as ADMISSION_FORMAT
        p["opsd"]["demonstration_admission"] = ADMISSION_FORMAT
        validate_config(p["opsd"]["demonstrator"])
        require(p["opsd"]["demonstrator"]["teacher_dispatch_tokens"] >= 4 * p["opsd"]["teacher_max_context"],
                "Teacher dispatch must fit four unchanged maximum-context rows")
    else:
        require(private_answers_path and private_answers_sha256 and demonstrator is None,
                "Answer mode requires the original private catalog only")
    require(runtime_profile in ("source", "frozen_v15"), "Unknown runtime profile")
    if runtime_profile == "frozen_v15":
        from .frozen_runtime import EXTENSIONS
        prior = source.get("resume", {})
        require(all(prior.get(k) == v for k, v in EXTENSIONS.items()) and prior.get("adapter_sha256"),
                "Frozen profile requires the actual deployed source run's adapter identities")
        p["opsd"]["runtime_extensions"] = {k: prior[k] for k in (*EXTENSIONS, "adapter_sha256")}
    return p


def training_command(base_argv, plan_path, plan, *, resume=""):
    from .topk import LOSS_NAME
    from opencode_rl.train_launch import training_command as native_command
    if resume:
        check_resume(resume, plan)
    argv = native_command(base_argv, plan_path, plan, steps=plan["requested_steps"], resume=resume)
    original = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in base_argv[3:] if "=" in x}
    require(original.get("trainer.train_batch_size") == original.get("trainer.policy_mini_batch_size") == "16",
            "Supply the actual prior batch16 command, not guessed optimizer defaults")
    values = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in argv[3:] if "=" in x}
    values.update({"generator.n_samples_per_prompt": "1", "trainer.algorithm.advantage_estimator": "answer_opsd",
        "trainer.algorithm.zero_variance_filter": "false", "trainer.algorithm.advantage_batch_normalize": "false",
        "trainer.algorithm.policy_loss_type": LOSS_NAME, "trainer.algorithm.use_kl_in_reward": "true",
        "trainer.algorithm.use_kl_loss": "false", "trainer.algorithm.kl_loss_coef": "0.0",
        "trainer.algorithm.use_entropy_loss": "false", "trainer.update_ref_every_epoch": "false",
        "trainer.ref.model.path": plan["opsd"]["teacher_model_path"],
        "trainer.policy.model.path": plan["model_path"],
        "trainer.ref.language_model_only": "false", "trainer.policy.language_model_only": "false",
        "trainer.fully_async.enabled": "true", "trainer.fully_async.sample_full_batch": "true",
        "trainer.fully_async.group_filter": "validity", "trainer.placement.colocate_policy_ref": "true",
        "trainer.placement.colocate_all": "false",
        "generator.sampling_params.temperature": "1.0", "generator.sampling_params.top_p": "1.0",
        "generator.sampling_params.top_k": "-1",
        "generator.sampling_params.max_generate_length": str(plan["contract"]["max_output"]),
        "trainer.run_name": Path(plan["run_root"]).name})
    # LR, optimizer, task batch, staleness, token budget and GPU scheduling inherit
    # the actual baseline. IS knobs remain recorded but are NOT applied to
    # categorical KL (the sampled action is already marginalized at each prefix).
    return [argv[0], "-m", MODULE, *[f"{k}={v}" for k, v in values.items()]]
