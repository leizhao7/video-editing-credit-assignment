"""Exact per-request training rows, grouped by original trajectory for GRPO/PRM."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .contract import ParityError, digest
from .records import load_turn, turn_from_events


@dataclass(frozen=True)
class AttemptResult:
    task_id: str
    repetition_id: int
    trajectory_id: str
    attempt_id: str
    session_id: str
    request_ids: tuple[str, ...]
    outcome_reward: float
    process_scores: tuple[float, ...]
    process_valid: tuple[bool, ...]
    runtime_completed: bool
    infrastructure_error: str | None = None


@dataclass(frozen=True)
class PreparedTrajectory:
    """One validated, sealed native snapshot, never retokenized or reprocessed."""
    trace_root: str
    attempt_id: str
    contract_sha256: str
    snapshot: dict
    snapshot_sha256: str

    def checked(self, store, attempt_id, contract):
        if (self.trace_root != str(store.root.resolve()) or self.attempt_id != attempt_id or
                self.contract_sha256 != contract.sha256 or digest(self.snapshot) != self.snapshot_sha256):
            raise ParityError('Prepared native trajectory changed or belongs to another input')
        return self.snapshot


def prepare_trajectory(store, attempt_id, contract):
    from .context_recovery import primary_inventory
    turns = []
    def complete(row, events, seal):
        turns.append(turn_from_events(store, row['request_id'], contract, events, seal))
    recorded, rejected = primary_inventory(store, attempt_id, contract, on_complete=complete)
    snapshot = {'recorded': recorded, 'rejected': rejected, 'turns': turns}
    return PreparedTrajectory(str(store.root.resolve()), attempt_id, contract.sha256,
                              snapshot, digest(snapshot))


def assemble_images(store, turn, contract):
    import torch
    pixels, grids = [], []
    occupied = set()
    ids = turn["input"]["prompt_token_ids"]
    for feature in turn["input"]["features"]:
        if set(feature["data"]) != {"pixel_values", "image_grid_thw"}:
            raise ParityError("Unexpected image input fields")
        values = torch.as_tensor(store.load_tensor(feature["data"]["pixel_values"]))
        grid = torch.as_tensor(store.load_tensor(feature["data"]["image_grid_thw"]))
        if grid.dtype not in (torch.int32, torch.int64) or grid.numel() != 3 or bool((grid <= 0).any()):
            raise ParityError("Invalid image grid")
        grid = grid.reshape(1, 3)
        if values.ndim != 2 or values.shape[1] != contract.vision_patch_dim or not bool(torch.isfinite(values).all()):
            raise ParityError("Invalid image patch tensor")
        patches = int(grid.prod())
        if len(values) != patches or patches % contract.spatial_merge_size**2:
            raise ParityError("Image pixels/grid mismatch")
        expected_length = patches // contract.spatial_merge_size**2
        start, count = feature["offset"], feature["length"]
        if count != expected_length or start < 0 or start + count > len(ids):
            raise ParityError("Image token positions/grid mismatch")
        span = set(range(start, start + count))
        if occupied & span or any(ids[pos] != contract.image_token_id for pos in span):
            raise ParityError("Image placeholders do not match actual model inputs")
        occupied |= span
        pixels.append(values)
        grids.append(grid)
    # A literal/generated image token is text unless EngineCore attached an
    # actual multimodal feature at that position. Never infer media from IDs.
    return (torch.cat(pixels, dim=0) if pixels else torch.empty((0, contract.vision_patch_dim), dtype=torch.float32),
            torch.cat(grids, dim=0) if grids else torch.empty((0, 3), dtype=torch.long))


def modality_ids(turn):
    types = [0] * (len(turn["input"]["prompt_token_ids"]) + len(turn["response_ids"]))
    for feature in turn["input"]["features"]:
        start, length = feature["offset"], feature["length"]
        types[start:start + length] = [1] * length
    return types


def attach_modality_ids(data, generated):
    import torch
    types = torch.zeros_like(data["sequences"], dtype=torch.long)
    if "mm_token_type_ids" not in generated:
        raise ParityError("Actual multimodal positions are missing")
    for i, values in enumerate(generated["mm_token_type_ids"]):
        if len(values) != len(generated["prompt_token_ids"][i]) + len(generated["response_ids"][i]):
            raise ParityError("Multimodal positions have the wrong sequence length")
        types[i, -len(values):] = types.new_tensor(values)
    # SkyRL pads the DP batch by cloning row 0 (including its image tensors),
    # not by inserting empty rows. Its zero-loss clones need row 0's media map.
    for i in range(len(generated['prompt_token_ids']), len(types)):
        if bool(data['loss_mask'][i].any()) or not torch.equal(data['sequences'][i], data['sequences'][0]):
            raise ParityError('Unreviewed distributed padding row')
        types[i] = types[0]
    data["mm_token_type_ids"] = types


def build_generator_output(store, attempts, contract, *, trajectory_id_factory=None, _pack_prefix=False,
                           prepared=None):
    from .contract import Contract
    if not isinstance(contract, Contract):
        raise ParityError('Evaluation-only model has not passed the training numerical gate')
    if contract.top_p != 1.0 or contract.top_k not in (0, -1):
        raise ParityError('Training raw policy probabilities requires the unchanged top_p=1/top_k=off recipe')
    if trajectory_id_factory is None:
        from skyrl.train.generators.base import TrajectoryID
        trajectory_id_factory = TrajectoryID
    if not attempts:
        raise ParityError("No completed trajectories")
    if prepared is not None and (not isinstance(prepared, PreparedTrajectory) or len(attempts) != 1):
        raise ParityError('A prepared trajectory must bind exactly one original attempt')
    names = ("prompt_token_ids", "response_ids", "rewards", "loss_masks", "stop_reasons",
             "rollout_logprobs", "trajectory_ids", "is_last_step", "pixel_values", "image_grid_thw", "mm_token_type_ids", "ocrl_rows")
    result = {key: [] for key in names}
    seen_requests, seen_trajectories, seen_samples = set(), set(), set()
    for attempt in attempts:
        if not attempt.runtime_completed or attempt.infrastructure_error:
            raise ParityError("Infrastructure attempt must be archived/retried, not labeled model reward zero")
        if not math.isfinite(attempt.outcome_reward) or not 0 <= attempt.outcome_reward <= 1:
            raise ParityError("Invalid outcome reward")
        n = len(attempt.request_ids)
        if not n or len(attempt.process_scores) != n or len(attempt.process_valid) != n:
            raise ParityError("Turn reward metadata is incomplete")
        sample_id = (attempt.task_id, attempt.repetition_id)
        if attempt.trajectory_id in seen_trajectories or sample_id in seen_samples:
            raise ParityError("Duplicate canonical trajectory/repetition")
        seen_trajectories.add(attempt.trajectory_id)
        seen_samples.add(sample_id)
        from .context_recovery import primary_inventory
        snapshot = prepared.checked(store, attempt.attempt_id, contract) if prepared is not None else None
        if snapshot is None:
            recorded, rejected = primary_inventory(store, attempt.attempt_id, contract)
        else:
            recorded, rejected = snapshot['recorded'], snapshot['rejected']
        if [r['request_id'] for r in recorded] != list(attempt.request_ids):
            raise ParityError("Attempt has missing, unfinished or unaccounted-for policy requests")
        calls, turns, rows = set(), [], []
        for t, request_id in enumerate(attempt.request_ids):
            if request_id in seen_requests:
                raise ParityError("Request reused in multiple training samples")
            seen_requests.add(request_id)
            turn = load_turn(store, request_id, contract) if snapshot is None else snapshot['turns'][t]
            if turn['request_id'] != request_id:
                raise ParityError('Prepared native request order changed')
            sampling = turn['input']['sampling']
            for key, neutral in (('presence_penalty', 0.), ('frequency_penalty', 0.), ('repetition_penalty', 1.),
                                 ('min_p', 0.), ('min_tokens', 0)):
                if sampling.get(key, neutral) != neutral:
                    raise ParityError(f'Unreviewed transformed sampling probability: {key}')
            for key in ('logit_bias', 'allowed_token_ids', 'bad_words_token_ids', 'structured_outputs', 'logits_processors'):
                if sampling.get(key):
                    raise ParityError(f'Diagnostic/constrained sampling cannot enter raw-policy RL: {key}')
            meta = turn["metadata"]
            if (meta["attempt_id"], meta["session_id"], meta["turn"]) != (
                    attempt.attempt_id, attempt.session_id, recorded[t]['metadata']['turn']):
                raise ParityError("Attempt/session/turn mismatch")
            if meta["logical_call_id"] in calls:
                raise ParityError("Ambiguous retry: same OpenCode call has multiple accepted responses")
            calls.add(meta["logical_call_id"])
            score, valid = attempt.process_scores[t], attempt.process_valid[t]
            if not math.isfinite(score) or not -1 <= score <= 1 or type(valid) is not bool:
                raise ParityError("Invalid process score")
            if _pack_prefix and (score != 0 or valid):
                raise ParityError('Causal packets require the explicit outcome-only recipe')
            turns.append(turn)
            rows.append({"task_id": attempt.task_id, "trajectory_id": attempt.trajectory_id, "turn": t,
                       "native_turn": meta['turn'], "context_rejected_calls": len(rejected),
                       "outcome": attempt.outcome_reward, "process": score, "process_valid": valid,
                       "request_id": request_id, "input_sha256": turn["input"]["input_sha256"],
                       "weight_version": turn["input"]["weight_version"], "response_length": len(turn["response_ids"]),
                       "contract_sha256": contract.sha256})
        if _pack_prefix:
            from .causal_prefix import plan_chains
            packets = plan_chains(turns)
            by_id = {turn['request_id']: (turn, row) for turn, row in zip(turns, rows)}
        else:
            packets = [None] * len(turns)
        for index, packet in enumerate(packets):
            if packet is None:
                turn, row = turns[index], rows[index]
                prompt, response = turn['input']['prompt_token_ids'], turn['response_ids']
                mask, logprobs, mm = turn['loss_mask'], turn['rollout_logprobs'], modality_ids(turn)
            else:
                turn = by_id[packet['terminal_request_id']][0]
                originals = [by_id[span['request_id']][1] for span in packet['spans']]
                row = {**originals[0], 'turn': index, 'original_rows': originals,
                    'causal_spans': packet['spans'], 'original_turn_count': len(originals),
                    'packed_response_length': len(packet['response_ids']), 'causal_format': packet['format']}
                prompt, response = packet['prompt_token_ids'], packet['response_ids']
                mask, logprobs, mm = packet['loss_mask'], packet['rollout_logprobs'], packet['mm_token_type_ids']
            # Only terminal media are materialized/transported for a proven
            # chain. load_turn and verify_packet already checked every input.
            pixels, grid = assemble_images(store, turn, contract)
            last = index == len(packets) - 1
            values = (prompt, response, attempt.outcome_reward if last else 0., mask, turn['finish_reason'],
                      logprobs, trajectory_id_factory(instance_id=attempt.task_id, repetition_id=attempt.repetition_id),
                      last, pixels, grid, mm, row)
            for key, value in zip(names, values):
                result[key].append(value)
    result["rollout_metrics"] = {"ocrl/trajectories": len(attempts), "ocrl/policy_turns": len(seen_requests),
        'ocrl/training_rows': len(result['response_ids']),
        'ocrl/original_response_tokens': sum(row['response_length'] for row in original_request_rows(result['ocrl_rows'])),
        'ocrl/context_gap_tokens': sum(len(ids) for ids in result['response_ids']) -
            sum(row['response_length'] for row in original_request_rows(result['ocrl_rows']))}
    return result


def build_packed_generator_output(store, attempts, contract, *, trajectory_id_factory=None, prepared=None):
    """Opt-in exact causal sharing; never changes the native agent loop."""
    return build_generator_output(store, attempts, contract,
        trajectory_id_factory=trajectory_id_factory, _pack_prefix=True, prepared=prepared)


def original_request_rows(rows):
    return [original for row in rows for original in row.get('original_rows', [row])]


def compute_turn_advantages(rows, loss_mask, *, process_weight=0.5, epsilon=1e-6, norm_by_std=True):
    """Same sample-std GRPO/turn-local PRM as the old estimator, without re-tokenizing."""
    import torch
    if not math.isfinite(process_weight) or process_weight < 0 or len(rows) > len(loss_mask):
        raise ParityError("Invalid process weight/row count")
    if bool(loss_mask[len(rows):].any()):
        raise ParityError("Padding rows must carry no policy loss")
    outcome, process = torch.zeros_like(loss_mask, dtype=torch.float32), torch.zeros_like(loss_mask, dtype=torch.float32)
    tasks, trajectories = defaultdict(dict), defaultdict(list)
    for i, row in enumerate(rows):
        if 'causal_spans' in row and process_weight != 0:
            raise ParityError('Per-turn PRM has not been wired for causal packets')
        if not bool(loss_mask[i].any()):
            continue
        tid = row["trajectory_id"]
        if tid in tasks[row["task_id"]] and tasks[row["task_id"]][tid] != row["outcome"]:
            raise ParityError("Conflicting trajectory outcome")
        tasks[row["task_id"]][tid] = row["outcome"]
        trajectories[tid].append(i)

    def centered(values):
        tensor = outcome.new_tensor(values)
        if len(values) < 2:
            return torch.zeros_like(tensor)
        delta = tensor - tensor.mean()
        if not norm_by_std:
            return delta
        std = tensor.std(unbiased=True)
        return delta / (std + epsilon) if float(std) > epsilon else torch.zeros_like(tensor)

    for task, members in tasks.items():
        for tid, advantage in zip(members, centered(list(members.values()))):
            for i in trajectories[tid]:
                outcome[i] = advantage * loss_mask[i]
        by_turn = defaultdict(list)
        for tid in members:
            seen = set()
            for i in trajectories[tid]:
                row = rows[i]
                if row["turn"] in seen:
                    raise ParityError("Duplicate semantic turn in trajectory")
                seen.add(row["turn"])
                if row["process_valid"]:
                    by_turn[row["turn"]].append(i)
        for indices in by_turn.values():
            for i, advantage in zip(indices, centered([rows[j]["process"] for j in indices])):
                process[i] = process_weight * advantage * loss_mask[i]
    details = []
    for tid, indices in trajectories.items():
        oa = float(outcome[indices].abs().sum())
        pa = float(process[indices].abs().sum())
        details.append({"trajectory_id": tid, "outcome_adv_abs_sum": oa, "process_adv_abs_sum": pa,
                        "process_contribution_fraction": pa / (pa + oa) if pa + oa else 0.0,
                        "outcome_reward": rows[indices[0]]["outcome"]})
    return outcome + process, outcome, process, details


def verify_training_batch(data, generated):
    """Check the actual post-padding SkyRL tensors, not an earlier JSON copy."""
    import torch
    for i, prompt in enumerate(generated["prompt_token_ids"]):
        response = generated["response_ids"][i]
        active = data["attention_mask"][i].bool()
        count = len(prompt) + len(response)
        if not bool(active[-count:].all()) or bool(active[:-count].any()):
            raise ParityError("Attention mask is not the original contiguous context")
        if data["sequences"][i][active].tolist() != prompt + response:
            raise ParityError("Training input differs from recorded engine sequence")
        types = data["mm_token_type_ids"][i]
        if types[active].tolist() != generated["mm_token_type_ids"][i] or bool(types[~active].any()):
            raise ParityError("Training multimodal positions differ from EngineCore")
        row = generated.get('ocrl_rows', [{}] * len(generated['prompt_token_ids']))[i]
        if 'causal_spans' in row:
            from .prefix_transport import verify_generated_packet
            verify_generated_packet(generated, i)
        elif bool(types[-len(response):].any()):
            raise ParityError("Generated tokens must not be reinterpreted as input images")
        rmask = data["response_mask"][i].bool()
        if not bool(rmask[-len(response):].all()) or bool(rmask[:-len(response)].any()):
            raise ParityError("Response mask does not select the original completion")
        if data["loss_mask"][i][rmask].tolist() != generated["loss_masks"][i] or bool(data["loss_mask"][i][~rmask].any()):
            raise ParityError("Prompt/padding leakage into policy loss")
        actual_lp = data["rollout_logprobs"][i][rmask]
        if not torch.equal(actual_lp, actual_lp.new_tensor(generated["rollout_logprobs"][i])):
            raise ParityError("Training behavior probabilities are misaligned")
        for key in ("pixel_values", "image_grid_thw"):
            tensors = data[key].tensors if hasattr(data[key], "tensors") else data[key]
            if tensors[i].dtype != generated[key][i].dtype or not torch.equal(tensors[i], generated[key][i]):
                raise ParityError(f"Training {key} differs from engine input")
