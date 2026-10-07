"""Exact per-request training rows, grouped by original trajectory for GRPO/PRM."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .contract import ParityError, digest
from .records import load_turn, turn_from_events

# Independent local overlay. The original repository file is never modified.
OVERLAY_SOURCE_SHA256 = '44c03123a2d6881ff797c6f66a809e0f015ea422724307f86899a66b375ac460'


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


def prepare_trajectory(store, attempt_id, contract, *, deduplicate_tensors=False):
    from .context_recovery import primary_inventory
    turns = []
    # Append-only contexts repeat the same content-addressed image references
    # on every turn. Validate each exact typed reference once per sealed
    # snapshot, while retaining every turn/token/logprob/provenance check.
    # Store only proof keys, never GBs of mutable pixel tensors. The terminal
    # training packet still loads and validates its actual media as before.
    validated_tensors = set() if deduplicate_tensors else None
    def complete(row, events, seal):
        turns.append(turn_from_events(store, row['request_id'], contract, events, seal,
                                      _validated_tensors=validated_tensors))
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
                           prepared=None, enable_packed_process=False):
    from .contract import Contract
    if type(enable_packed_process) is not bool or (enable_packed_process and not _pack_prefix):
        raise ParityError('Packed process requires explicit boolean causal-packet opt-in')
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
            if _pack_prefix and not enable_packed_process and (score != 0 or valid):
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
                if enable_packed_process:
                    row['causal_process_enabled'] = True
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


def build_packed_generator_output(store, attempts, contract, *, trajectory_id_factory=None, prepared=None,
                                  enable_packed_process=False):
    """Opt-in exact causal sharing; never changes the native agent loop."""
    return build_generator_output(store, attempts, contract,
        trajectory_id_factory=trajectory_id_factory, _pack_prefix=True, prepared=prepared,
        enable_packed_process=enable_packed_process)


def original_request_rows(rows):
    return [original for row in rows for original in row.get('original_rows', [row])]


def compute_turn_advantages(rows, loss_mask, *, process_weight=0.5, epsilon=1e-6, norm_by_std=True,
                            enable_packed_process=False, response_mask=None):
    """Same sample-std GRPO/turn-local PRM as the old estimator, without re-tokenizing."""
    import torch
    if type(enable_packed_process) is not bool:
        raise ParityError('Packed process opt-in must be boolean')
    if enable_packed_process:
        return _compute_packed_process_advantages(rows, loss_mask, response_mask,
            process_weight=process_weight, epsilon=epsilon, norm_by_std=norm_by_std)
    if any(row.get('causal_process_enabled') is True for row in rows):
        raise ParityError('Process packet requires explicit estimator opt-in')
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


def _packed_process_rows(rows):
    """Validate one causal packet per original trajectory without re-tokenizing."""
    seen_trajectories, seen_requests = set(), set()
    for row in rows:
        if row.get('causal_process_enabled') is not True or row.get('causal_format') != 'ocrl-causal-prefix-v1':
            raise ParityError('Packed PRM requires explicit per-packet opt-in and causal provenance')
        tid = row.get('trajectory_id')
        if tid in seen_trajectories:
            raise ParityError('Packed PRM requires one packet per complete original trajectory')
        seen_trajectories.add(tid)
        originals, spans = row.get('original_rows'), row.get('causal_spans')
        length = row.get('packed_response_length')
        if (not isinstance(originals, list) or not originals or not isinstance(spans, list) or
                len(originals) != len(spans) or row.get('original_turn_count') != len(originals) or
                type(length) is not int or length <= 0):
            raise ParityError('Incomplete original-turn metadata for packed PRM')
        if type(row.get('outcome')) not in (int, float) or not math.isfinite(row['outcome']) or not 0 <= row['outcome'] <= 1:
            raise ParityError('Invalid packed trajectory outcome')
        previous, ordinals = 0, set()
        for original, span in zip(originals, spans):
            lo, hi = span.get('response_start'), span.get('response_end')
            original_mask = span.get('original_loss_mask')
            if (type(lo) is not int or type(hi) is not int or not previous <= lo < hi <= length or
                    not isinstance(original_mask, list) or len(original_mask) != hi - lo or
                    any(value not in (0, 1) for value in original_mask) or original.get('response_length') != hi-lo):
                raise ParityError('Invalid original action span for packed PRM')
            if any(original.get(key) != row.get(key) for key in ('task_id', 'trajectory_id', 'outcome')):
                raise ParityError('Packed process reward crossed trajectory/outcome boundary')
            for key in ('request_id', 'input_sha256', 'weight_version', 'native_turn'):
                if key not in original or span.get(key) != original[key]:
                    raise ParityError('Packed process span changed original request identity')
            ordinal = original.get('turn')
            if type(ordinal) is not int or ordinal < 0 or ordinal in ordinals:
                raise ParityError('Duplicate or invalid original ordinal process turn')
            ordinals.add(ordinal)
            if original['request_id'] in seen_requests:
                raise ParityError('Original request repeated across process packets')
            seen_requests.add(original['request_id'])
            score, valid = original.get('process'), original.get('process_valid')
            if type(score) not in (float, int) or not math.isfinite(score) or not -1 <= score <= 1 or type(valid) is not bool:
                raise ParityError('Invalid original process score/validity')
            previous = hi
        if spans[0]['response_start'] != 0 or previous != length:
            raise ParityError('Unaccounted leading/trailing packed response')


def _packed_mask_activity(row, mask):
    """Check gaps/original masks and return per-turn actual policy activity."""
    if len(mask) != row['packed_response_length']:
        raise ParityError('Packed mask length differs from response provenance')
    if hasattr(mask, 'tolist'):
        mask = mask.tolist()
    if any(value not in (0, 1) for value in mask):
        raise ParityError('Policy loss mask must be binary')
    activity, previous = [], 0
    for span in row['causal_spans']:
        lo, hi = span['response_start'], span['response_end']
        if any(mask[previous:lo]) or any(value and not original for value, original in zip(mask[lo:hi], span['original_loss_mask'])):
            raise ParityError('Prompt/tool gap or masked action token entered packed policy loss')
        activity.append(any(mask[lo:hi]))
        previous = hi
    return activity


def _packed_signals(rows, activity, *, process_weight, epsilon, norm_by_std, diagnostics=None):
    """Same sample-std/ordinal algorithm as before, deduplicated by trajectory."""
    import torch
    if not math.isfinite(process_weight) or process_weight < 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ParityError('Invalid process coefficient/normalization epsilon')
    tasks, outcome_z, process_z = defaultdict(list), {}, {}
    for i, row in enumerate(rows):
        if any(activity[i]):
            tasks[row['task_id']].append(i)

    def centered(values):
        tensor = torch.tensor(values, dtype=torch.float32)
        if len(values) < 2:
            return torch.zeros_like(tensor)
        delta = tensor - tensor.mean()
        if not norm_by_std:
            return delta
        std = tensor.std(unbiased=True)
        return delta / (std + epsilon) if float(std) > epsilon else torch.zeros_like(tensor)

    for indices in tasks.values():
        for i, value in zip(indices, centered([rows[i]['outcome'] for i in indices])):
            outcome_z[i] = float(value)
        by_turn = defaultdict(list)
        for i in indices:
            for j, original in enumerate(rows[i]['original_rows']):
                if activity[i][j] and original['process_valid']:
                    by_turn[original['turn']].append((i, j))
        for members in by_turn.values():
            values = [rows[i]['original_rows'][j]['process'] for i, j in members]
            normalized = centered(values)
            if diagnostics is not None:
                raw = torch.tensor(values, dtype=torch.float32)
                raw_mean = float(raw.mean())
                raw_std = float(raw.std(unbiased=True)) if len(values) > 1 else 0.
            for index, (key, value) in enumerate(zip(members, normalized)):
                process_z[key] = float(process_weight) * float(value)
                if diagnostics is not None:
                    diagnostics[key] = {'process_group_size': len(values),
                        'process_group_mean': raw_mean, 'process_group_std': raw_std,
                        'process_group_range': max(values)-min(values),
                        'process_centered_reward': float(raw[index]-raw.mean()),
                        'process_advantage_unweighted': float(value),
                        'process_advantage_weighted': process_z[key],
                        'process_std_normalization': bool(norm_by_std),
                        'process_normalization_epsilon': epsilon,
                        'process_group_below_std_threshold': raw_std <= epsilon}
    return outcome_z, process_z


def causal_process_signal_groups(rows, loss_masks, *, process_weight=0.5, epsilon=1e-6,
                                 norm_by_std=True, tolerance=0.0):
    """Scalar admission check, never apply token spans to an N-by-1 proxy mask."""
    if len(rows) != len(loss_masks) or not math.isfinite(tolerance) or tolerance < 0:
        raise ParityError('Invalid packed signal admission inputs')
    _packed_process_rows(rows)
    activity = [_packed_mask_activity(row, mask) for row, mask in zip(rows, loss_masks)]
    outcome, process = _packed_signals(rows, activity, process_weight=process_weight,
        epsilon=epsilon, norm_by_std=norm_by_std)
    return {row['task_id'] for i, row in enumerate(rows)
            if any(live and abs(outcome.get(i, 0.) + process.get((i, j), 0.)) > tolerance
                   for j, live in enumerate(activity[i]))}


def _compute_packed_process_advantages(rows, loss_mask, response_mask, *, process_weight,
                                       epsilon, norm_by_std):
    import torch
    _packed_process_rows(rows)
    if (loss_mask.ndim != 2 or len(rows) > len(loss_mask) or response_mask is None or
            response_mask.shape != loss_mask.shape or bool(loss_mask[len(rows):].any())):
        raise ParityError('Packed PRM needs aligned response/loss masks and zero-loss DP padding')
    outcome = torch.zeros_like(loss_mask, dtype=torch.float32)
    process = torch.zeros_like(outcome)
    offsets, activity = [], []
    for i, row in enumerate(rows):
        positions = torch.nonzero(response_mask[i].bool(), as_tuple=False).flatten()
        length = row['packed_response_length']
        if len(positions) != length or int(positions[-1]) - int(positions[0]) + 1 != length:
            raise ParityError('Response mask does not identify the contiguous original packed response')
        offset = int(positions[0])
        if bool(loss_mask[i][~response_mask[i].bool()].any()):
            raise ParityError('Prompt or left-padding entered packed policy loss')
        offsets.append(offset)
        activity.append(_packed_mask_activity(row, loss_mask[i, offset:offset+length]))
    normalization = {}
    outcome_z, process_z = _packed_signals(rows, activity, process_weight=process_weight,
        epsilon=epsilon, norm_by_std=norm_by_std, diagnostics=normalization)
    details = []
    for i, row in enumerate(rows):
        if i not in outcome_z:
            continue
        outcome[i] = outcome_z[i] * loss_mask[i]
        turns = []
        for j, (original, span) in enumerate(zip(row['original_rows'], row['causal_spans'])):
            lo, hi = offsets[i] + span['response_start'], offsets[i] + span['response_end']
            process[i, lo:hi] = process_z.get((i, j), 0.) * loss_mask[i, lo:hi]
            masses = _advantage_masses(outcome[i, lo:hi], process[i, lo:hi], loss_mask[i, lo:hi])
            tokens = int(loss_mask[i, lo:hi].sum())
            norm = normalization.get((i, j), {'process_group_size': 0,
                'process_group_mean': 0., 'process_group_std': 0., 'process_group_range': 0.,
                'process_centered_reward': 0., 'process_advantage_unweighted': 0.,
                'process_advantage_weighted': 0., 'process_std_normalization': bool(norm_by_std),
                'process_normalization_epsilon': epsilon, 'process_group_below_std_threshold': True})
            turns.append({'request_id': original['request_id'], 'turn': original['turn'],
                'native_turn': original['native_turn'], 'process_score': original['process'],
                'process_valid': original['process_valid'], 'process_eligible': bool(activity[i][j] and original['process_valid']),
                'outcome_advantage': outcome_z[i],
                'training_tokens': tokens, **masses, **norm,
                'process_unweighted_adv_abs_sum': abs(norm['process_advantage_unweighted'])*tokens,
                'process_centered_raw_abs_sum': abs(norm['process_centered_reward'])*tokens})
        oa, pa = float(outcome[i].abs().sum()), float(process[i].abs().sum())
        valid = [r['process'] for r in row['original_rows'] if r['process_valid']]
        count = len(row['original_rows'])
        valid_tensor = torch.tensor(valid, dtype=torch.float64)
        unweighted_mass = sum(t['process_unweighted_adv_abs_sum'] for t in turns)
        raw_mass = sum(t['process_centered_raw_abs_sum'] for t in turns)
        details.append({'trajectory_id': row['trajectory_id'],
            **_advantage_masses(outcome[i], process[i], loss_mask[i]),
            'outcome_contribution_fraction': oa/(pa+oa) if pa+oa else 0.0,
            'outcome_reward': row['outcome'], 'process_reward_valid_mean': sum(valid)/len(valid) if valid else 0.0,
            'outcome_advantage': outcome_z[i],
            'process_reward_valid_std': float(valid_tensor.std(unbiased=True)) if len(valid) > 1 else 0.,
            'process_reward_valid_min': min(valid, default=0.),
            'process_reward_valid_max': max(valid, default=0.),
            'process_reward_valid_range': max(valid)-min(valid) if valid else 0.,
            'process_reward_mean': sum(valid)/count, 'process_valid_turns': len(valid),
            'process_fallback_turns': count-len(valid),
            'process_eligible_turns': sum(r['process_valid'] and live for r, live in zip(row['original_rows'], activity[i])),
            'process_coefficient': process_weight,
            'process_unweighted_adv_abs_sum': unweighted_mass,
            'process_centered_raw_abs_sum': raw_mass,
            'process_normalization_mass_gain': unweighted_mass/raw_mass if raw_mass else 0.,
            'process_coefficient_observed_on_mass': pa/unweighted_mass if unweighted_mass else 0.,
            'process_turn_details': turns,
            'contribution_definition': 'absolute pre-PPO advantage mass on actual generated loss-mask tokens; not gradient share'})
    return outcome + process, outcome, process, details


def _advantage_masses(outcome, process, mask):
    """Pre-PPO token advantage geometry; never claims an actual gradient share."""
    import torch
    oa, pa = outcome.abs(), process.abs()
    both = (oa > 0) & (pa > 0) & mask.bool()
    same_sign = (outcome > 0) == (process > 0)
    aligned, conflicting = both & same_sign, both & ~same_sign
    overlap = torch.minimum(oa, pa)
    outcome_mass, process_mass = float(oa.sum()), float(pa.sum())
    component_mass = outcome_mass + process_mass
    combined_mass = float((outcome+process).abs().sum())
    # Exact algebra: each opposing pair cancels twice min(|AO|, |AP|).
    # Use the direct overlap, avoiding subtraction of almost equal large sums.
    aligned_mass, conflict_mass = float(overlap[aligned].sum()), float(overlap[conflicting].sum())
    cancelled_mass = 2.*conflict_mass
    return {'outcome_adv_abs_sum': outcome_mass, 'process_adv_abs_sum': process_mass,
        'process_contribution_fraction': process_mass/component_mass if component_mass else 0.,
        'component_adv_abs_sum': component_mass, 'combined_adv_abs_sum': combined_mass,
        'cancelled_adv_abs_sum': cancelled_mass,
        'component_cancellation_fraction': cancelled_mass/component_mass if component_mass else 0.,
        'aligned_overlap_adv_abs_sum': aligned_mass, 'conflict_overlap_adv_abs_sum': conflict_mass,
        'aligned_token_count': int(aligned.sum()), 'conflicting_token_count': int(conflicting.sum()),
        'both_nonzero_token_count': int(both.sum()), 'policy_token_count': int(mask.sum()),
        'prm_only': outcome_mass == 0. and process_mass > 0.}


def packed_process_health_metrics(rows, details, *, process_weight):
    """Flat per-step scalars, plus stages; does not feed back into training.

    Reward statistics use original valid judgements (including masked rows).
    Ordinal and contribution statistics use actual live generated policy tokens.
    Zero-variance bucket rate has only size>=2 buckets in its denominator;
    singleton buckets are reported separately. None of these are gradient shares.
    """
    import torch

    def statistics(values):
        if not values:
            return {'count': 0, 'mean': 0., 'std': 0., 'min': 0., 'max': 0., 'range': 0.}
        tensor = torch.tensor(values, dtype=torch.float64)
        return {'count': len(values), 'mean': float(tensor.mean()),
            'std': float(tensor.std(unbiased=True)) if len(values) > 1 else 0.,
            'min': min(values), 'max': max(values), 'range': max(values)-min(values)}

    def aggregate(sample_rows, sample_details, suffix=''):
        result = {}
        reward_prefix = 'prm/reward/' + suffix
        contribution_prefix = 'prm/contribution/' + suffix
        originals = original_request_rows(sample_rows)
        valid = [row['process'] for row in originals if row['process_valid']]
        for key, value in statistics(valid).items():
            result[reward_prefix+'valid_'+key] = value
        result[reward_prefix+'total_turn_count'] = len(originals)
        result[reward_prefix+'fallback_turn_count'] = len(originals)-len(valid)
        eligible = [(detail, turn) for detail in sample_details for turn in detail['process_turn_details']
                    if turn['process_eligible']]
        for key, value in statistics([turn['process_score'] for _, turn in eligible]).items():
            result[reward_prefix+'eligible_'+key] = value
        count = len(eligible)
        result[reward_prefix+'eligible_zero_adv_fraction'] = sum(
            turn['process_advantage_weighted'] == 0. for _, turn in eligible)/max(1, count)
        result[reward_prefix+'eligible_zero_unweighted_adv_fraction'] = sum(
            turn['process_advantage_unweighted'] == 0. for _, turn in eligible)/max(1, count)
        buckets = {}
        for detail, turn in eligible:
            buckets[(detail['task_id'], turn['turn'])] = turn
        comparable = [turn for turn in buckets.values() if turn['process_group_size'] >= 2]
        variable = [turn for turn in comparable if turn['process_group_std'] > 0.]
        result[reward_prefix+'ordinal_bucket_count'] = len(buckets)
        result[reward_prefix+'ordinal_comparable_bucket_count'] = len(comparable)
        result[reward_prefix+'ordinal_singleton_bucket_fraction'] = (len(buckets)-len(comparable))/max(1, len(buckets))
        result[reward_prefix+'ordinal_comparable_zero_variance_fraction'] = sum(
            turn['process_group_std'] == 0. for turn in comparable)/max(1, len(comparable))
        result[reward_prefix+'ordinal_comparable_below_std_threshold_fraction'] = sum(
            turn['process_group_below_std_threshold'] for turn in comparable)/max(1, len(comparable))
        result[reward_prefix+'ordinal_zero_variance_eligible_turn_fraction'] = sum(
            turn['process_group_size'] >= 2 and turn['process_group_std'] == 0.
            for _, turn in eligible)/max(1, count)
        result[reward_prefix+'ordinal_variable_std_min'] = min(
            (turn['process_group_std'] for turn in variable), default=0.)
        result[reward_prefix+'ordinal_normalization_gain_max'] = max((
            1./(turn['process_group_std']+turn['process_normalization_epsilon'])
            if turn['process_std_normalization'] else 1. for turn in variable
            if not turn['process_std_normalization'] or not turn['process_group_below_std_threshold']), default=0.)
        summed = ('outcome_adv_abs_sum', 'process_adv_abs_sum', 'component_adv_abs_sum',
            'combined_adv_abs_sum', 'cancelled_adv_abs_sum', 'aligned_overlap_adv_abs_sum',
            'conflict_overlap_adv_abs_sum', 'aligned_token_count', 'conflicting_token_count',
            'both_nonzero_token_count', 'policy_token_count', 'process_unweighted_adv_abs_sum',
            'process_centered_raw_abs_sum')
        totals = {key: sum(detail[key] for detail in sample_details) for key in summed}
        for key, value in totals.items():
            result[contribution_prefix+key] = value
        oa, pa = totals['outcome_adv_abs_sum'], totals['process_adv_abs_sum']
        unweighted, raw = totals['process_unweighted_adv_abs_sum'], totals['process_centered_raw_abs_sum']
        aligned, conflict = totals['aligned_overlap_adv_abs_sum'], totals['conflict_overlap_adv_abs_sum']
        result.update({contribution_prefix+'coefficient_configured': process_weight,
            contribution_prefix+'process_weighted_fraction': pa/(oa+pa) if oa+pa else 0.,
            contribution_prefix+'process_unweighted_fraction': unweighted/(oa+unweighted) if oa+unweighted else 0.,
            contribution_prefix+'coefficient_observed_on_mass': pa/unweighted if unweighted else 0.,
            contribution_prefix+'normalization_mass_gain': unweighted/raw if raw else 0.,
            contribution_prefix+'active_trajectory_count': len(sample_details),
            contribution_prefix+'prm_only_trajectory_count': sum(d['prm_only'] for d in sample_details),
            contribution_prefix+'prm_only_trajectory_fraction': sum(d['prm_only'] for d in sample_details)/max(1, len(sample_details)),
            contribution_prefix+'conflict_overlap_fraction': conflict/(aligned+conflict) if aligned+conflict else 0.,
            contribution_prefix+'conflicting_token_fraction_of_both': totals['conflicting_token_count']/max(1, totals['both_nonzero_token_count']),
            contribution_prefix+'component_cancellation_fraction': totals['cancelled_adv_abs_sum']/(oa+pa) if oa+pa else 0.,
            contribution_prefix+'is_gradient_share': 0})
        return result

    metrics = aggregate(rows, details)
    for stage in sorted({row.get('stage', 'unknown') for row in rows}):
        metrics.update(aggregate([row for row in rows if row.get('stage', 'unknown') == stage],
            [detail for detail in details if detail.get('stage', 'unknown') == stage], suffix=f'stage/{stage}/'))
    return metrics


def verify_training_batch(data, generated, *, enable_packed_process=False):
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
            verify_generated_packet(generated, i, enable_packed_process=enable_packed_process)
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
