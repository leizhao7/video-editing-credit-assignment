from copy import deepcopy

import pytest
import torch

from opencode_rl.causal_prefix import plan_chains, verify_packet, original_action_slices
from opencode_rl.contract import ParityError


def turn(i, prompt, response, features=None):
    return {'request_id': str(i), 'metadata': {'attempt_id': 'a', 'session_id': 's',
        'contract_sha256': 'c', 'turn': i}, 'input': {'prompt_token_ids': prompt,
        'features': features or [], 'input_sha256': f'sha{i}', 'weight_version': f'w:{i}'},
        'response_ids': response, 'loss_mask': [1]*len(response), 'rollout_logprobs': [-.2]*len(response)}


def test_exact_inputs_actions_media_and_gradients_survive_packing():
    a = turn(0, [10, 11], [20, 21])
    b = turn(1, [10, 11, 20, 21, 99, 30], [22], [{'offset': 5, 'length': 1, 'data': {'pixels': 'b'}}])
    a['loss_mask'][1] = 0
    original = deepcopy([a, b])
    packets = plan_chains([a, b])
    assert len(packets) == 1
    packet = packets[0]
    assert packet['prompt_token_ids'] == [10, 11]
    assert packet['response_ids'] == [20, 21, 99, 30, 22]
    assert packet['loss_mask'] == [1, 0, 0, 0, 1]
    assert packet['mm_token_type_ids'] == [0, 0, 0, 0, 0, 1, 0]
    flat = torch.arange(5., requires_grad=True)
    slices = original_action_slices(packet, flat)
    (slices['0'].sum()+slices['1'].sum()).backward()
    assert flat.grad.tolist() == [1, 1, 0, 0, 1]
    assert [a, b] == original


@pytest.mark.parametrize('change', ['token', 'image', 'session', 'gap', 'contract'])
def test_unproven_prefixes_are_separate_packets(change):
    f = {'offset': 0, 'length': 1, 'data': {'pixels': 'a'}}
    a, b = turn(0, [1, 2], [3], [f]), turn(1, [1, 2, 3, 4], [5], [deepcopy(f)])
    if change == 'token': b['input']['prompt_token_ids'][0] = 8
    if change == 'image': b['input']['features'][0]['data']['pixels'] = 'different'
    if change == 'session': b['metadata']['session_id'] = 'other'
    if change == 'gap': b['metadata']['turn'] = 2
    if change == 'contract': b['metadata']['contract_sha256'] = 'other'
    assert len(plan_chains([a, b])) == 2


@pytest.mark.parametrize('change', ['mask', 'behavior', 'boundary', 'media', 'identity', 'count'])
def test_postpack_corruption_is_not_accepted(change):
    rows = [turn(0, [1, 2], [3]), turn(1, [1, 2, 3, 99], [4])]
    p = plan_chains(rows)[0]
    if change == 'mask': p['loss_mask'][1] = 1
    if change == 'behavior': p['rollout_logprobs'][0] = -.9
    if change == 'boundary': p['spans'][0]['end'] += 1
    if change == 'media': p['mm_token_type_ids'][1] = 1
    if change == 'identity': p['spans'][0]['input_sha256'] = 'wrong'
    if change == 'count': p['original_turn_count'] = 1
    with pytest.raises(ParityError): verify_packet(p, rows)


def test_duplicate_missing_and_truncated_actions_fail_closed():
    a = turn(0, [1, 2], [3])
    with pytest.raises(ParityError): plan_chains([a, a])
    with pytest.raises(ParityError): plan_chains([])
    packet = plan_chains([a])[0]
    with pytest.raises(ParityError): original_action_slices(packet, [])
    a['rollout_logprobs'] = []
    with pytest.raises(ParityError): plan_chains([a])
