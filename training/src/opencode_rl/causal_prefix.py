"""Exact causal-prefix packets, independent of the native rollout loop.

Planning alone does NOT enable training. Consumers must preserve every span's
original loss, geometric-IS and KL normalization. Tool/context gaps are inputs,
never new policy actions. No tokenization, image processing or I/O occurs here.
"""
from copy import deepcopy

from .contract import ParityError


def can_extend(previous, current):
    a, b = previous['input'], current['input']
    for key in ('attempt_id', 'session_id', 'contract_sha256'):
        if previous['metadata'].get(key) != current['metadata'].get(key):
            return False
    if current['metadata']['turn'] != previous['metadata']['turn'] + 1:
        return False
    sequence = a['prompt_token_ids'] + previous['response_ids']
    if b['prompt_token_ids'][:len(sequence)] != sequence:
        return False
    return [f for f in b['features'] if f['offset'] < len(sequence)] == a['features']


def plan_chains(turns):
    """Return ordered packets retaining a one-to-one inventory of real calls."""
    if not turns:
        raise ParityError('Cannot pack an empty policy trajectory')
    seen, groups = set(), []
    for turn in turns:
        key = turn['request_id']
        if key in seen:
            raise ParityError('Policy request would train twice')
        seen.add(key)
        n = len(turn['response_ids'])
        if not n or n != len(turn['loss_mask']) or n != len(turn['rollout_logprobs']):
            raise ParityError('Incomplete original policy actions')
        if not groups or not can_extend(groups[-1][-1], turn):
            groups.append([])
        groups[-1].append(turn)
    packets = [_packet(group) for group in groups]
    inventory = [s['request_id'] for p in packets for s in p['spans']]
    if inventory != [t['request_id'] for t in turns]:
        raise ParityError('Causal packing changed the request inventory')
    return packets


def _packet(group):
    first, terminal = group[0], group[-1]
    sequence = terminal['input']['prompt_token_ids'] + terminal['response_ids']
    offset = len(first['input']['prompt_token_ids'])
    features = terminal['input']['features']
    mask, logprobs = [0]*(len(sequence)-offset), [0.]*(len(sequence)-offset)
    spans, previous_end = [], offset
    for turn in group:
        prompt = turn['input']['prompt_token_ids']
        start, end = len(prompt), len(prompt)+len(turn['response_ids'])
        if start < previous_end or sequence[:end] != prompt+turn['response_ids']:
            raise ParityError('Overlapping actions or changed causal context')
        if [f for f in features if f['offset'] < end] != turn['input']['features']:
            raise ParityError('Image context differs inside a shared prefix')
        lo, hi = start-offset, end-offset
        mask[lo:hi], logprobs[lo:hi] = turn['loss_mask'], turn['rollout_logprobs']
        spans.append({'request_id': turn['request_id'], 'start': start, 'end': end,
            'response_start': lo, 'response_end': hi, 'native_turn': turn['metadata']['turn'],
            'input_sha256': turn['input']['input_sha256'],
            'weight_version': turn['input']['weight_version'],
            'original_loss_mask': list(turn['loss_mask'])})
        previous_end = end
    types = [0]*len(sequence)
    for feature in features:
        start, n = feature['offset'], feature['length']
        if type(start) is not int or type(n) is not int or start < 0 or n < 1 or start+n > len(terminal['input']['prompt_token_ids']):
            raise ParityError('Invalid captured image positions')
        if any(types[start:start+n]):
            raise ParityError('Captured image positions overlap')
        types[start:start+n] = [1]*n
    packet = {'terminal_request_id': terminal['request_id'],
        'prompt_token_ids': sequence[:offset], 'response_ids': sequence[offset:],
        'loss_mask': mask, 'rollout_logprobs': logprobs, 'features': deepcopy(features),
        'mm_token_type_ids': types, 'spans': spans,
        'original_turn_count': len(group), 'format': 'ocrl-causal-prefix-v1'}
    verify_packet(packet, group)
    return packet


def verify_packet(packet, original_turns):
    """Verify every action's exact original input/media and reject gap loss."""
    sequence = packet['prompt_token_ids'] + packet['response_ids']
    offset = len(packet['prompt_token_ids'])
    spans = packet['spans']
    if packet.get('format') != 'ocrl-causal-prefix-v1' or len(spans) != len(original_turns):
        raise ParityError('Incomplete causal packet provenance')
    if packet['original_turn_count'] != len(original_turns):
        raise ParityError('Original KL row count changed')
    size = len(packet['response_ids'])
    if len(packet['loss_mask']) != size or len(packet['rollout_logprobs']) != size:
        raise ParityError('Packed policy tensor lengths differ')
    covered, previous_end = [False]*size, offset
    for span, turn in zip(spans, original_turns):
        start, end = len(turn['input']['prompt_token_ids']), len(turn['input']['prompt_token_ids'])+len(turn['response_ids'])
        lo, hi = start-offset, end-offset
        identity = {'request_id': turn['request_id'], 'start': start, 'end': end,
            'response_start': lo, 'response_end': hi, 'native_turn': turn['metadata']['turn'],
            'input_sha256': turn['input']['input_sha256'], 'weight_version': turn['input']['weight_version'],
            'original_loss_mask': turn['loss_mask']}
        if span != identity or start < previous_end or not 0 <= lo < hi <= size:
            raise ParityError('Packed action boundary changed')
        if sequence[:start] != turn['input']['prompt_token_ids'] or sequence[start:end] != turn['response_ids']:
            raise ParityError('Packed training input differs from native policy input')
        if packet['loss_mask'][lo:hi] != turn['loss_mask'] or packet['rollout_logprobs'][lo:hi] != turn['rollout_logprobs']:
            raise ParityError('Original policy mask or behavior logprob changed')
        if [f for f in packet['features'] if f['offset'] < end] != turn['input']['features']:
            raise ParityError('Packed media differs from native policy input')
        covered[lo:hi] = [True]*(hi-lo)
        previous_end = end
    if previous_end != len(sequence):
        raise ParityError('Unaccounted trailing actions')
    if any(not yes and (packet['loss_mask'][i] != 0 or packet['rollout_logprobs'][i] != 0.)
           for i, yes in enumerate(covered)):
        raise ParityError('Tool/context gaps cannot become policy targets')
    expected_types = [0]*len(sequence)
    for f in packet['features']:
        expected_types[f['offset']:f['offset']+f['length']] = [1]*f['length']
    if packet['mm_token_type_ids'] != expected_types:
        raise ParityError('Packed media-position map changed')


def original_action_slices(packet, flat):
    """Preserve autograd when flat is a tensor; never detach reconstructed LPs."""
    if len(flat) != len(packet['response_ids']):
        raise ParityError('Forward does not cover the whole causal packet')
    return {s['request_id']: flat[s['response_start']:s['response_end']] for s in packet['spans']}
