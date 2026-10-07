"""Strict native context-overflow recovery; never turn HTTP failures into reward.

Only vLLM's pre-admission context rejection is recoverable. The original request
and HTTP error remain archived, but contain no sampled action to train on.
OpenCode itself owns compaction and the subsequent request, without re-tokenizing
or trimming its context in the harness.
"""
import json
import re

from .contract import ParityError

_CONTEXT = re.compile(r"^This model's maximum context length is [1-9][0-9]* tokens\. ")


def is_context_message(message):
    return isinstance(message, str) and bool(_CONTEXT.match(message)) and (
        'Please reduce the length of the input prompt or the number of requested output tokens.' in message)


def is_context_response(status, body):
    if status != 400 or len(body) > 65536:
        return False
    try:
        error = json.loads(body)['error']
        return (error.get('type') == 'BadRequestError' and error.get('code') == 400
                and error.get('param') == 'input_tokens' and is_context_message(error.get('message')))
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def primary_inventory(store, attempt_id, contract, *, on_complete=None):
    """Account for every native build call, including rejected zero-action calls.

This validates inventory/provenance. Callers must also validate every successful
turn's tokens/tensors/logprobs, either through load_turn() or the on_complete
callback over the already hash-verified journal. Do not reread large request
bodies solely to run the second set of checks.
"""
    refs = store.attempt_requests(attempt_id)
    rows = sorted((r for r in refs if r['metadata'].get('primary') is True),
                  key=lambda r: r['metadata']['turn'])
    if not rows or [r['metadata']['turn'] for r in rows] != list(range(len(rows))):
        raise ParityError('Missing/duplicated or unaccounted-for primary request turns')
    sessions = {r['metadata']['session_id'] for r in rows}
    if len(sessions) != 1:
        raise ParityError('Cross-session training attempt')
    session = next(iter(sessions))
    accepted, rejected, calls = [], [], set()
    for row in rows:
        events, seal = store.read_request(row['request_id'])
        requests = [e['data'] for e in events if e['kind'] == 'request']
        meta = row['metadata']
        if (len(requests) != 1 or requests[0]['metadata'] != meta
                or meta.get('contract_sha256') != contract.sha256
                or meta.get('attempt_id') != attempt_id or meta.get('agent') != 'build'):
            raise ParityError('Primary request identity/contract mismatch')
        if meta['logical_call_id'] in calls:
            raise ParityError('Ambiguous retry: duplicate native logical call')
        calls.add(meta['logical_call_id'])
        if seal['status'] == 'complete':
            accepted.append(row)
            if on_complete is not None:
                on_complete(row, events, seal)
            continue
        errors = [e['data'] for e in events if e['kind'] == 'error']
        markers = [e['data'] for e in events if e['kind'] == 'context_overflow_rejected']
        if (seal['status'] != 'context_overflow' or len(errors) != 1
                or errors[0].get('type') != 'VLLMValidationError'
                or not is_context_message(errors[0].get('message'))
                or markers != [{'before_engine_admission': True, 'generated_tokens': 0}]
                or any(e['kind'].startswith('engine_') or e['kind'].startswith('wire_') for e in events)):
            raise ParityError('Incomplete or unaccounted-for policy request cannot be discarded')
        rejected.append(row)
    if rejected:
        # The relay additionally proves chronological overflow -> successful
        # native compaction -> successful primary continuation before CLI exit.
        compacted = False
        for ref in refs:
            meta = ref['metadata']
            if meta.get('agent') != 'compaction' or meta.get('session_id') != session:
                continue
            events, seal = store.read_request(ref['request_id'])
            inputs = [e['data'] for e in events if e['kind'] == 'engine_input']
            outputs = [e['data'] for e in events if e['kind'] == 'engine_output']
            wires = [e['data'] for e in events if e['kind'] == 'wire_complete']
            if (meta.get('contract_sha256') == contract.sha256 and seal['status'] == 'complete'
                    and len(inputs) == 1 and inputs[0].get('contract_sha256') == contract.sha256
                    and outputs and outputs[-1].get('finished') is True
                    and wires == [{'done': True}]):
                compacted = True
        if not compacted or not accepted or accepted[-1]['metadata']['turn'] <= rejected[-1]['metadata']['turn']:
            raise ParityError('Context overflow has no successful native compaction/continuation')
    return accepted, rejected
