"""Pure observer metrics; never calibrates, edits, clips or replaces rewards.

Agreement is not correctness. Saturation flags nominate reviews, not bad labels.
Batch/active-loss scopes are not all generated rollouts or post-PPO gradients.
"""
from collections import Counter
import math

from .judge_v5 import EVENTS

VERSION = 'prm-v5-observation-metrics-2'
BINS = (('minus_one', -1.), ('minus_two_thirds', -2/3), ('minus_one_third', -1/3),
        ('zero', 0.), ('plus_one_third', 1/3), ('plus_two_thirds', 2/3), ('plus_one', 1.))
STAGES = ('s_1', 's_2', 's_3', 's_4e', 'unknown')
COUNT_KEYS = ('turns', 'valid_turns', 'invalid_turns', 'windows', 'unavailable_windows',
    'overflow_windows', 'unavailable_logical_votes', 'successful_logical_votes',
    'fully_valid_logical_votes', 'partial_logical_votes', 'invalid_assessment_rows',
    'successful_votes_after_retry', 'successful_votes_with_usage',
    'unavailable_turn_vote_slots', 'abstain_turn_vote_slots', 'raw_minus_one',
    'raw_zero', 'raw_plus_one', 'three_scored_turns', 'unanimous_turns',
    'disagreement_turns', 'unanimous_plus_one_turns', 'unanimous_zero_turns',
    'unanimous_minus_one_turns', 'vote_variance_sum', 'uncovered_turns')


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def _bucket(value):
    if not _finite(value):
        raise ValueError('Non-finite process observation')
    for label, number in BINS:
        if abs(value-number) < 1e-6:
            return label
    raise ValueError('Expected mean of exactly three ternary votes')


def summarize(result):
    """One completed original attempt, suitable for its admitted.json row."""
    scores, validity = result['scores'], result['valid']
    if len(scores) != len(validity) or len(result['request_ids']) != len(scores):
        raise ValueError('Metric request/score identity lengths differ')
    count, n = Counter({key: 0 for key in COUNT_KEYS}), len(scores)
    count['turns'] = n
    histogram = Counter({name: 0 for name, _ in BINS})
    samples = []
    for number, (score, valid) in enumerate(zip(scores, validity), 1):
        if type(valid) is not bool:
            raise ValueError('Metric validity is not a boolean')
        if valid:
            histogram[_bucket(score)] += 1
            samples.append(float(score))
            count['valid_turns'] += 1
        elif score != 0:
            raise ValueError('Unavailable process score must be masked zero')
    events, scopes, limitations = Counter(), Counter(), Counter()
    window_seconds, request_seconds, queue_seconds = [], [], []
    token_usage = Counter()
    covered, reviewed = set(), []
    for window in result.get('windows', []):
        lo, hi = window['start_turn'], window['end_turn']
        if not 1 <= lo <= hi <= n or covered.intersection(range(lo, hi+1)):
            raise ValueError('Duplicate or out-of-bound process window')
        covered.update(range(lo, hi+1))
        count['windows'] += 1
        status = window.get('status', 'unavailable')
        if status != 'complete':
            count['unavailable_windows'] += 1
        if status == 'admission_overflow':
            count['overflow_windows'] += 1
        if _finite(window.get('elapsed_s')):
            window_seconds.append(window['elapsed_s'])
        votes = window.get('votes') or [None]*3
        if len(votes) != 3:
            raise ValueError('Metric window needs three logical vote slots')
        for vote in votes:
            if vote is None:
                count['unavailable_logical_votes'] += 1
                continue
            count['successful_logical_votes'] += 1
            typed = vote['typed']['assessments']
            canonical = vote['canonical']['assessments']
            if len(typed) != hi-lo+1 or len(canonical) != len(typed):
                raise ValueError('Metric vote row inventory differs from window')
            invalid = []
            for number, row, plain in zip(range(lo, hi+1), typed, canonical):
                if (row is None) != (plain is None):
                    raise ValueError('Metric typed/canonical validity mismatch')
                if row is None:
                    invalid.append(number)
                elif row.get('turn') != number or plain.get('turn') != number:
                    raise ValueError('Metric votes crossed turn identity')
            errors = vote.get('row_errors', [])
            if [e['turn'] for e in errors] != invalid:
                raise ValueError('Metric invalid rows lack matching error provenance')
            count['partial_logical_votes' if invalid else 'fully_valid_logical_votes'] += 1
            count['invalid_assessment_rows'] += len(invalid)
            telemetry = vote.get('_transport', {})
            if telemetry.get('attempts', 1) > 1:
                count['successful_votes_after_retry'] += 1
            for field, collection in (('request_latency_s', request_seconds), ('queue_s', queue_seconds)):
                if _finite(telemetry.get(field)):
                    collection.append(telemetry[field])
            usage = telemetry.get('usage') or {}
            if usage:
                count['successful_votes_with_usage'] += 1
                for field in ('promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount', 'totalTokenCount'):
                    if _finite(usage.get(field)) and usage[field] >= 0:
                        token_usage[field] += usage[field]
            for row in vote['typed']['assessments']:
                if row is None:
                    continue
                events[row['event'] if row['event'] in EVENTS else 'unknown'] += 1
                scopes[row['scope']] += 1
                limitations.update(row.get('limitations', []))
        for offset, number in enumerate(range(lo, hi+1)):
            values = []
            for vote in votes:
                if vote is None:
                    count['unavailable_turn_vote_slots'] += 1
                    continue
                assessment = vote['canonical']['assessments'][offset]
                if assessment is None:
                    count['unavailable_turn_vote_slots'] += 1
                    continue
                if assessment['turn'] != number:
                    raise ValueError('Metric votes crossed turn identity')
                value = assessment['score']
                if value is None:
                    count['abstain_turn_vote_slots'] += 1
                elif type(value) is int and value in (-1, 0, 1):
                    values.append(value)
                    count[{-1:'raw_minus_one', 0:'raw_zero', 1:'raw_plus_one'}[value]] += 1
                else:
                    raise ValueError('Expected an unaveraged ternary vote')
            if len(values) == 3:
                count['three_scored_turns'] += 1
                if len(set(values)) == 1:
                    count['unanimous_turns'] += 1
                else:
                    count['disagreement_turns'] += 1
                    reviewed.append({'turn': number, 'reason': 'vote_disagreement', 'votes': values})
                if values == [1, 1, 1]:
                    count['unanimous_plus_one_turns'] += 1
                if values == [0, 0, 0]:
                    count['unanimous_zero_turns'] += 1
                if values == [-1, -1, -1]:
                    count['unanimous_minus_one_turns'] += 1
                mean = sum(values)/3.
                count['vote_variance_sum'] += sum((x-mean)**2 for x in values)/3.
    # Window omissions and API failures are unavailable, never ordinary neutral votes.
    count['uncovered_turns'] = n-len(covered)
    count['unavailable_turn_vote_slots'] += 3*count['uncovered_turns']
    count['invalid_turns'] = n-count['valid_turns']
    valid_n, full = count['valid_turns'], count['three_scored_turns']
    flags = {
        'all_valid_plus_one': valid_n >= 10 and valid_n == n and histogram['plus_one'] == valid_n,
        'all_valid_mean_zero': valid_n >= 10 and valid_n == n and histogram['zero'] == valid_n,
        'high_disagreement': full >= 10 and count['disagreement_turns']/full >= .35,
        'low_coverage': n >= 10 and valid_n/n < .8,
        'queue_overflow': count['overflow_windows'] > 0,
    }
    if flags['all_valid_plus_one']:
        reviewed += [{'turn': i, 'reason': 'all_plus_one_review'} for i in range(1, min(n, 5)+1)]
    return {'version': VERSION, **{key: result.get(key) for key in
            ('protocol', 'prompt_sha256', 'error', 'attempt_id')},
        'valid_turns': valid_n, 'total_turns': n, 'counts': dict(count),
        'mean_score_histogram': dict(histogram), 'score_sum': sum(samples),
        'score_squared_sum': sum(x*x for x in samples), 'event_counts': dict(events),
        'scope_counts': dict(scopes), 'limitation_counts': dict(limitations),
        'window_seconds': window_seconds, 'successful_request_seconds': request_seconds,
        'successful_vote_queue_seconds': queue_seconds, 'successful_vote_usage': dict(token_usage),
        'review_flags': flags, 'review_turns': reviewed[:8],
        'review_flags_are_not_quality_labels': True,
        'latency_scope': 'window includes queue; request latency is last successful attempt only; missing timing is unknown',
        'usage_scope': 'successful logical vote final attempts only; not total provider usage'}


def _quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    lower = int(position)
    upper = min(len(ordered)-1, lower+1)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(position-lower)


def _aggregate(summaries, prefix):
    count, hist, events, scopes, limitations, flags, usage = (Counter() for _ in range(7))
    latencies = {key: [] for key in ('window_seconds', 'successful_request_seconds', 'successful_vote_queue_seconds')}
    total, squared = 0., 0.
    for item in summaries:
        if item.get('version') != VERSION:
            raise ValueError('Unexpected process metrics version')
        count.update(item['counts']); hist.update(item['mean_score_histogram'])
        events.update(item['event_counts']); scopes.update(item['scope_counts'])
        limitations.update(item['limitation_counts']); flags.update(item['review_flags'])
        usage.update(item['successful_vote_usage'])
        total += item['score_sum']; squared += item['score_squared_sum']
        for key in latencies:
            latencies[key].extend(item[key])
    n, v, full = count['turns'], count['valid_turns'], count['three_scored_turns']
    raw = sum(count[key] for key in ('raw_minus_one', 'raw_zero', 'raw_plus_one'))
    mean = total/v if v else 0.
    result = {'trajectories': len(summaries), **dict(count), 'valid_fraction': v/max(1,n),
        'score_mean_valid': mean, 'score_std_valid': math.sqrt(max(0., squared/v-mean*mean)) if v else 0.,
        'score_mean_including_fallback_zero': total/max(1,n),
        'numeric_vote_slots': raw, 'disagreement_fraction': count['disagreement_turns']/max(1,full),
        'partial_logical_vote_fraction': count['partial_logical_votes']/max(1,count['successful_logical_votes']),
        'unanimous_fraction': count['unanimous_turns']/max(1,full),
        'mean_vote_variance': count['vote_variance_sum']/max(1,full),
        'abstain_vote_fraction': count['abstain_turn_vote_slots']/max(1,3*n),
        'unavailable_vote_fraction': count['unavailable_turn_vote_slots']/max(1,3*n)}
    for key in ('raw_minus_one', 'raw_zero', 'raw_plus_one'):
        result[key+'_fraction'] = count[key]/max(1,raw)
    for label, _ in BINS:
        result['score_hist/'+label+'/count'] = hist[label]
        result['score_hist/'+label+'/fraction'] = hist[label]/max(1,v)
    for label in EVENTS:
        result['event/'+label+'/count'] = events[label]
        result['event/'+label+'/fraction'] = events[label]/max(1,sum(events.values()))
    for label in ('visual','audio','metadata','artifact','execution','reasoning','io','unknown'):
        result['scope/'+label+'/count'] = scopes[label]
    for key, value in flags.items():
        result['review/'+key+'/trajectories'] = value
    for key, values in latencies.items():
        result['latency/'+key+'/count'] = len(values)
        if values:
            for label, fraction in (('p50', .5), ('p95', .95)):
                result['latency/'+key+'/'+label] = _quantile(values, fraction)
    for key, value in usage.items():
        result['successful_final_attempt_usage/'+key] = value
    # Engineering review thresholds, never automatic reward changes. Sample
    # floors avoid interpreting a small, perfectly good window as judge collapse.
    result.update({
        'review/high_plus_one_saturation': int(v >= 100 and hist['plus_one']/v >= .95),
        'review/high_mean_zero_saturation': int(v >= 100 and hist['zero']/v >= .95),
        'review/high_disagreement': int(full >= 100 and count['disagreement_turns']/full >= .35),
        'review/low_coverage': int(n >= 100 and v/n < .8),
        'review/queue_overflow': int(count['overflow_windows'] > 0),
    })
    return {prefix+'/'+key: value for key, value in result.items()}


def batch_judge_metrics(rows, *, active_trajectory_ids=None):
    """Deduplicate original trajectories; optional loss-active scope is separate."""
    selected = {}
    for row in rows:
        summary = row.get('process_judge_summary')
        if summary is None:
            continue
        key = row['trajectory_id']
        item = (row.get('stage', 'unknown'), summary)
        if key in selected and selected[key] != item:
            raise ValueError('Conflicting process summaries for one trajectory')
        selected[key] = item
    result = {}
    scopes = {'batch': selected}
    if active_trajectory_ids is not None:
        scopes['active_loss'] = {k:v for k,v in selected.items() if k in active_trajectory_ids}
    for scope, members in scopes.items():
        prefix = 'prm/judge/'+scope
        result.update(_aggregate([item[1] for item in members.values()], prefix))
        for stage in STAGES:
            subset = [item[1] for item in members.values() if
                      (item[0] if item[0] in STAGES else 'unknown') == stage]
            if subset:
                result.update(_aggregate(subset, prefix+'/stage/'+stage))
        result[prefix+'/rows_missing_summary'] = (len({r['trajectory_id'] for r in rows})-len(selected)
                                                  if scope == 'batch' else 0)
    return result
