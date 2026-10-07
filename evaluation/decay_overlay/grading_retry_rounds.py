"""Run the caller's finite scoring rounds inside one private grader task."""
import json
import math
from pathlib import Path
import time


def run_rounds(task, stage, runner, collect, deadline_unix, *, max_rounds,
               prohibited_evidence, clock=time.time):
    if type(max_rounds) is not int or not 1<=max_rounds<=3:
        raise ValueError('Expected at most the existing three scoring rounds')
    if type(deadline_unix) not in (int,float) or not math.isfinite(deadline_unix):
        raise ValueError('Invalid absolute deadline')
    if stage not in ('s_3','s_4e'):
        raise ValueError('Only judged stages use internal scoring rounds')
    rounds = []
    score, detail, artifacts = None, 'Total grading deadline exhausted', {}
    reason = 'deadline_exhausted'
    for index in range(max_rounds):
        if clock()>=deadline_unix:
            break
        started = clock()
        score, detail = runner(reuse_evidence=index>0)
        artifacts = collect()
        result = {'score':score,'detail':detail,'judge_files':artifacts}
        prefix = 's3' if stage=='s_3' else 's4e'
        refused = bool(prohibited_evidence(result,stage))
        transport_failed = any(isinstance(row,dict) and
            str(row.get('err','')).startswith('GRADING_TRANSPORT_FAILURE:')
            for name in (prefix+'_score.json',prefix+'_score_ref.json')
            for row in artifacts.get(name,{}).get('rubrics',[]))
        retry = not refused and (score is None or transport_failed)
        if not retry and not refused and (type(score) not in (int,float) or
                not math.isfinite(score) or not 0<=score<=1):
            raise ValueError('Invalid final score; cannot turn corruption into retries')
        snapshot = {'round':index+1,'started_at':started,'finished_at':clock(),
                    'deadline_unix':deadline_unix,'retryable':retry,'provider_refusal':refused,
                    **result}
        folder = Path(task)/'evidence/scoring_rounds'/('round_%02d'%(index+1))
        folder.mkdir(parents=True,exist_ok=False)
        (folder/'result.json').write_text(json.dumps(snapshot,ensure_ascii=False,allow_nan=False))
        rounds.append({k:v for k,v in snapshot.items() if k not in ('judge_files','detail')})
        if refused:
            reason = 'provider_refusal'
            break
        if not retry:
            reason = 'complete'
            break
        reason = 'rounds_exhausted'
    if reason!='complete' and reason!='provider_refusal' and clock()>=deadline_unix:
        reason = 'deadline_exhausted'
    return score,detail,artifacts,{'format':'outcome-internal-rounds-v1','rounds_limit':max_rounds,
        'rounds_used':len(rounds),'deadline_unix':deadline_unix,'reason':reason,'rounds':rounds}
