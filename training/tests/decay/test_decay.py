"""Behavioral tests for consumer-step weighting and the zero boundary."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch
from prm_v2.live_config_v5 import DECAY_V1, decay_weight, validate_decay
from opencode_rl.skyrl_adapter import ExactTurnTrainerMixin
from opencode_rl.training import compute_turn_advantages
from opencode_rl.train_generator import OpenCodeGenerator

@pytest.mark.parametrize('step,expected',[(20,.5),(21,.3645),(22,.256),(23,.1715),
    (24,.108),(25,.0625),(26,.032),(27,.0135),(28,.004),(29,.0005),(30,0),(31,0),(50,0)])
def test_exact_schedule(step,expected):
    assert decay_weight(DECAY_V1,step)==pytest.approx(expected,abs=1e-12)

@pytest.mark.parametrize('step',[-1,21.5,True,None])
def test_invalid_update_rejected(step):
    with pytest.raises(ValueError):decay_weight(DECAY_V1,step)

def row(i, outcome, process):
    identity={'task_id':'t','trajectory_id':f'traj{i}','outcome':outcome}
    request={'request_id':f'req{i}','input_sha256':'a'*64,'weight_version':'v1','native_turn':0}
    return {**identity,'causal_process_enabled':True,'causal_format':'ocrl-causal-prefix-v1',
        'packed_response_length':2,'original_turn_count':1,
        'original_rows':[{**identity,**request,'turn':0,'process':process,'process_valid':True,'response_length':2}],
        'causal_spans':[{**request,'response_start':0,'response_end':2,'original_loss_mask':[1,1]}]}

def trainer(tmp_path,step):
    p=tmp_path/'plan.json';p.write_text(json.dumps({'process_decay':DECAY_V1}))
    obj=ExactTurnTrainerMixin();obj.global_step=step;obj.max_staleness_steps=1
    obj.cfg=NS(generator=NS(ocrl_process_weight=.5,ocrl_plan=str(p),ocrl_packed_prm_enabled=True),
        trainer=NS(algorithm=NS(grpo_norm_by_std=True,zero_variance_filter_tol=1e-6)))
    return obj

def test_weight_uses_consumer_step_and_does_not_reset_on_resume(tmp_path):
    obj=trainer(tmp_path,25)
    assert obj._ocrl_effective_process_weight()==.0625
    obj.global_step=30
    assert obj._ocrl_effective_process_weight()==0
    resumed=trainer(tmp_path,27)
    assert resumed._ocrl_effective_process_weight()==pytest.approx(.0135)
    assert resumed.cfg.generator.ocrl_process_weight==.5

def test_relative_process_signal_and_outcome_only_boundary(tmp_path):
    # Group with equal outcomes remains eligible while PRM is positive.
    from opencode_rl.training import _packed_signals
    rows=[row(0,0,-1),row(1,0,1)]
    out={'ocrl_rows':rows,'loss_masks':[[1,1],[1,1]]}
    obj=trainer(tmp_path,29)
    assert obj._ocrl_signal_groups(out)=={'t'}
    obj.global_step=30
    assert obj._ocrl_signal_groups(out)==set()
    rows[1]['outcome']=1
    rows[1]['original_rows'][0]['outcome']=1
    assert obj._ocrl_signal_groups(out)=={'t'}
    o,p=_packed_signals(rows,[[True],[True]],process_weight=.5,epsilon=1e-6,norm_by_std=True)
    o2,p2=_packed_signals(rows,[[True],[True]],process_weight=.0625,epsilon=1e-6,norm_by_std=True)
    assert o==o2
    assert p2==pytest.approx({k:v/8 for k,v in p.items()})

@pytest.mark.parametrize('step,expected',[(29,True),(30,False),(50,False)])
def test_generation_stops_new_prm_collection_at_boundary(step,expected):
    # Stop at the real rollout boundary so no simulated outputs can bypass
    # native token/provenance checks. Test actual _one, not a mirrored flag.
    class Reached(Exception):pass
    obj=object.__new__(OpenCodeGenerator)
    obj.process_judge=object();obj.process_decay=dict(DECAY_V1);obj.dataset=object()
    @asynccontextmanager
    async def boundary(*args,process_enabled=False,**kwargs):
        assert process_enabled is expected
        raise Reached()
        yield
    obj._completed_rollout=boundary
    with pytest.raises(Reached):asyncio.run(obj._one('t',0,Path('/unused'),step))

def test_schedule_changes_rejected():
    with pytest.raises(ValueError):validate_decay({**DECAY_V1,'start_step':0})

def test_packed_loss_advantages_decay_after_normalization_and_preserve_masks():
    rows=[row(0,.1,-1),row(1,.9,1)]
    loss=torch.tensor([[0.,1.,1.],[0.,1.,1.],[0.,0.,0.]])
    response=torch.tensor([[0,1,1],[0,1,1],[0,1,1]])
    def calc(weight):return compute_turn_advantages(rows,loss,process_weight=weight,
        enable_packed_process=True,response_mask=response)
    combined,o,p,_=calc(.5)
    c2,o2,p2,d2=calc(.0625)
    cz,oz,pz,_=calc(0.)
    assert torch.equal(o,o2) and torch.equal(o,oz)
    assert torch.allclose(p2,p/8) and torch.equal(c2,o2+p2)
    assert torch.equal(cz,o) and not pz.any()
    assert not c2[:,0].any() and not c2[2].any()
    assert all(d['process_coefficient']==.0625 for d in d2)
