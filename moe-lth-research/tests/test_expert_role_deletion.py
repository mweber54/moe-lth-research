"""Scientific safeguards for the checkpoint-only Stage 1 diagnostic."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

SPEC=importlib.util.spec_from_file_location('role_study',Path(__file__).resolve().parents[1]/'scripts/expert_role_deletion_v1/study.py')
study=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(study)


def test_deletion_restores_even_when_evaluation_fails():
    model=torch.nn.Linear(3,2,bias=False)
    before=model.weight.detach().clone()
    with pytest.raises(RuntimeError):
        with study.deletion(model,{'weight':torch.tensor([0,4])}):
            assert model.weight.flatten()[0]==0
            assert model.weight.flatten()[4]==0
            raise RuntimeError('evaluation failed')
    assert torch.equal(model.weight,before)
    assert model.weight.grad is None


def test_empty_jaccard_and_retention_are_undefined():
    empty=np.zeros(4,dtype=bool);nonempty=np.array([True,False,False,False])
    assert study.overlap(empty,empty)==dict(jaccard=None,retention=None,inflow=None)
    assert study.overlap(empty,nonempty)==dict(jaccard=0.,retention=None,inflow=1.)
    assert study.overlap(nonempty,empty)==dict(jaccard=0.,retention=0.,inflow=None)


def test_js_uses_overall_probe_and_is_undefined_without_traffic():
    assert study.jsd(np.array([0,0]),np.array([1,1])) is None
    assert study.jsd(np.array([2,2]),np.array([1,1]))==0
    assert study.jsd(np.array([2,0]),np.array([1,1]))>0


def test_shuffle_preserves_each_complete_batch_population():
    r={'ids':torch.tensor([0,0,1,1,2,2,3,3]),'accepted':torch.tensor([True,False]*4),'gates':torch.arange(8)}
    shuffled=study.shuffle_route(r,np.random.default_rng(17),batch_tokens=4)
    for i in (0,4):
        before=list(zip(*(r[k][i:i+4].tolist() for k in r)))
        after=list(zip(*(shuffled[k][i:i+4].tolist() for k in r)))
        assert sorted(before)==sorted(after)


def test_forecast_never_selects_using_future_B_or_retrospective_filter():
    groups={'g':{'expert':0,'indices':torch.tensor([1,2])}}
    def panel(future_B):
        return [dict(kind=k,step=t,probe=p,group='g',subset='whole',damage=(future_B if p=='B' and t==1000 else 0.))
                for k in ('sparse','unpruned') for t in study.STEPS for p in ('A','B')]
    a,decisions_a=study.forecast(panel(100.),groups)
    b,decisions_b=study.forecast(panel(-100.),groups)
    assert decisions_a==decisions_b
    assert [r['selected'] for r in a]==[r['selected'] for r in b]
    selected=[r for r in a if r['decision_step']==250 and r['rule']=='persistent']
    assert selected and all(r['false_safe'] for r in selected)
    assert all(r['selected'] for r in selected)
    assert len({r['decision_step'] for r in a if r['rule']=='persistent'})==2


def test_protected_preserves_rng_modes_and_existing_gradients():
    m=torch.nn.Linear(2,1);m.weight.grad=torch.ones_like(m.weight)
    state=torch.get_rng_state().clone()
    with study.protected(m):
        assert not m.training
        torch.rand(10)
    assert m.training and torch.equal(state,torch.get_rng_state())
    assert torch.equal(m.weight.grad,torch.ones_like(m.weight))
