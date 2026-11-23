"""Tests for fixed scope, valid swaps, persistence and initialization equivalence."""
import importlib
import json
from pathlib import Path
import sys
import os
import pytest
import torch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts/imp_mask_information_v2'
sys.path.insert(0,str(SCRIPTS))
_previous_safe_load=os.environ.get('TORCH_FORCE_WEIGHTS_ONLY_LOAD')
import v2_common as shared
import degree
# Do not alter unrelated test modules through collection-time environment state.
if _previous_safe_load is None:os.environ.pop('TORCH_FORCE_WEIGHTS_ONLY_LOAD',None)
else:os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']=_previous_safe_load
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import expert_weight_parameters,save_masks,load_masks,apply_masks_


def tiny():
    return TinyMoELanguageModel(dict(vocab_size=16,max_seq_len=8,num_layers=2,num_heads=2,d_model=4,num_experts=2,expert_hidden_size=8,dropout=0.,top_k=1,capacity_factor=1.25))


def test_gate_is_all_five_strict_and_finite():
    effect={s:.03 for s in shared.SEEDS}
    assert shared.matrix_gate(effect)
    for value in [.02,.019,float('nan'),float('inf')]:
        bad=dict(effect);bad[61]=value;assert not shared.matrix_gate(bad)
    del effect[61];assert not shared.matrix_gate(effect)


def test_binary_swap_counts_reproducibility_and_persistence(tmp_path,monkeypatch):
    monkeypatch.setattr(degree,'OUT',tmp_path);monkeypatch.setattr(degree,'_LIB',None)
    gen=torch.Generator().manual_seed(2026)
    original=torch.rand((40,80),generator=gen)<.2
    a,da=degree.chain(original,17);b,db=degree.chain(original,17);c,dc=degree.chain(original,29)
    assert degree.same_degrees(original,a)
    assert torch.equal(a,b) and da==db
    assert not torch.equal(a,c)
    assert da['attempts']==20*int(original.sum()) and da['accepted_swaps']>0
    assert 0<=da['rejection_rate']<1
    assert [d['length'] for d in da['segments']]==[2,5,10,20]
    assert da['survivor_overlap']<.5
    path=tmp_path/'mask.pt';save_masks({'matrix':a},path)
    assert torch.equal(load_masks(path)['matrix'],a)
    assert not da['uniform_sampling_claimed'] and not da['independent_stationary_draws_claimed']


@pytest.mark.parametrize('value',[False,True])
def test_degenerate_masks_are_not_admitted(value):
    with pytest.raises(ValueError):degree.chain(torch.full((4,5),value,dtype=torch.bool),7)


def test_coordinate_identity_and_identical_mask_forward(tmp_path):
    torch.manual_seed(7);original=tiny();state={n:t.clone() for n,t in original.state_dict().items()}
    masks={n:(torch.arange(p.numel()).reshape(p.shape)%3==0) for n,p in expert_weight_parameters(original).items()}
    path=tmp_path/'mask.pt';save_masks(masks,path);persisted=load_masks(path)
    first=tiny();second=tiny();first.load_state_dict(state);second.load_state_dict(state)
    apply_masks_(first,masks);apply_masks_(second,persisted);first.eval();second.eval()
    for n,p in expert_weight_parameters(first).items():
        assert torch.equal(p[masks[n]],state[n][masks[n]])
        assert torch.count_nonzero(p[~masks[n]])==0
    hashes=shared.validate_coordinates(state,masks)
    actual=shared.initial_hashes(state,persisted)
    assert hashes==actual
    ids=torch.tensor([[1,2,3,4]])
    with torch.no_grad():assert torch.equal(first(ids).logits,second(ids).logits)
    for n,t in first.state_dict().items():
        if n not in masks:assert torch.equal(t,state[n])


def test_legacy_canonical_module_isolation():
    previous=sys.modules.get('common')
    module=shared.canonical()
    assert hasattr(module.common,'protocol_errors')
    assert sys.modules.get('common') is previous


def test_nested_analysis_counts_source_seeds(tmp_path,monkeypatch):
    import report
    monkeypatch.setattr(report,'OUT',tmp_path);monkeypatch.setattr(report,'figures',lambda *args:None)
    jobs=[];observed={};states={}
    for seed in shared.SEEDS:
        for condition,rep in [('IMP',0)]+[('matrix',r) for r in range(5)]:
            key=f'{seed}_{condition}_{rep}';j=dict(id=key,phase=1,seed=seed,condition=condition,replicate=rep);jobs.append(j)
            states[key]=dict(status='COMPLETE')
            observed[key]=dict(job=key,seed=seed,condition=condition,replicate=rep,final_ce=1.5 if condition=='IMP' else 1.6+rep*.01,historical_dense_ce=1.48,historical_imp_ce=1.5)
    monkeypatch.setattr(report,'job_status',lambda j:states[j['id']]);monkeypatch.setattr(report,'result',lambda j:observed[j['id']])
    report.analysis(dict(jobs=jobs))
    aggregate=json.loads((tmp_path/'aggregate_effects.json').read_text())['matrix']
    assert aggregate['n']==5
    assert aggregate['source_seeds']==list(shared.SEEDS)
    assert abs(aggregate['mean']-.12)<1e-10
    paired=shared.rows(tmp_path/'paired_effects.csv')
    assert len(paired)==5 and all(r['n_masks']=='5' for r in paired)


def transition_fixture(tmp_path,monkeypatch,one_failed_effect=False):
    import worker
    monkeypatch.setattr(worker,'OUT',tmp_path)
    jobs=[];records={}
    for seed in shared.SEEDS:
        for c,i in [('IMP',0)]+[('matrix',r) for r in range(5)]+[('degree',r) for r in range(5)]:
            key=f'{seed}_{c}_{i}';jobs.append(dict(id=key,phase=2 if c=='degree' else 1,seed=seed,condition=c,replicate=i))
            records[key]=dict(final_ce=1.5 if c=='IMP' else 1.51 if seed==61 and one_failed_effect else 1.6)
    monkeypatch.setattr(worker,'job_status',lambda j:dict(status='CONDITIONAL' if j['phase']==2 else 'COMPLETE'))
    monkeypatch.setattr(worker,'result',lambda j:records[j['id']])
    return worker,dict(jobs=jobs)


def test_one_failed_seed_blocks_degree_phase(tmp_path,monkeypatch):
    worker,manifest=transition_fixture(tmp_path,monkeypatch,True)
    monkeypatch.setattr(degree,'prepare_degree',lambda m:pytest.fail('Degree generation must not run'))
    worker.transition(manifest)
    assert shared.read(tmp_path/'phase2_gate.json')['status']=='NOT_ADMITTED_EFFECT_RULE'
    states=[shared.read(p) for p in (tmp_path/'status').glob('*.json')]
    assert len(states)==25 and all(s['status']=='NOT_ADMITTED' for s in states)


def test_mixing_failure_never_admits_recovery(tmp_path,monkeypatch):
    worker,manifest=transition_fixture(tmp_path,monkeypatch,False)
    monkeypatch.setattr(degree,'prepare_degree',lambda m:False)
    worker.transition(manifest)
    assert shared.read(tmp_path/'phase2_gate.json')['status']=='NOT_ADMITTED_MIXING'
    assert all(shared.read(p)['status']=='NOT_ADMITTED' for p in (tmp_path/'status').glob('*.json'))


def test_detached_launch_creates_fresh_log_directory(tmp_path,monkeypatch,capsys):
    import cluster
    import subprocess
    from types import SimpleNamespace
    calls=[]
    def fake_launch(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(pid=12345)
    monkeypatch.setattr(subprocess,'Popen',fake_launch)
    exec(cluster.detach_program(tmp_path),{})
    data=json.loads(capsys.readouterr().out)
    assert data['pid']==12345
    assert (tmp_path/'results/imp_mask_information_v2/workers').is_dir()
    assert len(calls)==1 and calls[0][1]['start_new_session'] is True
    assert calls[0][1]['env']['TORCH_FORCE_WEIGHTS_ONLY_LOAD']=='1'
