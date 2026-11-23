import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/token_expert_matching_v1'))
from replay import capture, install, forward, permute
from moe_lth.models import TinyMoELanguageModel


def fixture():
    torch.manual_seed(17)
    config=dict(vocab_size=16,d_model=8,max_seq_len=8,num_heads=2,num_layers=2,num_experts=4,expert_hidden_size=16,dropout=0.,capacity_factor=.5)
    model=TinyMoELanguageModel(config)
    tokens=torch.randint(0,16,(2,8))
    output=model(tokens)
    payload=capture(model,output,tokens,1,'batch')
    install(model)
    return model,tokens,output,payload


def test_identity_forward_rng_auxiliary_and_gradients():
    model,tokens,native,payload=fixture()
    for name,p in model.named_parameters():
        p.requires_grad_('.router.' not in name)
    rng=torch.get_rng_state().clone()
    replay=forward(model,tokens,payload,1,'batch')
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(native.logits,replay.logits)
    assert torch.equal(native.auxiliary_loss,replay.auxiliary_loss)
    replay.logits.square().mean().backward(retain_graph=True)
    assert all(p.grad is None for n,p in model.named_parameters() if '.router.' in n)
    assert any(p.grad is not None and p.grad.norm()>0 for n,p in model.named_parameters() if '.experts.' in n)
    assert model.token_embedding.weight.grad.norm()>0
    model.zero_grad()
    replay.auxiliary_loss.backward()
    assert model.token_embedding.weight.grad.norm()>0


def test_shuffle_conservation_and_shared_schedule():
    model,tokens,native,payload=fixture()
    rng=torch.get_rng_state().clone()
    for layer in payload['layers']:
        shuffled=permute(layer,101,17,1)
        assert not torch.equal(shuffled['token_positions'],layer['token_positions'])
        assert shuffled['permutation_hash']==permute(layer,101,17,1)['permutation_hash']
        assert shuffled['accepted'].sum()==layer['accepted'].sum()
    assert torch.equal(rng,torch.get_rng_state())
    output=forward(model,tokens,payload,1,'batch',101,17)
    assert not torch.equal(output.logits,native.logits)


def test_fail_closed():
    model,tokens,native,payload=fixture()
    with pytest.raises(ValueError):forward(model,tokens,payload,2,'batch')
    model.blocks[0].moe._replay_required=True
    with pytest.raises(RuntimeError):model(tokens)


def test_native_auxiliary_uses_recipient_decisions():
    model,tokens,native,payload=fixture()
    moe=model.blocks[0].moe
    hidden=native.pre_router_hidden_states[0]
    with torch.no_grad():moe.router.projection.weight.mul_(-3)
    reference=moe(hidden)[1]
    moe._replay_required=True
    moe._slot=permute(payload['layers'][0],101,17,1)
    replayed=moe(hidden)[1]
    assert torch.equal(reference,replayed)
    with pytest.raises(RuntimeError):moe(hidden)


def test_corrupt_acceptance_and_token_positions_rejected():
    import copy
    model,tokens,native,payload=fixture()
    altered=copy.deepcopy(payload)
    altered['layers'][0]['token_positions'][0]=altered['layers'][0]['token_positions'][1]
    with pytest.raises(ValueError,match='bijective'):forward(model,tokens,altered,1,'batch')
    altered=copy.deepcopy(payload)
    altered['layers'][0]['accepted'].logical_not_()
    with pytest.raises(ValueError):forward(model,tokens,altered,1,'batch')


def test_named_displacement_reports_frozen_and_missing():
    import run
    model,_,_,_=fixture()
    for n,p in model.named_parameters():p.requires_grad_('.router.' not in n)
    state={n:t.clone() for n,t in model.state_dict().items()}
    result=run.named_displacement(model,state,state)
    assert result['frozen'] and all('.router.' in n for n in result['frozen'])
    assert all(x['cumulative_from_initial_l2']==0 for x in result['components'])
    state.pop('token_embedding.weight')
    with pytest.raises(RuntimeError,match='Missing named'):run.named_displacement(model,state,state)


def test_analysis_averages_draws_within_seed_and_requires_complete_draws(tmp_path,monkeypatch):
    import report
    import run
    monkeypatch.setattr(report,'DEST',tmp_path)
    monkeypatch.setattr(report,'SEEDS',(7,17,29,43,61))
    jobs=[]
    for seed in (7,17,29,43,61):
        for sp,tag in ((0.,'dense'),(.8,'imp80'),(.85,'imp85')):
            variant=f'e32_seed{seed}_{tag}'
            for kind,draw in [('identity',None)]+[('shuffle',d) for d in (101,202,303)]:
                jid=variant+'_'+kind+(f'_{draw}' if draw else '')
                jobs.append(dict(id=jid,seed=seed,sparsity=sp,kind=kind,draw=draw))
                path=tmp_path/'attempts'/jid;path.mkdir(parents=True)
                ce=1.+(.1 if sp else 0.)+((.5 if sp else .3)+(draw-202)/1000 if draw else 0.)
                result=dict(variant=variant,seed=seed,sparsity=sp,kind=kind,draw=draw,final_ce=ce,applied_updates=5000,amp_skips=0,wall_seconds=1,checkpoint='mock')
                run.save(path/'result.json',result)
                run.save(path/'curves.json',[dict(step=0,ce=10),dict(step=2500,ce=ce),dict(step=5000,ce=ce)])
    run.save(tmp_path/'registration.json',dict(jobs=jobs))
    report.analyze()
    summaries=run.read(tmp_path/'descriptive_summaries.json')
    assert len(summaries)==4
    assert all(x['n_source_seeds']==5 and abs(x['mean']-.2)<1e-12 for x in summaries)
    (tmp_path/'attempts'/jobs[-1]['id']/'result.json').unlink()
    report.analyze()
    summaries=run.read(tmp_path/'descriptive_summaries.json')
    assert all(x['sparsity']==.8 for x in summaries)
    assert not run.read(tmp_path/'completion.json')['scientifically_complete']

    import finalize
    monkeypatch.setattr(finalize,'DEST',tmp_path)
    finalize.finalize()
    assert (tmp_path/'native_vs_replay_matching.csv').exists()
    assert (tmp_path/'figures/seed_differential_penalties.pdf').exists()
    assert not run.read(tmp_path/'reporting_complete.json')['scientifically_complete']
