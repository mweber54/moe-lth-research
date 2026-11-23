import math
import pytest
import torch
from scripts.imp_first_moe_direction_v1 import engine as e

def tensor(rows):return torch.tensor([rows],dtype=torch.float32)
def test_identical_directions_different_norms_and_reconstruction():
    a=tensor([[3.,4.],[0.,2.]]);b=2*a;z=e.common_parts(a,b)
    assert torch.equal(z['parts']['V00'],a.reshape(-1,2)) and torch.equal(z['parts']['V11'],b.reshape(-1,2))
    assert torch.equal(z['parts']['V00'],z['parts']['V01'])
    assert torch.equal(z['parts']['V10'],z['parts']['V11'])

def test_identical_norms_different_and_opposite_directions():
    a=tensor([[3.,4.],[1.,0.]]);b=tensor([[-4.,3.],[-1.,0.]]);z=e.common_parts(a,b)
    assert torch.equal(z['parts']['V00'],z['parts']['V10']) and torch.equal(z['parts']['V01'],z['parts']['V11'])
    assert z['cosine'][1]==-1 and abs(float(z['cosine'][0]))<1e-12
    same=e.common_parts(a,a)
    assert all(torch.equal(v,same['parts']['V00']) for v in same['parts'].values())

def test_one_sided_and_both_zero_backgrounds_are_recipient_native():
    a=tensor([[3.,4.],[1.,0.],[0.,0.],[0.,0.]]);b=tensor([[0.,5.],[0.,0.],[2.,0.],[0.,0.]])
    z=e.common_parts(a,b);assert z['coverage']['eligible_positions']==1 and z['coverage']['start_only_nonzero_positions']==1 and z['coverage']['end_only_nonzero_positions']==1 and z['coverage']['both_zero_positions']==1
    for own in (a,b):
        for name in e.VARIANTS:
            v=e.hybrid(z,{'c':own},name)['c'].reshape(-1,2)
            assert torch.equal(v[~z['S']],own.reshape(-1,2)[~z['S']])
    empty=e.common_parts(torch.zeros_like(a),torch.zeros_like(a));assert not empty['S'].any()
    assert all(c['maximum_relative_norm_error'] is None for c in empty['checks'])

def test_near_zero_fp64_stability_and_unusable_cast_fails_before_forward():
    tiny=tensor([[1e-30,2e-30,3e-30,4e-30]]);other=tensor([[-4e-30,3e-30,-2e-30,1e-30]])
    z=e.common_parts(tiny,other)
    assert z['coverage']['eligible_positions']==1 and all(c['maximum_relative_norm_error']<2e-6 for c in z['checks'])
    smallest=torch.nextafter(torch.tensor(0.,dtype=torch.float32),torch.tensor(1.,dtype=torch.float32)).item()
    # No epsilon or invented direction: tiny target cast along a diffuse direction
    # becomes unrepresentable; the fixed contract rejects it before any CE.
    with pytest.raises(RuntimeError,match='Numerically unusable|target-norm contract|direction contract'):
        e.common_parts(tensor([[smallest,0.,0.,0.]]),tensor([[1.,1.,1.,1.]]))

def fixture():
    torch.manual_seed(988)
    cfg=dict(vocab_size=19,max_seq_len=8,num_layers=4,num_heads=2,d_model=8,num_experts=3,expert_hidden_size=12,dropout=0.,top_k=1,capacity_factor=.5)
    models={t:e.c.TinyMoELanguageModel(cfg) for t in e.TIMES};models['end'].load_state_dict(models['start'].state_dict())
    with torch.no_grad():models['end'].token_embedding.weight.add_(torch.randn_like(models['end'].token_embedding.weight)*.25)
    states={t:{n:v.clone() for n,v in m.state_dict().items()} for t,m in models.items()};ident=dict(seed=17,model_type='sparse',checkpoints={t:dict(state_sha256=e.c.state_dict_hash(states[t])) for t in e.TIMES})
    x=torch.randint(0,19,(2,8));y=torch.randint(0,19,(2,8));return cfg,states,ident,x,y

def test_calibration_no_loss_then_partial_batches_sequence_sums_and_state(monkeypatch):
    cfg,states,ident,x,y=fixture();batches=[(x,y),(x[:1],y[:1])];rng=e.c._rng_snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(e.c,'_ce_vector',lambda *a,**k:(_ for _ in ()).throw(AssertionError('No calibration CE')))
        cal=e.evaluate(states,{t:{} for t in e.TIMES},cfg,batches,ident,torch.device('cpu'),calibration=True)
    assert cal['losses']==[] and cal['checks']['status']=='PASS'
    result=e.evaluate(states,{t:{} for t in e.TIMES},cfg,batches,ident,torch.device('cpu'))
    assert len(result['losses'])==32 and e.c._rng_equal(rng,e.c._rng_snapshot())
    for row in result['losses']:assert abs(math.fsum(row['sequence_nll'])-row['nll_sum'])<1e-10
    for r in e.TIMES:
        for p in e.MODES:
            for v in e.VARIANTS:assert sum(z['target_tokens'] for z in result['losses'] if (z['recipient'],z['mode'],z['variant'])==(r,p,v))==24

def test_direction_norm_contrast_signs_and_both_interactions():
    from scripts.imp_first_moe_direction_v1 import report as r
    z=r.contrasts(dict(zip(e.VARIANTS,[8.,6.,5.,2.])))
    assert z==dict(D_start_norm=2.,D_end_norm=3.,N_start_dir=3.,N_end_dir=4.,B_common=6.,I=1.)
    assert z['B_common']==z['D_start_norm']+z['N_end_dir']==z['N_start_dir']+z['D_end_norm']
    assert z['I']==z['D_end_norm']-z['D_start_norm']==z['N_end_dir']-z['N_start_dir']
    negative=r.contrasts(dict(zip(e.VARIANTS,[3.,4.,6.,5.])))
    assert negative['D_start_norm']==-1 and negative['N_start_dir']==-3 and negative['B_common']==-2

def test_prepared_replay_exact_against_original_and_payload_mutation_refused():
    import copy
    from scripts.imp_first_moe_direction_v1 import fast_replay
    cfg,states,ident,x,y=fixture();a=e.c.TinyMoELanguageModel(cfg).eval();a.load_state_dict(states['start']);b=copy.deepcopy(a);e.old.replay.install(a);fast_replay.install(b);bh=e.c.common.batch_hash([(x,y)])
    with torch.inference_mode():
        baseline,cache=e.old.capture(a,x,y,dict(seed=17,model_type='sparse',step=1000,batch_hash=bh))
        assert torch.equal(b(x).logits,baseline.logits)
        changed=dict(cache,c=cache['c']*.7+.2)
        for _ in range(3):
            u,_=e.old.substitute(a,x,y,cache,changed,1.,e.old.MODES[1]);v,_=e.old.substitute(b,x,y,cache,changed,1.,e.old.MODES[1]);assert torch.equal(u.logits,v.logits)
            for ru,rv in zip(u.routes,v.routes):
                assert torch.equal(ru.selected_expert_indices,rv.selected_expert_indices)
                assert torch.equal(ru.selected_probabilities,rv.selected_probabilities)
                assert torch.equal(ru.accepted_mask,rv.accepted_mask)
        cache['routing']['layers'][1]['gates'].mul_(.5)
        with pytest.raises(ValueError,match='immutable payload changed'):e.old.substitute(b,x,y,cache,changed,1.,e.old.MODES[1])

def test_prepared_replay_rejects_invalid_cpu_acceptance_and_order():
    import copy
    from scripts.imp_first_moe_direction_v1 import fast_replay
    cfg,states,ident,x,y=fixture();m=e.c.TinyMoELanguageModel(cfg).eval();m.load_state_dict(states['start']);bh=e.c.common.batch_hash([(x,y)])
    with torch.inference_mode():
        _,cache=e.old.capture(m,x,y,dict(seed=17,model_type='sparse',step=1000,batch_hash=bh));slot=cache['routing']['layers'][1];capacity=slot['capacity']
        bad=copy.deepcopy(slot);bad['accepted'].logical_not_()
        with pytest.raises(ValueError):fast_replay.prepare(bad,x.numel(),cfg['num_experts'],capacity,torch.device('cpu'),torch.float32)
        bad=copy.deepcopy(slot);bad['token_positions'][0]=bad['token_positions'][1]
        with pytest.raises(ValueError):fast_replay.prepare(bad,x.numel(),cfg['num_experts'],capacity,torch.device('cpu'),torch.float32)
