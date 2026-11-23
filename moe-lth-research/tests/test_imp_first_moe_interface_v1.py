import copy
import pytest
import torch
from scripts.imp_first_moe_interface_v1 import engine as e

def fixture():
    torch.manual_seed(328)
    config=dict(vocab_size=19,max_seq_len=8,num_layers=4,num_heads=2,d_model=8,num_experts=3,expert_hidden_size=12,dropout=0.,top_k=1,capacity_factor=.5)
    models={t:e.c.TinyMoELanguageModel(config).eval() for t in e.TIMES}
    # Identical expert/router weights, genuinely different surrounding context.
    state=models['start'].state_dict();models['end'].load_state_dict(state)
    with torch.no_grad():
        models['end'].token_embedding.weight.add_(torch.randn_like(models['end'].token_embedding.weight)*.3)
        models['end'].blocks[0].moe_norm.bias.add_(.4)
        for m in models.values():
            for expert in m.blocks[0].moe.experts:expert.fc2.bias.add_(1.)
    for m in models.values():e.replay.install(m)
    x=torch.randint(0,19,(2,8));y=torch.randint(0,19,(2,8));bh=e.c.common.batch_hash([(x,y)])
    return config,models,x,y,bh

def get(m,x,y,bh,t):return e.capture(m,x,y,dict(seed=7,model_type='sparse',step=1000 if t=='start' else 1500,batch_hash=bh,checkpoint=t))

def test_distinct_donor_residual_linear_no_double_gate_and_replay():
    _,models,x,y,bh=fixture()
    with torch.inference_mode(),e.c._precision_guard():
        outputs={};cache={}
        for t in e.TIMES:outputs[t],cache[t]=get(models[t],x,y,bh,t)
        assert not torch.equal(cache['start']['u'],cache['end']['u'])
        assert not torch.equal(cache['start']['c'],cache['end']['c'])
        for r in e.TIMES:
            for mode in e.MODES:
                own,_=e.substitute(models[r],x,y,cache[r],cache[r],1.,mode)
                assert torch.equal(own.logits,outputs[r].logits)
                zeros=[]
                for d in e.TIMES:
                    for alpha in e.ALPHAS:
                        out,obs=e.substitute(models[r],x,y,cache[r],cache[d],alpha,mode)
                        assert torch.equal(obs['u'],cache[r]['u'])
                        assert torch.equal(obs['inserted'],alpha*cache[d]['c'])
                        assert torch.equal(obs['h'],cache[r]['u']+alpha*cache[d]['c'])
                        assert torch.equal(out.pre_router_hidden_states[0],cache[r]['z'])
                        if alpha==0:zeros.append(out.logits)
                assert torch.equal(*zeros)
        # Identity inactive hook preserves untouched model (all replay flags off).
        m=models['start'];untouched=m(x)
        h=m.blocks[0].moe.register_forward_hook(lambda mod,args,res:res)
        try:assert torch.equal(m(x).logits,untouched.logits)
        finally:h.remove()
        # Wrong source, token order or targets must fail closed.
        bad=dict(cache['end']);bad['identity']=dict(bad['identity'],seed=17)
        with pytest.raises(RuntimeError):e.substitute(m,x,y,cache['start'],bad,.9,e.MODES[0])
        with pytest.raises(RuntimeError):e.substitute(m,x.flip(1),y,cache['start'],cache['end'],.9,e.MODES[0])
        with pytest.raises(RuntimeError):e.substitute(m,x,y.flip(1),cache['start'],cache['end'],.9,e.MODES[0])

def test_alpha_zero_matches_previous_zero_l0_and_current_input_replay():
    from scripts.imp_expert_contribution_ablation_v1 import run as old
    _,models,x,y,bh=fixture();m=models['start']
    with torch.inference_mode(),e.c._precision_guard():
        native,c=get(m,x,y,bh,'start')
        for mode,previous in zip(e.MODES,old.MODES):
            out,_=e.substitute(m,x,y,c,c,0.,mode)
            expected,_=old.run_forward(m,x,'ZERO_L0',previous,c['routing'],dict(step=1000,batch_hash=bh))
            assert torch.equal(out.logits,expected.logits)
        # Deliberately alter downstream hidden inputs while preserving baseline slots.
        moe=m.blocks[1].moe;slot=c['routing']['layers'][1]
        changed=native.pre_router_hidden_states[1]+.8
        moe._replay_required=True;moe._slot=slot
        actual,_,_=moe(changed);moe._replay_required=False
        flat=changed.reshape(-1,8);expected=torch.zeros_like(flat)
        for i,expert in enumerate(moe.experts):
            pos=slot['order'][i];expected=expected.index_add(0,pos,expert(flat[pos])*slot['gates'][pos])
        assert torch.equal(actual.reshape_as(expected),expected)

def test_chunking_partial_batch_state_and_token_weighting():
    config,models,x,y,bh=fixture()
    with torch.inference_mode(),e.c._precision_guard():
        n,c=get(models['start'],x,y,bh,'start')
        chunked,d=e.capture(models['start'],x,y,c['identity'],chunk=1)
        assert torch.allclose(c['ungated'],d['ungated'],atol=1e-6,rtol=1e-5)
        a,_=e.substitute(models['start'],x,y,c,c,.75,e.MODES[0])
        b,_=e.substitute(models['start'],x,y,c,c,.75,e.MODES[0],expert_chunk_size=1)
        assert torch.allclose(a.logits,b.logits,atol=1e-6,rtol=1e-5)
    states={t:{n:p.clone() for n,p in m.state_dict().items()} for t,m in models.items()}
    batches=[(x,y),(x[:1],y[:1])];rng=e.c._rng_snapshot()
    result=e.evaluate_pair(states,{t:{} for t in e.TIMES},config,batches,dict(seed=7,model_type='sparse',checkpoints={t:t for t in e.TIMES}),torch.device('cpu'))
    assert e.c._rng_equal(rng,e.c._rng_snapshot())
    assert result['checks']['status']=='PASS' and len(result['batches'])==80
    for mode in e.MODES:
        for r in e.TIMES:
            for d in e.TIMES:
                for alpha in e.ALPHAS:
                    rows=[z for z in result['batches'] if (z['mode'],z['recipient'],z['donor'],z['alpha'])==(mode,r,d,alpha)]
                    assert sum(z['target_tokens'] for z in rows)==24
                    assert sum(z['nll_sum'] for z in rows)/24>0

def test_zero_cosine_exclusions_and_changed_expert_populations():
    a=torch.tensor([[0.,0.],[1.,0.],[0.,1.]])
    b=torch.tensor([[0.,0.],[0.,0.],[1.,0.]])
    row=e.compare(a,b)
    assert row['both_zero_positions']==1 and row['end_only_zero_positions']==1
    assert row['cosine_valid_positions']==1 and row['mean_position_cosine']==0
    empty=e.compare(a,b,torch.zeros(3,dtype=torch.bool));assert empty['relative_l2'] is None and empty['positions']==0

def test_prespecified_estimand_signs_for_known_four_cell_matrix():
    from scripts.imp_first_moe_interface_v1 import analysis as a
    records=[]
    matrix={('start','start'):4.,('start','end'):5.,('end','start'):2.,('end','end'):1.}
    for seed in e.c.SEEDS:
        for m in e.c.MODEL_TYPES:
            for p in e.MODES:
                for r in e.TIMES:
                    for d in e.TIMES:
                        for alpha in e.ALPHAS:
                            extra=(1-alpha)*(4 if r=='end' else 1)
                            records.append(dict(seed=seed,model_type=m,mode=p,recipient=r,donor=d,alpha=alpha,task_ce=matrix[r,d]+extra))
    sub,atten,sensitivity,sparse,routing=a.estimands(records)
    assert all(r['V_start']==-1 and r['V_end']==1 and r['Q']==2 and r['S_start']==1 and r['S_end']==1 for r in sub)
    assert all(abs(r['T']-.3)<1e-12 for r in sensitivity if r['alpha']==.9)
    assert all(r['sparse_minus_unpruned']==0 for r in sparse)
    assert all(r['native_minus_replay']==0 for r in routing)

def test_native_response_can_change_downstream_routing():
    _,models,x,y,bh=fixture();m=models['start']
    with torch.inference_mode(),e.c._precision_guard():
        native,cache=get(m,x,y,bh,'start')
        donor=dict(cache);donor['c']=torch.randn_like(cache['c'])*25
        out,obs=e.substitute(m,x,y,cache,donor,1.,e.MODES[0])
        assert any(not torch.equal(a.selected_expert_indices,b.selected_expert_indices) for a,b in zip(native.routes[1:],out.routes[1:]))
        # A separate minimal first-output hook must reproduce the intervention.
        h=m.blocks[0].moe.register_forward_hook(lambda mod,args,result:(donor['c'],result[1],result[2]))
        try:assert torch.equal(m(x).logits,out.logits)
        finally:h.remove()

def test_checkpoint_configuration_schemas_are_explicit_and_fail_closed():
    from scripts.imp_first_moe_interface_v1.run import checkpoint_model_configuration
    cfg={'d_model':8,'num_experts':3}
    assert checkpoint_model_configuration({'identity':{'config':{'model':cfg}}},0)==cfg
    assert checkpoint_model_configuration({'config':{'model':cfg}},500)==cfg
    with pytest.raises(KeyError):checkpoint_model_configuration({'config':{'model':cfg}},0)
    with pytest.raises(KeyError):checkpoint_model_configuration({'identity':{'config':{'model':cfg}}},500)
