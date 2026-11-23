import torch
from scripts.imp_expert_contribution_ablation_v1 import run as a
from scripts.imp_expert_contribution_ablation_v1 import replay

def config():
    return dict(vocab_size=19,max_seq_len=8,num_layers=4,num_heads=2,d_model=8,num_experts=3,expert_hidden_size=12,dropout=0.,top_k=1,capacity_factor=.5)

def test_complete_contribution_residual_bias_and_replay_current_inputs():
    torch.manual_seed(72)
    model=a.c.TinyMoELanguageModel(config()).eval()
    x=torch.randint(0,19,(2,8))
    with torch.no_grad():
        for block in model.blocks:
            for expert in block.moe.experts:
                # Strong nonzero biases make an output-only ablation distinguishable
                # from masking only expert matrix products.
                for name,p in expert.named_parameters():
                    if name.endswith('bias'): p.fill_(.4)
    replay.install(model)
    with torch.inference_mode(), a.c._precision_guard():
        untouched=model(x)
        baseline,_=a.run_forward(model,x,'BASELINE',a.MODES[0])
        assert torch.equal(untouched.logits,baseline.logits)
        payload=replay.capture(model,baseline,x,1000,'batch')
        rep,_=a.run_forward(model,x,'BASELINE',a.MODES[1],payload,dict(step=1000,batch_hash='batch'))
        assert torch.equal(rep.logits,baseline.logits)
        for q in a.QS[1:]:
            native,ns=a.run_forward(model,x,q,a.MODES[0])
            imposed,_=a.run_forward(model,x,q,a.MODES[1],payload,dict(step=1000,batch_hash='batch'))
            selected=range(4) if q=='ZERO_ALL' else [int(q[-1])]
            assert all(ns[i]['post_sumsq']==0 and ns[i]['pre_sumsq']>0 for i in selected)
            if q=='ZERO_ALL': assert torch.equal(native.logits,imposed.logits)
            else:
                for i in range(int(q[-1])+1): assert torch.equal(native.pre_router_hidden_states[i],baseline.pre_router_hidden_states[i])
        # Manual attention/norm/residual-only forward is exactly ZERO_ALL.
        positions=torch.arange(x.shape[1]); hidden=model.token_embedding(x)+model.position_embedding(positions)[None]
        mask=torch.full((8,8),float('-inf')).triu(1)
        for b in model.blocks:
            norm=b.attention_norm(hidden); attended,_=b.attention(norm,norm,norm,attn_mask=mask)
            hidden=hidden+attended
        expected=model.lm_head(model.final_norm(hidden))
        assert torch.equal(native.logits,expected)
        # Same imposed slots, deliberately changed current inputs: outputs must
        # equal explicit frozen-expert evaluation on those changed inputs.
        moe=model.blocks[0].moe; changed=baseline.pre_router_hidden_states[0]+.7
        slot=payload['layers'][0];moe._replay_required=True;moe._slot=slot
        actual,_,trace=moe(changed)
        moe._replay_required=False
        flat=changed.reshape(-1,8); expected=torch.zeros_like(flat)
        for i,expert in enumerate(moe.experts):
            pos=slot['order'][i]
            expected=expected.index_add(0,pos,expert(flat[pos])*slot['gates'][pos])
        assert torch.equal(actual.reshape_as(expected),expected)
        assert torch.equal(trace.accepted_mask.cpu(),slot['accepted'])

def test_partial_batch_aggregation_state_masks_rng_and_modes():
    torch.manual_seed(23)
    model=a.c.TinyMoELanguageModel(config());state={k:v.clone() for k,v in model.state_dict().items()}
    name=next(k for k in state if '.experts.' in k and k.endswith('weight'))
    mask=torch.ones_like(state[name],dtype=torch.bool);mask.flatten()[::2]=False;state[name][~mask]=0
    batches=[(torch.randint(0,19,(2,8)),torch.randint(0,19,(2,8))),(torch.randint(0,19,(1,8)),torch.randint(0,19,(1,8)))]
    rng=a.c._rng_snapshot()
    r=a.evaluate(state,{name:mask},config(),batches,dict(seed=7,model_type='sparse',offset=0,checkpoint_id='test'),torch.device('cpu'))
    assert a.c._rng_equal(rng,a.c._rng_snapshot())
    assert len(r['batches'])==24
    for mode in a.MODES:
        for q in a.QS:
            rows=[b for b in r['batches'] if b['mode']==mode and b['intervention']==q]
            assert sum(b['target_tokens'] for b in rows)==24
            weighted=sum(b['nll_sum'] for b in rows)/24
            assert weighted>0
    assert r['checks']['status']=='PASS'

def test_replay_identity_rejects_wrong_batch():
    import pytest
    model=a.c.TinyMoELanguageModel(config()).eval();replay.install(model)
    x=torch.randint(0,19,(1,8));out=model(x);p=replay.capture(model,out,x,1000,'batch')
    with pytest.raises(ValueError): replay.forward(model,x,p,1500,'batch')
    with pytest.raises(ValueError): replay.forward(model,x,p,1000,'other')
