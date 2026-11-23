"""Fixed-budget sampling, paired bootstrap, weighting and no-pooling safeguards."""
import sys
from pathlib import Path
import numpy as np
import torch
import pytest
P=Path(__file__).resolve().parents[1]/'scripts/expert_deletion_reliability_v1'
sys.path.insert(0,str(P))
import reliability_common as c
import analyze as a


class Windows:
    def __len__(self):return 200
    def __getitem__(self,i):
        z=torch.arange(i*128,i*128+129,dtype=torch.int64)
        return z[:-1],z[1:]


def test_probe_sampler_excludes_source_overlap_and_content_without_model_inputs():
    ds=Windows();old={5,13};hashes={c.content_hash(ds,i) for i in old}
    x,why=c.choose_sequences(ds,old,hashes,seed=17,needed=32)
    y,_=c.choose_sequences(ds,old,hashes,seed=17,needed=32)
    assert x==y and len(x)==32
    ids={i for i,h in x}
    assert len(ids)==32 and not ids&old
    assert not any(i-1 in ids or i+1 in ids or i-1 in old or i+1 in old for i in ids)
    assert len({h for i,h in x})==32


def test_unavailable_probe_never_relaxes_budget_or_overlap_rules():
    x,_=c.choose_sequences(Windows(),set(range(200)),set(),needed=4096)
    assert x==[]


def test_conditional_bootstrap_resamples_sums_and_counts_not_means():
    nums=np.array([[0.,0.],[100.,200.],[3.,6.],[4.,8.]])
    den=np.array([1.,100.,1.,1.])
    p=a.summarize_phase(nums,den,np.arange(4),10)
    assert p['mean'][0]==pytest.approx(107/103)
    assert p['mean'][0]!=pytest.approx(np.mean(nums[:,0]/den))
    valid=np.isfinite(p['boot'][:,0])
    assert np.allclose(p['boot'][valid,1],2*p['boot'][valid,0])
    assert a.bootstrap_counts(4,10).shape==(2000,4)


def test_zero_denominator_bootstrap_replicates_stay_undefined():
    p=a.summarize_phase(np.zeros((4,2)),np.zeros(4),np.arange(4),12)
    assert np.isnan(p['mean']).all() and np.isnan(p['boot']).all()
    assert a.interval(p['boot'][:,0])==(None,None,1.)


def test_energies_pool_before_ratio_not_mean_of_ratios():
    rows=[dict(output_elements=1,output_difference_energy=1.,baseline_output_energy=1.,gate_weighted_difference_energy=1.,baseline_gate_weighted_energy=1.,relative_output_rms_difference=1.),dict(output_elements=1,output_difference_energy=1.,baseline_output_energy=100.,gate_weighted_difference_energy=1.,baseline_gate_weighted_energy=100.,relative_output_rms_difference=.1)]
    z=c.aggregate_energies(rows)
    assert z['pooled_relative_output_rms_difference']==pytest.approx(np.sqrt(2/101))
    assert z['mean_defined_batch_relative_rms']==pytest.approx(.55)


def test_paired_token_losses_reconcile_with_conditional_and_sequence_sums():
    base=torch.arange(256,dtype=torch.float32);deleted=base+torch.arange(256,dtype=torch.float32)/1024;delta=deleted.double()-base.double();mask=torch.arange(256)%3==0
    rows,seq=c.subset_statistics(delta,base,deleted,mask)
    assert rows[0]['damage_sum']==rows[1]['damage_sum']+rows[2]['damage_sum']
    for r,q in zip(rows,seq):
        assert r['damage_sum']==float(q['paired_loss_sums'].sum())
        assert r['target_tokens']==int(q['counts'].sum())


def test_small_and_constant_rank_panels_are_undefined():
    assert a.spearman([1,2],[2,1]) is None
    assert a.spearman([0,0,0],[0,0,0]) is None
    assert a.spearman([1,2,3],[3,2,1])==-1
    assert a.ties([0,0,1])==1


def test_bootstrap_rank_preserves_within_panel_pairing():
    x=np.tile([1.,2.,3.],(2000,1));y=x[:,::-1]
    z=a.bootstrap_rank(x,y)
    assert np.allclose(z,-1)
    assert a.tolerance_state(-1,1,.001)=='unresolved_crossing_or_boundary'
    assert a.tolerance_state(-1,0,.001)=='compatible_below_convention_on_this_probe'


def test_full_analysis_keeps_discovery_choices_when_confirmation_reverses(tmp_path,monkeypatch):
    import json
    import pandas as pd
    panel=[x for x in c.interventions() if x['expert']==0]
    ids=sorted(x['intervention'] for x in panel);index={k:i for i,k in enumerate(ids)}
    rows=[]
    for batch in range(32):
      for item in panel:
        i=index[item['intervention']];effect=(i if batch<16 else len(ids)-i)*1e-5
        for subset,count in [('whole',16384),('baseline_accepted',100+batch%7),('complement',16384-100-batch%7)]:
            rows.append(dict(trajectory='sparse',step=250,expert=0,subset=subset,intervention=item['intervention'],batch=batch,damage_sum=effect*count,target_tokens=count,output_difference_energy=float((i+1)**2),baseline_output_energy=1000.,output_elements=25600,gate_weighted_difference_energy=float(i+1),baseline_gate_weighted_energy=10.,relative_output_rms_difference=np.sqrt((i+1)**2/1000)))
    pd.DataFrame(rows).to_csv(tmp_path/'deletion_damage_by_batch.csv',index=False)
    pd.DataFrame([dict(level='block',trajectory='sparse',step=250,expert=0,block=b,accepted_tokens=400) for b in range(1,9)]).to_csv(tmp_path/'coverage_by_batch_and_block.csv',index=False)
    (tmp_path/'measurement_completion.json').write_text('{"complete":true}')
    (tmp_path/'compute_accounting.json').write_text('{}')
    monkeypatch.setattr(a,'D',tmp_path);monkeypatch.setattr(a,'verify_frozen',lambda:{})
    monkeypatch.setattr(a,'interventions',lambda:panel)
    a.main()
    r=pd.read_csv(tmp_path/'discovery_confirmation_rankings.csv')
    q=r[(r.subset=='whole')&(r.dose_groups==1)].iloc[0]
    values=json.loads(q.discovery_values);these=json.loads(q.intervention_ids)
    assert q.selected_high_discovery==these[int(np.argmax(values))]
    assert q.selected_low_discovery==these[int(np.argmin(values))]
    assert q.confirmation_selected_contrast<0 and q.confirmation_selected_ci95_high<0
    assert q.interpretation=='directional_contrast_reversed_candidate_context_dependence'
    assert (r.expert==0).all() and (r.step==250).all()


def test_integer_dispatch_acceleration_matches_native_with_capacity_and_ties():
    import accelerated
    from moe_lth.models import TinyMoELanguageModel
    for capacity in (.25,1.25):
      for tied in (False,True):
        cfg=dict(vocab_size=8,max_seq_len=4,d_model=8,num_heads=2,num_layers=2,num_experts=4,expert_hidden_size=12,top_k=1,capacity_factor=capacity,dropout=0.)
        with c.s.protected():
            torch.manual_seed(112);model=TinyMoELanguageModel(cfg).eval()
        if tied:
            with torch.no_grad():
                for b in model.blocks:b.moe.router.projection.weight.zero_()
        x=torch.arange(8).reshape(2,4)
        with torch.inference_mode():
            original=model(x)
            with accelerated.enabled():fast=model(x)
        assert torch.equal(original.logits,fast.logits)
        for u,v in zip(original.routes,fast.routes):
            assert torch.equal(u.selected_expert_indices,v.selected_expert_indices)
            assert torch.equal(u.accepted_mask,v.accepted_mask)
            assert torch.equal(u.selected_probabilities,v.selected_probabilities)


def test_invariant_prefix_preserves_ablations_controls_and_batch_changes():
    import prefix_engine
    from moe_lth.models import TinyMoELanguageModel
    cfg=dict(vocab_size=8,max_seq_len=4,d_model=8,num_heads=2,num_layers=3,num_experts=4,expert_hidden_size=12,top_k=1,capacity_factor=.5,dropout=0.)
    with c.s.protected():torch.manual_seed(112);m=TinyMoELanguageModel(cfg).eval()
    x=torch.arange(8).reshape(2,4);y=x.flip(0);n='blocks.1.moe.experts.0.fc1.weight';idx=torch.tensor([0,1,2])
    with torch.inference_mode():
        baseline=m(x)
        with prefix_engine.enabled():
            assert torch.equal(m(x).logits,baseline.logits)
            with c.s.deletion(m,{n:idx}):fast=m(x)
            with c.dc.output_hooks(m,[0],{},zero_expert=0):fastcontrol=m(x)
            changed=m(y)
        assert not hasattr(m,'_diagnostic_fixed_prefix')
        with c.s.deletion(m,{n:idx}):original=m(x)
        with c.dc.output_hooks(m,[0],{},zero_expert=0):originalcontrol=m(x)
        assert torch.equal(fast.logits,original.logits)
        assert torch.equal(fastcontrol.logits,originalcontrol.logits)
        assert torch.equal(changed.logits,m(y).logits)
