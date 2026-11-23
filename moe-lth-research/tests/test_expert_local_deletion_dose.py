"""Scientific invariants of expert-local dose and reliability diagnostics."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

DIR=Path(__file__).resolve().parents[1]/'scripts/expert_local_deletion_dose_v1'
sys.path.insert(0,str(DIR))
import dose_common as c
spec=importlib.util.spec_from_file_location('local_dose_report',DIR/'report.py');report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def groups():
    return {f'e{e}_g{i}':dict(expert=e,indices=torch.arange(i*3,(i+1)*3),parameter=c.s.name(e)) for e in c.s.EXPERTS for i in range(8)}


def test_panel_prefixes_are_nested_deduplicated_and_expert_local():
    panel,aliases,orders=c.build_panel(groups());seen=set()
    for item in panel:
        if item['kind']!='weight_union':continue
        identity=(item['expert'],tuple(item['indices'].tolist()));assert identity not in seen;seen.add(identity)
        assert len(item['indices'])==item['dose_groups']*3
        assert all(groups()[k]['expert']==item['expert'] for k in item['groups'])
    for e in c.s.EXPERTS:
        for seed,order in orders.items():
            previous=set()
            for dose in c.DOSES:
                entry=next(a for a in aliases if a['expert']==e and a['ordering_seed']==int(seed) and a['dose_groups']==dose)
                item=next(x for x in panel if x['intervention']==entry['intervention']);current=set(item['indices'].tolist());assert previous<=current;previous=current
        assert sum(x['expert']==e and x['dose_groups']==8 for x in panel)==1
        assert sum(x['expert']==e and x['dose_groups']==0 for x in panel)==1


def test_panel_does_not_depend_on_weights_losses_or_checkpoint():
    a=c.build_panel(groups());b=c.build_panel(groups())
    assert a[1:]==b[1:]
    assert all(torch.equal(x['indices'],y['indices']) for x,y in zip(a[0],b[0]))


def test_output_rms_uses_gates_and_explicit_empty_nearzero_rules():
    base=[dict(output=torch.tensor([[3.,4.]]),gates=torch.tensor([.25]))]
    m=c.output_metrics(base,[torch.zeros(1,2)])
    assert m['output_rms_difference']==pytest.approx(np.sqrt(12.5))
    assert m['relative_output_rms_difference']==1
    assert m['gate_weighted_rms_difference']==pytest.approx(np.sqrt(12.5)*.25)
    near=c.output_metrics([dict(output=torch.zeros(1,2),gates=torch.ones(1))],[torch.ones(1,2)])
    assert near['relative_output_rms_difference'] is None and near['relative_rms_status']=='undefined_near_zero_baseline'
    empty=c.output_metrics([dict(output=torch.empty(0,2),gates=torch.empty(0))],[torch.empty(0,2)])
    assert empty['output_rms_difference'] is None and empty['accepted_positions']==0


def fixture_model():
    cfg=dict(vocab_size=8,max_seq_len=4,d_model=8,num_heads=2,num_layers=2,num_experts=2,expert_hidden_size=12,top_k=1,capacity_factor=1.25,dropout=0.)
    with c.s.protected():torch.manual_seed(112);model=c.s.TinyMoELanguageModel(cfg).eval()
    return model


def test_output_zero_control_keeps_routing_gates_and_restores_hooks():
    m=fixture_model();x=torch.arange(8).reshape(2,4);before=c.s.state_dict_hash(m.state_dict())
    with torch.inference_mode():
        baseline=m(x);e=int(baseline.routes[1].selected_expert_indices.flatten()[0]);records={}
        with c.output_hooks(m,[e],records,zero_expert=e):control=m(x)
        assert torch.count_nonzero(records[e][1])==0
        assert torch.equal(baseline.routes[1].selected_expert_indices,control.routes[1].selected_expert_indices)
        assert torch.equal(baseline.routes[1].selected_probabilities,control.routes[1].selected_probabilities)
        assert torch.equal(baseline.routes[1].accepted_mask,control.routes[1].accepted_mask)
        assert not torch.equal(baseline.logits,control.logits)
        assert torch.equal(m(x).logits,baseline.logits)
    assert not m.blocks[1].moe.experts[e]._forward_hooks
    assert c.s.state_dict_hash(m.state_dict())==before


def test_exception_removes_output_zero_hook_and_restores_weights():
    m=fixture_model();e=0;n=c.s.name(e);p=dict(m.named_parameters())[n];before=p.clone()
    with pytest.raises(RuntimeError):
        with c.s.deletion(m,{n:torch.tensor([0,1])}),c.output_hooks(m,[e],{},zero_expert=e):raise RuntimeError('failure')
    assert torch.equal(p,before) and not m.blocks[1].moe.experts[e]._forward_hooks


def test_sequence_damage_conditional_weighting_and_empty_coverage():
    delta=torch.arange(256,dtype=torch.float64);mask=torch.zeros(256,dtype=torch.bool);mask[:2]=True
    rows=c.sequence_rows(delta,mask,[5,9]);assert rows[0]['accepted_tokens']==2 and rows[1]['accepted_damage'] is None
    assert np.mean([r['whole_damage'] for r in rows])==float(delta.mean())
    assert rows[0]['accepted_damage']*2+rows[0]['complement_damage']*126==pytest.approx(rows[0]['whole_damage']*128)


def test_agreement_reports_reversed_equal_dose_ranks_and_undefined_constants():
    r=report.agreement([1,2,3,4],[4,3,2,1]);assert r['spearman']==-1 and r['pairwise_concordance']==0
    z=report.agreement([0,0,0],[0,0,0]);assert z['spearman'] is None and z['pairwise_concordance'] is None and z['both_tied_pairs']==3
    assert report.agreement([1],[2])['spearman'] is None


def test_reliability_never_pools_doses_or_ordering_aliases():
    import pandas as pd
    rows=[]
    for dose in (1,2):
        for i in range(3):
            for probe in ('A','B'):
                rows.append(dict(trajectory='sparse',step=250,intervention_kind='weight_union',dose_groups=dose,subset='whole',intervention=f'd{dose}i{i}',probe=probe,expert=0,accepted_positions=100,damage=(dose*100+i if probe=='A' else dose*100-i),output_rms_difference=i+1,relative_output_rms_difference=i+1,gate_weighted_rms_difference=i+1))
    result=report.reliability(pd.DataFrame(rows),1e-12)
    main=[r for r in result if r['metric']=='damage' and r['coverage']=='all_retained' and r['scope']=='across_experts_equal_dose']
    assert len(main)==2 and all(r['spearman']==-1 for r in main) and all(r['pairs']==3 for r in main)
