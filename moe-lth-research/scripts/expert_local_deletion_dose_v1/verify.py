"""Audit new outputs and reused provenance without modifying Stage 1."""
from dose_common import *
import pandas as pd


def main():
    started=time.perf_counter();reg=verify_registration();seal=s.read(D/'output_seal.json')
    for p,h in seal['files'].items():assert sha(D/p)==h,p
    manifest=pd.read_csv(D/'intervention_manifest.csv');aliases=pd.read_csv(D/'ordering_aliases.csv');panel=load(D/'intervention_coordinates.pt');groups=load(s.D/'group_coordinates.pt')
    assert len(panel)==len(manifest)==reg['unique_interventions']
    assert len(aliases)==4*4*5
    for (e,ordering),z in aliases.groupby(['expert','ordering_seed']):
        prev=set()
        for row in z.sort_values('dose_groups').itertuples():
            current=set(row.ordered_groups.split(';')) if isinstance(row.ordered_groups,str) else set()
            assert len(current)==row.dose_groups and prev<=current;prev=current
    seen=set();byid={x['intervention']:x for x in panel}
    for item in panel:
        e=item['expert'];idx=item['indices'];identity=(e,item['kind'],tuple(idx.tolist()))
        assert identity not in seen;seen.add(identity)
        assert all(groups[k]['expert']==e for k in item['groups'])
        if item['kind']=='weight_union':
            union=torch.cat([groups[k]['indices'] for k in item['groups']]).sort().values if item['groups'] else torch.empty(0,dtype=torch.long)
            assert torch.equal(idx,union) and len(idx)==len(torch.unique(idx))
            r=manifest.set_index('intervention').loc[item['intervention']]
            assert r.coordinates==len(idx) and r.coordinate_sha256==digest(idx)
            assert np.isclose(r.sparse_active_matrix_fraction,len(idx)/r.sparse_active_matrix_coordinates)
            assert np.isclose(r.unpruned_active_matrix_fraction,len(idx)/r.unpruned_active_matrix_coordinates)
    df=pd.read_csv(D/'measurements.csv');sequences=pd.read_csv(D/'paired_sequence_damage.csv');rel=pd.read_csv(D/'ab_reliability.csv');checks=s.read(D/'numerical_checks.json')
    assert len(df)==len(panel)*16*3 and len(sequences)==len(panel)*16*256
    assert not df.duplicated(['trajectory','step','probe','intervention','subset']).any()
    assert checks['passed'] and checks['training_updates']==0
    assert len(checks['checks'])==len(panel)*16
    for r in checks['checks']:
        assert all(r[k] for k in ['restored_exactly','gradient_none','hooks_removed','targeted_layer_routing_preserved'])
        if r['repeat_tested']:assert r['repeat_loss_max_abs']==0
    for key,z in df.groupby(['trajectory','step','probe','intervention']):
        whole=z[z.subset=='whole'].iloc[0];parts=z[z.subset!='whole'];valid=parts[parts.target_tokens>0]
        assert parts.target_tokens.sum()==whole.target_tokens
        assert np.isclose((valid.damage*valid.target_tokens).sum()/whole.target_tokens,whole.damage,atol=1e-13,rtol=0)
        if whole.intervention_kind=='weight_union':
            assert np.isclose(whole.damage-whole.sum_individual_damage,whole.nonadditivity,atol=1e-13,rtol=0)
            assert whole.coordinates==len(byid[whole.intervention]['indices'])
            if whole.dose_groups==0:assert whole.damage==0 and (whole.output_rms_difference==0 or pd.isna(whole.output_rms_difference))
        else:assert z.sum_individual_damage.isna().all() and z.nonadditivity.isna().all() and z.dose_groups.isna().all()
        if whole.accepted_positions==0:assert pd.isna(whole.output_rms_difference) and pd.isna(z[z.subset=='baseline_accepted'].damage.iloc[0])
    keys=['trajectory','step','probe','intervention']
    seqsummary=sequences.groupby(keys).agg(target_tokens=('target_tokens','sum'),accepted_tokens=('accepted_tokens','sum'),whole_damage=('whole_damage','mean'))
    joined=df[df.subset=='whole'].set_index(keys).join(seqsummary,rsuffix='_seq')
    assert (joined.target_tokens==joined.target_tokens_seq).all() and (joined.accepted_positions==joined.accepted_tokens).all()
    assert np.allclose(joined.damage,joined.whole_damage,atol=1e-13,rtol=0)
    assert rel[rel.intervention_kind=='weight_union'].dose_groups.isin(DOSES).all()
    assert rel[rel.intervention_kind=='expert_output_zero'].dose_groups.isna().all()
    assert rel.scope.isin(['across_experts_equal_dose','within_expert_equal_dose']).all()
    for path in (D/'cache').glob('*.pt'):
        cache=load(path);assert cache['verified'];kind=cache['identity']['kind'];t=cache['identity']['step'];p=cache['identity']['probe'];old=load(s.cache_path(kind,t,p));check_baseline(old['baseline'],cache['baseline'])
        assert cache['identity']['stage1_cache_sha256']==sha(s.cache_path(kind,t,p))
        for iid,item in cache['interventions'].items():
            e=byid[iid]['expert'];baseline=cache['baseline'];blocks=baseline['outputs'][e]
            assert sum(len(b['output']) for b in blocks)==int(((baseline['ids']==e)&baseline['accepted']).sum())
            if byid[iid]['kind']=='weight_union':
                component=sum((old['paired_token_damage'][k].double() for k in byid[iid]['groups']),torch.zeros_like(baseline['loss'],dtype=torch.float64))
                measured=next(r for r in item['rows'] if r['subset']=='whole')
                assert np.isclose(float(component.mean()),measured['sum_individual_damage'],atol=1e-14,rtol=0)
    save(D/'final_audit.json',dict(passed=True,measurement_rows=len(df),sequence_rows=len(sequences),unique_interventions=len(panel),ordering_aliases=len(aliases),output_files_verified=len(seal['files']),source_files_unchanged=True,nested_unions_deduplicated=True,coordinate_matched=True,no_global_union_cap=True,no_cross_dose_rank_pooling=True,paired_sequence_and_conditional_weighting_verified=True,representative_repeats_bitwise_equal=True,verification_wall_seconds=time.perf_counter()-started,storage_bytes_at_audit=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())))
    print('Expert-local dose final audit passed',flush=True)
if __name__=='__main__':
    with worker():main()
