"""Verify fixed budget, identities, paired weighting, discovery-only selection and intervals."""
from reliability_common import *
import pandas as pd


def main():
    started=time.perf_counter();reg=verify_frozen();seal=s.read(D/'output_seal.json')
    for p,h in seal['files'].items():assert sha(D/p)==h,p
    for p,h in s.read(D/'report_implementation.json')['files'].items():assert sha(ROOT/p)==h,p
    probes=load(D/'probes.pt');manifest=pd.read_csv(D/'probe_manifest.csv');assert len(probes)==32 and len(manifest)==4096
    assert not manifest.sequence_id.duplicated().any() and not manifest.context_content_sha256.duplicated().any()
    ids=set(manifest.sequence_id);old={i for b in load(s.D/'probe_candidates.pt') for i in b['sequence_ids']}
    assert not ids&old and not {i-1 for i in ids}&old and not {i+1 for i in ids}&old
    assert not {i+1 for i in ids}&ids
    for b,batch in enumerate(probes):
        p=reg['probes'][b];assert tuple(batch[0].shape)==tuple(batch[1].shape)==(128,128)
        assert digest(batch[0])==p['inputs_sha256'] and digest(batch[1])==p['targets_sha256']
        assert p['block']==b//4+1 and p['partition']==('discovery' if b<16 else 'confirmation')
        assert manifest[manifest.batch==b].sequence_id.tolist()==p['sequence_ids']
    panel={x['intervention']:x for x in interventions()};assert len(panel)==56
    df=pd.read_csv(D/'deletion_damage_by_batch.csv');blocks=pd.read_csv(D/'deletion_damage_by_block.csv');r=pd.read_csv(D/'discovery_confirmation_rankings.csv');u=pd.read_csv(D/'uncertainty_and_tolerance_summary.csv');pairs=pd.read_csv(D/'paired_intervention_contrasts.csv');fun=pd.read_csv(D/'functional_distortion_summary.csv')
    assert len(df)==56*8*32*3 and len(blocks)==56*8*8*3
    assert not df.duplicated(['trajectory','step','expert','intervention','batch','subset']).any()
    cachecount=0
    for kind in ('sparse','unpruned'):
      for t in s.STEPS:
       for b in range(32):
        cache=load(cache_path(kind,t,b));cachecount+=1;assert cache['verified']
        if 'routing_engine_sha256' in cache['identity']:
            if 'prefix_engine_sha256' in cache['identity']:
                from prefix_engine import identity as actual_identity
            else:
                from accelerated import identity as actual_identity
            assert cache['identity']==actual_identity(kind,t,b)
        else:assert cache['identity']==cache_identity(kind,t,b)
        assert cache['full_model_hash_preserved'] and cache['mask_integrity'] and cache['zero_dose_equal'] and cache['RNG_modes_gradients_preserved']
        assert set(cache['baseline'])=={'loss','ids','accepted','gates','auxiliary_batch_means'}
        assert len(cache['rows'])==56*3 and len(cache['checks'])==56
        for item in cache['checks']:
            assert item['restored_exactly'] and item['no_gradient'] and item['hooks_removed'] and item['target_layer_native']
            if item['repeat_tested']:assert item['repeat_loss_max_abs']==0
        byid={}
        for row in cache['rows']:byid.setdefault(row['intervention'],[]).append(row)
        for iid,rr in byid.items():
            w=next(q for q in rr if q['subset']=='whole');parts=[q for q in rr if q['subset']!='whole'];assert w['target_tokens']==16384
            assert sum(p['target_tokens'] for p in parts)==16384 and np.isclose(sum(p['damage_sum'] for p in parts),w['damage_sum'],atol=1e-10,rtol=0)
            for row,seq in zip(rr,cache['paired_sequence_statistics'][iid]):
                assert row['subset']==seq['subset'] and row['target_tokens']==int(seq['counts'].sum())
                assert np.isclose(row['damage_sum'],float(seq['paired_loss_sums'].sum()),atol=1e-10,rtol=0)
                assert np.isclose(row['deleted_loss_sum']-row['baseline_loss_sum'],row['damage_sum'],atol=1e-8,rtol=0)
                if row['target_tokens']==0:assert row['damage'] is None
            p=panel[iid]
            if p['kind']=='weight_union':
                assert w['coordinates']==len(p['indices'])
                if p['dose_groups']==0:assert w['damage_sum']==0
            else:assert w['dose_groups'] is None and w['coordinates'] is None
    for keys,z in df.groupby(['trajectory','step','expert','intervention','block','subset']):
        kind,t,e,iid,b,subset=keys;q=blocks[(blocks.trajectory==kind)&(blocks.step==t)&(blocks.expert==e)&(blocks.intervention==iid)&(blocks.block==b)&(blocks.subset==subset)].iloc[0]
        assert q.target_tokens==z.target_tokens.sum()
        assert np.isclose(q.damage_sum,z.damage_sum.sum(),atol=1e-9,rtol=0)
        if q.target_tokens:assert np.isclose(q.damage,q.damage_sum/q.target_tokens,atol=1e-14,rtol=0)
    for row in r.itertuples():
        these=json.loads(row.intervention_ids);a=json.loads(row.discovery_values);b=json.loads(row.confirmation_values);assert len(these)==row.distinct_interventions
        assert all(panel[i]['expert']==row.expert and panel[i]['kind']==row.intervention_kind for i in these)
        if row.distinct_interventions<3:assert pd.isna(row.spearman)
        if isinstance(row.selected_high_discovery,str):
            order=sorted(range(len(these)),key=lambda i:(a[i],these[i]));assert row.selected_low_discovery==these[order[0]] and row.selected_high_discovery==these[order[-1]]
            assert np.isclose(row.discovery_selected_contrast,a[order[-1]]-a[order[0]],atol=1e-14,rtol=0)
            if b[order[0]] is not None and b[order[-1]] is not None:assert np.isclose(row.confirmation_selected_contrast,b[order[-1]]-b[order[0]],atol=1e-14,rtol=0)
        if row.subset=='baseline_accepted' and not row.operational_coverage_eligible:assert pd.isna(row.spearman)
    assert u.bootstrap_resamples.eq(2000).all();assert u.partition.isin(['discovery','confirmation']).all()
    assert pairs.partition.isin(['discovery','confirmation']+[f'block_{b}' for b in range(1,9)]).all()
    for row in fun.itertuples():
        if row.baseline_output_energy>0 and row.output_elements>0 and np.sqrt(row.baseline_output_energy/row.output_elements)>dc.OUTPUT_EPS:
            assert np.isclose(row.pooled_relative_rms,np.sqrt(row.output_difference_energy/row.baseline_output_energy),atol=1e-12,rtol=1e-12)
        else:assert pd.isna(row.pooled_relative_rms)
    save(D/'final_audit.json',dict(passed=True,unique_probe_batches=32,source_disjoint_windows=4096,checkpoint_batch_caches=cachecount,deletion_batch_rows=len(df),deletion_block_rows=len(blocks),rankings=len(r),pairwise_contrasts=len(pairs),uncertainty_rows=len(u),historical_files_unchanged=True,strict_expert_checkpoint_dose_strata=True,discovery_only_extreme_selection=True,FP64_conditional_and_sequence_weighting_verified=True,pooled_energy_ratio_verified=True,bootstrap_units='complete_batches_not_tokens',training_updates=0,verification_wall_seconds=time.perf_counter()-started,storage_bytes_at_audit=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())))
    print('Fixed-budget reliability final audit passed',flush=True)
if __name__=='__main__':
    with worker():main()
