"""Reuse recorded SVD measurements, with fixed expert membership."""
from common import *
METRICS=('source_subspace_overlap','alignment_centered','alignment_above_shuffled','retained_energy','relative_cutoff_gap')
def valid_reason(row):
    reasons=[]
    for k in ('zero_matrix','rank_deficient','degenerate_cutoff'):
        if bool(row[k]):reasons.append(k)
    if not np.isfinite(row['source_subspace_overlap']):reasons.append('missing_source_overlap')
    return ';'.join(reasons) or 'valid'
def run():
    jobs=pd.read_csv(BASE/'run_manifest.csv');raw=[];audit=[];summaries=[];changes=[];pairs=[]
    lookup={}
    for _,j in jobs.iterrows():
        parts=[]
        for step in STEPS:
            p=Path(j['attempt'])/'geometry'/f'step_{step:04d}'/'weights.csv'
            if not p.exists():audit.append(dict(condition_id=j.condition_id,step=step,status='missing_svd_table'));continue
            d=pd.read_csv(p);d['step']=step;d['condition_id']=j.condition_id
            for k in ('seed','router_age','sparsity','mask_type'):d[k]=j[k]
            d['valid_reason']=d.apply(valid_reason,axis=1);d['valid']=d.valid_reason.eq('valid')
            d['row_id']=d.layer.astype(str)+':'+d.expert.astype(str)
            source=BASE/'existing'/j.source_id/'source_age_R100/weights.csv'
            ref=pd.read_csv(source)[['layer','expert','rank','zero_matrix','rank_deficient','degenerate_cutoff']]
            ref=ref.rename(columns={k:'source_'+k for k in ('zero_matrix','rank_deficient','degenerate_cutoff')})
            d=d.merge(ref,on=['layer','expert','rank'],validate='one_to_one')
            d['source_reference_valid']=~(d.source_zero_matrix|d.source_rank_deficient|d.source_degenerate_cutoff)
            parts.append(d)
        if not parts:continue
        d=pd.concat(parts,ignore_index=True);raw.append(d);lookup[j.condition_id]=d
        for rank in (8,16,32):
            r=d[d['rank'].eq(rank)]
            bystep={s:r[r.step.eq(s)].set_index('row_id') for s in STEPS}
            end=set(bystep[0].query('valid').index)&set(bystep[2500].query('valid').index)
            fixed=set.intersection(*[set(bystep[s].query('valid').index) for s in STEPS])
            refend=set(bystep[0].query('valid & source_reference_valid').index)&set(bystep[2500].query('valid & source_reference_valid').index)
            reffixed=set.intersection(*[set(bystep[s].query('valid & source_reference_valid').index) for s in STEPS])
            for membership,ids in [('endpoint_matched',end),('trajectory_fixed',fixed),('endpoint_reference_valid',refend),('trajectory_reference_valid',reffixed)]:
                for layer in (-1,0,1,2,3):
                    chosen=sorted(i for i in ids if layer==-1 or int(i.split(':')[0])==layer)
                    for step in (STEPS if membership.startswith('trajectory') else (0,2500)):
                        sel=bystep[step].loc[chosen]
                        row=dict(condition_id=j.condition_id,seed=j.seed,router_age=j.router_age,sparsity=j.sparsity,
                                 mask_type=j.mask_type,rank=rank,layer=layer,membership=membership,step=step,n_experts=len(chosen),
                                 n_excluded=(128 if layer==-1 else 32)-len(chosen))
                        row.update({k:float(sel[k].mean()) if len(sel) else None for k in METRICS});summaries.append(row)
                    a=bystep[0].loc[chosen];b=bystep[2500].loc[chosen]
                    row=dict(condition_id=j.condition_id,seed=j.seed,router_age=j.router_age,sparsity=j.sparsity,
                             mask_type=j.mask_type,rank=rank,layer=layer,membership=membership,n_experts=len(chosen))
                    for k in METRICS:row[k+'_change']=float((b[k]-a[k]).mean()) if len(a) else None
                    row['source_overlap_initial']=float(a.source_subspace_overlap.mean()) if len(a) else None
                    row['source_overlap_final']=float(b.source_subspace_overlap.mean()) if len(a) else None
                    vals=[float(bystep[s].loc[chosen].source_subspace_overlap.mean()) if chosen else np.nan for s in STEPS]
                    row['monotonic_source_overlap_decrease']=bool(np.all(np.diff(vals)<=0)) if membership.startswith('trajectory') and chosen else None
                    changes.append(row)
            for step in STEPS:
                a=bystep[step]
                for reason,n in a.valid_reason.value_counts().items():audit.append(dict(condition_id=j.condition_id,step=step,rank=rank,reason=reason,n_rows=int(n)))
    for _,j in jobs[jobs.mask_type.eq('learned')].iterrows():
        random_id=j.condition_id.replace('_learned_','_random_')
        if j.condition_id not in lookup or random_id not in lookup:continue
        ld,rd=lookup[j.condition_id],lookup[random_id]
        for rank in (8,16,32):
            l={s:ld[ld['rank'].eq(rank)&ld.step.eq(s)].set_index('row_id') for s in STEPS}
            r={s:rd[rd['rank'].eq(rank)&rd.step.eq(s)].set_index('row_id') for s in STEPS}
            for membership,steps in [('endpoint_matched',(0,2500)),('trajectory_fixed',STEPS),('endpoint_reference_valid',(0,2500)),('trajectory_reference_valid',STEPS)]:
                query='valid & source_reference_valid' if 'reference' in membership else 'valid'
                ids=set.intersection(*[set(t[s].query(query).index) for t in (l,r) for s in steps])
                for layer in (-1,0,1,2,3):
                    chosen=sorted(i for i in ids if layer==-1 or int(i.split(':')[0])==layer)
                    for step in steps:
                        row=dict(seed=j.seed,router_age=j.router_age,sparsity=j.sparsity,rank=rank,layer=layer,
                                 membership=membership,step=step,n_experts=len(chosen))
                        for k in METRICS:
                            row[k+'_difference']=float((l[step].loc[chosen,k]-r[step].loc[chosen,k]).mean()) if chosen else None
                            row[k+'_change_difference']=float(((l[step].loc[chosen,k]-l[0].loc[chosen,k])-(r[step].loc[chosen,k]-r[0].loc[chosen,k])).mean()) if chosen else None
                        pairs.append(row)
    out=pd.concat(raw,ignore_index=True);out.to_parquet(OUT/'svd_expert_raw.parquet',index=False)
    frame(OUT/'svd_inclusion_exclusion.csv',audit);frame(OUT/'svd_seed_trajectories.csv',summaries)
    frame(OUT/'svd_endpoint_changes.csv',changes);frame(OUT/'svd_learned_random_paired.csv',pairs)
    c=pd.DataFrame(changes);c=c[c['rank'].eq(16)&c.layer.eq(-1)&c.membership.eq('endpoint_matched')&c.mask_type.eq('learned')]
    old=pd.read_csv(BASE/'analysis/functional_geometry_cases.csv');old=old[old.condition_id.str.contains('_learned_')]
    correction=dict(original_learned_decreases=int((old.source_overlap_change<0).sum()),endpoint_matched_learned_decreases=int((c.source_subspace_overlap_change<0).sum()),n_learned=len(c),
                    exceptions=c[c.source_subspace_overlap_change.ge(0)].to_dict('records'),
                    validity='same original current-matrix flags, with identical layer/expert membership at both endpoints; source-reference flags additionally exported for sensitivity')
    save(OUT/'svd_correction.json',correction);print(json.dumps(correction,indent=2),flush=True)
if __name__=='__main__':run()
