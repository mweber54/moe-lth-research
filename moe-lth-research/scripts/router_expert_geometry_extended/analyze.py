"""Five-seed paired summaries, separate references, and exploratory figures."""
from common import *
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summarize_seed_effects(data,keys,metrics,path):
    rows=[]
    for key,g in data.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        for metric in metrics:
            valid=g.dropna(subset=[metric]);assert not valid.seed.duplicated().any(),(keys,key,metric)
            mean,lo,hi=bootstrap(valid[metric]);rows.append(dict(zip(keys,key),metric=metric,n_seeds=len(valid),mean=mean,bootstrap_lo=lo,bootstrap_hi=hi,
                 seed_values=json.dumps({str(int(r.seed)):float(r[metric]) for _,r in valid.iterrows()})))
    frame(OUT/path,rows);return pd.DataFrame(rows)

def load_measurements():
    gram=[];groups=[];pools=[];drift=[];within=[];inventory=[]
    expected_code=code_hash()
    for p in sorted((OUT/'cache').glob('*/**/completed.json')):
        done=read(p);meta=done['meta']
        expected_key=hashlib.sha256(json.dumps(dict(origin=done['origin'],refs=done['reference_identities'],probe=done['probe'],
                        code=expected_code,config=CONFIG,meta=meta),sort_keys=True).encode()).hexdigest()
        assert done['cache_key']==expected_key,'Measurement/config/input cache identity mismatch: '+str(p)
        # Validate outputs independently of trusting completion flag alone.
        for name,h in done['output_hashes'].items():
            assert sha(p.parent/name)==h,'Cache output altered '+str(p.parent/name)
        inv=dict(**meta,path=str(p.parent),origin=json.dumps(done['origin'],sort_keys=True),probe_sha256=done['probe']['sha256'],
                 model_tensor_sha256=done['model_tensor_sha256'],checkpoint_identity='hash established by this pass; diagnostic intermediate hashes had no prior independent attestation',
                 mask_verified=done['mask_applied_verified'],initialization_verified=done['expected_initialization_verified'],seconds=done['elapsed_seconds'])
        inventory.append(inv)
        for filename,dest in [('gram_experts.csv',gram),('activation_groups.csv',groups),('activation_layer_pools.csv',pools),('input_drift.csv',drift),('activation_within_token.csv',within)]:
            d=pd.read_csv(p.parent/filename)
            for k,v in meta.items():d[k]=v
            dest.append(d)
    def concat(xs):return pd.concat(xs,ignore_index=True) if xs else pd.DataFrame()
    for name,ds in [('gram_expert_raw',gram),('activation_expert_raw',groups),('activation_layer_pools',pools),('input_coordinate_drift',drift),('within_token_association',within)]:
        concat(ds).to_parquet(OUT/(name+'.parquet'),index=False)
    frame(OUT/'state_coverage_inventory.csv',inventory)
    return concat(gram),concat(groups),concat(pools),concat(drift),concat(within),pd.DataFrame(inventory)

def gram_analysis(raw):
    r=raw[raw.category.eq('recovery')];metrics=['shape_distance','cosine','trace','trace_ratio','relative_unnormalized_distance']
    parts=[]
    for layer in (-1,0,1,2,3):
        d=r if layer==-1 else r[r.layer.eq(layer)]
        keys=['condition_id','seed','router_age','sparsity','mask_type','step','reference']
        g=d.groupby(keys,dropna=False)[metrics].mean().reset_index();g['layer']=layer;parts.append(g)
    means=pd.concat(parts,ignore_index=True);frame(OUT/'gram_seed_trajectories.csv',means)
    end=means[means.step.eq(2500)].merge(means[means.step.eq(0)],on=['condition_id','seed','router_age','sparsity','mask_type','reference','layer'],suffixes=('_final','_initial'))
    for metric in metrics:end[metric+'_change']=end[metric+'_final']-end[metric+'_initial']
    frame(OUT/'gram_endpoint_changes.csv',end)
    summaries=summarize_seed_effects(end,['router_age','sparsity','mask_type','reference','layer'],[k+'_change' for k in metrics],'gram_endpoint_seed_summary.csv')
    pair=[]
    l=means[means.mask_type.eq('learned')];random=means[means.mask_type.eq('random')]
    d=l.merge(random,on=['seed','router_age','sparsity','step','reference','layer'],suffixes=('_learned','_random'))
    for metric in metrics:d[metric+'_difference']=d[metric+'_learned']-d[metric+'_random']
    frame(OUT/'gram_learned_random_seed_pairs.csv',d)
    summarize_seed_effects(d,['router_age','sparsity','step','reference','layer'],[k+'_difference' for k in metrics],'gram_learned_random_paired_summary.csv')
    return means,end

def activation_analysis(raw,pools,within):
    r=raw[raw.category.eq('recovery')].copy();r['row_id']=r.layer.astype(str)+':'+r.expert.astype(str)
    trajectories=[];endpoints=[];pair=[];coverage=[]
    for (cid,mode,selection),group in r.groupby(['condition_id','mode','selection']):
        bystep={s:group[group.step.eq(s)].set_index('row_id') for s in STEPS}
        endpoints_ids=set(bystep[0].query('status=="valid"').index)&set(bystep[2500].query('status=="valid"').index)
        fixed_ids=set.intersection(*[set(bystep[s].query('status=="valid"').index) for s in STEPS])
        meta=group.iloc[0][['condition_id','seed','router_age','sparsity','mask_type']].to_dict()
        for membership,ids,steps in [('endpoint_matched',endpoints_ids,(0,2500)),('trajectory_fixed',fixed_ids,STEPS)]:
            for layer in (-1,0,1,2,3):
                chosen=sorted(i for i in ids if layer==-1 or int(i.split(':')[0])==layer)
                vals=[]
                for step in steps:
                    d=bystep[step].loc[chosen]
                    row=dict(**meta,mode=mode,selection=selection,membership=membership,layer=layer,step=step,
                        n_experts=len(d),n_excluded=(128 if layer==-1 else 32)-len(d),n_pairs=int(d.n_samples.sum()),
                        pearson_expert_mean=float(d.pearson.mean()) if len(d) else None,spearman_expert_mean=float(d.spearman.mean()) if len(d) else None,
                        standardized_pooled_pearson=float(np.average(d.pearson,weights=d.n_samples)) if len(d) else None)
                    trajectories.append(row);vals.append(row)
                if len(vals)>=2:
                    row=dict(**meta,mode=mode,selection=selection,membership=membership,layer=layer,n_experts=len(chosen))
                    for metric in ('pearson_expert_mean','spearman_expert_mean','standardized_pooled_pearson'):
                        row[metric+'_initial']=vals[0][metric];row[metric+'_final']=vals[-1][metric]
                        row[metric+'_change']=vals[-1][metric]-vals[0][metric] if chosen else None
                    row['monotonic_increase']=bool(np.all(np.diff([v['pearson_expert_mean'] for v in vals])>=0)) if membership=='trajectory_fixed' and chosen else None
                    endpoints.append(row)
        for (step,layer,status),g in group.groupby(['step','layer','status']):coverage.append(dict(**meta,mode=mode,selection=selection,step=step,layer=layer,status=status,n_experts=len(g),n_pairs=int(g.n_samples.sum())))
    # Additional learned/random expert matching before any comparison.
    for (seed,age,sp,mode,selection),group in r[r.mask_type.isin(['learned','random'])].groupby(['seed','router_age','sparsity','mode','selection']):
        ls={k:{s:group[group.mask_type.eq(k)&group.step.eq(s)].set_index('row_id') for s in STEPS} for k in ('learned','random')}
        for membership,steps in [('endpoint_matched',(0,2500)),('trajectory_fixed',STEPS)]:
            ids=set.intersection(*[set(ls[k][s].query('status=="valid"').index) for k in ls for s in steps])
            for layer in (-1,0,1,2,3):
                chosen=sorted(i for i in ids if layer==-1 or int(i.split(':')[0])==layer)
                for step in steps:
                    row=dict(seed=seed,router_age=age,sparsity=sp,mode=mode,selection=selection,membership=membership,layer=layer,step=step,n_experts=len(chosen))
                    for metric in ('pearson','spearman'):
                        l=ls['learned'][step].loc[chosen,metric];rr=ls['random'][step].loc[chosen,metric]
                        row[metric+'_difference']=float((l-rr).mean()) if chosen else None
                        row[metric+'_change_difference']=float(((l-ls['learned'][0].loc[chosen,metric])-(rr-ls['random'][0].loc[chosen,metric])).mean()) if chosen else None
                    pair.append(row)
    trajectory=pd.DataFrame(trajectories);endpoint=pd.DataFrame(endpoints);paired=pd.DataFrame(pair)
    frame(OUT/'activation_seed_trajectories.csv',trajectory);frame(OUT/'activation_endpoint_changes.csv',endpoint)
    frame(OUT/'activation_coverage.csv',coverage);frame(OUT/'activation_learned_random_pairs.csv',paired)
    summarize_seed_effects(endpoint,['router_age','sparsity','mask_type','mode','selection','membership','layer'],
                          ['pearson_expert_mean_change','spearman_expert_mean_change','standardized_pooled_pearson_change'],'activation_endpoint_seed_summary.csv')
    summarize_seed_effects(paired,['router_age','sparsity','mode','selection','membership','layer','step'],
                          ['pearson_difference','spearman_difference','pearson_change_difference'],'activation_learned_random_paired_summary.csv')
    # Shuffled controls retain the exact common expert membership across identity and all shifts.
    pp=pools[pools.category.eq('recovery')];parts=[]
    for layer in (-1,0,1,2,3):
        d=pp if layer==-1 else pp[pp.layer.eq(layer)]
        keys=['condition_id','seed','router_age','sparsity','mask_type','step','mode','selection']
        for key,g in d.groupby(keys):
            valid=g.dropna(subset=['matched_pearson','matched_shuffle_mean']);n=valid.matched_shuffle_n_pairs.sum()
            row=dict(zip(keys,key),layer=layer,n_pairs=int(n),n_experts=int(valid.matched_shuffle_n_experts.sum()))
            for metric in ('matched_pearson','matched_shuffle_mean','pearson_above_shuffle'):
                row[metric]=float(np.average(valid[metric],weights=valid.matched_shuffle_n_pairs)) if n else None
            parts.append(row)
    shuffle=pd.DataFrame(parts);frame(OUT/'activation_shuffled_seed_trajectories.csv',shuffle)
    summarize_seed_effects(shuffle,['router_age','sparsity','mask_type','step','mode','selection','layer'],['pearson_above_shuffle'],'activation_shuffled_seed_summary.csv')
    ww=within[within.category.eq('recovery')]
    keys=['condition_id','seed','router_age','sparsity','mask_type','step','mode','layer']
    wt=ww.groupby(keys)[['pearson','spearman']].mean().reset_index();frame(OUT/'within_token_seed_trajectories.csv',wt)
    return trajectory,endpoint,shuffle

def svd_summaries():
    end=pd.read_csv(OUT/'svd_endpoint_changes.csv');paired=pd.read_csv(OUT/'svd_learned_random_paired.csv')
    summarize_seed_effects(end,['router_age','sparsity','mask_type','rank','layer','membership'],
        ['source_subspace_overlap_change','alignment_centered_change','alignment_above_shuffled_change','retained_energy_change'],'svd_endpoint_seed_summary.csv')
    summarize_seed_effects(paired,['router_age','sparsity','rank','layer','membership','step'],
        ['source_subspace_overlap_difference','source_subspace_overlap_change_difference','alignment_centered_change_difference'],'svd_learned_random_paired_summary.csv')
    # Reference-valid sensitivity: current/source must both satisfy spectral flags.
    raw=pd.read_parquet(OUT/'svd_expert_raw.parquet');out=[]
    for (cid,rank),g in raw.groupby(['condition_id','rank']):
        bs={s:g[g.step.eq(s)].set_index('row_id') for s in STEPS}
        ids=sorted(set(bs[0].query('valid & source_reference_valid').index)&set(bs[2500].query('valid & source_reference_valid').index))
        m=g.iloc[0];out.append(dict(condition_id=cid,seed=m.seed,router_age=m.router_age,sparsity=m.sparsity,mask_type=m.mask_type,rank=rank,n_experts=len(ids),
             source_overlap_change=float((bs[2500].loc[ids].source_subspace_overlap-bs[0].loc[ids].source_subspace_overlap).mean()) if ids else None))
    frame(OUT/'svd_reference_valid_sensitivity.csv',out)

def existing_svd_analysis(inventory):
    data=[]
    for _,rec in inventory[~inventory.category.eq('recovery')].iterrows():
        p=BASE/'existing'/f'source_e32_seed{int(rec.seed)}_T2500'/rec.condition_id/'weights.csv'
        if not p.exists():continue
        d=pd.read_csv(p)
        for k in ('condition_id','seed','router_age','sparsity','mask_type','category','step'):d[k]=rec[k]
        data.append(d)
    raw=pd.concat(data,ignore_index=True)
    source=raw[raw.category.eq('source_age')&raw.router_age.eq(100)][['seed','layer','expert','rank','zero_matrix','rank_deficient','degenerate_cutoff']]
    source=source.rename(columns={k:'source_'+k for k in ('zero_matrix','rank_deficient','degenerate_cutoff')})
    raw=raw.merge(source,on=['seed','layer','expert','rank'],validate='many_to_one')
    raw['valid']=~(raw.zero_matrix|raw.rank_deficient|raw.degenerate_cutoff|raw.source_zero_matrix|raw.source_rank_deficient|raw.source_degenerate_cutoff)
    raw.to_parquet(OUT/'existing_svd_expert_raw.parquet',index=False)
    metrics=['source_subspace_overlap','alignment_centered','alignment_above_shuffled','retained_energy','relative_cutoff_gap'];summaries=[]
    keys=['condition_id','seed','router_age','sparsity','mask_type','category','rank']
    for layer in (-1,0,1,2,3):
        d=raw[raw.valid] if layer==-1 else raw[raw.valid&raw.layer.eq(layer)]
        a=d.groupby(keys,dropna=False)[metrics].mean().reset_index();a['layer']=layer;a['n_experts']=d.groupby(keys,dropna=False).size().values;summaries.append(a)
    frame(OUT/'existing_svd_seed_summaries.csv',pd.concat(summaries,ignore_index=True))
    d=raw[raw.category.eq('masked_trained')&raw.valid]
    keys=['seed','sparsity','layer','expert','rank']
    l=d[d.mask_type.eq('learned')];r=d[d.mask_type.eq('random')]
    pair=l[keys+metrics].merge(r[keys+metrics],on=keys,suffixes=('_learned','_random'),validate='one_to_one')
    for metric in metrics:pair[metric+'_difference']=pair[metric+'_learned']-pair[metric+'_random']
    pair.to_parquet(OUT/'pruning_learned_random_svd_expert_pairs.parquet',index=False)
    summaries=[];keys=['seed','sparsity','rank'];ms=[m+'_difference' for m in metrics]
    for layer in (-1,0,1,2,3):
        d=pair if layer==-1 else pair[pair.layer.eq(layer)]
        a=d.groupby(keys)[ms].mean().reset_index();a['layer']=layer;a['n_experts']=d.groupby(keys).size().values;summaries.append(a)
    d=pd.concat(summaries,ignore_index=True);frame(OUT/'pruning_learned_random_svd_seed_pairs.csv',d)
    summarize_seed_effects(d,['sparsity','rank','layer'],ms,'pruning_learned_random_svd_paired_summary.csv')

def combine(gram_end,act_end,shuffle,inventory):
    svd=pd.read_csv(OUT/'svd_endpoint_changes.csv');svd=svd[svd['rank'].eq(16)&svd.layer.eq(-1)&svd.membership.eq('endpoint_reference_valid')]
    d=svd[['condition_id','seed','router_age','sparsity','mask_type','source_subspace_overlap_change','source_overlap_initial','source_overlap_final','alignment_above_shuffled_change']].copy()
    for ref in ('source_final','matched_dense'):
        g=gram_end[gram_end.reference.eq(ref)&gram_end.layer.eq(-1)]
        columns=['condition_id','shape_distance_initial','shape_distance_final','shape_distance_change','trace_ratio_final','trace_ratio_change']
        d=d.merge(g[columns].rename(columns={k:ref+'_'+k for k in columns if k!='condition_id'}),on='condition_id')
    for mode in ('fixed','native'):
        for selection in ('proposed','accepted','all_expert'):
            a=act_end[act_end['mode'].eq(mode)&act_end.selection.eq(selection)&act_end.membership.eq('endpoint_matched')&act_end.layer.eq(-1)]
            columns=['condition_id','pearson_expert_mean_initial','pearson_expert_mean_final','pearson_expert_mean_change','n_experts']
            d=d.merge(a[columns].rename(columns={k:mode+'_'+selection+'_'+k for k in columns if k!='condition_id'}),on='condition_id')
            a=shuffle[shuffle['mode'].eq(mode)&shuffle.selection.eq(selection)&shuffle.step.eq(2500)&shuffle.layer.eq(-1)]
            d=d.merge(a[['condition_id','pearson_above_shuffle']].rename(columns={'pearson_above_shuffle':mode+'_'+selection+'_above_shuffle_final'}),on='condition_id')
    losses=inventory[inventory.category.eq('recovery')&inventory.step.eq(2500)][['condition_id','validation_loss']]
    d=d.merge(losses,on='condition_id');dense=d[d.mask_type.eq('dense')][['seed','router_age','validation_loss']].rename(columns={'validation_loss':'dense_loss'})
    d=d.merge(dense,on=['seed','router_age']);d['matched_dense_gap']=d.validation_loss-d.dense_loss
    frame(OUT/'combined_seed_cases.csv',d)
    correlations=[]
    for (age,sp,kind),g in d.groupby(['router_age','sparsity','mask_type']):
        for metric in ('source_overlap_final','source_final_shape_distance_final','fixed_proposed_pearson_expert_mean_final','native_proposed_pearson_expert_mean_final'):
            x=g[metric];y=g.matched_dense_gap
            valid=x.notna()&y.notna();xx=x[valid];yy=y[valid]
            rho=float(spearmanr(xx,yy).statistic) if len(xx)>=3 and xx.std()>0 and yy.std()>0 else None
            correlations.append(dict(router_age=age,sparsity=sp,mask_type=kind,metric=metric,n_seeds=int(valid.sum()),spearman_with_dense_gap=rho,
                seed_values=json.dumps({str(int(r.seed)):[None if pd.isna(r[metric]) else float(r[metric]),float(r.matched_dense_gap)] for _,r in g.iterrows()}),
                interpretation='post hoc n=5 descriptive sensitivity; no significance testing'))
    frame(OUT/'within_condition_seed_correlations.csv',correlations)
    return d

def joint_expert_analysis(gram_raw,act_raw):
    """Compare all three metrics on exactly the same endpoint-valid experts."""
    s=pd.read_parquet(OUT/'svd_expert_raw.parquet');s=s[s['rank'].eq(16)&s.step.isin([0,2500])&s.valid&s.source_reference_valid]
    keys=['condition_id','seed','router_age','sparsity','mask_type','layer','expert']
    cols=keys+['source_subspace_overlap','alignment_above_shuffled']
    endpoints=s[s.step.eq(0)][cols].merge(s[s.step.eq(2500)][cols],on=keys,suffixes=('_initial','_final'))
    for metric in ('source_subspace_overlap','alignment_above_shuffled'):
        endpoints[metric+'_change']=endpoints[metric+'_final']-endpoints[metric+'_initial']
    g=gram_raw[gram_raw.category.eq('recovery')&gram_raw.reference.eq('source_final')&gram_raw.step.isin([0,2500])]
    cols=keys+['shape_distance','trace_ratio']
    ge=g[g.step.eq(0)][cols].merge(g[g.step.eq(2500)][cols],on=keys,suffixes=('_initial','_final'))
    for metric in ('shape_distance','trace_ratio'):ge[metric+'_change']=ge[metric+'_final']-ge[metric+'_initial']
    endpoints=endpoints.merge(ge,on=keys,validate='one_to_one')
    a=act_raw[act_raw.category.eq('recovery')&act_raw.step.isin([0,2500])&act_raw.status.eq('valid')]
    ak=keys+['mode','selection'];cols=ak+['pearson','spearman','n_samples']
    ae=a[a.step.eq(0)][cols].merge(a[a.step.eq(2500)][cols],on=ak,suffixes=('_initial','_final'))
    for metric in ('pearson','spearman'):ae[metric+'_change']=ae[metric+'_final']-ae[metric+'_initial']
    joined=endpoints.merge(ae,on=keys,validate='one_to_many');joined.to_parquet(OUT/'joint_expert_endpoint_effects.parquet',index=False)
    metrics=['source_subspace_overlap_change','alignment_above_shuffled_change','shape_distance_change','trace_ratio_change','pearson_change','spearman_change']
    strata=['condition_id','seed','router_age','sparsity','mask_type','mode','selection'];summaries=[]
    for layer in (-1,0,1,2,3):
        d=joined if layer==-1 else joined[joined.layer.eq(layer)]
        grouped=d.groupby(strata)[metrics].mean().reset_index();grouped['layer']=layer
        grouped['n_experts']=d.groupby(strata).size().values;summaries.append(grouped)
    summaries=pd.concat(summaries,ignore_index=True);frame(OUT/'joint_matched_expert_seed_effects.csv',summaries)
    summarize_seed_effects(summaries,['router_age','sparsity','mask_type','mode','selection','layer'],metrics,'joint_matched_expert_seed_summary.csv')
    l=joined[joined.mask_type.eq('learned')];rr=joined[joined.mask_type.eq('random')]
    pk=['seed','router_age','sparsity','layer','expert','mode','selection']
    paired=l[pk+metrics].merge(rr[pk+metrics],on=pk,suffixes=('_learned','_random'),validate='one_to_one')
    for metric in metrics:paired[metric+'_difference']=paired[metric+'_learned']-paired[metric+'_random']
    paired.to_parquet(OUT/'joint_learned_random_expert_effects.parquet',index=False)
    ps=[];mk=[m+'_difference' for m in metrics];groups=['seed','router_age','sparsity','mode','selection']
    for layer in (-1,0,1,2,3):
        d=paired if layer==-1 else paired[paired.layer.eq(layer)]
        g=d.groupby(groups)[mk].mean().reset_index();g['layer']=layer;g['n_experts']=d.groupby(groups).size().values;ps.append(g)
    ps=pd.concat(ps,ignore_index=True);frame(OUT/'joint_learned_random_seed_effects.csv',ps)
    summarize_seed_effects(ps,['router_age','sparsity','mode','selection','layer'],mk,'joint_learned_random_seed_summary.csv')
    # Age contrasts use exact common experts and retain paired source-seed blocks.
    age_metrics=metrics+['source_subspace_overlap_final','shape_distance_final','pearson_final']
    ages=[];ak=['seed','sparsity','mask_type','layer','expert','mode','selection']
    for a,b in ((50,0),(100,0),(100,50)):
        left=joined[joined.router_age.eq(a)];right=joined[joined.router_age.eq(b)]
        d=left[ak+age_metrics].merge(right[ak+age_metrics],on=ak,suffixes=('_left','_right'),validate='one_to_one')
        for metric in age_metrics:d[metric+'_age_difference']=d[metric+'_left']-d[metric+'_right']
        keys=['seed','sparsity','mask_type','mode','selection'];ms=[m+'_age_difference' for m in age_metrics]
        for layer in (-1,0,1,2,3):
            ld=d if layer==-1 else d[d.layer.eq(layer)]
            g=ld.groupby(keys)[ms].mean().reset_index();g['layer']=layer;g['age_comparison']=f'R{a}-R{b}';g['n_experts']=ld.groupby(keys).size().values;ages.append(g)
    ages=pd.concat(ages,ignore_index=True);frame(OUT/'joint_router_age_seed_pairs.csv',ages)
    summarize_seed_effects(ages,['sparsity','mask_type','mode','selection','layer','age_comparison'],
                          [m+'_age_difference' for m in age_metrics],'joint_router_age_paired_summary.csv')

def consistency_audit(grams):
    source=grams[grams.reference.eq('source_final')].copy()
    old=pd.read_parquet(OUT/'svd_expert_raw.parquet');old=old[old['rank'].eq(16)]
    existing=pd.read_parquet(OUT/'existing_svd_expert_raw.parquet');existing=existing[existing['rank'].eq(16)].copy()
    old=pd.concat([old,existing],ignore_index=True)
    keys=['condition_id','seed','layer','expert','step']
    energy=source[keys+['trace']].merge(old[keys+['singular_frobenius_energy']],on=keys,validate='one_to_one')
    assert len(energy)==len(source),'Some checkpoint weights have no matching recorded SVD identity'
    energy['relative_energy_error']=(energy.trace-energy.singular_frobenius_energy).abs()/energy.trace.clip(lower=1e-30)
    available=energy.singular_frobenius_energy.notna()
    energy['consistency_status']=np.where(available,np.where(energy.relative_energy_error.le(2e-6),'consistent','mismatch'),'original_svd_energy_field_unavailable')
    assert energy.loc[available,'relative_energy_error'].le(2e-6).all(),'Recorded SVD energy disagrees with actual checkpoint Gram trace'
    frame(OUT/'svd_gram_weight_identity_consistency.csv',energy)
    valid=grams.dropna(subset=['shape_distance','cosine'])
    discrepancy=np.abs(valid.shape_distance**2-(2-2*valid.cosine))
    assert float(discrepancy.max())<1e-10
    save(OUT/'metric_consistency_validation.json',dict(passed=True,svd_gram_rows=len(energy),independent_energy_checks=int(available.sum()),
              original_svd_energy_unavailable_rows=int((~available).sum()),
              original_svd_energy_unavailable_conditions=sorted(energy.loc[~available,'condition_id'].unique().tolist()),
              max_relative_energy_error=float(energy.relative_energy_error.max()),
              gram_rows=len(valid),max_shape_cosine_algebra_error=float(discrepancy.max())))

def figures(grams,activation):
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    svd=pd.read_csv(OUT/'svd_seed_trajectories.csv');svd=svd[svd.membership.eq('trajectory_reference_valid')]
    # Preserve all seed curves; different age/sparsity strata are separate panels.
    def plot(data,metric,name,selection=None):
        data=data.copy();fig,axes=plt.subplots(3,5,figsize=(17,9),sharex=True)
        kinds=[('dense',0.),('learned',.5),('random',.5),('learned',.65),('random',.65)]
        for i,age in enumerate((0,50,100)):
            for j,(kind,sp) in enumerate(kinds):
                ax=axes[i,j];g=data[data.router_age.eq(age)&data.mask_type.eq(kind)&data.sparsity.eq(sp)]
                for seed in SEEDS:
                    d=g[g.seed.eq(seed)].sort_values('step');ax.plot(d.step,d[metric],'.-',label=str(seed),linewidth=.9)
                ax.set_title(f'R{age} {kind} s={sp:g}');ax.set_xlabel('Recovery step');ax.set_ylabel(metric)
        axes[0,0].legend(title='Source seed',fontsize=7);fig.tight_layout();fig.savefig(figures/(name+'.png'),dpi=150);fig.savefig(figures/(name+'.pdf'));plt.close(fig)
    for rank in (8,16,32):
        for layer in (-1,0,1,2,3):
            plot(svd[svd['rank'].eq(rank)&svd.layer.eq(layer)],'source_subspace_overlap',f'svd_rank{rank}_layer{layer}')
    for ref in ('source_final','matched_dense'):
        for layer in (-1,0,1,2,3):
            g=grams[grams.reference.eq(ref)&grams.layer.eq(layer)]
            plot(g,'shape_distance',f'gram_shape_{ref}_layer{layer}');plot(g,'trace_ratio',f'gram_energy_{ref}_layer{layer}')
    for mode in ('fixed','native'):
        for selection in ('proposed','accepted','all_expert'):
            for layer in (-1,0,1,2,3):
                a=activation[activation['mode'].eq(mode)&activation.selection.eq(selection)&activation.membership.eq('trajectory_fixed')&activation.layer.eq(layer)]
                plot(a,'pearson_expert_mean',f'activation_{mode}_{selection}_layer{layer}')

def score_activation_figures(inventory):
    """Paper-like standardized binned curves; raw signed histograms remain separate."""
    from measurements import association
    figures=OUT/'figures';stats=[];panels={}
    selected=inventory[inventory.category.eq('recovery')&inventory.step.isin([0,2500])]
    for _,rec in selected.iterrows():
        with np.load(Path(rec.path)/'token_pairs.npz') as data:
            for mode in ('fixed','native'):
                for selection in ('proposed','accepted','all_expert'):
                    xs=[];ys=[];raws=[];rawa=[]
                    for layer in range(4):
                        prefix=f'{mode}_L{layer}';scores=data[prefix+'_scores'];acts=data[prefix+'_activations']
                        sel=data[prefix+'_selected'];acc=data[prefix+'_accepted']
                        for expert in range(32):
                            if selection=='all_expert':mask=np.arange(len(scores))<128
                            elif selection=='proposed':mask=(sel==expert).any(1)
                            else:mask=((sel==expert)&acc).any(1)
                            x=scores[mask,expert].astype(float);y=acts[mask,expert].astype(float)
                            if association(x,y)[2]!='valid':continue
                            xs.append((x-x.mean())/x.std());ys.append((y-y.mean())/y.std());raws.append(x);rawa.append(y)
                    if not xs:continue
                    x=np.concatenate(xs);y=np.concatenate(ys);p,s,_=association(x,y)
                    raw_y=np.concatenate(rawa);raw_x=np.concatenate(raws)
                    stats.append(dict(condition_id=rec.condition_id,seed=rec.seed,router_age=rec.router_age,sparsity=rec.sparsity,mask_type=rec.mask_type,
                                      step=rec.step,mode=mode,selection=selection,pooled_pearson=p,pooled_spearman=s,n_pairs=len(x),n_valid_experts=len(xs)))
                    bins=np.linspace(-3,3,21);inds=np.digitize(x,bins)-1
                    binned=[float(y[inds==i].mean()) if np.any(inds==i) else np.nan for i in range(20)]
                    counts=[int((inds==i).sum()) for i in range(20)]
                    # Visual token distributions are descriptive, never independent seed confidence bands.
                    panels[(mode,selection,int(rec.step),int(rec.router_age),rec.mask_type,float(rec.sparsity),int(rec.seed))]=(binned,counts,
                         np.histogram(raw_y,bins=np.linspace(-.5,3,71),density=False)[0],float((raw_y<0).mean()),p,s)
    frame(OUT/'activation_global_pooled_endpoint_statistics.csv',stats)
    for mode in ('fixed','native'):
        for selection in ('proposed','accepted','all_expert'):
            for step in (0,2500):
                fig,axes=plt.subplots(3,5,figsize=(17,9),sharex=True,sharey=True)
                hist,haxes=plt.subplots(3,5,figsize=(17,9),sharex=True)
                kinds=[('dense',0.),('learned',.5),('random',.5),('learned',.65),('random',.65)]
                for i,age in enumerate((0,50,100)):
                    for j,(kind,sp) in enumerate(kinds):
                        ax=axes[i,j];ha=haxes[i,j]
                        for seed in SEEDS:
                            vals=panels.get((mode,selection,step,age,kind,sp,seed))
                            if vals is None:continue
                            b,n,h,negative,p,s=vals
                            ax.plot(np.linspace(-2.85,2.85,20),b,'.-',label=f'{seed} r={p:.2f}',linewidth=.9)
                            ha.plot(np.linspace(-.475,2.975,70),h/max(h.sum(),1),label=f'{seed} neg={negative:.3f}',linewidth=.9)
                        ax.axhline(0,color='gray',linewidth=.5);ax.set_title(f'R{age} {kind} s={sp:g}');ax.set(xlabel='Within-expert standardized raw score',ylabel='Standardized signed GELU mean')
                        ha.axvline(0,color='gray',linewidth=.5);ha.set_title(f'R{age} {kind} s={sp:g}');ha.set(xlabel='Raw signed mean GELU activation',ylabel='Fraction in display bins')
                        ax.legend(fontsize=6);ha.legend(fontsize=6)
                for chart,name in [(fig,f'score_activation_{mode}_{selection}_step{step}'),(hist,f'raw_signed_activation_{mode}_{selection}_step{step}')]:
                    chart.tight_layout();chart.savefig(figures/(name+'.png'),dpi=150);chart.savefig(figures/(name+'.pdf'));plt.close(chart)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--finish',action='store_true');args=p.parse_args()
    if args.finish:
        assert read(OUT/'analysis_status.json')['complete'] and read(OUT/'implementation_config.json')['code_hash']==code_hash()
        raw=pd.read_parquet(OUT/'gram_expert_raw.parquet');act=pd.read_parquet(OUT/'activation_expert_raw.parquet')
        inventory=pd.read_csv(OUT/'state_coverage_inventory.csv')
        gs=pd.read_csv(OUT/'gram_seed_trajectories.csv');ge=pd.read_csv(OUT/'gram_endpoint_changes.csv')
        at=pd.read_csv(OUT/'activation_seed_trajectories.csv');ae=pd.read_csv(OUT/'activation_endpoint_changes.csv');sh=pd.read_csv(OUT/'activation_shuffled_seed_trajectories.csv')
        save(OUT/'aggregation_resume.json',dict(stage='joint comparisons, consistency, figures and report; measurements not repeated',
             input_hashes={n:sha(OUT/n) for n in ['gram_expert_raw.parquet','activation_expert_raw.parquet','gram_endpoint_changes.csv','activation_endpoint_changes.csv']}))
        finish(raw,act,inventory,gs,ge,at,ae,sh);return
    raw,act,pools,drift,within,inventory=load_measurements()
    recovery=inventory[inventory.category.eq('recovery')]
    expected_existing={(int(p.parent.parent.name.split('seed')[1].split('_')[0]),p.parent.name) for p in (BASE/'existing').glob('*/*/done.json') if (p.parent/'weights.csv').exists()}
    actual_existing={(int(r.seed),r.condition_id) for _,r in inventory[~inventory.category.eq('recovery')].iterrows()}
    complete=len(recovery)==75*len(STEPS) and recovery.groupby('condition_id').step.nunique().eq(len(STEPS)).all() and expected_existing==actual_existing
    save(OUT/'analysis_status.json',dict(recovery_states=len(recovery),expected_recovery_states=525,complete=bool(complete),
         measured_existing_states=len(inventory)-len(recovery),expected_existing_states=len(expected_existing),missing_existing=sorted(expected_existing-actual_existing),
         missing='see coverage/seed_*.json and state_coverage_inventory.csv'))
    if not complete:raise RuntimeError('Incomplete recovery measurements; preserve raw exports, do not issue final claims')
    svd_summaries();existing_svd_analysis(inventory);gs,ge=gram_analysis(raw);at,ae,sh=activation_analysis(act,pools,within);cases=combine(ge,ae,sh,inventory)
    finish(raw,act,inventory,gs,ge,at,ae,sh)
def finish(raw,act,inventory,gs,ge,at,ae,sh):
    cases=combine(ge,ae,sh,inventory);joint_expert_analysis(raw,act);consistency_audit(raw)
    figures(gs,at);score_activation_figures(inventory)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from report import write_report
    write_report(cases,inventory)
    print('FINAL ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':main()
