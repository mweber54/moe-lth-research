"""Pruning-only and original reference summaries, separate from recovery trajectories."""
from common import *
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze import summarize_seed_effects

def run():
    gram=pd.read_parquet(OUT/'gram_expert_raw.parquet');act=pd.read_parquet(OUT/'activation_expert_raw.parquet')
    g=gram[~gram.category.eq('recovery')];a=act[~act.category.eq('recovery')]
    keys=['condition_id','seed','router_age','sparsity','mask_type','category','reference']
    summaries=[]
    for layer in (-1,0,1,2,3):
        d=g if layer==-1 else g[g.layer.eq(layer)]
        mean=d.groupby(keys,dropna=False)[['shape_distance','cosine','trace_ratio','relative_unnormalized_distance']].mean().reset_index()
        mean['layer']=layer;mean['n_experts']=d.groupby(keys,dropna=False).size().values;summaries.append(mean)
    gs=pd.concat(summaries,ignore_index=True);frame(OUT/'existing_gram_seed_summaries.csv',gs)
    summaries=[];keys=['condition_id','seed','router_age','sparsity','mask_type','category','mode','selection']
    for layer in (-1,0,1,2,3):
        d=a[a.status.eq('valid')];d=d if layer==-1 else d[d.layer.eq(layer)]
        mean=d.groupby(keys,dropna=False)[['pearson','spearman']].mean().reset_index();mean['layer']=layer
        mean['n_experts']=d.groupby(keys,dropna=False).size().values;summaries.append(mean)
    ass=pd.concat(summaries,ignore_index=True);frame(OUT/'existing_activation_seed_summaries.csv',ass)
    # Pruning-only comparisons jointly match experts for each selection/mode.
    pg=g[g.category.eq('masked_trained')&g.reference.eq('source_final')]
    pa=a[a.category.eq('masked_trained')&a.status.eq('valid')]
    ps=pd.read_parquet(OUT/'existing_svd_expert_raw.parquet');ps=ps[ps.category.eq('masked_trained')&ps.valid&ps['rank'].eq(16)]
    keys=['condition_id','seed','sparsity','mask_type','layer','expert']
    joined=pg[keys+['shape_distance','trace_ratio']].merge(pa[keys+['mode','selection','pearson','spearman']],on=keys,validate='one_to_many')
    joined=joined.merge(ps[keys+['source_subspace_overlap','alignment_above_shuffled']],on=keys,validate='many_to_one')
    pk=['seed','sparsity','layer','expert','mode','selection'];metrics=['shape_distance','trace_ratio','pearson','spearman','source_subspace_overlap','alignment_above_shuffled']
    l=joined[joined.mask_type.eq('learned')];r=joined[joined.mask_type.eq('random')]
    pair=l[pk+metrics].merge(r[pk+metrics],on=pk,suffixes=('_learned','_random'),validate='one_to_one')
    for metric in metrics:pair[metric+'_difference']=pair[metric+'_learned']-pair[metric+'_random']
    pair.to_parquet(OUT/'pruning_joint_expert_pairs.parquet',index=False)
    summaries=[];keys=['seed','sparsity','mode','selection'];ms=[m+'_difference' for m in metrics]
    for layer in (-1,0,1,2,3):
        d=pair if layer==-1 else pair[pair.layer.eq(layer)]
        mean=d.groupby(keys)[ms].mean().reset_index();mean['layer']=layer;mean['n_experts']=d.groupby(keys).size().values;summaries.append(mean)
    summary=pd.concat(summaries,ignore_index=True);frame(OUT/'pruning_joint_seed_pairs.csv',summary)
    summarize_seed_effects(summary,['sparsity','mode','selection','layer'],ms,'pruning_joint_paired_summary.csv')
    lines=['# Existing source and pruning-only references','',
      'These states are not recovery trajectories or extra seed replications. See existing_gram_seed_summaries.csv and existing_activation_seed_summaries.csv for every source, masked-trained, masked-initialization and historical endpoint state. Pruning-only learned/random comparisons jointly match valid experts across SVD, Gram and activation measurements; all five seeds and paired bootstrap uncertainty are in pruning_joint_paired_summary.csv.','',
      'Source-final activation reference (equal eligible-expert Pearson means):','',
      '| Mode | Selection | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    refs=ass[ass.category.eq('source_age')&ass.router_age.eq(100)&ass.layer.eq(-1)]
    for (mode,selection),d in refs.groupby(['mode','selection']):
        vals={int(r.seed):float(r.pearson) for _,r in d.iterrows()}
        lines.append('| '+mode+' | '+selection+' | '+' | '.join(f'{vals[s]:.5f}' if s in vals else 'NA' for s in SEEDS)+' |')
    lines+=['','Gram pruning-only source-final shape and energy (all experts, separate from joint paired eligible-expert analysis):','',
            '| Mask | Sparsity | Mean shape distance | Mean trace ratio |','| --- | ---: | ---: | ---: |']
    for (kind,sp),d in gs[gs.category.eq('masked_trained')&gs.reference.eq('source_final')&gs.layer.eq(-1)].groupby(['mask_type','sparsity']):
        lines.append(f'| {kind} | {sp:.0%} | {d.shape_distance.mean():.5f} | {d.trace_ratio.mean():.5f} |')
    lines+=['','These projection statistics do not quantify the complete nonlinear expert or imply that pruning preserves its full function. Native and fixed-input activation modes remain distinct. Individual expert distributions are descriptive. The exact eligible sample counts and exclusions are retained in the raw tables.']
    (OUT/'existing_reference_findings.md').write_text('\n'.join(lines)+'\n')
    print('EXISTING REFERENCE SUMMARY COMPLETE',flush=True)
if __name__=='__main__':run()
