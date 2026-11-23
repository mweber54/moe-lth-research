"""Seed-paired trajectories and matched dense gaps; final report only at completion."""
from gcommon import *
from summarize import mean
import numpy as np

def analyze():
 jobs=rows(OUT/'run_manifest.csv');complete=[j for j in jobs if j['status']=='COMPLETE'];records=[]
 for j in complete:
  for step in STEPS:
   root=Path(j['attempt'])/'geometry'/f'step_{step:04d}'
   meta=read(root/'metadata.json');w=root/'weights.csv'
   records.append(dict(condition_id=j['condition_id'],seed=int(j['seed']),age=int(j['router_age']),sparsity=float(j['sparsity']),mask_type=j['mask_type'],step=step,loss=meta['validation_loss'],alignment=mean(w,'alignment_centered'),source_overlap=mean(w,'source_subspace_overlap'),applied_updates=meta['applied_updates'],amp_skips=meta['amp_skips']))
 lookup={(r['seed'],r['age'],r['sparsity'],r['mask_type'],r['step']):r for r in records}
 for r in records:
  original=mean(OUT/'existing'/f"source_e32_seed{r['seed']}_T2500"/'source_age_R100/weights.csv','alignment_centered');r['source_original_alignment']=original;r['alignment_distance_to_source']=None if original is None or r['alignment'] is None else abs(r['alignment']-original)
  dense=lookup.get((r['seed'],r['age'],0.,'dense',r['step']));r['sparse_minus_dense_loss']=None if dense is None else r['loss']-dense['loss']
 table(OUT/'analysis/trajectory_seed_values.csv',records);paired=[];age_effects=[]
 for seed in SEEDS:
  for age in AGES:
   for sp in SP:
    for step in STEPS:
     l=lookup.get((seed,age,sp,'learned',step));r=lookup.get((seed,age,sp,'random',step));l0=lookup.get((seed,age,sp,'learned',0));r0=lookup.get((seed,age,sp,'random',0))
     if l and r and l0 and r0:paired.append(dict(seed=seed,age=age,sparsity=sp,step=step,alignment_difference=l['alignment']-r['alignment'],alignment_change_difference=(l['alignment']-l0['alignment'])-(r['alignment']-r0['alignment']),loss_difference=l['loss']-r['loss'],source_overlap_difference=l['source_overlap']-r['source_overlap']))
  for sp in (0.,)+SP:
   for kind in (('dense',) if sp==0 else ('learned','random')):
    for a,b in ((50,0),(100,0),(100,50)):
     for step in STEPS:
      ra=lookup.get((seed,a,sp,kind,step));rb=lookup.get((seed,b,sp,kind,step))
      if ra and rb:age_effects.append(dict(seed=seed,age_comparison=f'R{a}-R{b}',sparsity=sp,mask_type=kind,step=step,alignment_difference=ra['alignment']-rb['alignment'],loss_difference=ra['loss']-rb['loss']))
 table(OUT/'analysis/learned_random_paired.csv',paired);table(OUT/'analysis/router_age_paired.csv',age_effects)
 # Bootstrap across source seeds only, each family stratum kept separate.
 summaries=[];gen=np.random.default_rng(8103)
 for family,rs,keys,metrics in [('learned_random',paired,('age','sparsity','step'),('alignment_change_difference','loss_difference')),('router_age',age_effects,('age_comparison','sparsity','mask_type','step'),('alignment_difference','loss_difference'))]:
  strata=sorted({tuple(r[k] for k in keys) for r in rs})
  for stratum in strata:
   values=[r for r in rs if tuple(r[k] for k in keys)==stratum]
   for metric in metrics:
    v=np.array([r[metric] for r in values]);ci=np.quantile(gen.choice(v,(10000,len(v)),replace=True).mean(1),[.025,.975]);summaries.append(dict(family=family,stratum=str(stratum),metric=metric,n_seeds=len(v),mean=float(v.mean()),lo=float(ci[0]),hi=float(ci[1]),seed_values=json.dumps({r['seed']:r[metric] for r in values})))
 table(OUT/'analysis/trajectory_paired_summary.csv',summaries)
 cases=[]
 for r in records:
  if r['step']!=2500 or r['mask_type']=='dense':continue
  zero=lookup[r['seed'],r['age'],r['sparsity'],r['mask_type'],0];cases.append(dict(condition_id=r['condition_id'],loss_change=r['loss']-zero['loss'],source_overlap_change=r['source_overlap']-zero['source_overlap'],alignment_change=r['alignment']-zero['alignment'],alignment_distance_change=r['alignment_distance_to_source']-zero['alignment_distance_to_source'],loss_improves_without_alignment_restoration=r['loss']<zero['loss'] and r['alignment_distance_to_source']>=zero['alignment_distance_to_source'],alignment_restores_without_loss_gain=r['alignment_distance_to_source']<zero['alignment_distance_to_source'] and r['loss']>=zero['loss'],loss_improves_without_subspace_restoration=r['loss']<zero['loss'] and r['source_overlap']<=zero['source_overlap'],subspace_returns_without_loss_gain=r['source_overlap']>zero['source_overlap'] and r['loss']>=zero['loss']))
 table(OUT/'analysis/functional_geometry_cases.csv',cases)
 correlations=[]
 for j in complete:
  if j['job_type']=='dense':continue
  trajectory=[r for r in records if r['condition_id']==j['condition_id'] and r['sparse_minus_dense_loss'] is not None]
  for metric in ('alignment','source_overlap'):
   x=np.array([r[metric] for r in trajectory]);y=np.array([r['sparse_minus_dense_loss'] for r in trajectory]);corr=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>0 and y.std()>0 else None
   correlations.append(dict(condition_id=j['condition_id'],seed=j['seed'],age=j['router_age'],sparsity=j['sparsity'],mask_type=j['mask_type'],metric=metric,correlation=corr,n_timepoints=len(trajectory),interpretation='exploratory within-condition trajectory; time is a confound, no causal inference'))
 table(OUT/'analysis/within_condition_correlations.csv',correlations)
 complete_all=len(complete)==75
 report=['# Recovery findings' if complete_all else '# Interim recovery status','',f'{len(complete)}/75 validated runs complete. Seed-level paired effects and uncertainty are exported; all strata retain five-seed replication when complete.','', 'Geometry/performance relationships are descriptive. No pooled correlation or causal conclusion is justified. Review functional_geometry_cases.csv and paired seed values for discordant recovery and inconsistent age effects. Similar learned/random geometry with distinct losses is visible in paired effect tables; no unregistered numerical threshold defines similarity.','', 'Final conclusions require all registered runs.']
 if complete_all:
  report+=['',f"Loss improves without approaching source alignment: {sum(r['loss_improves_without_alignment_restoration'] for r in cases)}/60 sparse runs.",f"Alignment approaches source without improved loss: {sum(r['alignment_restores_without_loss_gain'] for r in cases)}/60 sparse runs.",f"Loss improves without increased source subspace overlap: {sum(r['loss_improves_without_subspace_restoration'] for r in cases)}/60 sparse runs.",f"Source overlap increases without improved loss: {sum(r['subspace_returns_without_loss_gain'] for r in cases)}/60 sparse runs.",'','Final paired effects (all seed values in trajectory_paired_summary.csv):']
  for r in summaries:
   if '2500' in r['stratum']:report.append(f"- {r['family']} {r['stratum']} {r['metric']}: mean {r['mean']:.6g}, seed bootstrap [{r['lo']:.6g}, {r['hi']:.6g}], n={r['n_seeds']}.")
  inconsistent=[r for r in summaries if r['family']=='router_age' and '2500' in r['stratum'] and min(json.loads(r['seed_values']).values())<0<max(json.loads(r['seed_values']).values())]
  report+=['',f"Router-age endpoint effect strata with mixed signs across seeds: {len(inconsistent)}. See individual seed effects; no assumption of an optimal interior age."]
 (OUT/'analysis'/('final_findings.md'  if complete_all else 'interim_findings.md')).write_text('\n'.join(report)+'\n')
if __name__=='__main__':
 analyze()
 from domain_analysis import analyze as domains
 domains()
