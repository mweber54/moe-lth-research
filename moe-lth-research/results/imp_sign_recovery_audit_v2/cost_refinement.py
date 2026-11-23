from common import *
import os
# Split sign runs at the native 2500-step reset boundary for an earlier decision.
a=pd.read_csv(OLD/'optimizer_compute_accounting.csv');a=a[~a.reused.astype(str).str.lower().eq('true')]
full=pd.read_csv(OUT/'runtime_by_seed.csv').set_index('seed')
seg=a.groupby(['seed','segment']).seconds.agg(['mean','median',lambda v:v.quantile(.9)]).reset_index();seg.columns=['seed','segment','mean','median','q90'];seg.to_csv(OUT/'runtime_by_segment.csv',index=False)
cost=[]
stages=[('1_sign_to_2500',(7,17,29),('sign',),2500,False),('2_sign_completion',(7,17,29),('sign',),2500,True),('2_first_three_other_arms',(7,17,29),('full_shuffle_rerun','magnitude','matched_sign'),5000,False),('3_last_two_all_arms',(43,61),('full_shuffle_rerun','sign','magnitude','matched_sign'),5000,False)]
for stage,seeds,arms,steps,continued in stages:
 for seed in seeds:
  if steps==5000:
   t=full.loc[seed];m=t.mean_seconds;md=t.median_seconds;q=t.q90_seconds
  else:
   t=seg[(seg.seed==seed)&(seg.segment==('segment_2500' if continued else 'segment_0'))].iloc[0];m=t['mean'];md=t['median'];q=t.q90
  for arm in arms:
   for mask in ('imp','random1','random2'):
    n=6
    cost.append(dict(stage=stage,seed=seed,arm=arm,mask_type=mask,orders=2,draws=3,run_invocations=n,new_distinct_runs=0 if continued else n,continued_runs=n if continued else 0,attempted_steps_per_invocation=steps,total_attempted_steps=n*steps,mean_seconds_per_invocation=m,median_seconds_per_invocation=md,q90_seconds_per_invocation=q,estimated_segment_hours_mean=n*m/3600,estimated_segment_hours_median=n*md/3600,estimated_segment_hours_q90=n*q/3600,planning_hours_mean_with_15pct_overhead=n*m/3600*1.15))
cost=pd.DataFrame(cost);assert cost.new_distinct_runs.sum()==360 and cost.total_attempted_steps.sum()==1800000
cost.to_csv(OUT/'phase2_cost_estimate.csv',index=False)
cost.groupby('stage')[['run_invocations','new_distinct_runs','continued_runs','total_attempted_steps','estimated_segment_hours_mean','estimated_segment_hours_median','estimated_segment_hours_q90','planning_hours_mean_with_15pct_overhead']].sum().to_csv(OUT/'phase2_staging_cost.csv')
# Measure the retained native first-segment file footprint; do not change retention.
manifest=rows(OLD/'variant_manifest.csv');sizes=[]
for r in manifest:
 if r['reused']=='True':continue
 attempt=Path(read(OLD/r['status_path'])['attempt']);first=0;total=0
 for path in attempt.rglob('*'):
  if not path.is_file():continue
  sz=path.stat().st_size;total+=sz
  rel=path.relative_to(attempt)
  if rel.parts[0]=='segment_0':first+=sz
  elif rel.parts[0]=='snapshots' and path.stem.startswith('step_'):
   token=path.stem.split('_')[1]
   if token.isdigit() and int(token)<=2500 and 'after_reset' not in path.stem:first+=sz
 sizes.append(dict(job_id=r['job_id'],seed=int(r['seed']),first_segment_retained_bytes=first,full_attempt_retained_bytes=total))
sizes=pd.DataFrame(sizes);sizes.to_csv(OUT/'observed_storage.csv',index=False)
fs=os.statvfs(OUT);free=fs.f_bavail*fs.f_frsize
first_est=sum(18*sizes[sizes.seed==s].first_segment_retained_bytes.mean() for s in (7,17,29))
full_est=sum(72*sizes[sizes.seed==s].full_attempt_retained_bytes.mean() for s in SEEDS)
# An extra 15% planning allowance covers variants, RNG/permutation provenance and atomic writes.
write('phase2_storage_estimate.csv',[dict(available_bytes=free,first_stage_native_files_bytes=first_est,first_stage_with_15pct_allowance_bytes=first_est*1.15,full_grid_native_files_bytes=full_est,full_grid_with_15pct_allowance_bytes=full_est*1.15,first_stage_fits_current_free_space=first_est*1.15<free,full_grid_fits_current_free_space=full_est*1.15<free,retention='same native segment files and snapshots; no deletion of existing results')])
print(cost.groupby('stage')[['new_distinct_runs','continued_runs','total_attempted_steps','planning_hours_mean_with_15pct_overhead']].sum().to_string())
print('storage',free,first_est*1.15,full_est*1.15)
