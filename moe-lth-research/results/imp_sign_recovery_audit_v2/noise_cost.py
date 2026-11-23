from common import *
d=pd.read_csv(OLD/'realization_level_effects.csv')
assert set(d.seed)==set(SEEDS) and set(d.step)==set(STEPS)
d.to_csv(OUT/'legacy_penalties.csv',index=False)
wide=d.pivot(index=['seed','order','step','realization'],columns='family',values='penalty').reset_index();wide['paired_interaction']=wide.imp-wide.random
wide['replication_note']='legacy random realizations differ in masks, not repeated shuffles within one mask'
wide.to_csv(OUT/'legacy_paired_interactions.csv',index=False)
# Orders are averaged within each realization before any seed-level spread is computed.
z=wide.groupby(['seed','step','realization'])[['imp','random','paired_interaction']].mean().reset_index()
z.to_csv(OUT/'legacy_order_averaged_realizations.csv',index=False)
bys=z.groupby(['seed','step']).agg(random_mean=('random','mean'),random_sd=('random','std'),random_min=('random','min'),random_max=('random','max'),interaction_mean=('paired_interaction','mean'),interaction_draw_sd=('paired_interaction','std'),interaction_min=('paired_interaction','min'),interaction_max=('paired_interaction','max'),paired_realizations=('realization','nunique')).reset_index();bys.to_csv(OUT/'noise_floor_by_seed.csv',index=False)
wide.groupby(['seed','order','step'])[['random','paired_interaction']].agg(['mean','std','min','max']).to_csv(OUT/'noise_floor_by_seed_order.csv')
z=z.merge(bys[['seed','step','interaction_mean']],on=['seed','step']);z['interaction_within_seed_deviation']=z.paired_interaction-z.interaction_mean
floor=[]
for step,g in z.groupby('step'):
 seeds=bys[bys.step==step];center=seeds.random_mean.mean()
 floor.append(dict(step=int(step),source_seeds=5,order_averaged_paired_realizations=len(g),random_mean=g.random.mean(),random_sd=g.random.std(),random_min=g.random.min(),random_max=g.random.max(),random_q05=g.random.quantile(.05),random_q95=g.random.quantile(.95),random_abs_q95=g.random.abs().quantile(.95),random_seed_mean_sd=seeds.random_mean.std(),random_seed_centered_abs_q95=(seeds.random_mean-center).abs().quantile(.95),interaction_mean=seeds.interaction_mean.mean(),interaction_across_seed_sd=seeds.interaction_mean.std(),interaction_across_seed_min=seeds.interaction_mean.min(),interaction_across_seed_max=seeds.interaction_mean.max(),interaction_within_seed_abs_deviation_q95=g.interaction_within_seed_deviation.abs().quantile(.95),interaction_within_seed_sd_mean=seeds.interaction_draw_sd.mean(),interaction_within_seed_sd_max=seeds.interaction_draw_sd.max(),legacy_imp_draws_per_seed=2,legacy_shuffle_draws_per_random_mask=1,legacy_random_masks_per_seed=2))
floor=pd.DataFrame(floor);floor.to_csv(OUT/'noise_floor_per_step.csv',index=False)
post=floor[floor.step>0]
components={c:float(post[c].max()) for c in ['random_abs_q95','random_seed_centered_abs_q95','interaction_within_seed_abs_deviation_q95']}
X=max(components.values());Y=2*X
write('decision_thresholds.csv',[dict(near_zero_X=X,reopens_Y=Y,required_seeds_K=4,total_seeds=5,stage1_required_seeds=2,stage1_total_seeds=3,draws=3,stage1_step=2000,about_as_much_margin=X,matched_sign_minus_magnitude_margin=X,threshold_formula='X=max over post-init logged steps of q95 absolute random penalties, q95 centered seed-mean random penalties, q95 within-seed paired-interaction deviations; Y=2X',**components)])
# Observed segment wall time is a planning proxy for GPU time, not measured GPU-active time.
a=pd.read_csv(OLD/'optimizer_compute_accounting.csv');a=a[~a.reused.astype(str).str.lower().eq('true')]
runtime=a.groupby(['job_id','seed','order','variant']).seconds.sum().reset_index().rename(columns={'seconds':'segment_wall_seconds_per_5000_attempts'})
manifest=pd.read_csv(OLD/'variant_manifest.csv');hosts=[]
for _,r in runtime.iterrows():
 status=read(OLD/f'status/{r.job_id}.json');hosts.append(status['host'])
runtime['host']=hosts;runtime.to_csv(OUT/'observed_runtime.csv',index=False)
rt=runtime.groupby(['seed','host']).segment_wall_seconds_per_5000_attempts.agg(['count','mean','median','min','max',lambda v:v.quantile(.9)]).reset_index();rt.columns=['seed','host','observed_runs','mean_seconds','median_seconds','min_seconds','max_seconds','q90_seconds'];rt.to_csv(OUT/'runtime_by_seed.csv',index=False)
cost=[]
for stage,seeds,arms in [('1_sign_only',(7,17,29),('sign',)),('2_first_three_remaining_arms',(7,17,29),('full_shuffle_rerun','magnitude','matched_sign')),('3_last_two_all_arms',(43,61),('full_shuffle_rerun','sign','magnitude','matched_sign'))]:
 for seed in seeds:
  t=rt[rt.seed==seed].iloc[0]
  for arm in arms:
   for mask_type in ('imp','random1','random2'):
    runs=2*3
    cost.append(dict(stage=stage,seed=seed,arm=arm,mask_type=mask_type,orders=2,draws=3,runs=runs,steps_per_run=5000,total_attempted_steps=runs*5000,reference_host=t.host,mean_seconds_per_run=t.mean_seconds,median_seconds_per_run=t.median_seconds,q90_seconds_per_run=t.q90_seconds,estimated_segment_hours_mean=runs*t.mean_seconds/3600,estimated_segment_hours_median=runs*t.median_seconds/3600,estimated_segment_hours_q90=runs*t.q90_seconds/3600,planning_overhead_fraction=.15,planning_hours_mean_with_overhead=runs*t.mean_seconds/3600*1.15,reference='observed new jobs; excludes reused compute; elapsed segment time proxy'))
cost=pd.DataFrame(cost);cost.to_csv(OUT/'phase2_cost_estimate.csv',index=False)
cost.groupby('stage')[['runs','total_attempted_steps','estimated_segment_hours_mean','estimated_segment_hours_median','estimated_segment_hours_q90','planning_hours_mean_with_overhead']].sum().to_csv(OUT/'phase2_staging_cost.csv')
print('THRESHOLDS',X,Y);print(cost.groupby('stage')[['runs','planning_hours_mean_with_overhead']].sum().to_string());print('TOTAL RUNS',cost.runs.sum())
