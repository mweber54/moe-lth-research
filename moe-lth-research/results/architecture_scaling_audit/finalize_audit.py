import json,csv,hashlib
from pathlib import Path
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;R=O.parents[1]
def df(n):return pd.read_csv(O/n)
def save(n,x):x.to_csv(O/n,index=False,na_rep='NA')
def table(d):
 return '| '+' | '.join(map(str,d.columns))+' |\n|'+'|'.join('---' for _ in d.columns)+'|\n'+'\n'.join('| '+' | '.join('NA' if pd.isna(v) else f'{v:.6g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
samp=df('sampled_source_exposure.csv');end=samp[samp.router_age==100];cap=df('capacity_audit.csv');rec=df('recovery_exposure.csv');disp=df('expert_parameter_displacement.csv');late=df('late_training_slopes.csv');pairs=df('dense_control_audit.csv');mask=df('mask_audit.csv');mapping=df('checkpoint_mapping_audit.csv');gap=df('24e_ticket_gap_recomputation.csv');summary=df('24e_age_summary_independent.csv')
# Explicit NA columns for unavailable complete source distributions.
actual=df('actual_expert_exposure.csv')
for col in ('accepted_assignments_per_expert','median_accepted_per_expert','min_accepted_per_expert','max_accepted_per_expert','sd_accepted_per_expert','cv_accepted_per_expert','fraction_assignments_per_expert','zero_token_steps_per_expert','nonzero_gradient_steps','cumulative_gradient_norm','cumulative_update_norm','accepted_assignments_per_expert_parameter'):
 actual[col]=np.where(actual.router_age==0,0,np.nan)
save('actual_expert_exposure.csv',actual)
er=df('recovery_expert_exposure.csv')
extremes=er.groupby(['experts','kind']).agg(min_active_steps=('active_applied_steps','min'),median_active_steps=('active_applied_steps','median'),mean_active_steps=('active_applied_steps','mean'),max_active_steps=('active_applied_steps','max'),fraction_below_half_budget=('active_applied_steps',lambda x:(x<1250).mean()),zero_token_attempts_mean=('zero_token_attempts','mean')).reset_index()
save('recovery_active_step_extremes.csv',extremes)
rs=er.groupby(['experts','seed','router_age','kind']).accepted.agg(accepted_median='median',accepted_min='min',accepted_max='max',accepted_sd=lambda x:np.std(x,ddof=0),accepted_cv=lambda x:np.std(x,ddof=0)/np.mean(x)).reset_index()
rec=rec.drop(columns=[c for c in rs if c.startswith('accepted_') and c in rec.columns],errors='ignore').merge(rs,on=['experts','seed','router_age','kind'])
save('recovery_exposure.csv',rec)
checks=[]
for row in gap.to_dict('records'):
 for kind in ('dense','sparse'):
  p=R/f"results/24e_router_age_localization/recovery/seed_{row['seed']}/R{row['router_age']}/{kind}/metrics.jsonl";entries=[json.loads(x) for x in p.read_text().splitlines()];last=entries[-1]
  checks.append(dict(seed=row['seed'],router_age=row['router_age'],kind=kind,raw_step=last['step'],raw_final_loss=last['loss'],summary_final_loss=row[kind+'_final_loss'],matches=last['step']==2500 and abs(last['loss']-row[kind+'_final_loss'])<1e-12))
save('raw_loss_crosscheck.csv',pd.DataFrame(checks))
old=pd.read_csv(R/'results/24e_router_age_localization/age_summary.csv');print('OLD SUMMARY COLUMNS',old.columns.tolist())
merged=summary.merge(old,on='router_age',suffixes=('_audit','_published'))
for f in ('mean_ticket_gap','sd_ticket_gap'):
 assert np.allclose(merged[f+'_audit'],merged[f+'_published'],rtol=0,atol=1e-12)
assert all(x['matches'] for x in checks)
assert mapping.status.eq('PASS').all()
for key in ('router_hash','expert_hash','shared_hash'):
 assert not mapping.duplicated(['seed',key]).any(),key
original_manifest=pd.read_csv(R/'results/24e_router_age_localization/checkpoint_manifest.csv');joined=original_manifest.merge(mapping,on=['seed','router_age'],suffixes=('_historical','_audit'))
assert len(joined)==27
for g in ('router','expert','shared'):assert (joined[g+'_hash_historical']==joined[g+'_hash_audit']).all()
assert (joined.checkpoint_hash==joined.sha256).all()
assert mask.drop(columns=['experts','seed','router_age','target_sparsity','actual_weight_sparsity','mask_hash','mask_source','rewind']).eq(True).all().all()
boolean_cols=[c for c in pairs if c.endswith('_matched') or c.startswith(('dense_audit','sparse_audit')) or c in ('fresh_optimizer','matched_dense_loss')]
assert pairs[pairs.experts==24][boolean_cols].eq(True).all().all()
assert rec.complete_counters.all()
assert (rec.proposed_assignments==rec.potential_assignments).all()
assert cap.max_reconstruction_error.max()<1e-7
save('checkpoint_manifest_crosscheck.csv',joined)
# Aggregate seeds equally; recovery comparison restricts all architectures to common ages 0/20/100.
source=end.groupby('experts').sampled_accepted_mean.mean();source_ratio=source/source.loc[8]
relative=pd.DataFrame({'experts':source.index,'EXPECTED_FROM_SIMPLE_BALANCE':[8/e for e in source.index],'ACTUAL_OR_BEST_AVAILABLE':source_ratio.values,'DEVIATION':source_ratio.values-np.array([8/e for e in source.index]),'measurement':'sampled accepted source assignments; complete source ratio NA'})
save('source_relative_exposure.csv',relative)
common=rec[rec.router_age.isin([0,20,100])];rg=common.groupby(['experts','kind']).agg(accepted_per_expert=('accepted_per_expert','mean'),accepted_assignments=('accepted_assignments','mean'),active_applied_steps=('active_applied_steps_mean','mean'),loss=('final_validation_loss','mean')).reset_index()
rg['relative_to_8E']=rg.apply(lambda x:x.accepted_per_expert/rg[(rg.experts==8)&(rg.kind==x.kind)].accepted_per_expert.iloc[0],axis=1);rg['exposure_matched_budget_multiplier']=1/rg.relative_to_8E;save('recovery_scaling_common_ages.csv',rg)
rr=rec.pivot(index=['experts','seed','router_age'],columns='kind',values='accepted_assignments');rr['sparse_dense_accepted_ratio']=rr['sparse']/rr['dense'];save('recovery_pair_exposure.csv',rr.reset_index())
params=df('architecture_parameter_counts.csv').groupby('experts').first();vl=late[late.start_age==80].groupby('experts').end_loss.mean()
primary=pd.DataFrame([{'Experts':e,'Total tokens':40960000,'Total accepted assignments':'NA','Mean accepted/expert':'NA','Relative exposure vs 8E':'NA','Active steps/expert':'NA','Expert params':int(params.loc[e,'expert_parameters']),'Dense R100 loss':vl.loc[e]} for e in (8,12,16,24)])
save('primary_source_exposure.csv',primary)
# Same-seed within-age slopes of recovery late windows, without executing recovery.
rsl=[]
for x in rec.to_dict('records'):
 path=Path(x['source'])/'metrics.jsonl';m={z['step']:z['loss'] for z in map(json.loads,path.read_text().splitlines())}
 for st in (1500,2000):
  if st in m:rsl.append(dict(experts=x['experts'],seed=x['seed'],router_age=x['router_age'],kind=x['kind'],start=st,end=2500,absolute_improvement=m[st]-m[2500],relative_improvement=(m[st]-m[2500])/m[st]))
save('recovery_late_slopes.csv',pd.DataFrame(rsl))
# H4, descriptive same-seed covariance across ages (not independent replicates).
h4=[]
for seed,g in gap.groupby('seed'):
 h4.append(dict(seed=seed,sparse_dense_correlation=g.sparse_final_loss.corr(g.dense_final_loss),dense_range=g.dense_final_loss.max()-g.dense_final_loss.min(),sparse_range=g.sparse_final_loss.max()-g.sparse_final_loss.min(),gap_range=g.ticket_gap.max()-g.ticket_gap.min()))
save('comparator_covariation.csv',pd.DataFrame(h4))
print('SOURCE',relative.to_string(index=False));print('CAPACITY',cap.groupby('experts').sampled_source_drop_rate.mean().to_string());print('LATE',late[late.start_age==80].to_string(index=False));print('DISPLACEMENT',disp[(disp.router_age==100)&(disp.component=='expert')].groupby('experts')[['absolute_mean','relative_mean']].mean().to_string());print('RECOVERY',rg.to_string(index=False));print('PAIR RATIOS',rr.groupby('experts').sparse_dense_accepted_ratio.agg(['min','mean','max']).to_string());print('SUMMARY',summary.to_string(index=False));print('H4',h4)
# Save report inputs for a concise, evidence-grounded report.
(O/'report_statistics.json').write_text(json.dumps(dict(source_relative=relative.to_dict('records'),source_drop=cap.groupby('experts').sampled_source_drop_rate.mean().to_dict(),displacement=disp[(disp.router_age==100)&(disp.component=='expert')].groupby('experts')[['absolute_mean','relative_mean']].mean().to_dict(),recovery=rg.to_dict('records'),pair_ratio=rr.groupby('experts').sparse_dense_accepted_ratio.agg(['min','mean','max']).to_dict(),late=late[late.start_age==80].to_dict('records'),H4=h4),indent=2))
