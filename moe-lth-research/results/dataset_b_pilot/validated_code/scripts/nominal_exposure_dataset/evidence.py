"""Existing Dataset A evidence for the prespecified Dataset B design."""
from pilot_common import *
import itertools,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 f=pd.read_csv(AD/'canonical/dataset_a_canonical.csv');d=pd.read_csv(AD/'canonical/dataset_a_dense.csv');out=OUT/'analysis';out.mkdir(exist_ok=True);rows=[];tails=[];timings=[]
 for r in d.to_dict('records'):
  e=int(r['architecture']);p=Path(r['path']);curve=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()];rows.extend(dict(num_experts=e,seed=int(r['seed']),router_age=int(r['router_age']),model_type='dense',sparsity=0.,**z) for z in curve)
 for r in f[f.sparsity.isin(SP)].to_dict('records'):
  p=Path(r['sparse_artifact_path']);curve=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()];rows.extend(dict(num_experts=r['num_experts'],seed=r['seed'],router_age=r['router_age'],model_type='sparse',sparsity=r['sparsity'],**z) for z in curve)
  if r['num_experts']==32:
   start=p/'component_checkpoints/initial_router.pt';end=p/'metadata.json'
   if start.exists():
    seconds=end.stat().st_mtime-start.stat().st_mtime
    if seconds>0:timings.append({'artifact_path':str(p),'seed':r['seed'],'seconds':seconds,'seconds_per_attempt':seconds/2500,'timing_source':'metadata mtime minus initial-router checkpoint mtime; excludes setup and may include interruptions'})
 curves=pd.DataFrame(rows);csv(out/'dataset_a_selected_recovery_curves.csv',curves)
 for key,g in curves.groupby(['num_experts','seed','router_age','model_type','sparsity']):
  z=g.set_index('step').loss.sort_index();late=z.loc[2000:2500];diff=late.diff().dropna();coef=np.polyfit(late.index,late.values,1);resid=late.values-np.polyval(coef,late.index);tails.append(dict(zip(['num_experts','seed','router_age','model_type','sparsity'],key))|{'evaluation_spacing':50,'tail_points':len(late),'improvement_2000_to_2500':z[2000]-z[2500],'improvement_2450_to_2500':z[2450]-z[2500],'decreasing_tail_intervals':int((diff<0).sum()),'tail_intervals':len(diff),'adjacent_difference_sample_sd':diff.std(ddof=1),'detrended_tail_sample_sd':np.std(resid,ddof=1)})
 tail=pd.DataFrame(tails);csv(out/'dataset_a_tail_improvement_by_seed.csv',tail);ts=tail.groupby(['num_experts','router_age','model_type','sparsity']).agg(n_seeds=('seed','nunique'),mean_tail_improvement=('improvement_2000_to_2500','mean'),sd_tail_improvement=('improvement_2000_to_2500','std'),seeds_improving=('improvement_2000_to_2500',lambda x:int((x>0).sum())),mean_last_interval_improvement=('improvement_2450_to_2500','mean'),mean_tail_residual_sd=('detrended_tail_sample_sd','mean')).reset_index();csv(out/'dataset_a_tail_summary.csv',ts)
 zero=f[f.router_age==0].set_index(['num_experts','seed','sparsity']);last=f[f.router_age==100].set_index(['num_experts','seed','sparsity']);paired=last[['sparse_final_validation_loss','dense_final_validation_loss','ticket_gap']]-zero[['sparse_final_validation_loss','dense_final_validation_loss','ticket_gap']];paired=paired.rename(columns=lambda c:'r100_minus_r0_'+c).reset_index();assert np.allclose(paired.r100_minus_r0_ticket_gap,paired.r100_minus_r0_sparse_final_validation_loss-paired.r100_minus_r0_dense_final_validation_loss,rtol=0,atol=1e-10);csv(out/'dataset_a_age_contrasts_paired.csv',paired)
 summary=paired.groupby(['num_experts','sparsity']).agg({c:['mean','std',lambda x:int((x>0).sum())] for c in paired if c.startswith('r100_minus')});summary.columns=['_'.join(c).replace('<lambda_0>','positive_seeds') for c in summary.columns];csv(out/'dataset_a_age_contrasts_summary.csv',summary.reset_index())
 retained=f[['num_experts','seed','router_age','sparsity','mask_keep_count','sparse_final_validation_loss','ticket_gap']];pairs=retained.merge(retained,on=['seed','router_age'],suffixes=('_a','_b'));pairs=pairs[pairs.num_experts_a<pairs.num_experts_b].copy();pairs['relative_retained_count_difference']=(pairs.mask_keep_count_b-pairs.mask_keep_count_a)/pairs.mask_keep_count_a;pairs['sparse_loss_b_minus_a']=pairs.sparse_final_validation_loss_b-pairs.sparse_final_validation_loss_a;pairs['gap_b_minus_a']=pairs.ticket_gap_b-pairs.ticket_gap_a;csv(out/'equal_sparsity_comparisons.csv',pairs[pairs.sparsity_a==pairs.sparsity_b]);matched=pairs[pairs.relative_retained_count_difference.abs()<=.01];csv(out/'approximately_equal_retained_parameter_comparisons.csv',matched)
 plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
 for e in (8,32):
  for late in (False,True):
   fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
   for i,(kind,sp) in enumerate([('dense',0.),('sparse',.75),('sparse',.95)]):
    for j,age in enumerate(AGES):
     ax=axes[i,j];g=curves[(curves.num_experts==e)&(curves.model_type==kind)&(curves.sparsity==sp)&(curves.router_age==age)]
     if late:g=g[g.step>=2000]
     for seed,h in g.groupby('seed'):ax.plot(h.step,h.loss,lw=.6,alpha=.5,label=f'Seed {seed}')
     z=g.groupby('step').loss.agg(['mean','std']);ax.plot(z.index,z['mean'],'k-',lw=1.2,label='Mean');ax.fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],color='gray',alpha=.15);ax.set_title(f'{kind} {sp:.0%}, R{age}');ax.set_xlabel('Recovery attempts');ax.set_ylabel('CE, nats/token')
     if not late:ax.set_yscale('log')
   axes[0,0].legend(fontsize=6);fig.suptitle(f'Dataset A {e}E — '+('late recovery, mean ± seed SD' if late else 'full curves, logarithmic loss axis'))
   for ext in ('pdf','png'):
    p=out/f'recovery_{e}E_{"tail" if late else "full"}.{ext}';q=p.with_suffix('.'+ext+'.tmp');fig.savefig(q,format=ext,dpi=220);q.replace(p)
   plt.close(fig)
 csv(out/'historical_32e_runtime_samples.csv',timings);times=pd.DataFrame(timings);med=float(times.seconds_per_attempt.median());quant=[float(x) for x in times.seconds_per_attempt.quantile([.1,.9])];attempt_total=20*2500+45*10000+5*10000+45*2500+65*10000 # A random, C, long source, B, D incl random
 estimate={'new_jobs':180,'reused_jobs':45,'new_recovery_jobs':175,'new_source_jobs':5,'total_attempts':attempt_total,'historical_seconds_per_attempt_median':med,'historical_seconds_per_attempt_p10_p90':quant,'estimated_gpu_hours_median':attempt_total*med/3600,'estimated_gpu_hours_p10_p90':[attempt_total*q/3600 for q in quant],'caveat':'mtime-derived on existing hardware, includes some logging and probe cost; source timing approximated by recovery. Queue waiting excluded. No runtime guarantee.','expected_storage_gib':{'recovery_final_model_checkpoints':175*0.26,'source_checkpoints':5*5*.26,'masks_and_components':3,'compact_per_expert_logs_and_reports':20,'reserve':30},'storage_location':str(OUT),'no_checkpoint_deletion_policy':'Keep final recoveries and all 5 source checkpoint steps; interrupted attempts preserved; investigate storage gate rather than deleting A.'};jsave(OUT/'audits/resource_estimate.json',estimate)
 selected=ts[(ts.num_experts.isin([8,32]))&ts.router_age.isin(AGES)]
 def md(df):return df.to_csv(index=False)
 pairs_unique=matched[['num_experts_a','sparsity_a','num_experts_b','sparsity_b']].drop_duplicates()
 text=f'''# Existing evidence before Dataset B\n\nDataset A canonical rows: {len(f)}. Independent source seeds are replication units. All six sparsities appear in paired R100−R0 tables; sparse loss, dense loss and gap are separate. No results change the prespecified A/B/C/D contrasts.\n\n## Recovery endpoint\n\nValidation spacing is 50 attempts on the fixed validation set. Tail improvement is loss at 2000 minus loss at 2500; positive means continued improvement. Adjacent differences and residual SD around a descriptive linear tail trend measure observed variation, not independently sampled validation noise. A flat segment is not proof of convergence.\n\n```csv\n{md(selected)}\n```\n\n## Router age × sparsity\n\nSee all 30 architecture/sparsity rows in `dataset_a_age_contrasts_summary.csv` and all 150 paired seed rows. Gap change equals sparse loss change minus dense loss change; repeated grid cells are not independent replicates.\n\n## Retained capacity\n\nCounts are actual stored-mask ones, not nominal density estimates. The approximately equal-retained comparison uses a declared 1% relative count tolerance and the same seed and age. Matching retained parameters does not match active top-1 compute, routing, shared parameters, optimization or convergence. Available architecture/sparsity combinations:\n\n```csv\n{md(pairs_unique)}\n```\n\nThese descriptive comparisons motivate the controlled follow-up but do not establish capacity limits or a router optimum.\n''';atomic(out/'dataset_a_evidence.md',text)
 print('EVIDENCE COMPLETE; runtime estimate',estimate['estimated_gpu_hours_median'],'GPU hours',flush=True)
if __name__=='__main__':main()
