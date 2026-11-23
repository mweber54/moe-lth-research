"""Predeclared seed-level factorial estimands; no token-level inference."""
from factorial import *
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 endpoint=[];budgets=[];trajectories=[];diagnostic=[];updates=[]
 old=pd.read_csv(BASE/'outcomes.csv');old=old[(old.experts==8)&old.age.isin([20,100])]
 for _,r in old.iterrows():
  for arm,level in [('A','assign'),('B','gate')]:
   for kind in ('dense','sparse'):
    endpoint.append(dict(arm=arm,seed=int(r.seed),age=int(r.age),kind=kind,primary_loss=r[f'{level}_fixedgate_{kind}_loss'],native_loss=r[f'{level}_native_{kind}_loss'],assignment_loss=r[f'{level}_assign_{kind}_loss'],source='audited_reuse',source_file=str(BASE/'outcomes.csv')))
 oldcurves=pd.read_csv(BASE/'common_readout_snapshot_losses.csv')
 for _,r in oldcurves[(oldcurves.experts==8)&oldcurves.age.isin([20,100])&oldcurves.level.isin(['assign','gate'])].iterrows():trajectories.append(dict(arm='A' if r.level=='assign' else 'B',seed=int(r.seed),age=int(r.age),kind=r.kind,step=int(r.step),primary_loss=r.loss))
 for arm in ('C','D'):
  for seed in (7,17,29):
   for age in (20,100):
    for kind in ('dense','sparse'):
     tag=f'{arm}/seed{seed}/R{age}/{kind}';job=OUT/'recoveries'/tag;extra=OUT/'measurements'/tag
     cp=read(job/'complete.json');summary=read(job/'summary.json');meta=read(job/'metadata.json');readouts=pd.read_csv(extra/'readouts.csv');last=readouts[readouts.step==2500].iloc[0]
     assert cp['replay']['counts']=={'train':2500,'validation':1581}
     assert cp['replay']['all_assignment_and_acceptance_checks_passed']
     endpoint.append(dict(arm=arm,seed=seed,age=age,kind=kind,primary_loss=summary['final_validation_loss'],native_loss=last.native_loss,assignment_loss=last.assignment_loss,source='new_recovery',source_file=str(job/'complete.json')))
     budgets.append(dict(arm=arm,seed=seed,age=age,kind=kind,attempted_updates=meta['optimizer_step_attempts'],applied_updates=meta['optimizer_steps_applied'],AMP_skips=meta['amp_overflow_step_count'],seconds=cp['elapsed_seconds']))
     for line in (job/'metrics.jsonl').read_text().splitlines():
      r=json.loads(line);trajectories.append(dict(arm=arm,seed=seed,age=age,kind=kind,step=r['step'],primary_loss=r['loss']))
     for filename,collector in [('diagnostics.csv',diagnostic),('updates.csv',updates)]:
      df=pd.read_csv(extra/filename)
      for col,val in dict(arm=arm,seed=seed,age=age,kind=kind).items():df[col]=val
      collector.append(df)
     mutation=pd.read_csv(extra/'mutation_audit.csv');assert len(mutation)==9 and mutation.drop(columns='step').all().all()
 df=pd.DataFrame(endpoint);assert len(df)==48;df.to_csv(OUT/'full_factorial_results.csv',index=False)
 pd.DataFrame(budgets).to_csv(OUT/'recovery_budgets.csv',index=False)
 pd.concat(diagnostic).to_csv(OUT/'gradient_diagnostics.csv',index=False);pd.concat(updates).to_csv(OUT/'optimizer_update_diagnostics.csv',index=False)
 contrasts=[]
 for readout in ('primary','native','assignment'):
  for arm in ARMS:
   for seed in (7,17,29):
    cell=df[(df.arm==arm)&(df.seed==seed)].set_index(['age','kind'])[readout+'_loss']
    g20=cell[20,'sparse']-cell[20,'dense'];g100=cell[100,'sparse']-cell[100,'dense']
    contrasts.append(dict(readout=readout,arm=arm,seed=seed,gap_R20=g20,gap_R100=g100,age_contrast=g20-g100))
 co=pd.DataFrame(contrasts);co.to_csv(OUT/'router_age_contrasts.csv',index=False)
 attenuation=[];effects=[]
 for readout in ('primary','native','assignment'):
  for seed in (7,17,29):
   s=co[(co.readout==readout)&(co.seed==seed)].set_index('arm').age_contrast;A,B,C,D=(s[a] for a in ARMS)
   for arm in ARMS:attenuation.append(dict(readout=readout,seed=seed,arm=arm,baseline=A,age_contrast=s[arm],absolute_attenuation=1-abs(s[arm])/(abs(A)+1e-12),signed_reduction=A-s[arm],near_zero_baseline=abs(A)<1e-5))
   for name,v in dict(value_native_J=C-A,detach_native_values=D-A,combined=B-A,interaction=B-C-D+A,marginal_value=.5*((C-A)+(B-D)),marginal_detach=.5*((D-A)+(B-C))).items():effects.append(dict(readout=readout,seed=seed,effect=name,nats=v))
 at=pd.DataFrame(attenuation);at.to_csv(OUT/'per_seed_attenuation.csv',index=False)
 ef=pd.DataFrame(effects);ef.to_csv(OUT/'factorial_effects_per_seed.csv',index=False)
 agg=ef.groupby(['readout','effect']).nats.agg(['mean','std','min','max']);agg.to_csv(OUT/'factorial_effects_aggregate.csv')
 at.groupby(['readout','arm']).absolute_attenuation.agg(['mean','std','min','max']).to_csv(OUT/'attenuation_aggregate.csv')
 co.groupby(['readout','arm']).age_contrast.agg(['mean','std','min','max']).to_csv(OUT/'contrasts_aggregate.csv')
 decisions=[]
 for effect in ('value_native_J','detach_native_values'):
  p=ef[(ef.readout=='primary')&(ef.effect==effect)].sort_values('seed').nats.to_numpy();n=ef[(ef.readout=='native')&(ef.effect==effect)].sort_values('seed').nats.to_numpy()
  decisions.append(dict(effect=effect,primary_values=p.tolist(),native_values=n.tolist(),same_direction_all_three=bool(np.all(p>0) or np.all(p<0)),exceeds_predeclared_floor_all_three=bool(np.all(p>.02) or np.all(p<-.02)),native_effect_sign_reverses_all_three=bool(np.all(p*n<0))))
 write(OUT/'stopping_rule_evaluation.json',dict(screen_complete=True,new_recoveries=24,isolated_effects=decisions,fresh_seeds_launched=0))
 traj=pd.DataFrame(trajectories);traj.to_csv(OUT/'observed_primary_trajectories.csv',index=False)
 curves=traj.pivot_table(index=['arm','seed','step'],columns=['age','kind'],values='primary_loss');curves['age_contrast']=(curves[20,'sparse']-curves[20,'dense'])-(curves[100,'sparse']-curves[100,'dense']);curves=curves['age_contrast'].dropna().reset_index()
 figs=OUT/'figures';figs.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
 for ax,seed in zip(axes,(7,17,29)):
  for arm in ARMS:
   q=curves[(curves.seed==seed)&(curves.arm==arm)&(curves.step.isin([0,100,250,500,750,1000,1500,2000,2500]))];ax.plot(q.step,q.age_contrast,label=arm,marker='.',lw=1)
  ax.axhline(0,color='gray',lw=.6);ax.set_title(f'Seed {seed}');ax.set_xlabel('Attempted recovery updates');ax.set_ylim(-.5,.5)
 axes[0].set_ylabel('Router-age contrast (nats/token)');axes[-1].legend();fig.tight_layout()
 for suffix in ('png','pdf'):fig.savefig(figs/f'primary_contrasts.{suffix}',dpi=180)
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=False)
 for ax,readout in zip(axes,('primary','native','assignment')):
  for seed in (7,17,29):
   q=co[(co.seed==seed)&(co.readout==readout)].set_index('arm').loc[list(ARMS)];ax.plot(list(ARMS),q.age_contrast,marker='o',label=str(seed))
  ax.axhline(0,color='gray',lw=.6);ax.set_title(readout+' readout');ax.set_xlabel('Factorial training arm')
 axes[0].set_ylabel('Router-age contrast (nats/token)');axes[-1].legend(title='Seed');fig.tight_layout()
 for suffix in ('png','pdf'):fig.savefig(figs/f'readout_sensitivity.{suffix}',dpi=180)
 plt.close(fig)
 from preservation import audit_all
 preservation=audit_all()
 write(OUT/'analysis_preservation_check.json',preservation)
 write(OUT/'analysis_complete.json',dict(complete=True,conditions=48,new_recoveries=24,scientific_inputs_unchanged=True,all_prior_bytes_identical=preservation['all_prior_bytes_identical'],outputs={p.name:file_hash(p) for p in OUT.glob('*.csv')}))
 print(agg.to_string());print(json.dumps(decisions,indent=2))
if __name__=='__main__':main()
