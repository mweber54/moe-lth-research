import sys
sys.path.insert(0,'results/router_gate_factorial_v1')
from support import *
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 endpoint=pd.read_csv(OUT/'full_factorial_results.csv');co=pd.read_csv(OUT/'router_age_contrasts.csv');at=pd.read_csv(OUT/'per_seed_attenuation.csv')
 ce_effects=[];gap_effects=[];components=[]
 for readout in ('primary','native','assignment'):
  for seed in (7,17,29):
   selected=endpoint[endpoint.seed==seed]
   for arm in ARMS:
    cell=selected[selected.arm==arm].set_index(['age','kind'])[readout+'_loss'];sparse_age=cell[20,'sparse']-cell[100,'sparse'];dense_age=cell[20,'dense']-cell[100,'dense'];components.append(dict(readout=readout,seed=seed,arm=arm,sparse_age_CE=sparse_age,dense_age_CE=dense_age,age_contrast=sparse_age-dense_age))
   def effects(values):
    A,B,C,D=[values[a] for a in ARMS];return dict(value_native_J=C-A,detach_native_values=D-A,combined=B-A,interaction=B-C-D+A,marginal_value=.5*((C-A)+(B-D)),marginal_detach=.5*((D-A)+(B-C)))
   for age in (20,100):
    for kind in ('dense','sparse'):
     vals=selected[(selected.age==age)&(selected.kind==kind)].set_index('arm')[readout+'_loss']
     for effect,value in effects(vals).items():ce_effects.append(dict(readout=readout,seed=seed,age=age,kind=kind,effect=effect,nats=value))
    vals=co[(co.readout==readout)&(co.seed==seed)].set_index('arm')['gap_R'+str(age)]
    for effect,value in effects(vals).items():gap_effects.append(dict(readout=readout,seed=seed,age=age,effect=effect,nats=value))
 pd.DataFrame(ce_effects).to_csv(OUT/'factorial_effects_on_condition_CE.csv',index=False);pd.DataFrame(gap_effects).to_csv(OUT/'factorial_effects_on_ticket_gap.csv',index=False);pd.DataFrame(components).to_csv(OUT/'age_effect_components.csv',index=False)
 gate=[]
 for arm in ('C','D'):
  for seed in (7,17,29):
   for age in (20,100):
    for kind in ('dense','sparse'):
     g=pd.read_csv(OUT/f'measurements/{arm}/seed{seed}/R{age}/{kind}/gates.csv')
     for k,v in dict(arm=arm,seed=seed,age=age,kind=kind).items():g[k]=v
     assert not g.maximum_forward_error.any();gate.append(g)
 gd=pd.concat(gate);gd.to_csv(OUT/'forward_gate_diagnostics.csv',index=False)
 budget=pd.read_csv(OUT/'recovery_budgets.csv');budget['source']='new'
 old=[]
 for arm,p in [('A',ROOT/'results/cross_arch_optimization_geometry/fixed_schedule_intervention/replay_audit.csv'),('B',BASE/'replay_audit.csv')]:
  a=pd.read_csv(p);a=a[(a.experts==8)&a.age.isin([20,100])]
  for _,r in a.iterrows():old.append(dict(arm=arm,seed=int(r.seed),age=int(r.age),kind=r.kind,attempted_updates=int(r.training_forwards),applied_updates=int(r.optimizer_steps_applied),AMP_skips=int(r.amp_overflow_steps),seconds=None,source='audited_reuse'))
 allbudget=pd.concat([pd.DataFrame(old),budget]);assert len(allbudget)==48;assert (allbudget.attempted_updates==2500).all();assert (allbudget.applied_updates+allbudget.AMP_skips==2500).all();allbudget.to_csv(OUT/'all_factorial_budgets.csv',index=False)
 aggregate=[]
 for readout in ('primary','native','assignment'):
  a=co[co.readout==readout].groupby('arm').age_contrast.mean()
  for arm in ARMS:aggregate.append(dict(readout=readout,arm=arm,mean_age_contrast=a[arm],attenuation_of_mean_contrasts=1-abs(a[arm])/(abs(a['A'])+1e-12),mean_of_seed_attenuations=at[(at.readout==readout)&(at.arm==arm)].absolute_attenuation.mean()))
 pd.DataFrame(aggregate).to_csv(OUT/'aggregate_attenuation_estimands.csv',index=False)
 schedule=read(OUT/'execution_scheduling.json');pause=schedule['events'][0]['utc_seconds'];resumed={e['resumed_pid']:e['utc_seconds'] for e in schedule['events'] if 'resumed_pid' in e};runtime=budget.copy();runtime['paused_seconds']=0.
 for seed,pid in [(17,3229179),(29,3229183)]:
  mask=(runtime.seed==seed)&(runtime.arm=='C')&(runtime.age==100)&(runtime.kind=='dense');runtime.loc[mask,'paused_seconds']=resumed[pid]-pause
 runtime['active_process_wall_seconds']=runtime.seconds-runtime.paused_seconds;runtime.to_csv(OUT/'runtime_accounting.csv',index=False)
 # Full-range primary trajectory figure. Keep the predeclared limited-axis rendering as a labeled zoom.
 traj=pd.read_csv(OUT/'observed_primary_trajectories.csv');p=traj.pivot_table(index=['arm','seed','step'],columns=['age','kind'],values='primary_loss');p['age_contrast']=(p[20,'sparse']-p[20,'dense'])-(p[100,'sparse']-p[100,'dense']);cu=p['age_contrast'].dropna().reset_index()
 fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
 for ax,seed in zip(axes,(7,17,29)):
  for arm in ARMS:
   q=cu[(cu.seed==seed)&(cu.arm==arm)&cu.step.isin([0,100,250,500,750,1000,1500,2000,2500])];ax.plot(q.step,q.age_contrast,label=arm,marker='.',lw=1)
  ax.set_title(f'Seed {seed}');ax.set_xlabel('Attempted updates');ax.axhline(0,c='gray',lw=.5)
 axes[0].set_ylabel('Router-age contrast (nats/token)');axes[-1].legend();fig.tight_layout()
 for suffix in ('png','pdf'):fig.savefig(OUT/f'figures/primary_contrasts_full_range.{suffix}',dpi=180)
 plt.close(fig)
 diag=pd.read_csv(OUT/'gradient_diagnostics.csv');fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,field in zip(axes,('router_task_norm','router_aux_norm','expert_task_norm')):
  for arm in ('C','D'):
   s=diag[diag.arm==arm].groupby('step')[field].mean();ax.plot(s.index,s.values,marker='.',label=arm)
  ax.set_title(field);ax.set_xlabel('Attempted updates');ax.set_ylabel('Mean fixed-probe L2 norm');ax.legend()
 fig.tight_layout()
 for suffix in ('png','pdf'):fig.savefig(OUT/f'figures/gradient_policies.{suffix}',dpi=180)
 plt.close(fig)
 from preservation import audit_all
 preservation=audit_all()
 preservation.update(original_audit_inputs_verified=len(read(OUT/'audited_reuse_hash_verification.json')['files']))
 write(OUT/'preservation_audit.json',preservation)
 print('All factorial budgets, forward diagnostics, aggregate estimands and original-input preservation passed.')
if __name__=='__main__':main()
