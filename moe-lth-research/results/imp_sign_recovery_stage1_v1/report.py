from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'results/imp_sign_recovery_audit_v2'))
from common import pd,np,rows,read,sha,p
from gate import seed_result,interim_gate
import json,csv,datetime,subprocess

def table(name,data):
 data=list(data)
 with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
 manifest=rows(OUT/'job_manifest.csv');assert len(manifest)==54
 baseline=pd.read_csv(p.OUT/'trajectory_losses.csv');comparators=pd.read_csv(ROOT/'results/imp_sign_recovery_audit_v2/legacy_unpruned_comparator.csv')
 allrows=[];account=[]
 for job in manifest:
  status=read(OUT/'status'/f"{job['job_id']}.json");assert status['status']=='COMPLETE' and status['attempted_updates']==2500
  path=Path(status['attempt']);ledger=[json.loads(l) for l in (path/'attempt_ledger.jsonl').read_text().splitlines()];assert [r['attempt'] for r in ledger]==list(range(1,2501));assert not (path/'segment_2500').exists()
  original='imp_original' if job['mask_type']=='imp' else job['mask_type']+'_original'
  for ev in [json.loads(l) for l in (path/'common_evaluations.jsonl').read_text().splitlines()]:
   seed=int(job['seed']);order=job['order'];step=ev['step'];ref=baseline[(baseline.seed==seed)&(baseline.order==order)&(baseline.variant==original)&(baseline.step==step)]
   assert len(ref)==1 and step<=2500
   comp=comparators[(comparators.seed==seed)&(comparators.order==order)&(comparators.step==step)].iloc[0].unpruned_CE
   allrows.append(dict(seed=seed,order=order,draw=int(job['draw']),mask_type=job['mask_type'],step=step,CE=ev['cross_entropy'],original_CE=float(ref.cross_entropy.iloc[0]),delta_CE=ev['cross_entropy']-float(ref.cross_entropy.iloc[0]),unpruned_CE=comp,realized_distance=float(job['realized_distance']),frac_sign_flipped=float(job['frac_sign_flipped']),job_id=job['job_id']))
  account.append(dict(job_id=job['job_id'],seed=int(job['seed']),order=job['order'],draw=int(job['draw']),mask_type=job['mask_type'],host=status['host'],attempted_updates=status['attempted_updates'],applied_updates=status['applied_updates'],amp_skipped_updates=status['amp_skipped_updates'],job_slot_seconds=status['slot_seconds'],native_segment_seconds=status['native_segment_seconds'],retained_bytes=sum(f.stat().st_size for f in path.rglob('*') if f.is_file()),attempt_ledger_sha256=sha(path/'attempt_ledger.jsonl'),second_segment_started=False))
 df=pd.DataFrame(allrows);assert len(df)==378 and not df.duplicated(['seed','order','draw','mask_type','step']).any();df.to_csv(OUT/'losses.csv',index=False);table('run_accounting.csv',account)
 interactions=[];perorderdraw=[]
 for (seed,order,draw),g in df.groupby(['seed','order','draw']):
  bystep={}
  for step,h in g.groupby('step'):
   z=h.set_index('mask_type');assert set(z.index)=={'imp','random1','random2'}
   r=dict(seed=int(seed),order=order,draw=int(draw),step=int(step))
   for m in ('imp','random1','random2'):
    r['CE_'+m]=float(z.loc[m,'CE']);r['original_CE_'+m]=float(z.loc[m,'original_CE']);r['penalty_'+m]=float(z.loc[m,'delta_CE'])
   r['CE_random_equal_mean']=(r['CE_random1']+r['CE_random2'])/2;r['penalty_random_equal_mean']=(r['penalty_random1']+r['penalty_random2'])/2
   r['I_S']=r['penalty_imp']-r['penalty_random_equal_mean'];r['unpruned_CE']=float(z.loc['imp','unpruned_CE']);interactions.append(r);bystep[int(step)]=r
  changes={}
  # Difference checkpoint values within each same mask/order/draw recovery run first.
  for m in ('imp','random1','random2'):
   v=g[g.mask_type==m].set_index('step')
   changes[m]=(float(v.loc[2000,'CE'])-float(v.loc[250,'CE']))-(float(v.loc[2000,'original_CE'])-float(v.loc[250,'original_CE']))
  R=changes['imp']-(changes['random1']+changes['random2'])/2
  assert abs(R-(bystep[2000]['I_S']-bystep[250]['I_S']))<1e-12
  r=dict(seed=int(seed),order=order,draw=int(draw),I_S_250=bystep[250]['I_S'],I_S_2000=bystep[2000]['I_S'],R=R)
  for step in (250,2000):
   for key,value in bystep[step].items():
    if key.startswith(('CE_','original_CE_','penalty_')) or key=='unpruned_CE':r[f'{key}_{step}']=value
  for m in ('imp','random1','random2'):r['within_run_penalty_change_'+m]=changes[m]
  perorderdraw.append(r)
 wide=pd.DataFrame(interactions);wide.to_csv(OUT/'interactions_per_seed_order_draw_step.csv',index=False)
 values=pd.DataFrame(perorderdraw);values.to_csv(OUT/'per_seed_order_draw.csv',index=False)
 numeric=[c for c in values.columns if c not in ('seed','order','draw')];values.groupby(['seed','order'])[numeric].mean().reset_index().to_csv(OUT/'per_seed_order_summary.csv',index=False)
 nums=[c for c in wide.columns if c not in ('seed','order','draw','step')];wide.groupby(['seed','order','step'])[nums].mean().reset_index().to_csv(OUT/'per_seed_order_step.csv',index=False)
 # Actual draws are paired over the two orders; never treat orders as extra draws.
 for (seed,mask,draw),z in pd.DataFrame(manifest).groupby(['seed','mask_type','draw']):assert z.initial_state_hash.nunique()==1 and set(z.order)=={'A','B'}
 paired=values.groupby(['seed','draw'])[['I_S_250','I_S_2000','R']].mean().reset_index();paired['L']=paired.I_S_2000;assert len(paired)==9;paired.to_csv(OUT/'individual_paired_draw_values.csv',index=False)
 gate_results={};seeds=[]
 for seed,g in paired.groupby('seed'):
  g=g.sort_values('draw');assert list(g.draw)==[1,2,3];result=seed_result(list(g.L),list(g.R));gate_results[int(seed)]=result;seeds.append(dict(seed=int(seed),I_S_250=g.I_S_250.mean(),**result))
 table('per_seed_gate.csv',seeds);gate=interim_gate(gate_results);table('stage1_gate.csv',[{**gate,'same_seeds_passing_both':json.dumps(gate['same_seeds_passing_both']),'tested_seeds':'7,17,29'}])
 workers=[read(f) for f in (OUT/'workers').glob('*_accounting.json')];allocated=sum(r['worker_slot_seconds'] for r in workers);prep=read(OUT/'registration.json')['preparation_elapsed_seconds'];attempts=sum(r['attempted_updates'] for r in account);assert attempts==135000
 table('actual_resources.csv',[dict(completed_runs=54,attempted_updates=attempts,applied_updates=sum(r['applied_updates'] for r in account),amp_skipped_updates=sum(r['amp_skipped_updates'] for r in account),production_worker_slot_hours=allocated/3600,sum_job_slot_hours=sum(r['job_slot_seconds'] for r in account)/3600,native_segment_hours=sum(r['native_segment_seconds'] for r in account)/3600,preparation_elapsed_hours=prep/3600,total_worker_plus_preparation_hours=(allocated+prep)/3600,logical_retained_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),slot_definition='allocated worker wall time, including model I/O and readout; GPU-wait time before allocation excluded',training_continuation_authorized=False)])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,(seed,g) in zip(axes,wide.groupby('seed')):
  for order,z in g.groupby('order'):
   p0=z.groupby('step').I_S.mean();ax.plot(p0.index,p0.values,label='Order '+order)
  ax.set_title(f'Seed {seed}');ax.set_xlabel('Attempted update');ax.axhline(0,color='gray',lw=.5);ax.legend()
 axes[0].set_ylabel('Sign interaction (nats)');fig.suptitle('E32/IMP80 Stage1 only; paired draws, no continuation');fig.tight_layout();fig.savefig(OUT/'stage1_interaction.png',dpi=170);fig.savefig(OUT/'stage1_interaction.pdf');plt.close(fig)
 heading=['Stage 1 stopped after all 54 approved sign-only runs; no second segment or other arm was run. ([run_accounting.csv](run_accounting.csv): `attempted_updates`, `second_segment_started`.)','','Per-seed joint decision uses three paired draws and the committed amendment. These intervals are conditional on the tested source seed, masks and fixed orders. ([per_seed_gate.csv](per_seed_gate.csv): `L`, `R`, `L_CI_low`, `R_CI_low`, `outcome`, `interval_scope`.)','','|Seed|L = I_S(2000)|R = I_S(2000)-I_S(250)|Outcome|','|---|---|---|---|']
 for r in seeds:heading.append(f"|{r['seed']}|{r['L']:.8f}|{r['R']:.8f}|{r['outcome']}|")
 heading+=['',f"The joint interim gate {'passed' if gate['stage1_gate_pass'] else 'did not pass'}; seeds passing both criteria: {gate['same_seeds_passing_both']}. Failure to pass is not proof of no effect. The gate does not authorize further training. ([stage1_gate.csv](stage1_gate.csv): `stage1_gate_pass`, `same_seeds_passing_both`, `training_continuation_authorized`.)",'', 'Per-seed/order absolute IMP and separate random-mask losses, separate shuffle penalties, both interaction levels and their within-run difference are in [per_seed_order_summary.csv](per_seed_order_summary.csv); all individual draws are in [per_seed_order_draw.csv](per_seed_order_draw.csv) and [individual_paired_draw_values.csv](individual_paired_draw_values.csv). Actual attempts, storage and elapsed slot-hours are in [actual_resources.csv](actual_resources.csv): `attempted_updates`, `logical_retained_bytes`, `production_worker_slot_hours`.']
 (OUT/'findings.md').write_text('\n'.join(heading)+'\n')
 # Finish only here: no continuation invocation is present.
 completion=dict(status='COMPLETE_STAGE1_STOPPED',completed_runs=54,attempted_updates=135000,stage1_gate_pass=gate['stage1_gate_pass'],continuation_authorized=False,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (OUT/'completion.json').write_text(json.dumps(completion,indent=2));print(json.dumps(completion))
if __name__=='__main__':main()
