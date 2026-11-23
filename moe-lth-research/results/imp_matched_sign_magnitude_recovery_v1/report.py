"""Paired draw analysis only after all 108 registered recoveries finish."""
from pathlib import Path
import sys,json,csv,datetime
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'results/imp_sign_recovery_audit_v2'))
from common import pd,np,rows,read,sha,p
from gate import interval,seed_result,consistency,MARGIN

def table(name,data):
 data=list(data)
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def main():
 manifest=rows(OUT/'job_manifest.csv');assert len(manifest)==108
 baseline=pd.read_csv(p.OUT/'trajectory_losses.csv');allrows=[];account=[]
 for job in manifest:
  status=read(OUT/'status'/f"{job['job_id']}.json");assert status['status']=='COMPLETE' and status['attempted_updates']==2500
  path=Path(status['attempt']);ledger=[json.loads(l) for l in (path/'attempt_ledger.jsonl').read_text().splitlines()]
  assert [r['attempt'] for r in ledger]==list(range(1,2501));assert not (path/'segment_2500').exists()
  evs=[json.loads(l) for l in (path/'common_evaluations.jsonl').read_text().splitlines()];assert [e['step'] for e in evs]==[0,250,500,1000,1500,2000,2500]
  for ev in evs:
   seed=int(job['seed']);order=job['order'];step=ev['step'];ref=baseline[(baseline.seed==seed)&(baseline.order==order)&(baseline.variant==job['mask_type']+'_original')&(baseline.step==step)]
   assert len(ref)==1;original=float(ref.cross_entropy.iloc[0]);ce=ev['cross_entropy'];assert np.isfinite(ce)
   allrows.append(dict(seed=seed,order=order,draw=int(job['draw']),arm=job['arm'],mask_type=job['mask_type'],step=step,CE=ce,original_CE=original,delta_CE=ce-original,target_distance=float(job['target_distance']),realized_distance=float(job['realized_distance']),frac_sign_flipped=float(job['frac_sign_flipped']),job_id=job['job_id']))
  account.append(dict(job_id=job['job_id'],seed=int(job['seed']),order=job['order'],draw=int(job['draw']),arm=job['arm'],mask_type=job['mask_type'],host=status['host'],attempted_updates=status['attempted_updates'],applied_updates=status['applied_updates'],amp_skipped_updates=status['amp_skipped_updates'],job_slot_seconds=status['slot_seconds'],native_segment_seconds=status['native_segment_seconds'],retained_bytes=sum(f.stat().st_size for f in path.rglob('*') if f.is_file()),attempt_ledger_sha256=sha(path/'attempt_ledger.jsonl'),second_segment_started=False))
 df=pd.DataFrame(allrows);assert len(df)==756 and not df.duplicated(['seed','order','draw','arm','mask_type','step']).any()
 df.to_csv(OUT/'losses.csv',index=False);table('run_accounting.csv',account)
 for (_,_,_,_),z in pd.DataFrame(manifest).groupby(['seed','mask_type','draw','arm']):assert z.initial_state_hash.nunique()==1 and set(z.order)=={'A','B'}
 interactions=[];changes=[]
 for (seed,order,draw,arm),g in df.groupby(['seed','order','draw','arm']):
  steps={}
  for step,h in g.groupby('step'):
   z=h.set_index('mask_type');assert set(z.index)=={'imp','random1','random2'}
   r=dict(seed=int(seed),order=order,draw=int(draw),arm=arm,step=int(step))
   for m in ('imp','random1','random2'):
    for field,col in [('CE','CE'),('original_CE','original_CE'),('penalty','delta_CE')]:r[field+'_'+m]=float(z.loc[m,col])
   r['penalty_random_equal_mean']=(r['penalty_random1']+r['penalty_random2'])/2
   r['CE_random_equal_mean']=(r['CE_random1']+r['CE_random2'])/2
   r['I']=r['penalty_imp']-r['penalty_random_equal_mean'];interactions.append(r);steps[int(step)]=r
  # Preserve within-run endpoint covariance before mask contrasts or arm comparisons.
  within={}
  for m in ('imp','random1','random2'):
   z=g[g.mask_type==m].set_index('step');within[m]=(float(z.loc[2000,'CE'])-float(z.loc[250,'CE']))-(float(z.loc[2000,'original_CE'])-float(z.loc[250,'original_CE']))
  R=within['imp']-(within['random1']+within['random2'])/2
  assert abs(R-(steps[2000]['I']-steps[250]['I']))<1e-12
  r=dict(seed=int(seed),order=order,draw=int(draw),arm=arm,I_250=steps[250]['I'],L=steps[2000]['I'],R=R)
  for t in (250,2000):
   for k,v in steps[t].items():
    if k.startswith(('CE_','original_CE_','penalty_')):r[f'{k}_{t}']=v
  for m,v in within.items():r['within_run_penalty_change_'+m]=v
  changes.append(r)
 wide=pd.DataFrame(interactions);values=pd.DataFrame(changes)
 wide.to_csv(OUT/'interactions_per_seed_order_draw_step.csv',index=False);values.to_csv(OUT/'per_seed_order_arm_draw.csv',index=False)
 nums=[c for c in values if c not in ('seed','order','draw','arm')];values.groupby(['seed','order','arm'])[nums].mean().reset_index().to_csv(OUT/'per_seed_order_arm_summary.csv',index=False)
 nums=[c for c in wide if c not in ('seed','order','draw','arm','step')];wide.groupby(['seed','order','arm','step'])[nums].mean().reset_index().to_csv(OUT/'per_seed_order_arm_step.csv',index=False)
 direct=[]
 for (seed,order,draw),g in values.groupby(['seed','order','draw']):
  z=g.set_index('arm');assert set(z.index)=={'M','Bm'}
  r=dict(seed=int(seed),order=order,draw=int(draw))
  for a in ('M','Bm'):
   for k in ('I_250','L','R'):r[k+'_'+a]=float(z.loc[a,k])
  r['C']=r['R_Bm']-r['R_M'];direct.append(r)
 direct=pd.DataFrame(direct);direct.to_csv(OUT/'per_seed_order_draw_contrast.csv',index=False)
 nums=[c for c in direct if c not in ('seed','order','draw')];direct.groupby(['seed','order'])[nums].mean().reset_index().to_csv(OUT/'per_seed_order_contrast.csv',index=False)
 paired=direct.groupby(['seed','draw'])[nums].mean().reset_index();assert len(paired)==9;paired.to_csv(OUT/'individual_paired_draw_values.csv',index=False)
 decisions={};seedrows=[]
 for seed,g in paired.groupby('seed'):
  g=g.sort_values('draw');assert list(g.draw)==[1,2,3]
  result=seed_result(list(g.C),list(g.R_M),list(g.R_Bm));decisions[int(seed)]=result
  for arm in ('M','Bm'):
   for endpoint in ('I_250','L'):
    result.update({f'{endpoint}_{arm}_{k}':v for k,v in interval(list(g[endpoint+'_'+arm])).items()})
  seedrows.append(dict(seed=int(seed),**result))
 table('per_seed_results.csv',seedrows);decision=consistency(decisions);table('seed_consistency.csv',[{**decision,'seeds_direct_pass':json.dumps(decision['seeds_direct_pass'])}])
 workers=[read(f) for f in (OUT/'workers').glob('*_accounting.json')];allocated=sum(r['worker_slot_seconds'] for r in workers);prep=read(OUT/'registration.json')['preparation_elapsed_seconds'];attempts=sum(r['attempted_updates'] for r in account);assert attempts==270000
 table('actual_resources.csv',[dict(completed_runs=108,attempted_updates=attempts,applied_updates=sum(r['applied_updates'] for r in account),amp_skipped_updates=sum(r['amp_skipped_updates'] for r in account),production_worker_slot_hours=allocated/3600,sum_job_slot_hours=sum(r['job_slot_seconds'] for r in account)/3600,native_segment_hours=sum(r['native_segment_seconds'] for r in account)/3600,preparation_elapsed_hours=prep/3600,logical_retained_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),slot_definition='allocated worker wall time including I/O/readout, before-allocation GPU waiting excluded',continuation_authorized=False)])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,(seed,g) in zip(axes,wide.groupby('seed')):
  for arm,h in g.groupby('arm'):
   z=h.groupby('step').I.mean();ax.plot(z.index,z.values,label=arm)
  ax.axhline(0,color='gray',lw=.5);ax.set_title(f'Seed {seed}');ax.set_xlabel('Attempted updates');ax.legend()
 axes[0].set_ylabel('Extra perturbation cost to IMP (nats)');fig.tight_layout();fig.savefig(OUT/'interactions.png',dpi=170);fig.savefig(OUT/'interactions.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4))
 for i,r in enumerate(seedrows):
  ax.errorbar(i,r['C_mean'],yerr=[[r['C_mean']-r['C_CI_low']],[r['C_CI_high']-r['C_mean']]],fmt='o',capsize=5,color='black')
  z=paired[paired.seed==r['seed']];ax.scatter([i-.09,i,i+.09],z.C,alpha=.5)
 ax.axhline(0,color='gray',lw=.7);ax.axhline(MARGIN,color='red',ls='--',label='Prospective practical margin');ax.set_xticks(range(3),[str(r['seed']) for r in seedrows]);ax.set_xlabel('Source seed');ax.set_ylabel('Paired reopening contrast Bm − M (nats)');ax.legend();fig.tight_layout();fig.savefig(OUT/'paired_contrast.png',dpi=170);fig.savefig(OUT/'paired_contrast.pdf');plt.close(fig)
 m=pd.read_csv(OUT/'matrix_invariants.csv');bm=m[m.arm=='Bm'];fig,ax=plt.subplots(figsize=(5,4));ax.scatter(bm.target_distance,bm.realized_distance,s=3,alpha=.2);lo=min(bm.target_distance.min(),bm.realized_distance.min());hi=max(bm.target_distance.max(),bm.realized_distance.max());ax.plot([lo,hi],[lo,hi],color='black');ax.set_xlabel('M matrix distance');ax.set_ylabel('Bm matrix distance');fig.tight_layout();fig.savefig(OUT/'distance_matching.png',dpi=170);fig.savefig(OUT/'distance_matching.pdf');plt.close(fig)
 summary=['E32/IMP80 matched-distance recovery stopped after all 108 registered runs (270,000 attempted updates). Each arm preserved the same masks, with fixed original controls. ([run_accounting.csv](run_accounting.csv): `attempted_updates`, `second_segment_started`.)','','|Seed|Magnitude reopening R_M|Matched-sign reopening R_Bm|Bm−M contrast|95% paired interval|Direct criterion|','|---|---|---|---|---|---|']
 for r in seedrows:summary.append(f"|{r['seed']}|{r['R_M_mean']:.6f}|{r['R_Bm_mean']:.6f}|{r['C_mean']:.6f}|[{r['C_CI_low']:.6f}, {r['C_CI_high']:.6f}]|{r['direct_pass']}|")
 summary+=['',f"The prespecified seed-consistency rule {'passed' if decision['consistency_pass'] else 'did not pass'}. This compares arms directly; three-draw intervals are conditional on each tested seed, masks and orders. Failure does not prove no difference. ([per_seed_results.csv](per_seed_results.csv): `C_mean`, `C_CI_low`, `C_CI_high`, `direct_pass`, `arm_outcome`, `M_near_zero`; [seed_consistency.csv](seed_consistency.csv): `consistency_pass`.)"]
 if decision['consistency_pass']:
  summary+=['','At matched displacement, disrupting sign placement produced a larger measured IMP-specific reopening under these interventions. The criterion uses a point-estimate practical margin and a directional interval; it does not require the entire interval to exceed the margin.']
 else:summary+=['','These data do not establish a consistent practically larger sign effect at this matched dose; inspect per-seed intervals and arm outcomes before interpreting similarity or imprecision.']
 summary+=['','All individual paired draws are reported in [individual_paired_draw_values.csv](individual_paired_draw_values.csv): `C`, `R_M`, `R_Bm`, `I_250_M`, `I_250_Bm`, `L_M`, `L_Bm`. Absolute IMP and distinct random-mask losses and penalties, by seed/order/arm/checkpoint, are in [per_seed_order_arm_step.csv](per_seed_order_arm_step.csv): `CE_imp`, `CE_random1`, `CE_random2`, `original_CE_imp`, `original_CE_random1`, `original_CE_random2`, `penalty_imp`, `penalty_random1`, `penalty_random2`, `I`. Matching and flips are in [matrix_invariants.csv](matrix_invariants.csv): `target_distance`, `realized_distance`, `absolute_distance_mismatch`, `frac_sign_flipped`. Actual costs are in [actual_resources.csv](actual_resources.csv).','','The three tested seeds and fixed masks constrain generalization. The high-CE original initialization regime remains a caveat from the prior audit. This experiment does not establish which subnetworks ultimately beat dense, whether sign alignment causes IMP selection, or what happens beyond step 2500. No extension is enabled.']
 (OUT/'findings.md').write_text('\n'.join(summary)+'\n')
 completion=dict(status='COMPLETE_MATCHED_DISTANCE_STOPPED',completed_runs=108,attempted_updates=270000,consistency_pass=decision['consistency_pass'],continuation_authorized=False,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(OUT/'completion.json').write_text(json.dumps(completion,indent=2));print(json.dumps(completion))
if __name__=='__main__':main()
