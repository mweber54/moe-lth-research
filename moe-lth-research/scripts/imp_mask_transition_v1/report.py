"""Live exact-cohort tables; missing outcomes stay missing until verified completion."""
from common import *
import pandas as pd,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'dense':'#555555','fixed':'#2469aa','abrupt':'#c34d45','gradual':'#3b9863'}
def write(name,rows,fields):
 fields=list(dict.fromkeys(fields+[k for row in rows for k in row]))
 table(OUT/f'{name}.csv',rows,fields)
def lines(path):
 if not path.exists():return []
 text=path.read_text().splitlines();result=[]
 for line in text:
  try:result.append(json.loads(line))
  except json.JSONDecodeError:pass # an active append may be incomplete; completion audit requires all rows
 return result

def analyze():
 if not (OUT/'manifest.json').exists():return
 manifest=read(OUT/'manifest.json');endpoints=[];trajectories=[];events=[];routing=[];updates=[];capacity=[];coverage=[];budgets=[];states=[]
 schedules=pd.read_csv(OUT/'event_schedule.csv');vmap={v['seed']:v for v in manifest['variants']};work_counts={}
 for (seed,level),z in schedules.groupby(['seed','event']):
  ec={}
  for x in z.itertuples():
   parts=x.parameter.split('.');key=f'layer_{parts[1]}_expert_{parts[4]}';ec[key]=ec.get(key,0)+int(x.surviving_coordinates)
  work_counts[(seed,level)]=ec
 for seed in SEEDS:work_counts[(seed,0)]={k:524288 for k in work_counts[(seed,1)]}
 for j in manifest['jobs']:
  st=read(OUT/'status'/f"{j['id']}.json");states.append(dict(j,status=st['status']));v=vmap[j['seed']];base={k:j[k] for k in ('id','seed','condition','reuse')}
  assignment_counts=schedules[schedules.seed==j['seed']].groupby('event').surviving_coordinates.sum().to_dict();assignment_counts[0]=67108864
  counts=np.array([assignment_counts[event_count(j['condition'],t-1)] for t in range(1,5001)],dtype=np.int64)
  budgets.append(dict(base,active_at_update1=int(counts[0]),active_at_update1000=int(counts[999]),active_at_update1001=int(counts[1000]),cumulative_active_coordinate_attempts=int(counts.sum()),attempted_tokens=5000*128*128))
  # Complete deterministic planned coordinate counts at every attempt, including reused controls.
  for t,c in enumerate(counts,1):capacity.append(dict(base,attempt=t,active_expert_weight_coordinates=int(c),cumulative_active_coordinate_attempts=int(counts[:t].sum())))
  attempt=Path(st['attempt']) if 'attempt' in st else None;result=read(st['result']) if st.get('result') else None
  grids=[]
  if attempt:
   for p in list(attempt.glob('grid_*.json'))+list(attempt.glob('segment_*/grid_*.json')):
    d=read(p)
    if d.get('phase')=='pre_pruning':continue
    grids.append(d);trajectories.append(dict(base,**d,source_path=str(p)))
  bystep={d['step']:d for d in grids}
  if j['reuse'] and not result and st['status']=='PENDING':reason='pending common readout'
  else:reason='missing_reference_snapshot' if j['reuse'] else 'not_yet_measured'
  for step in GRID:coverage.append(dict(base,step=step,available=step in bystep,reason='observed common readout' if step in bystep else reason))
  endpoints.append(dict(base,status=st['status'],ce_5000=bystep.get(5000,{}).get('ce') if result else None,ce_2500_prereset=bystep.get(2500,{}).get('ce'),
       final_mask_hash=result.get('final_mask_hash') if result else None,checkpoint=result.get('checkpoint') if result else None,
       attempted_updates=result.get('attempted_updates') if result else None,applied_updates=result.get('applied_updates') if result else None,
       amp_skips=result.get('amp_skips') if result else None,wall_seconds=result.get('wall_seconds') if result else None,
       diagnostic_seconds=sum(r.get('diagnostic_seconds',0) for r in result.get('records',[])) if result else None,
       common_evaluation_seconds=sum(r.get('common_evaluation_seconds',0) for r in result.get('records',[])) if result else None,
       peak_allocated_bytes=result.get('peak_allocated_bytes') if result else None,peak_reserved_bytes=result.get('peak_reserved_bytes') if result else None,
       runtime=json.dumps(result.get('runtime')) if result else None))
  if attempt:
   for p in attempt.glob('segment_*/event_*.json'):
    d=read(p);events.append(dict(base,step=d['event']['attempt'],event_after=d['event']['event_after'],task_ce_pre=d['pre']['task_ce'],task_ce_post=d['post']['task_ce'],task_damage=d['damage'],aux_pre=d['pre']['auxiliary_layer_mean'],aux_post=d['post']['auxiliary_layer_mean'],mask_hash=d['mask_hash'],parameter_norms_pre=json.dumps(d['pre']['parameter_norms']),parameter_norms_post=json.dumps(d['post']['parameter_norms'])))
    for phase in ('pre','post'):
     for row in d[phase]['routing']:routing.append(dict(base,step=d['event']['attempt'],phase=phase,**row))
    for row in d['overlap']:routing.append(dict(base,step=d['event']['attempt'],phase='event_overlap',**row))
   for p in attempt.glob('segment_*/routing_*.json'):
    d=read(p)
    for row in d['routing']:routing.append(dict(base,step=d['step'],phase='interval',**row))
  # Full native training logs are available for both executed and reused work.
  source=Path(v['references'][j['condition']]['attempt']) if j['reuse'] and result else (None if j['reuse'] else attempt)
  if source:
   net={}
   if not j['reuse']:
    for p in source.glob('segment_*/updates.jsonl'):
     net.update({d['step']:d['actual_update_l2'] for d in lines(p)})
   for offset in (0,2500):
    for row in lines(source/f'segment_{offset}/gradient_stats/gradient_stats.jsonl'):
     if row['step']==0:continue
     step=offset+row['step'];un=net.get(step,{})
     accepted=row.get('accepted_expert_token_counts');work=None
     if accepted:
      level=event_count(j['condition'],step-1);ec=work_counts[(j['seed'],level)]
      work=sum(int(accepted[k])*ec[k] for k in accepted)
     updates.append(dict(base,step=step,optimizer_step_applied=row.get('optimizer_step_applied'),loss_scale_before=row.get('loss_scale_before'),loss_scale_after=row.get('loss_scale_after'),
          unscaled_expert_gradient_norm=row.get('expert_grad_norm'),unscaled_router_gradient_norm=row.get('router_grad_norm'),unscaled_shared_gradient_norm=row.get('shared_grad_norm'),
          gradient_status=row.get('measurement_status'),actual_expert_update_l2=un.get('E'),actual_router_update_l2=un.get('R'),actual_shared_update_l2=un.get('S'),
          update_norm_availability='actual_one_attempt_pre_prune' if un else 'unavailable_at_this_attempt',
          active_expert_weight_coordinates=int(counts[step-1]),accepted_token_weight_coordinate_work_estimate=work,
          training_ce=row.get('train_cross_entropy'),native_auxiliary_loss=row.get('auxiliary_routing_loss')))
 write('endpoint_metrics_by_condition',endpoints,['id','seed','condition','reuse','status','ce_5000','ce_2500_prereset'])
 write('trajectory_metrics',trajectories,['id','seed','condition','reuse','step','ce','target_tokens','phase'])
 write('pruning_event_damage',events,['id','seed','condition','step','task_damage'])
 write('routing_diagnostics',routing,['id','seed','condition','step','phase','layer','expert'])
 write('update_and_amp_accounting',updates,['id','seed','condition','step','optimizer_step_applied'])
 write('active_coordinate_schedule',capacity,['id','seed','condition','attempt','active_expert_weight_coordinates','cumulative_active_coordinate_attempts'])
 write('coverage_and_exclusions',coverage,['id','seed','condition','step','available','reason'])
 write('job_status',states,['id','seed','condition','reuse','status'])
 effects=[];summary=[];ed=pd.DataFrame(endpoints)
 for seed in SEEDS:
  z=ed[ed.seed==seed].set_index('condition');d=dict(seed=seed)
  for step,col in [(5000,'ce_5000'),(2500,'ce_2500_prereset')]:
   for a,b,label in [('abrupt','fixed','abrupt_minus_fixed'),('gradual','fixed','gradual_minus_fixed'),('gradual','abrupt','gradual_minus_abrupt')]+[(c,'dense',c+'_minus_dense') for c in ('fixed','abrupt','gradual')]:
    va,vb=z.loc[a,col],z.loc[b,col];d[f'{label}_{step}']=float(va-vb) if pd.notna(va) and pd.notna(vb) else None
  for cond in ('abrupt','gradual'):
   a={x['step']:x['ce'] for x in trajectories if x['seed']==seed and x['condition']==cond and x['step']>=1000};b={x['step']:x['ce'] for x in trajectories if x['seed']==seed and x['condition']=='fixed' and x['step']>=1000}
   common=sorted(a.keys()&b.keys());ok=len(common)>=3 and common[0]==1000 and common[-1]==5000
   d[f'{cond}_excess_AUC_vs_fixed']=float(np.trapz([a[t]-b[t] for t in common],common)) if ok else None
   d[f'{cond}_AUC_observed_grid']=json.dumps(common);d[f'{cond}_AUC_status']='common_observed_grid_only' if ok else 'unavailable_inadequate_common_grid'
  effects.append(d)
 ef=pd.DataFrame(effects)
 for col in ef.columns:
  if col=='seed' or 'status' in col or 'grid' in col:continue
  a=pd.to_numeric(ef[col],errors='coerce').dropna()
  summary.append(dict(contrast=col,n_observed_seeds=len(a),mean=float(a.mean()) if len(a) else None,median=float(a.median()) if len(a) else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None,SD=float(a.std(ddof=1)) if len(a)>1 else None,replication_unit='source_seed',cohort_complete=len(a)==5))
 write('paired_seed_effects',effects,['seed']);write('descriptive_seed_summaries',summary,['contrast','n_observed_seeds'])
 compute=[]
 for b,e in zip(budgets,endpoints):
  work=[u['accepted_token_weight_coordinate_work_estimate'] for u in updates if u['id']==b['id'] and u['accepted_token_weight_coordinate_work_estimate'] is not None]
  compute.append(dict(b,executed_work='common_readout_only_reused_training' if b['reuse'] else 'new_training_and_readouts',
     wall_seconds=e.get('wall_seconds'),diagnostic_seconds=e.get('diagnostic_seconds'),common_evaluation_seconds=e.get('common_evaluation_seconds'),
     accepted_token_weight_coordinate_work=sum(work) if work else None,work_attempts_observed=len(work),
     peak_allocated_bytes=e.get('peak_allocated_bytes'),peak_reserved_bytes=e.get('peak_reserved_bytes'),runtime=e.get('runtime'),
     IMP_mask_discovery_cost='excluded privileged historical IMP rounds; not free',dense_tensor_zero_compute_savings_claim=False))
  compute[-1]['historical_IMP_discovery_attempts_excluded']=8*2500
  compute[-1]['historical_IMP_discovery_recorded_wall_seconds_excluded']=sum(read(Path(vmap[b['seed']]['mask']).parent/f'mask_{k:02d}.json')['duration_seconds'] for k in range(1,9))
 write('compute_and_capacity_accounting',compute,['id','seed','condition'])
 complete=sum(x['status'] in ('COMPLETE','REUSED') for x in states);active=sum(x['status']=='RUNNING' for x in states);failed=sum(x['status']=='FAILED' for x in states);pending=20-complete-active-failed
 save(OUT/'completion.json',dict(status='FINALIZING' if complete==20 else 'FAILED' if (OUT/'STOP.json').exists() else 'RUNNING' if active else 'READY',scientifically_complete=False,completed=complete,active=active,pending=pending,failed=failed,registered=20,unix=time.time()))
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
 fig,axes=plt.subplots(5,1,figsize=(10,17))
 for ax,seed in zip(axes,SEEDS):
  for cond in CONDITIONS:
   rows=sorted([r for r in trajectories if r['seed']==seed and r['condition']==cond],key=lambda x:x['step'])
   if rows:ax.plot([r['step'] for r in rows],[r['ce'] for r in rows],'o-',label=cond,color=COLORS[cond],markersize=3)
  ax.axvline(1000,color='.5',ls=':');ax.axvline(2500,color='.5',ls='--');ax.set_title(f'seed{seed}');ax.set_ylabel('common task validation CE')
  if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(figdir/'validation_trajectories.png');plt.close(fig)
 for name,columns in [('endpoint_effects',['abrupt_minus_fixed_5000','gradual_minus_fixed_5000']),('sparse_minus_dense',['fixed_minus_dense_5000','abrupt_minus_dense_5000','gradual_minus_dense_5000'])]:
  fig,ax=plt.subplots(figsize=(9,5))
  for col in columns:ax.plot(ef.seed,pd.to_numeric(ef[col],errors='coerce'),'o',label=col)
  ax.axhline(0,color='.5');ax.set_xlabel('source seed');ax.set_ylabel('paired common CE difference');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/f'{name}.png');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for cond in CONDITIONS:
  z=[r for r in capacity if r['seed']==17 and r['condition']==cond]
  axes[0].plot([r['attempt'] for r in z],[r['active_expert_weight_coordinates'] for r in z],label=cond,color=COLORS[cond]);axes[1].plot([r['attempt'] for r in z],[r['cumulative_active_coordinate_attempts'] for r in z],label=cond,color=COLORS[cond])
 axes[0].set_title('Active expert coordinates before each attempt');axes[1].set_title('Cumulative active-coordinate-attempt budget');axes[0].legend();fig.tight_layout();fig.savefig(figdir/'capacity_budgets.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5))
 for seed in SEEDS:
  for cond in ('abrupt','gradual'):
   z=sorted([r for r in events if r['seed']==seed and r['condition']==cond],key=lambda x:x['step'])
   if z:ax.plot([r['step'] for r in z],[r['task_damage'] for r in z],'o-',label=f'{seed}/{cond}')
 ax.axhline(0,color='.5');ax.set_title('Immediate event damage on fixed training diagnostic batches');ax.set_xlabel('attempt');ax.set_ylabel('post−pre task CE');
 if events:ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(figdir/'event_damage.png');plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,5))
 for seed in SEEDS:
  for cond in CONDITIONS:
   z=[r for r in routing if r['seed']==seed and r['condition']==cond and r['phase']=='interval']
   if z:
    rr=pd.DataFrame(z).groupby('step').agg(proposed=('proposed','sum'),accepted=('accepted','sum'));ax.plot(rr.index,1-rr.accepted/rr.proposed,label=f'{seed}/{cond}',lw=.8)
 ax.set_title('Training diagnostic accepted-capacity drop fraction (native routing)');ax.set_xlabel('attempt');ax.set_ylabel('dropped/proposed');
 if routing:ax.legend(fontsize=6,ncol=4)
 fig.tight_layout();fig.savefig(figdir/'routing_capacity.png');plt.close(fig)
 for seed in SEEDS:
  fig,axes=plt.subplots(1,2,figsize=(13,7))
  for ax,cond in zip(axes,('abrupt','gradual')):
   z=[r for r in routing if r['seed']==seed and r['condition']==cond and r['phase']=='interval']
   if z:
    rr=pd.DataFrame(z).groupby(['layer','expert','step']).accepted.sum().unstack('step');ax.imshow(rr.to_numpy(),aspect='auto',cmap='Blues');ax.set_xticks(range(len(rr.columns)),rr.columns,rotation=60);ax.set_yticks([0,32,64,96],['layer0','layer1','layer2','layer3'])
   ax.set_title(f'seed{seed}/{cond}: native accepted loads');ax.set_xlabel('attempt')
  fig.tight_layout();fig.savefig(figdir/f'expert_loads_seed{seed}.png');plt.close(fig)
 fig,axes=plt.subplots(5,1,figsize=(10,16))
 for ax,seed in zip(axes,SEEDS):
  for cond in ('abrupt','gradual'):
   a={x['step']:x['ce'] for x in trajectories if x['seed']==seed and x['condition']==cond};b={x['step']:x['ce'] for x in trajectories if x['seed']==seed and x['condition']=='fixed'};grid=sorted(t for t in a.keys()&b.keys() if t>=1000)
   if grid:ax.plot(grid,[a[t]-b[t] for t in grid],'o-',label=cond,color=COLORS[cond])
  ax.axhline(0,color='.5');ax.axvline(2500,color='.5',ls='--');ax.set_title(f'seed{seed}: post-mask excess validation CE vs fixed');ax.set_ylabel('transition−fixed')
  if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(figdir/'post_event_recovery.png');plt.close(fig)
 header=f"# Known-mask transitions — {'COMPLETE' if complete==20 else 'RUNNING' if active else 'FAILED' if failed else 'READY'}\n\n{complete}/20 complete, {active} active, {pending} pending, {failed} failed.\n"
 text='''
Prospective controlled experiment motivated by previous exploratory findings; not
project-level preregistration. The existing IMP mask is privileged information.
No outcome-driven changes or additional conditions are authorized.

The conditions share initialization, attempted-token budget and final sparse mask,
but have unequal cumulative active-parameter budgets. Abrupt keeps dense capacity
longer than gradual; both exceed fixed early capacity. Zeros in dense tensors imply
no FLOP or wall-clock savings. IMP discovery cost remains an excluded oracle cost.

Source seed is the replication unit(n=5). Individual seed effects precede descriptive
means, medians, ranges and SD. Missing states/readouts remain missing. Historical
TF32 CE values are never substituted for the common FP32 TF32-off readout.

'''
 text+='## Each seed\n\n|Seed|Abrupt−fixed5000|Gradual−fixed5000|Gradual−abrupt5000|\n|---|---:|---:|---:|\n'
 for effect in effects:
  vals=[effect.get(k+'_5000') for k in ('abrupt_minus_fixed','gradual_minus_fixed','gradual_minus_abrupt')]
  text+='|'+str(effect['seed'])+'|'+'|'.join(f'{v:.6g}' if v is not None else 'pending' for v in vals)+'|\n'
 text+='\n## Descriptive source-seed summaries\n\n|Contrast|Observed seeds|Mean|Median|Range|SD|\n|---|---:|---:|---:|---|---:|\n'
 for r in summary:
  if r['contrast'] not in ('abrupt_minus_fixed_5000','gradual_minus_fixed_5000','gradual_minus_abrupt_5000'):continue
  f=lambda v:f'{v:.6g}' if v is not None else 'missing'
  text+=f"|{r['contrast']}|{r['n_observed_seeds']}|{f(r['mean'])}|{f(r['median'])}|{f(r['min'])} to {f(r['max'])}|{f(r['SD'])}|\n"
 if complete<20:
  text+='''
## Required conclusions: pending verified cohort completion

1. Schedule construction reaches exactly each original IMP80 mask; executed path
verification is pending for incomplete jobs. A schedule identity is not a completed run.
2–4. Persistent endpoint effects, consistency across all five seeds, and recovery
around the common2500 reset remain unresolved until the registered outcomes exist.
5. Capacity, deletion timing, dense-training history and retained optimizer state
remain combined; the2500 reset also coincides with additional training.
6. No follow-up is recommended from an incomplete cohort and none is launched.
'''
 else:
  assert all(e['final_mask_hash']==vmap[e['seed']]['mask_hash'] for e in endpoints if e['condition']!='dense')
  text+='\n## Required conclusions\n\n1. All three sparse paths reached the identical registered final IMP80 mask within each seed.\n\n'
  for cond in ('abrupt','gradual'):
   a=ef[f'{cond}_minus_fixed_5000'].astype(float);pre=ef[f'{cond}_minus_fixed_2500'].astype(float)
   text+=f"2–3. {cond}:5000 transition−fixed mean={a.mean():.6g}, median={a.median():.6g}, range[{a.min():.6g},{a.max():.6g}], SD={a.std(ddof=1):.6g}; {int((a>0).sum())}/5 positive deficits and {int((a<0).sum())}/5 negative differences. Pre-reset2500 mean difference={pre.mean():.6g}. These continuous seed effects describe persistence and sign consistency; no equivalence margin was registered.\n\n"
   before=[];after=[];early=[]
   for seed in SEEDS:
    aa={x['step']:x['ce'] for x in trajectories if x['seed']==seed and x['condition']==cond};bb={x['step']:x['ce'] for x in trajectories if x['seed']==seed and x['condition']=='fixed'}
    before.append(aa.get(2500,np.nan)-bb.get(2500,np.nan));after.append(aa.get(2525,np.nan)-bb.get(2525,np.nan));early.append(aa.get(1000,np.nan)-bb.get(1000,np.nan))
   text+=f"4. {cond} observed post1000 transition−fixed gaps by seed={early}; pre-reset2500={before};2525={after};5000={a.to_list()}. Compare these individual trajectories with immediate training-probe event damage; they measure different populations. Changes across2500 combine optimizer reset and further training and do not isolate either.\n\n"
  text+='''5. Capacity, removal timing, dense-training history and retained Adam state remain combined. Routing changes are accompanying diagnostics, not a causal routing intervention. Better transition results may reflect useful dense history at a higher capacity cost. Known-mask reachability does not establish online discovery or training savings.\n\n'''
  consistent=any((ef[f'{c}_minus_fixed_5000']>0).all() for c in ('abrupt','gradual'))
  text+=('6. One focused follow-up is justified: separately specified transition-dynamics work matching cumulative capacity budgets, because at least one path has positive endpoint deficits in all five seeds. No mask-selection or continuation experiment is launched.\n' if consistent else '6. These mixed path effects do not yet isolate a mechanism or justify expanding mask selection. Neither a new mask-selection experiment nor an additional transition intervention is recommended from these results alone. No follow-up is launched.\n')
 (OUT/'findings.md').write_text(header+text)
 save(OUT/'storage_accounting.json',dict(actual_bytes_at_report=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),unix=time.time(),estimated_storage_bytes=read(OUT/'resource_estimate.json')['estimated_storage_bytes']))
 if complete==20:
  verified={}
  for job in manifest['jobs']:
   st=read(OUT/'status'/f"{job['id']}.json");cp=Path(st['attempt'])/'completion.json';assert sha(cp)==st['completion_sha256'];assert read(cp)['verified'];verified[job['id']]=sha(cp)
  save(OUT/'completion.json',dict(status='COMPLETE',scientifically_complete=True,completed=20,active=0,pending=0,failed=0,registered=20,verified_attempt_inventories=verified,unix=time.time()))
  save(OUT/'output_inventory.json',dict(deliverables={str(p.relative_to(OUT)):sha(p) for p in list(OUT.glob('*.csv'))+list(OUT.glob('*.md'))+list(figdir.glob('*.png'))},verified_attempt_inventories=verified,seal_sha256=sha(OUT/'seal.json')))

if __name__=='__main__':
 with lock(OUT/'report.lock'):analyze()
