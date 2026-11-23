"""Read-only historical analysis. Writes only to this new audit directory. No forwards/backwards/train calls."""
import csv,json,hashlib,math,re,sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from moe_lth.models.transformer import TinyMoELanguageModel
from moe_lth.pruning.router_age import state_dict_hash,parameter_group
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks

torch.set_num_threads(2)
SEEDS=(7,17,29); ES=(8,12,16,24)
R=ROOT/'results'; OLD=R/'expert_count_replication'; NEW12=R/'router_development_12expert_retry4_20260902'
inputs={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def track(p):
 p=Path(p); s=p.stat(); inputs[str(p)]={'size':s.st_size,'mtime_ns':s.st_mtime_ns}; return p
def read(p):return json.loads(track(p).read_text())
def lines(p):
 if not p.exists():return []
 return [json.loads(x) for x in track(p).read_text().splitlines() if x]
def load(p):return torch.load(track(p),map_location='cpu',weights_only=False)
def write(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:('NA' if v is None else json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in r.items()})
def stats(a,prefix=''):
 a=np.asarray(a,dtype=float);m=float(a.mean())
 return {prefix+'mean':m,prefix+'median':float(np.median(a)),prefix+'sd':float(a.std()),prefix+'min':float(a.min()),prefix+'max':float(a.max()),prefix+'cv':float(a.std()/m) if m else 0}
def group(s,g):return {k:v for k,v in s.items() if parameter_group(k)==g}
def ref(e,s):return (OLD/f'e{e}/reference' if e==8 else NEW12/'references/reference' if e==12 else R/f'{e}e_router_age_localization/references')/f'seed_{s}'
def condition(e,s,a,k):
 if e>=16:return R/f'{e}e_router_age_localization/recovery/seed_{s}/R{a}/{k}'
 root=NEW12 if e==12 and a not in (0,20,100) else OLD/f'e{e}'
 return root/(f'sparse/s0.85/seed_{s}/R{a}' if k=='sparse' else f'dense/seed_{s}/R{a}')
configs={};counts=[];budgets=[];theory=[];actual=[];sampled=[];capacity=[];trajectories=[];slopes=[];displacements=[];mapping=[];masksout=[];pairs=[];gaps=[];recovery=[];expertrec=[];gradrows=[];initial_hashes=[]
for e in ES:
 for seed in SEEDS:
  root=ref(e,seed);c=yaml.safe_load(track(root/'resolved_config.yaml').read_text());configs[e,seed]=c
  p0=load(root/'checkpoints/step_0.pt');init=p0['model'];model=TinyMoELanguageModel(c['model']);model.load_state_dict(init)
  pc=Counter()
  for n,p in model.named_parameters():pc[parameter_group(n)]+=p.numel()
  counts.append(dict(experts=e,seed=seed,total_parameters=sum(pc.values()),router_parameters=pc['router'],expert_parameters=pc['expert'],shared_parameters=pc['shared'],parameters_per_expert=pc['expert']//(4*e),active_parameters_per_token=pc['shared']+pc['router']+pc['expert']//e))
  initial_hashes.append(dict(experts=e,seed=seed,**{g:state_dict_hash(group(init,g)) for g in ('expert','shared','router')}))
  t=c['training'];b=t['batch_size']*c['data']['seq_len'];total=t['steps']*b;cap=math.ceil(c['model']['capacity_factor']*b/e)
  budgets.append(dict(experts=e,seed=seed,source=str(root),training_step_attempts=t['steps'],optimizer_steps_applied=None,batch_size=t['batch_size'],sequence_length=c['data']['seq_len'],gradient_accumulation=1,effective_batch_size=t['batch_size'],tokens_per_attempt=b,total_token_occurrences=total,total_routed_token_occurrences=total*4,top_k_assignments_generated=total*4,capacity_dropped_assignments=None,accepted_expert_assignments=None))
  theory.append(dict(experts=e,seed=seed,total_training_tokens=total,top_k=1,moe_layers=4,potential_expert_assignments=total*4,ideal_assignments_per_expert=total/e,relative_to_8E=8/e,definition='one physical expert in one layer; divide all-layer assignments by 4E'))
  usage=lines(root/'logs/expert_usage.jsonl');vals=lines(root/'logs/validation_metrics.jsonl');tr=lines(root/'logs/train_metrics.jsonl');vmap={x['step']:x['loss'] for x in vals}
  for x in tr:trajectories.append(dict(experts=e,seed=seed,router_age=x['step']/25,step=x['step'],kind='train_sample',training_loss=x['loss'],language_loss=x.get('language_loss'),auxiliary_loss=x.get('auxiliary_loss'),validation_loss=None))
  for a in range(0,101,10):trajectories.append(dict(experts=e,seed=seed,router_age=a,step=a*25,kind='validation',validation_loss=vmap.get(a*25),training_loss=None,language_loss=None,auxiliary_loss=None))
  for start in (60,80):
   first=vmap.get(start*25);last=vmap.get(2500);imp=first-last if first is not None else None
   slopes.append(dict(experts=e,seed=seed,start_age=start,end_age=100,start_loss=first,end_loss=last,absolute_improvement=imp,relative_improvement=imp/first if imp is not None else None,slope_per_attempt=-imp/((100-start)*25) if imp is not None else None,slope_per_age_percent=-imp/(100-start) if imp is not None else None))
  for a in range(0,101,5):
   us=[x for x in usage if x['step']<=a*25];pro=np.zeros((4,e),dtype=np.int64);acc=pro.copy();active=pro.copy();dropcheck=[]
   for x in us:
    l,j=x['layer_id'],x['expert_id'];n=x['token_count'];pro[l,j]+=n;acc[l,j]+=min(n,cap);active[l,j]+=n>0
   for step in sorted({x['step'] for x in us}):
    for l in range(4):
     rr=[x for x in us if x['step']==step and x['layer_id']==l];n=sum(x['token_count'] for x in rr)
     if n:dropcheck.append(abs((1-sum(min(x['token_count'],cap) for x in rr)/n)-rr[0]['dropped_fraction']))
   base=dict(experts=e,seed=seed,router_age=a)
   actual.append(dict(**base,total_assignments_proposed=a*25*b*4,total_assignments_accepted=0 if a==0 else None,total_assignments_dropped=0 if a==0 else None,mean_accepted_per_expert=0 if a==0 else None,active_steps_per_expert=0 if a==0 else None,status='exact zero at initialization' if a==0 else 'HISTORICAL ACTIVE-STEP EXPOSURE UNAVAILABLE; source only sampled'))
   sampled.append(dict(**base,logged_steps=len({x['step'] for x in us}),sampled_proposed=int(pro.sum()),sampled_accepted=int(acc.sum()),sampled_dropped=int((pro-acc).sum()),sampled_proposed_mean=float(pro.mean()),**stats(acc.flatten(),'sampled_accepted_'),sampled_accepted_per_expert=acc.tolist(),sampled_assignment_fractions=(acc/acc.sum()).tolist() if acc.sum() else [],sampled_active_steps=active.tolist(),drop_fraction_max_reconstruction_error=max(dropcheck,default=0),status='EXACT FOR LOGGED BATCHES ONLY; min(proposed,capacity) reconstructed'))
  capacity.append(dict(experts=e,seed=seed,capacity_per_expert=cap,capacity_fraction_batch_tokens=cap/b,sampled_source_drop_rate=1-acc.sum()/pro.sum(),sampled_accepted_proposed_ratio=acc.sum()/pro.sum(),max_reconstruction_error=max(dropcheck,default=0),complete_source_drop_rate=None))
  ckpts=sorted((root/'checkpoints').glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
  for path in ckpts:
   payload=load(path);state=payload['model'];step=payload['step'];a=step/25
   groups={}
   for n,v in state.items():
    if n=='lm_head.weight' or n.endswith('temperature'):continue
    key=n.split('.moe.experts.')[0]+'.moe.experts.'+n.split('.moe.experts.')[1].split('.')[0] if '.moe.experts.' in n else parameter_group(n)
    d=(v.double()-init[n].double()).square().sum().item();z=init[n].double().square().sum().item()
    old=groups.get(key,(0,0));groups[key]=(old[0]+d,old[1]+z)
   for g in ('expert','router','shared'):
    dd=[math.sqrt(d) for k,(d,z) in groups.items() if ('.moe.experts.' in k if g=='expert' else k==g)];rel=[math.sqrt(d)/(math.sqrt(z)+1e-12) for k,(d,z) in groups.items() if ('.moe.experts.' in k if g=='expert' else k==g)]
    displacements.append(dict(experts=e,seed=seed,router_age=a,step=step,component=g,**stats(dd,'absolute_'),**stats(rel,'relative_'),per_expert_absolute=dd,per_expert_relative=rel))
   if e==24:
    hashes={g:state_dict_hash(group(state,g)) for g in ('router','expert','shared')}
    mapping.append(dict(experts=e,seed=seed,router_age=a,checkpoint_step=step,expected_step=int(path.stem.split('_')[1]),config_seed=payload['config']['seed'],config_experts=payload['config']['model']['num_experts'],path=str(path),resolved_path=str(path.resolve()),is_symlink=path.is_symlink(),sha256=sha(path),**{g+'_hash':v for g,v in hashes.items()},status='PASS' if step==int(path.stem.split('_')[1]) and payload['config']['seed']==seed and payload['config']['model']['num_experts']==24 else 'FAIL'))
   if step==2500:final=state
  if e==24:
   model.load_state_dict(final);expected=expert_local_magnitude_masks(model,.85);mask=load(R/f'24e_router_age_localization/masks/seed_{seed}/pruning_mask.pt')
   maskhash=state_dict_hash(mask);valid=set(mask)==set(expected) and all(torch.equal(mask[n],expected[n]) for n in mask)
   sparseinit={n:v*mask[n] if n in mask else v for n,v in init.items()}
  ages=sorted(int(p.name[1:]) for p in (R/f'{e}e_router_age_localization/recovery/seed_{seed}').glob('R*')) if e>=16 else ([0,10,20,40,60,80,100] if e==12 else [0,20,100])
  for a in ages:
   records={}
   for kind in ('dense','sparse'):
    dest=condition(e,seed,a,kind);meta=read(dest/'metadata.json');summary=read(dest/'summary.json');record={**meta,**summary};records[kind]=record
    logs=lines(dest/'gradient_stats/gradient_stats.jsonl');logs=[x for x in logs if x['step']>0]
    accepted=Counter();proposed=Counter();active=Counter();applied=Counter();appliedacc=Counter();gn=defaultdict(list);gn_group=defaultdict(list)
    complete=len(logs)==2500 and len({x['step'] for x in logs})==2500 and all('accepted_expert_token_counts' in x for x in logs)
    for x in logs:
     ac=x.get('accepted_expert_token_counts',{});accepted.update(ac);proposed.update(x.get('expert_token_counts',{}));active.update({k:int(v>0) for k,v in ac.items()})
     if x.get('optimizer_step_applied'):
      applied.update({k:int(v>0) for k,v in ac.items()});appliedacc.update(ac)
     for g in ('expert','router','shared'):
      v=x.get(g+'_grad_norm')
      if v is not None and math.isfinite(v):gn_group[g].append(v)
     for k,v in (x.get('per_expert_grad_norm') or {}).items():
      if isinstance(v,(float,int)) and math.isfinite(v):gn[k].append(v)
    base=dict(experts=e,seed=seed,router_age=a,kind=kind,source=str(dest))
    rr=dict(**base,recovery_step_attempts=record['recovery_steps'],optimizer_steps_applied=record.get('optimizer_steps_applied'),total_tokens=record['recovery_steps']*b,potential_assignments=record['recovery_steps']*b*4,accepted_assignments=sum(accepted.values()) if complete else None,proposed_assignments=sum(proposed.values()) if complete else None,dropped_assignments=sum(proposed.values())-sum(accepted.values()) if complete else None,accepted_per_expert=sum(accepted.values())/(4*e) if complete else None,applied_update_accepted_assignments=sum(appliedacc.values()) if complete else None,complete_counters=complete,final_validation_loss=record['final_validation_loss'],**stats(list(active.values()),'active_attempts_'),**stats(list(applied.values()),'active_applied_steps_'))
    recovery.append(rr)
    for k in accepted:expertrec.append(dict(**base,expert=k,accepted=accepted[k],proposed=proposed[k],dropped=proposed[k]-accepted[k],fraction_assignments=accepted[k]/sum(accepted.values()),active_attempts=active[k],active_applied_steps=applied[k],zero_token_attempts=2500-active[k],fraction_attempts_active=active[k]/2500,accepted_per_expert_parameter=accepted[k]/(pc['expert']/(4*e)),active_steps_per_expert_parameter=applied[k]/(pc['expert']/(4*e)),complete_counters=complete))
    for g,vs in {**gn_group,**gn}.items():
     if vs:gradrows.append(dict(**base,component=g,n_finite_observations=len(vs),nonzero_observations=sum(v>0 for v in vs),sum_observed_norms=sum(vs),cumulative_update_norm=None,**stats(vs),coverage='every attempted step, finite only' if g in ('expert','shared','router') else 'sampled per expert; NOT cumulative full-training norm'))
    if e==24:
     validated=read(dest/'validated_result.json');payload=load(dest/'checkpoints/final_recovered.pt');state=payload['model'];mc=next(x for x in mapping if x['seed']==seed and x['router_age']==a)
     records[kind]['audit_final_file_hash']=sha(dest/'checkpoints/final_recovered.pt')==validated['final_checkpoint_sha256'];records[kind]['audit_final_model_hash']=state_dict_hash(state)==validated['final_model_hash'];records[kind]['audit_final_step']=payload['step']==2500
     assembled={**(sparseinit if kind=='sparse' else init)};routerstate=load(Path(meta['router_checkpoint']))['model'];assembled.update(group(routerstate,'router'))
     records[kind]['audit_initial_model_hash']=state_dict_hash(assembled)==validated['initial_model_hash'];records[kind]['audit_checkpoint_hash']=mc['sha256']==validated['router_checkpoint_hash'];records[kind]['audit_router_hash']=mc['router_hash']==meta['initial_router_state_hash']
     if kind=='sparse':
      masksout.append(dict(experts=e,seed=seed,router_age=a,target_sparsity=.85,actual_weight_sparsity=1-sum(int(v.sum()) for v in mask.values())/sum(v.numel() for v in mask.values()),mask_hash=maskhash,hash_matches_record=maskhash==meta['mask_hash'],matches_recomputed_R100_magnitude=valid,mask_source='source R100 expert weights; rounded keep count per expert; biases excluded',rewind='source R0 expert/shared; age-specific router',expert_rewind_hash_matches=state_dict_hash(group(sparseinit,'expert'))==meta['expert_state_hash'],shared_rewind_hash_matches=state_dict_hash(group(init,'shared'))==meta['shared_state_hash'],final_pruned_weights_zero=all(not bool(state[n][~v.bool()].any()) for n,v in mask.items())))
   d,s=records['dense'],records['sparse'];fields=['seed','router_age_percent','router_step','shared_state_hash','initial_router_state_hash','training_batch_sequence_hash','validation_batch_sequence_hash','recovery_steps','optimizer','scheduler','precision','recovery_eval_interval']
   row=dict(experts=e,seed=seed,router_age=a,**{f'{f}_matched':d.get(f)==s.get(f) for f in fields},fresh_optimizer=d.get('optimizer_state_entries_at_start')==s.get('optimizer_state_entries_at_start')==0,matched_dense_loss=abs(s['dense_reference_loss']-d['final_validation_loss'])<1e-10,source_dense=str(condition(e,seed,a,'dense')),source_sparse=str(condition(e,seed,a,'sparse')))
   if e==24:
    row.update({kind+'_'+f:r[f] for kind,r in records.items() for f in r if f.startswith('audit_')});gaps.append(dict(seed=seed,router_age=a,dense_final_loss=d['final_validation_loss'],sparse_final_loss=s['final_validation_loss'],ticket_gap=s['final_validation_loss']-d['final_validation_loss']))
   pairs.append(row)
  print('AUDITED',e,seed,flush=True)
  del model,init,final,p0,payload,state
for name,rows in [('architecture_parameter_counts.csv',counts),('source_training_budget.csv',budgets),('theoretical_expert_exposure.csv',theory),('actual_expert_exposure.csv',actual),('sampled_source_exposure.csv',sampled),('capacity_audit.csv',capacity),('dense_training_trajectories.csv',trajectories),('late_training_slopes.csv',slopes),('expert_parameter_displacement.csv',displacements),('checkpoint_mapping_audit.csv',mapping),('mask_audit.csv',masksout),('dense_control_audit.csv',pairs),('24e_ticket_gap_recomputation.csv',gaps),('recovery_exposure.csv',recovery),('recovery_expert_exposure.csv',expertrec),('historical_gradient_exposure.csv',gradrows),('initialization_hashes.csv',initial_hashes)]:write(name,rows)
def flatten(d,p=''):
 return {kk:vv for k,v in d.items() for kk,vv in (flatten(v,p+k+'.').items() if isinstance(v,dict) else [(p+k,v)])}
flat={e:flatten(configs[e,7]) for e in ES};diff=[]
for key in sorted(set().union(*(x.keys() for x in flat.values()))):
 values=[flat[e].get(key) for e in ES];same=all(v==values[0] for v in values)
 diff.append(dict(field=key,**{str(e)+'E':v for e,v in zip(ES,values)},expected_relationship='same except expert count, output paths and checkpoint save grid',actual_relationship='identical' if same else 'different',status='MATCHED' if same else 'INTENTIONALLY SCALED' if key in ('model.num_experts','output_dir','training.checkpoint_steps','training.checkpoint_only_steps') else 'UNKNOWN'))
for key,val in [('gradient_accumulation',1),('scheduler','none'),('warmup',0),('recovery_steps',2500),('recovery_sparsity',.85),('auxiliary_loss_type','E*sum(detached_proposal_fraction*mean_probability), mean over layers')]:diff.append(dict(field=key,**{str(e)+'E':val for e in ES},expected_relationship='same',actual_relationship='same code path',status='MATCHED'))
write('architecture_config_diff.csv',diff)
summary=[]
for a in sorted({x['router_age'] for x in gaps}):
 v=[x['ticket_gap'] for x in gaps if x['router_age']==a];summary.append(dict(router_age=a,mean_ticket_gap=np.mean(v),sd_ticket_gap=np.std(v,ddof=1)))
write('24e_age_summary_independent.csv',summary)
contr=[]
for seed in SEEDS:
 rows=[x for x in gaps if x['seed']==seed];m={x['router_age']:x['ticket_gap'] for x in rows};contr.append(dict(seed=seed,winner=min(m,key=m.get),R50_minus_R0=m[50]-m[0],R50_minus_R100=m[50]-m[100]))
write('24e_seed_contrasts.csv',contr)
unchanged=all(Path(p).stat().st_size==v['size'] and Path(p).stat().st_mtime_ns==v['mtime_ns'] for p,v in inputs.items())
(OUT/'artifact_integrity.json').write_text(json.dumps(dict(historical_inputs_unchanged_size_mtime=unchanged,inputs=inputs,training_launched=False,analysis='CPU checkpoint arithmetic and persisted logs only',outputs={p.name:sha(p) for p in OUT.glob('*') if p.is_file() and p.name!='artifact_integrity.json'}),indent=2))
print('DATA ANALYSIS COMPLETE',flush=True)
