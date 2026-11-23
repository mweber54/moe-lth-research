"""Existing artifacts only. Run from repo root; output isolated in results/final_dataset_a.
Reuses canonical metrics/exposure functions and never calls experiment execution.
"""
import os,sys,json,hashlib,itertools,datetime,io,ast,subprocess
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,yaml
ROOT=Path.cwd(); DATA=ROOT/'results/router_conditioned_fixed_budget_dataset'; BACK=ROOT/'results/router_conditioned_fixed_budget_dataset.HOME_BACKUP'; OUT=ROOT/'results/final_dataset_a'
sys.path[:0]=[str(ROOT/'scripts/fixed_budget_dataset'),str(ROOT/'src')]
import common as C
import torch
from moe_lth.pruning.router_age import state_dict_hash,parameter_group
ES,SEEDS,AGES,SP=C.ES,C.SEEDS,C.AGES,C.SP
for d in ['canonical','summaries','trajectories','audits','figures','reports','scripts','inputs','cache']:(OUT/d).mkdir(parents=True,exist_ok=True)
def atomic(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(b if isinstance(b,bytes) else b.encode());tmp.replace(p)
def jsave(p,x):atomic(p,json.dumps(x,indent=2,default=str,allow_nan=True)+'\n')
def csv(p,x):
 f=x if isinstance(x,pd.DataFrame) else pd.DataFrame(x);atomic(p,f.to_csv(index=False,na_rep='NA'));return f
def table(p,f):
 csv(p.with_suffix('.csv'),f);tmp=p.with_suffix('.parquet.tmp');f.to_parquet(tmp,index=False);tmp.replace(p.with_suffix('.parquet'))
inputs={}; cache=OUT/'cache/hashes.json'; hcache=json.loads(cache.read_text()) if cache.exists() else {}
def sha(p):
 p=Path(p);st=p.stat();key=str(p.resolve());identity=[st.st_size,st.st_mtime_ns];old=hcache.get(key,{})
 if old.get('identity')==identity:digest=old['sha256']
 else:
  digest=C.sha(p);hcache[key]={'identity':identity,'sha256':digest}
 inputs[str(p)]={'sha256':digest,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns};return digest
def read(p):sha(p);return json.loads(Path(p).read_text())
def frame(p):sha(p);return pd.read_csv(p)
def relocated(p):
 p=Path(p)
 if p.exists():return p
 try:q=BACK/p.relative_to(DATA)
 except ValueError:return p
 return q if q.exists() else p
# Reuse the exact historical metric/exposure implementation without importing a runner.
mod=ast.parse((ROOT/'scripts/fixed_budget_dataset/finalize.py').read_text());env=dict(C.__dict__);env.update(OUT=OUT/'cache',Counter=Counter,np=np,math=__import__('math'))
exec(compile(ast.Module(body=[n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','exposures']],type_ignores=[]),'reused_finalizer_functions','exec'),env)
metrics,exposures=env['metrics'],env['exposures']
protocol=[]
def check(cid,scope,field,actual,expected):
 ok=actual==expected;protocol.append(dict(condition_id=cid,scope=scope,field=field,actual=str(actual),expected=str(expected),passed=bool(ok)));return bool(ok)
def main():
 torch.set_num_threads(2)
 stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();jobs=frame(DATA/'run_manifest.csv');atomic(OUT/'inputs/run_manifest.csv',(DATA/'run_manifest.csv').read_bytes())
 F={}
 for n in ['source_trajectory_manifest','source_checkpoint_manifest','mask_manifest','historical_recovery_audit','dense_recovery_manifest','sparse_recovery_manifest']:
  F[n]=frame(DATA/(n+'.csv'));atomic(OUT/'inputs'/(n+'.csv'),(DATA/(n+'.csv')).read_bytes())
 for p in [DATA/'README.md',DATA/'historical_source_identities.json',DATA/'canonical_data_hashes.json',DATA/'mask_grid_validation.json',DATA/'implementation_validation.json',DATA/'audit_handoff_validation.json']:
  if p.exists():sha(p);atomic(OUT/'inputs'/p.name,p.read_bytes())
 for p in list((ROOT/'scripts/fixed_budget_dataset').glob('*.py'))+[ROOT/'src/moe_lth/experiments/run_router_age_recovery.py',ROOT/'src/moe_lth/pruning/magnitude_prune.py',ROOT/'src/moe_lth/pruning/masks.py',ROOT/'src/moe_lth/training/train.py',ROOT/'configs/multidomain_gpu.yaml']:
  sha(p);atomic(OUT/'inputs/code'/p.relative_to(ROOT),p.read_bytes())
 cpmap=F['source_checkpoint_manifest'].set_index(['num_experts','seed','router_age']).to_dict('index');smap=F['source_trajectory_manifest'].set_index(['num_experts','seed']).to_dict('index');mmap=F['mask_manifest'].set_index(['num_experts','seed','target_sparsity']).to_dict('index')
 hist=F['historical_recovery_audit'];ident=read(DATA/'historical_source_identities.json');datahash=read(DATA/'canonical_data_hashes.json')
 jobmap={k:g.to_dict('records') for k,g in jobs.groupby('condition_id')};expected=[]
 for e,s in itertools.product(ES,SEEDS):
  expected.append(dict(condition_id=C.cid('source',e,s),architecture=e,seed=s,router_age=None,sparsity=None,job_type='source'))
  for a in AGES:
   expected.append(dict(condition_id=C.cid('dense',e,s,a),architecture=e,seed=s,router_age=a,sparsity=0.,job_type='dense'))
   expected.extend(dict(condition_id=C.cid('sparse',e,s,a,sp),architecture=e,seed=s,router_age=a,sparsity=sp,job_type='sparse') for sp in SP)
 # Index all existing recovery directories, including restart attempts and preserved backup copies.
 candidates={};orphans=[]
 for base in [DATA/'recoveries',BACK/'recoveries']:
  for p in sorted(base.rglob('metadata.json')):
   try:m=json.loads(p.read_text());k=m.get('condition');candidates.setdefault(k,[]).append(p.parent)
   except Exception as ex:orphans.append({'path':str(p),'reason':str(ex)})
 # Optional user-provided additional read-only roots.
 for base in os.environ.get('DATASET_A_EXTRA_ROOTS','').split(os.pathsep):
  if not base:continue
  for p in sorted(Path(base).rglob('metadata.json')):
   try:m=json.loads(p.read_text());candidates.setdefault(m.get('condition'),[]).append(p.parent)
   except Exception:pass
 audit=[];records={};sources={};maskstates={};rewound={};candidate_audit=[];trajectories=[]
 datafiles={k:sha(ROOT/C.config(8,7)['data'][k]) for k in ['train_path','validation_path']}
 for e,s in itertools.product(ES,SEEDS):
  cid=C.cid('source',e,s);jr=jobmap.get(cid,[]);root=relocated(jr[0]['artifact_path']) if jr else Path('/nonexistent');errors=[];good=True
  try:
   c=yaml.safe_load((root/'resolved_config.yaml').read_text());sha(root/'resolved_config.yaml');err=C.protocol_errors(c,e,s)
   if err:errors+=err
   initial=C.load(root/'checkpoints/step_0.pt');config_hash=hashlib.sha256(json.dumps(initial['config'],sort_keys=True).encode()).hexdigest()
   for a in AGES:
    p=relocated(Path(cpmap[e,s,a]['path']));h=sha(p);check(cid,'source_checkpoint',f'R{a}_sha256',h,cpmap[e,s,a]['sha256']) or errors.append(f'R{a} hash mismatch')
   for sp in SP:
    info=mmap[e,s,sp];p=relocated(info['path']);sha(p);mask=torch.load(p,map_location='cpu',weights_only=True);mh=state_dict_hash(mask)
    if not check(cid,'mask',str(sp),mh,info['mask_hash']):errors.append(f'mask {sp} mismatch')
    if len(mask)!=8*e:errors.append('mask tensor count')
    state={n:v*mask[n] if n in mask else v for n,v in initial['model'].items() if parameter_group(n)=='expert'}
    rewound[e,s,sp]=state_dict_hash(state);maskstates[e,s,sp]=str(p);del state,mask
   ep=root/'logs/training_exposure.jsonl';sx=exposures(ep,True,e)
   if ep.exists():sha(ep)
   sources[e,s]={**smap[e,s],**sx,'path':str(root),'config':c,'config_hash':config_hash,'valid':not errors}
   del initial
  except Exception as ex:errors.append(repr(ex));sources[e,s]={**smap.get((e,s),{}),'path':str(root),'valid':False}
  audit.append(dict(condition_id=cid,architecture=f'{e}E',seed=s,router_age=None,sparsity=None,job_type='source',manifest_status=jr[0]['status'] if jr else 'MISSING',artifact_exists=root.exists(),artifact_valid=not errors,protocol_valid=not errors,comparator_valid=True,condition_valid=not errors and len(jr)==1,duplicate_manifest_rows=max(len(jr)-1,0),artifact_path=str(root),failure_reason='; '.join(errors)))
  jsave(cache,dict(hcache));print('SOURCE CHECKED',e,s,'PASS' if not errors else errors,flush=True)
 def validate(x,p):
  cid=x['condition_id'];e,s,a,sp=x['architecture'],x['seed'],x['router_age'],x['sparsity'];err=[];meta={};digest=None
  try:
   for n in ['metadata.json','summary.json','metrics.jsonl']:sha(p/n)
   m,su,ini,fin,auc=metrics(p);meta=m;curve=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
   if [z['step'] for z in curve]!=list(range(0,2501,50)):err.append('curve cadence/order')
   expect={'seed':s,'router_step':a*25,'router_age_percent':a,'sparsity':sp,'recovery_steps':2500,'recovery_eval_interval':50,'early_auc_window_steps':1250,'optimizer':'fresh_AdamW','optimizer_state_entries_at_start':0,'scheduler':'none','precision':'fp16','pruning_method':'expert_local_magnitude','confidence_control':False,'integrity_checks_passed':True,'training_batch_sequence_hash':datahash[str(s)]['train'],'validation_batch_sequence_hash':datahash[str(s)]['validation'],'initial_router_state_hash':cpmap[e,s,a]['router'],'shared_state_hash':cpmap[e,s,0]['shared'],'expert_state_hash':rewound[e,s,sp] if sp else cpmap[e,s,0]['expert'],'mask_hash':mmap[e,s,sp]['mask_hash'] if sp else 'dense_no_mask'}
   for k,v in expect.items():
    if not check(cid,'recovery_metadata',k,m.get(k),v):err.append(k)
   if m.get('router_mode','trainable')!='trainable' or m.get('temperature',1)!=1 or m.get('initial_router_state_hash')==m.get('final_router_state_hash'):err.append('native routing')
   if not sources[e,s]['valid']:err.append('source invalid')
   if abs(su['initial_validation_loss']-ini)>1e-10 or abs(su['early_auc']-auc)>1e-8:err.append('summary mismatch')
   att=read(p/'validated_result.json') if (p/'validated_result.json').exists() else {}
   # Historical audit path remains the original path even when the physical copy moved.
   oldpath=str(p).replace(str(BACK),str(DATA));hh=hist[(hist.path==oldpath)&hist.valid.eq(True)]
   att_hash=att.get('final_checkpoint_sha256') or (hh.iloc[0].final_checkpoint_hash if len(hh) else None)
   final=p/'checkpoints/final_recovered.pt';digest=sha(final)
   if not att_hash or digest!=att_hash:err.append('final checkpoint attestation/hash mismatch')
   ck=C.load(final);err.extend(C.protocol_errors(ck.get('config',{}),e,s))
   if ck.get('step')!=2500:err.append('checkpoint final step')
   if sp:
    mask=torch.load(maskstates[e,s,sp],map_location='cpu',weights_only=True)
    if any(bool(ck['model'][n][~v.bool()].any()) for n,v in mask.items()):err.append('nonzero pruned weights')
    del mask
   if any(not bool(torch.isfinite(v).all()) for v in ck['model'].values()):err.append('nonfinite checkpoint tensor')
   cfg_hash=hashlib.sha256(json.dumps(ck['config'],sort_keys=True).encode()).hexdigest();del ck
   if 'optimizer_step_attempts' in m and m['optimizer_step_attempts']!=2500:err.append('optimizer attempts')
   if 'optimizer_steps_applied' in m and m.get('amp_overflow_step_count') is not None and m['optimizer_steps_applied']+m['amp_overflow_step_count']!=2500:err.append('AMP accounting')
   return dict(metadata=m,summary=su,curve=curve,initial=ini,final=fin,auc=auc,path=str(p),final_checkpoint_sha256=digest,config_hash=cfg_hash),err
  except Exception as ex:err.append(repr(ex));return None,err
 from concurrent.futures import ThreadPoolExecutor
 pool=ThreadPoolExecutor(max_workers=6)
 futures={}
 for x in expected:
  if x['job_type']=='source':continue
  jr=jobmap.get(x['condition_id'],[])
  pp=([relocated(jr[0]['artifact_path'])] if jr else [])+candidates.get(x['condition_id'],[])
  for p in pp:
   key=(x['condition_id'],str(p.resolve()))
   if key not in futures and (p/'summary.json').exists() and (p/'checkpoints/final_recovered.pt').exists():futures[key]=pool.submit(validate,x,p)
 for i,x in enumerate(expected):
  if x['job_type']=='source':continue
  cid=x['condition_id'];jr=jobmap.get(cid,[]);paths=[]
  if jr:
   paths.append(relocated(jr[0]['artifact_path']))
  paths+=candidates.get(cid,[]);unique=[]
  for p in paths:
   if str(p.resolve()) not in [str(q.resolve()) for q in unique]:unique.append(p)
  selected=None;allerr=[];nvalid=0
  for p in unique:
   if not (p/'summary.json').exists() or not (p/'checkpoints/final_recovered.pt').exists():
    candidate_audit.append(dict(condition_id=cid,path=str(p),valid=False,reason='missing endpoint artifacts'));continue
   record,err=futures[cid,str(p.resolve())].result();candidate_audit.append(dict(condition_id=cid,path=str(p),valid=not err,reason='; '.join(err),checkpoint_sha256=record.get('final_checkpoint_sha256') if record else None));allerr.extend(err)
   if not err:
    nvalid+=1
    if selected is None:selected=record
    elif selected['final_checkpoint_sha256']!=record['final_checkpoint_sha256']:allerr.append('ambiguous valid duplicate checkpoints')
  duplicate_conflict='ambiguous valid duplicate checkpoints' in allerr
  valid=selected is not None and not duplicate_conflict and len(jr)==1
  if valid:records[cid]={**x,**selected}
  exists=any(p.exists() for p in unique)
  reasons=[] if valid else (allerr or ['missing complete recovery artifacts'])
  audit.append(dict(condition_id=cid,architecture=f"{x['architecture']}E",seed=x['seed'],router_age=x['router_age'],sparsity=x['sparsity'],job_type=x['job_type'],manifest_status=jr[0]['status'] if jr else 'MISSING',artifact_exists=exists,artifact_valid=selected is not None,protocol_valid=selected is not None,comparator_valid=True,condition_valid=valid,duplicate_manifest_rows=max(len(jr)-1,0),valid_artifact_copies=nvalid,artifact_path=selected['path'] if selected else str(unique[0]) if unique else None,failure_reason='; '.join(reasons)))
  if i%25==0:
   csv(OUT/'audits/dataset_a_completion_audit.csv',audit);jsave(cache,dict(hcache));print('AUDITED',i,'/',len(expected),'valid recoveries',len(records),flush=True)
 pool.shutdown(wait=True)
 # Read raw training diagnostics once for exposure and full trajectories. Preserve unlogged fields as NA.
 for i,(cid,r) in enumerate(records.items()):
  e=r['architecture'];p=Path(r['path']);gp=p/'gradient_stats/gradient_stats.jsonl';curve={z['step']:z for z in r['curve']};train={};applied=0;have_flags=True
  if gp.exists():
   sha(gp)
   with gp.open() as h:
    for line in h:
     z=json.loads(line);step=z['step'];have_flags=have_flags and ('optimizer_step_applied' in z)
     if step>0:applied+=int(z.get('optimizer_step_applied',False))
     train[step]={'training_loss':z.get('train_loss'),'auxiliary_loss':z.get('auxiliary_loss'),'optimizer_steps_applied':applied if have_flags else None}
  r['exposure']=exposures(gp,False,e)
  for step in sorted(set(curve)|set(train)):
   trajectories.append(dict(condition_id=cid,architecture=f'{e}E',seed=r['seed'],router_age=r['router_age'],sparsity=r['sparsity'],model_type=r['job_type'],recovery_step=step,validation_loss=curve.get(step,{}).get('loss'),training_loss=train.get(step,{}).get('training_loss'),auxiliary_loss=train.get(step,{}).get('auxiliary_loss'),learning_rate=.0003,optimizer_steps_applied=train.get(step,{}).get('optimizer_steps_applied'),artifact_path=str(p)))
  if i%100==0:print('TRAJECTORIES',i,'/',len(records),flush=True)
 # Existing developmental measurements only; use identity-validated interim table and preserve provenance.
 old=DATA/'interim_analysis/20260921T133833582668Z';dev=frame(old/'interim_router_development.csv');devmap=dev.set_index(['num_experts','seed','router_age']).to_dict('index');routercols=[c for c in dev if c.startswith('router_') and c!='router_age']
 rows=[];gaps=[];stored_candidates=[]
 for name,e in [('expert_count_replication/ticket_gap_seed_level.csv',None),('router_development_12expert_retry4_20260902/ticket_gap_seed_level.csv',12),('16e_router_age_localization/outcomes.csv',16),('24e_router_age_localization/outcomes.csv',24),('32e_router_age_localization/matched_ticket_results.csv',32)]:
  p=ROOT/'results'/name
  if p.exists():
   for z in frame(p).to_dict('records'):stored_candidates.append((e,z,str(p)))
 for x in expected:
  if x['job_type']!='sparse':continue
  cid=x['condition_id'];e,s,a,sp=x['architecture'],x['seed'],x['router_age'],x['sparsity'];dcid=C.cid('dense',e,s,a);r=records.get(cid);d=records.get(dcid);ar=next(v for v in audit if v['condition_id']==cid);ar['comparator_valid']=d is not None;ar['condition_valid']=ar['condition_valid'] and d is not None
  if d is None:ar['failure_reason']+='; no valid matched dense comparator'
  gaprow=dict(condition_id=cid,architecture=f'{e}E',seed=s,router_age=a,sparsity=sp,stored_gap=None,recomputed_gap=None,absolute_difference=None,dense_artifact_path=d['path'] if d else None,sparse_artifact_path=r['path'] if r else None,comparison_status='MISSING_OR_INVALID_ENDPOINT')
  if not r or not d:gaps.append(gaprow);continue
  gap=r['final']-d['final'];stored=r['summary'].get('ticket_gap');storedpath=r['path']+'/summary.json' if stored is not None else None
  for he,z,hp in stored_candidates:
   ze=int(z.get('expert_count',z.get('experts',he)));za=int(z.get('router_age',z.get('router_age_percent')));zs=int(z['seed']);zsp=float(z.get('sparsity',.85))
   if (ze,zs,za,zsp)!=(e,s,a,sp):continue
   if abs(z.get('sparse_final_loss',z.get('final_validation_loss'))-r['final'])<1e-10 and abs(z.get('dense_final_loss',z.get('matched_dense_final_loss'))-d['final'])<1e-10:stored=z['ticket_gap'];storedpath=hp;break
  diff=abs(stored-gap) if stored is not None else None;gaprow.update(stored_gap=stored,recomputed_gap=gap,absolute_difference=diff,stored_gap_artifact=storedpath,comparison_status='MATCH' if diff is not None and diff<=1e-10 else 'MISMATCH' if diff is not None else 'NO_STORED_GAP');gaps.append(gaprow)
  src=sources[e,s];cp=cpmap[e,s,a];mm=mmap[e,s,sp];dv=devmap.get((e,s,a),{});dvvalid=dv.get('sha256')==cp['sha256'];maskp=maskstates[e,s,sp]
  row=dict(condition_id=cid,architecture=f'{e}E',num_experts=e,seed=s,router_age=a,router_age_percent=a,source_checkpoint_step=a*25,sparsity=sp,retained_fraction=1-sp,protocol_version='historical-native-fixed-budget-v1',source_config_id=src['config_hash'],recovery_config_id=r['config_hash'],source_attempt_budget=2500,recovery_attempt_budget=2500,batch_size=128,sequence_length=128,top_k=1,capacity_factor=1.25,aux_loss_weight=.1,source_artifact_path=src['path'],source_checkpoint_path=str(relocated(cp['path'])),source_checkpoint_hash=cp['sha256'],router_checkpoint_hash=cp['router'],mask_artifact_path=maskp,mask_hash=mm['mask_hash'],dense_artifact_path=d['path'],sparse_artifact_path=r['path'],dense_condition_id=dcid,dense_config_hash=d['config_hash'],sparse_config_hash=r['config_hash'],data_hash=r['metadata']['training_batch_sequence_hash'],validation_hash=r['metadata']['validation_batch_sequence_hash'],dense_final_checkpoint_sha256=d['final_checkpoint_sha256'],sparse_final_checkpoint_sha256=r['final_checkpoint_sha256'],dense_initial_validation_loss=d['initial'],sparse_preprune_validation_loss=d['initial'],sparse_postprune_validation_loss=r['initial'],pruning_shock=r['initial']-d['initial'],dense_early_auc=d['auc'],sparse_early_auc=r['auc'],matched_early_auc_gap=r['auc']-d['auc'],dense_recovery_improvement=d['initial']-d['final'],sparse_recovery_improvement=r['initial']-r['final'],dense_final_validation_loss=d['final'],sparse_final_validation_loss=r['final'],ticket_gap=gap,source_valid=True,dense_valid=True,sparse_valid=True,mask_valid=True,comparator_valid=True,ticket_gap_valid=diff is None or diff<=1e-10,condition_valid=diff is None or diff<=1e-10,development_artifact=dv.get('development_artifact') if dvvalid else None,actual_sparsity=mm['actual_sparsity'],mask_keep_count=mm['keep_count'],mask_pruned_count=mm['pruned_count'])
  for k in routercols:row[k]=dv.get(k) if dvvalid else None
  for k in ['proposed_assignments','accepted_assignments','dropped_assignments','acceptance_rate','mean_accepted_per_expert','mean_active_attempt_steps_per_expert','mean_active_applied_steps_per_expert']:row['source_'+k]=src.get(k)
  for kind,rr in [('dense',d),('sparse',r)]:
   for k,v in rr['exposure'].items():row[kind+'_'+k]=v
   row[kind+'_optimizer_steps_applied']=rr['metadata'].get('optimizer_steps_applied');row[kind+'_amp_skipped_updates']=rr['metadata'].get('amp_overflow_step_count')
  rows.append(row)
 f=pd.DataFrame(rows);audit=csv(OUT/'audits/dataset_a_completion_audit.csv',audit);csv(OUT/'audits/dataset_a_ticket_gap_audit.csv',gaps);csv(OUT/'audits/dataset_a_candidate_audit.csv',candidate_audit);csv(OUT/'audits/dataset_a_protocol_audit.csv',protocol)
 csv(OUT/'audits/dataset_a_unexpected_manifest_cells.csv',jobs[~jobs.condition_id.isin([x['condition_id'] for x in expected])]);csv(OUT/'audits/dataset_a_missing_or_invalid_cells.csv',audit[~audit.condition_valid])
 table(OUT/'canonical/dataset_a_canonical',f);t=pd.DataFrame(trajectories);table(OUT/'trajectories/dataset_a_recovery_trajectories',t)
 csv(OUT/'canonical/dataset_a_sources.csv',[
   {'num_experts': e, 'seed': s,
    **{k:v for k,v in x.items() if k!='config'}}
   for (e,s),x in sources.items()
  ]);csv(OUT/'canonical/dataset_a_dense.csv',[{k:v for k,v in x.items() if k not in ['metadata','summary','curve','exposure']} for x in records.values() if x['job_type']=='dense'])
 csv(OUT/'canonical/dataset_a_router_development.csv',dev)
 jsave(OUT/'dataset_a_provenance.json',dict(timestamp_utc=stamp,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),data_file_sha256=datafiles,artifact_roots=[str(DATA),str(BACK)],selection='Manifest path first, then stable existing candidate path order; byte-identical copies allowed; conflicting valid checkpoint copies block condition',hash_scope='Every selected source checkpoint, mask, recovery checkpoint and raw log freshly SHA-256 checked or cache reused only with unchanged size and mtime; checkpoint configs, finite tensors and masked zeros checked',metric_reuse='AST-extracted metrics and exposures from scripts/fixed_budget_dataset/finalize.py',development='Only existing September 21 identity-validated measurements; NA for unmeasured source ages/seeds',training_started=False,dataset_b_started=False,raw_artifacts_modified=False))
 jsave(cache,dict(hcache));jsave(OUT/'dataset_a_input_hashes.json',inputs)
 print('ASSEMBLY DONE',len(f),len(t),flush=True)
if __name__=='__main__':main()
