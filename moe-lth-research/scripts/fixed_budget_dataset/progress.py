"""Incremental, read-only artifact validation and manifest exports during Dataset A."""
from common import *
import argparse,fcntl,os,time,traceback
import pandas as pd
from finalize import exposures,metrics

CACHE=OUT/'progress_cache'
_RECORDS={}

def source_record(job,history,dataidentity):
 e=int(job.architecture);seed=int(job.seed);root=Path(job.artifact_path)
 cache=CACHE/(job.condition_id+'.json')
 key=(job.condition_id,str(root))
 if key in _RECORDS:return _RECORDS[key]
 if cache.exists():
  record=jread(cache)
  if record['root']==str(root):
   _RECORDS[key]=record;return record
 initial=load(root/'checkpoints/step_0.pt');c=initial['config']
 assert not protocol_errors(c,e,seed)
 confighash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
 checkpoints=[]
 for age in AGES:
  path=root/f'checkpoints/step_{age*25}.pt';assert path.is_file()
  old=history['checkpoints'].get(f'{e}:{seed}:{age}')
  if job.status=='REUSED' and old and Path(old['path'])==path:
   hashes={g:old[g+'_hash'] for g in ('router','expert','shared')};filehash=old['sha256'];modelhash=old['model_hash']
  else:
   ck=load(path);assert ck['step']==age*25 and not protocol_errors(ck['config'],e,seed)
   assert all(bool(torch.isfinite(v).all()) for v in ck['model'].values())
   hashes={g:state_dict_hash(group(ck['model'],g)) for g in ('router','expert','shared')}
   filehash=sha(path);modelhash=state_dict_hash(ck['model'])
   if old:assert modelhash==old['model_hash'],f'Historical overlap mismatch: {job.condition_id} R{age}'
  checkpoints.append({'num_experts':e,'seed':seed,'router_age':age,'source_checkpoint_step':age*25,'path':str(path),'sha256':filehash,'model_hash':modelhash,**hashes,'config_hash':confighash})
 assert len({x['model_hash'] for x in checkpoints})==11,f'Unexpected identical checkpoint states: {job.condition_id}'
 sx=exposures(root/'logs/training_exposure.jsonl',True,e)
 source={'condition_id':job.condition_id,'num_experts':e,'seed':seed,'path':str(root),'status':job.status,'training_steps_attempted':2500,'training_steps_applied':sx.get('applied_steps'),'batch_size':128,'sequence_length':128,'gradient_accumulation':1,'effective_batch_size':128,'input_tokens_processed':40960000,'optimizer':'AdamW','learning_rate':.0003,'scheduler':'none','warmup':0,'weight_decay':.01,'grad_clip':1,'precision':'fp16','GradScaler':True,'dataset_hashes':json.dumps(dataidentity,sort_keys=True),'config_hash':confighash,**sx}
 record={'root':str(root),'source':source,'checkpoints':checkpoints};jwrite(cache,record);_RECORDS[key]=record;return record

def recovery_record(job,historical):
 path=Path(job.artifact_path);cache=CACHE/(job.condition_id+'.json')
 key=(job.condition_id,str(path))
 if key in _RECORDS:return _RECORDS[key]
 if cache.exists():
  record=jread(cache)
  if record['path']==str(path):
   _RECORDS[key]=record;return record
 e=int(job.architecture);seed=int(job.seed);age=int(job.router_age)
 m,s,initial,final,auc=metrics(path)
 checked=path/'validated_result.json'
 digest=jread(checked).get('final_checkpoint_sha256') if checked.exists() else None
 digest=digest or historical.get(str(path)) or sha(path/'checkpoints/final_recovered.pt')
 x=exposures(path/'gradient_stats/gradient_stats.jsonl',False,e)
 record={'condition_id':job.condition_id,'num_experts':e,'seed':seed,'router_age':age,'sparsity':float(job.sparsity),'path':str(path),'status':job.status,'initial_validation_loss':initial,'early_auc':auc,'final_loss':final,'recovery_improvement':initial-final,'final_checkpoint_sha256':digest,'router_hash':m['initial_router_state_hash'],'expert_hash':m['expert_state_hash'],'shared_hash':m['shared_state_hash'],'mask_hash':m['mask_hash'],'data_hash':m['training_batch_sequence_hash'],'validation_data_hash':m['validation_batch_sequence_hash'],'recovery_attempts':2500,'native_routing':True,**x}
 if job.job_type=='sparse':record['dense_condition_id']=cid('dense',e,seed,age)
 jwrite(cache,record);_RECORDS[key]=record;return record

def sync():
 if (OUT/'dataset_manifest.json').exists() and jread(OUT/'dataset_manifest.json').get('frozen'):return True
 jobs=pd.read_csv(OUT/'run_manifest.csv').fillna('');ready=jobs[jobs.status.isin(['COMPLETE','REUSED'])]
 history=jread(OUT/'historical_source_identities.json')
 historical={x['path']:x['final_checkpoint_hash'] for x in pd.read_csv(OUT/'historical_recovery_audit.csv').fillna('').to_dict('records') if x['valid']}
 identity_path=CACHE/'dataset_hashes.json'
 if identity_path.exists():dataidentity=jread(identity_path)
 else:
  dataidentity={k:sha(ROOT/config(8,7)['data'][k]) for k in ('train_path','validation_path')};jwrite(identity_path,dataidentity)
 sources=[];checkpoints=[]
 for job in ready[ready.job_type=='source'].itertuples():
  record=source_record(job,history,dataidentity);sources.append(record['source']);checkpoints.extend(record['checkpoints'])
  writecsv(OUT/'source_trajectory_manifest.csv',sources);writecsv(OUT/'source_checkpoint_manifest.csv',checkpoints)
 for kind in ('dense','sparse'):
  records=[recovery_record(job,historical) for job in ready[ready.job_type==kind].itertuples()]
  writecsv(OUT/f'{kind}_recovery_manifest.csv',records)
 # Masks are created by the execution pipeline; export existing masks without creating alternatives.
 mask_cache_path=CACHE/'masks.json';mask_cache=jread(mask_cache_path) if mask_cache_path.exists() else {}
 for path in sorted((OUT/'masks').glob('e*/seed_*/*.pt')):
  identity=f'{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
  if identity in mask_cache:continue
  e=int(path.parent.parent.name[1:]);seed=int(path.parent.name[5:]);sp=float(path.stem[1:]);mask=torch.load(path,map_location='cpu',weights_only=True)
  assert len(mask)==4*e*2
  keep=sum(int(v.sum()) for v in mask.values());total=sum(v.numel() for v in mask.values());digest=state_dict_hash(mask)
  assert abs(1-keep/total-sp)<1/total+1e-6
  old=history['masks'].get(f'{e}:{seed}:{sp:.2f}')
  if old:assert old['hash']==digest
  source=jobs[(jobs.job_type=='source') & (jobs.architecture==e) & (jobs.seed==seed)].iloc[0]
  root=Path(source.artifact_path)
  mask_cache[identity]={'num_experts':e,'seed':seed,'target_sparsity':sp,'actual_sparsity':1-keep/total,'keep_count':keep,'pruned_count':total-keep,'mask_hash':digest,'path':str(path),'mask_checkpoint':str(root/'checkpoints/step_2500.pt'),'mask_source_age':100,'bias_handling':'excluded','scope':'expert_local','method':'magnitude','rewind_checkpoint':str(root/'checkpoints/step_0.pt')}
 jwrite(mask_cache_path,mask_cache);writecsv(OUT/'mask_manifest.csv',list(mask_cache.values()))
 blocks=[]
 for e in ES:
  for seed in SEEDS:
   block=ready[(ready.architecture==e)&(ready.seed==seed)]
   dense=block[block.job_type=='dense'];sparse=block[block.job_type=='sparse']
   dense_ages={int(x) for x in dense.router_age};sparse_cells={(int(x.router_age),float(x.sparsity)) for x in sparse.itertuples()}
   assert len(dense)==len(dense_ages) and len(sparse)==len(sparse_cells)
   assert dense_ages<=set(AGES) and sparse_cells<={(a,sp) for a in AGES for sp in SP}
   source_ready=len(block[block.job_type=='source'])==1
   checkpoint_count=sum(x['num_experts']==e and x['seed']==seed for x in checkpoints)
   mask_count=sum(x['num_experts']==e and x['seed']==seed for x in mask_cache.values())
   complete=source_ready and checkpoint_count==11 and mask_count==6 and len(dense)==11 and len(sparse)==66
   blocks.append({'num_experts':e,'seed':seed,'source_validated':source_ready,'checkpoints_validated':checkpoint_count,'masks_validated':mask_count,'dense_validated':len(dense),'sparse_validated':len(sparse),'missing_dense':11-len(dense),'missing_sparse':66-len(sparse),'status':'PASS' if complete else 'INCOMPLETE'})
 writecsv(OUT/'block_integrity_manifest.csv',blocks)
 jwrite(OUT/'incremental_manifest_status.json',{'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'sources':len(sources),'source_checkpoints':len(checkpoints),'dense':int(((ready.job_type=='dense')).sum()),'sparse':int(((ready.job_type=='sparse')).sum()),'masks':len(mask_cache),'status':'PASS','dataset_frozen':False})
 print('MANIFESTS UPDATED',len(sources),len(checkpoints),len(ready),flush=True)
 return len(ready)==1950

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
 lock=(OUT/'progress.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 last_identity=None
 while True:
  identity=((OUT/'run_manifest.csv').stat().st_mtime_ns,len(list((OUT/'masks').glob('e*/seed_*/*.pt'))))
  if identity==last_identity:
   time.sleep(15);continue
  try:
   done=sync();last_identity=identity
  except Exception as ex:
   jwrite(OUT/'incremental_manifest_failure.json',{'error':repr(ex),'traceback':traceback.format_exc()});raise
  if done or not args.watch:return
  time.sleep(15)
if __name__=='__main__':main()
