"""Execute only audited missing Dataset A jobs; preserve historical artifacts."""
from common import *
import os,time,shutil,traceback,fcntl,math
import pandas as pd
from moe_lth.training.train import train_from_config
from moe_lth.data import build_dataloaders
from moe_lth.utils import seed_everything,configure_device
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.pruning.router_age import selected_experts_per_batch
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 lock=(OUT/'build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 audit=jread(OUT/'audit_complete.json');assert audit['status']=='COMPLETE' and sha(OUT/'canonical_grid_audit.csv')==audit['grid_sha256']
 if (OUT/'dataset_manifest.json').exists() and jread(OUT/'dataset_manifest.json').get('frozen'):return
 jobs=pd.read_csv(OUT/'run_manifest.csv').fillna('')
 # Read-only prelaunch check: all chosen reused endpoints must be finite.
 for item in jobs[(jobs.status=='REUSED') & (jobs.job_type!='source')].itertuples():
  points=[json.loads(x) for x in (Path(item.artifact_path)/'metrics.jsonl').read_text().splitlines()]
  assert all(math.isfinite(x['loss']) for x in points),f'Nonfinite historical endpoint: {item.condition_id}'
 history=jread(OUT/'historical_source_identities.json');datahash=jread(OUT/'canonical_data_hashes.json')
 device=torch.device('cuda');assert torch.__version__.startswith('2.7.0') and torch.cuda.is_available();configure_device(device);torch.set_num_threads(4)
 def save():
  writecsv(OUT/'run_manifest.csv',jobs)
  jwrite(OUT/'build_status.json',{'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'counts':{f'{k[0]}:{k[1]}':int(v) for k,v in jobs.groupby(['job_type','status']).size().items()},'running_conditions':jobs.loc[jobs.status=='RUNNING','condition_id'].tolist(),'dataset_frozen':False})
 def idx(kind,e,s,a=None,sp=None):
  matches=jobs.index[jobs.condition_id==cid(kind,e,s,a,sp)];assert len(matches)==1;return matches[0]
 def begin(i):
  if shutil.disk_usage(OUT).free<12*1024**3:raise RuntimeError('Storage reserve below 12GiB; preserved existing artifacts and stopped before new job')
  p=Path(jobs.at[i,'artifact_path']);attempt=int(jobs.at[i,'attempt_count'])+1
  if p.exists() and any(p.iterdir()):p=p.parent/(p.name+f'_attempt_{attempt}')
  assert not p.exists() or not any(p.iterdir())
  jobs.at[i,'artifact_path']=str(p);jobs.at[i,'attempt_count']=attempt;jobs.at[i,'status']='RUNNING';jobs.at[i,'last_error']='';save();return p
 def complete(i):jobs.at[i,'status']='COMPLETE';save()
 def error(i,ex):jobs.at[i,'status']='FAILED';jobs.at[i,'last_error']=repr(ex);save()
 # Source jobs first. Original seed reconstructions must exactly reproduce all available historical states.
 for e in ES:
  for seed in SEEDS:
   i=idx('source',e,seed);root=Path(jobs.at[i,'artifact_path'])
   if jobs.at[i,'status'] not in ('COMPLETE','REUSED'):
    # Recover a fully completed output if interruption occurred between artifact completion and manifest update.
    finished=(root/'summary.json').exists() and all((root/f'checkpoints/step_{a*25}.pt').exists() for a in AGES)
    try:
     if not finished:
      root=begin(i);c=config(e,seed,root);train_from_config(c)
     for a in AGES:
      p=root/f'checkpoints/step_{a*25}.pt';ck=load(p);assert ck['step']==25*a and not protocol_errors(ck['config'],e,seed)
      previous=history['checkpoints'].get(f'{e}:{seed}:{a}')
      if previous:assert state_dict_hash(ck['model'])==previous['model_hash'],f'Reconstructed source differs: E{e} seed{seed} R{a}; historical reuse halted'
     complete(i)
    except Exception as ex:error(i,ex);raise
   print('SOURCE READY',e,seed,root,flush=True)
 # One source/model/mask bank per E,seed; dense reused across all six sparsities.
 for e in ES:
  for seed in SEEDS:
   root=Path(jobs.at[idx('source',e,seed),'artifact_path']);c=load(root/'checkpoints/step_0.pt')['config'];assert not protocol_errors(c,e,seed)
   initial=load(root/'checkpoints/step_0.pt')['model'];state={n:v.clone() for n,v in initial.items()}
   final=r.load_model_from_checkpoint(c['model'],str(root/'checkpoints/step_2500.pt'),device)
   seed_everything(seed);tr,va=build_dataloaders(c['data'],128,seed);train=r._materialize_batches(tr,2500);val=r._materialize_validation_batches(va,32);probe=r._calibration_batches(val)
   trainhash=r._batch_sequence_hash(train);valhash=r._batch_sequence_hash(val);assert trainhash==datahash[str(seed)]['train'] and valhash==datahash[str(seed)]['validation']
   selected=selected_experts_per_batch(final,probe,device);source_loss=r.evaluate_language_model(final,val,device,max_batches=len(val))['loss'];maskbank={}
   for sp in SP:
    mm=expert_local_magnitude_masks(final,sp);p=OUT/f'masks/e{e}/seed_{seed}/s{sp:.2f}.pt';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert state_dict_hash(torch.load(p,map_location='cpu',weights_only=True))==state_dict_hash(mm)
    else:torch.save(mm,p)
    maskbank[sp]=mm
   del final;torch.cuda.empty_cache()
   for a in AGES:
    cp=root/f'checkpoints/step_{25*a}.pt';router=group(load(cp)['model'],'router');routerhash=state_dict_hash(router)
    for kind,sp in [('dense',None)]+[('sparse',sp) for sp in SP]:
     i=idx(kind,e,seed,a,sp);dest=Path(jobs.at[i,'artifact_path']);mask={} if sp is None else maskbank[sp];base=state if sp is None else {n:v*mask[n] if n in mask else v for n,v in state.items()}
     expert_hash=state_dict_hash(group(base,'expert'));shared_hash=state_dict_hash(group(base,'shared'))
     def validate(p):
      m=jread(p/'metadata.json');s=jread(p/'summary.json');ck=load(p/'checkpoints/final_recovered.pt')
      assert ck['step']==2500 and not protocol_errors(ck['config'],e,seed)
      assert m['seed']==seed and m['router_step']==a*25 and m['initial_router_state_hash']==routerhash
      assert m['expert_state_hash']==expert_hash and m['shared_state_hash']==shared_hash and m['recovery_steps']==2500
      assert m['training_batch_sequence_hash']==trainhash and m['validation_batch_sequence_hash']==valhash
      if mask:
       assert m['mask_hash']==state_dict_hash(mask)
       assert all(not bool(ck['model'][n][~v.bool()].any()) for n,v in mask.items())
      rows=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()];assert rows[-1]['step']==2500 and abs(rows[-1]['loss']-s['final_validation_loss'])<1e-10
      return {**m,**s}
     try:
      if jobs.at[i,'status'] in ('COMPLETE','REUSED'):
       record=validate(dest);print('REUSE',jobs.at[i,'condition_id'],flush=True);continue
      if (dest/'summary.json').exists() and (dest/'metadata.json').exists() and (dest/'checkpoints/final_recovered.pt').exists():record=validate(dest);complete(i);continue
      dest=begin(i)
      dense_loss=source_loss if sp is None else jread(Path(jobs.at[idx('dense',e,seed,a),'artifact_path'])/'summary.json')['final_validation_loss']
      record=r._run_recovery_condition(config=c,condition_name=jobs.at[i,'condition_id'],pruned_base_state=base,router_checkpoint=str(cp),router_age_percent=a,router_step=a*25,masks=mask,expert_hash=expert_hash,shared_hash=shared_hash,mask_hash=state_dict_hash(mask) if mask else 'dense_no_mask',reference_selected=selected,calibration_batches=probe,train_batches=train,validation_batches=val,train_batch_hash=trainhash,validation_batch_hash=valhash,device=device,recovery_steps=2500,dense_loss=dense_loss,output_dir=dest,confidence_control=False,target_confidence=None,seed=seed,sparsity=sp or 0.,router_mode='trainable')
      validate(dest);record.update(native_routing=True,forward_callbacks=None,final_checkpoint_sha256=sha(dest/'checkpoints/final_recovered.pt'),source_checkpoint_sha256=sha(cp));jwrite(dest/'validated_result.json',record);complete(i)
      print('COMPLETE',jobs.at[i,'condition_id'],'loss',record['final_validation_loss'],flush=True)
     except Exception as ex:error(i,ex);raise
   del maskbank,state,initial,train,val,probe;torch.cuda.empty_cache()
 # Exact finalization never starts Dataset B or any mechanism experiment.
 import finalize
 finalize.main()

if __name__=='__main__':
 try:main()
 except Exception as ex:
  jwrite(OUT/'build_failure.json',{'error':repr(ex),'traceback':traceback.format_exc(),'time':time.time(),'action':'Stopped; historical and completed artifacts preserved. Resume after resolving the recorded failure.'});raise
