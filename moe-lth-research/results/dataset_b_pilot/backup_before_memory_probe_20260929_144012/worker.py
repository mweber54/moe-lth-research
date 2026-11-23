"""One pilot job per process, no historical continuation or destructive retries."""
from pilot_common import *
import argparse,copy,time
from moe_lth.experiments import run_router_age_recovery as R
from moe_lth.data import build_dataloaders
from moe_lth.pruning.router_age import selected_experts_per_batch,component_state_dict
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.models import TinyMoELanguageModel
from moe_lth.utils import seed_everything,configure_device

def protected(fn):
 def wrapper(model,*args,**kwargs):
  mode=model.training
  with preserve_rng():
   try:return fn(model,*args,**kwargs)
   finally:model.train(mode)
 return wrapper

def source_info(seed,budget):
 if budget==2500:return read(OUT/'manifests/short_sources.json')[str(seed)]
 row=next(r for r in pd.read_csv(OUT/'manifests/run_manifest.csv').to_dict('records') if r['condition_id']==f'source_e32_seed{seed}_T10000')
 if row['status']!='COMPLETE':raise RuntimeError('Long source dependency not complete')
 return read(Path(row['artifact_path'])/'pilot_source_identity.json')

def source_job(job,dest):
 from moe_lth.training import train as T
 if dest.exists():raise FileExistsError('Fresh source attempts only')
 cfg=read(OUT/f"configs/source_e32_seed{int(job['seed'])}_T10000.json");cfg['output_dir']=str(dest)
 # Retain forward/evaluation cadence; suppress only unneeded token-level file exports.
 T.save_validation_routes=lambda *args,**kwargs:None
 T.sampled_context_records=lambda *args,**kwargs:[]
 T.train_from_config(cfg)
 checkpoints={};cps={}
 for step in [0,1250,2500,5000,10000]:
  p=dest/f'checkpoints/step_{step}.pt';ck=A.load(p);assert ck['step']==step
  cps[str(step)]={'path':str(p),'sha256':sha(p),'model_hash':state_dict_hash(ck['model'])}
  if step in (0,5000,10000):checkpoints[str(step//100)]={**cps[str(step)],'router_state_hash':state_dict_hash(A.group(ck['model'],'router'))}
 src=A.load(dest/'checkpoints/step_10000.pt');assert np.isfinite(src['validation_loss']) and all(bool(torch.isfinite(v).all()) for v in src['model'].values());seed=int(job['seed']);masks={}
 with preserve_rng():
  model=TinyMoELanguageModel(cfg['model']);model.load_state_dict(src['model'])
  for sp in SP:
   learned=expert_local_magnitude_masks(model,sp);randomized=random_mask(learned,job['condition_id'],sp);assert check_masks(learned,randomized);rec={}
   for kind,mask in [('learned',learned),('random',randomized)]:
    p=OUT/f'masks/long_seed{seed}_s{sp:.2f}_{kind}.pt'
    if p.exists():raise FileExistsError('Refusing to replace existing B mask')
    q=p.with_suffix('.pt.tmp');torch.save(mask,q);q.replace(p);rec.update({kind+'_path':str(p),kind+'_state_hash':state_dict_hash(mask),kind+'_file_sha256':sha(p)})
   rec['random_seed']=mask_seed(job['condition_id'],sp);masks[str(sp)]=rec
 exposure=exposure_at(dest/'logs/training_exposure.jsonl',[0,1250,2500,5000,10000],True);jsave(dest/'checkpoint_exposure.json',exposure)
 short=read(OUT/'manifests/short_sources.json')[str(seed)]
 identity={'path':str(dest),'config':cfg,'checkpoints':checkpoints,'absolute_checkpoints':cps,'masks':masks,'source_2500_equals_short_r100':cps['2500']['model_hash']==short['checkpoints']['100']['model_hash'],'source_r0_equals_short_r0':cps['0']['model_hash']==short['checkpoints']['0']['model_hash']}
 jsave(dest/'pilot_source_identity.json',identity)
 return {'final_checkpoint_sha256':cps['10000']['sha256'],'source_identity_sha256':sha(dest/'pilot_source_identity.json'),'attempts':10000}

def recovery_job(job,dest):
 seed,age,sb,rb=int(job['seed']),int(job['router_age']),int(job['source_attempts']),int(job['recovery_attempts']);info=source_info(seed,sb);cfg=copy.deepcopy(info['config']);device=torch.device('cuda:0');configure_device(device)
 initial=A.load(info['checkpoints']['0']['path']);cpath=Path(info['checkpoints'][str(age)]['path']);assert sha(cpath)==info['checkpoints'][str(age)]['sha256'];assert A.load(cpath)['step']==router_step(sb,age)
 mask={};mask_hash='dense_no_mask'
 if job['job_type']=='sparse':
  m=info['masks'][str(float(job['sparsity']))];kind=job['mask_type'];p=Path(m[kind+'_path']);assert sha(p)==m[kind+'_file_sha256'];mask=torch.load(p,map_location='cpu',weights_only=True);mask_hash=state_dict_hash(mask);assert mask_hash==m[kind+'_state_hash']
 base={n:v*mask[n] if n in mask else v for n,v in initial['model'].items()}
 seed_everything(seed);train,val=build_dataloaders(cfg['data'],128,seed,reshuffle_each_epoch=False);batches=R._materialize_batches(train,rb);validation=R._materialize_validation_batches(val,32);trainhash=R._batch_sequence_hash(batches);valhash=R._batch_sequence_hash(validation)
 canonical=read(A.OUT/'canonical_data_hashes.json')[str(seed)];assert R._batch_sequence_hash(batches[:2500])==canonical['train'];assert valhash==canonical['validation'];probe=R._calibration_batches(validation)
 R.routing_statistics=protected(R.routing_statistics);R.selected_experts_per_batch=protected(R.selected_experts_per_batch)
 with preserve_rng():
  ref=R.load_model_from_checkpoint(cfg['model'],info['checkpoints']['100']['path'],device);selected=R.selected_experts_per_batch(ref,probe,device);del ref
 if job['job_type']=='dense':
  ck=A.load(info['checkpoints']['100']['path']);dense_loss=ck['validation_loss'];assert dense_loss is not None
 else:
  df=pd.read_csv(OUT/'manifests/run_manifest.csv');dr=df[df.condition_id==job['dense_condition_id']].iloc[0];assert dr.status in ('COMPLETE','REUSED');dense_loss=read(Path(dr.artifact_path)/'summary.json')['final_validation_loss']
 diagnostics=tuple(sorted({0,min(2500,rb),rb}));checks=[]
 def snapshot(model,optimizer,scaler,step):
  if step not in diagnostics:return
  for n,p in model.named_parameters():
   if n not in mask:continue
   off=~mask[n].to(p.device).bool();assert not bool(p.detach()[off].any())
   for k in ('exp_avg','exp_avg_sq'):
    if k in optimizer.state.get(p,{}):assert not bool(optimizer.state[p][k][off].any())
  assert state_dict_hash(mask)==mask_hash if mask else mask_hash=='dense_no_mask'
  checks.append({'step':step,'masked_weights_zero':True,'masked_optimizer_moments_zero':True,'mask_hash':mask_hash})
  jsave(dest/'pilot_sparsity_checks.json',checks)
 R.EARLY_AUC_WINDOW_FRACTION=1250/rb
 start=time.time()
 result=R._run_recovery_condition(config=cfg,condition_name=job['condition_id'],pruned_base_state=base,router_checkpoint=str(cpath),router_age_percent=age,router_step=router_step(sb,age),masks=mask,expert_hash=state_dict_hash(A.group(base,'expert')),shared_hash=state_dict_hash(A.group(base,'shared')),mask_hash=mask_hash,reference_selected=selected,calibration_batches=probe,train_batches=batches,validation_batches=validation,train_batch_hash=trainhash,validation_batch_hash=valhash,device=device,recovery_steps=rb,dense_loss=dense_loss,output_dir=dest,confidence_control=False,target_confidence=None,seed=seed,sparsity=float(job['sparsity']),router_mode='trainable',diagnostic_steps=diagnostics,state_snapshot_callback=snapshot)
 meta=read(dest/'metadata.json');meta.update(dataset_b_protocol='dataset-b-nominal-exposure-v1',arm=job['arm'],source_attempts=sb,recovery_attempts=rb,mask_type=job['mask_type'],pruning_method='per_tensor_count_matched_random' if job['mask_type']=='random' else meta['pruning_method'],absolute_router_step=router_step(sb,age),early_auc_comparable_window=1250,elapsed_seconds=time.time()-start,short_training_prefix_hash=R._batch_sequence_hash(batches[:2500]),diagnostic_rng_preserved=True,initial_shared_expert_policy='source R0; router from requested source checkpoint')
 jsave(dest/'metadata.json',meta);jsave(dest/'checkpoint_exposure.json',exposure_at(dest/'gradient_stats/gradient_stats.jsonl',sorted({0,2500,rb})))
 curve=[json.loads(line) for line in (dest/'metrics.jsonl').read_text().splitlines()]
 assert [x['step'] for x in curve]==list(range(0,rb+1,50))
 assert all(np.isfinite(x['loss']) for x in curve)
 assert abs(curve[-1]['loss']-result['final_validation_loss'])<=1e-10
 assert meta['optimizer_step_attempts']==rb and meta['optimizer_steps_applied']+meta['amp_overflow_step_count']==rb
 final=A.load(dest/'checkpoints/final_recovered.pt');assert final['step']==rb and all(bool(torch.isfinite(v).all()) for v in final['model'].values())
 result['final_checkpoint_sha256']=sha(dest/'checkpoints/final_recovered.pt');result['mask_hash']=mask_hash;jsave(dest/'validated_result.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--attempt',required=True);a=p.parse_args();torch.set_num_threads(4)
 jobs=pd.read_csv(OUT/'manifests/run_manifest.csv');job=jobs[jobs.condition_id==a.condition].iloc[0].to_dict();dest=Path(a.attempt)
 if OUT.resolve() not in dest.resolve().parents or dest.exists():raise RuntimeError('Unsafe output/attempt path')
 result=source_job(job,dest) if job['job_type']=='source' else recovery_job(job,dest);jsave(dest/'pilot_completed.json',{'condition_id':a.condition,**result,'completed_at':time.time()});print('PILOT JOB COMPLETE',a.condition,flush=True)
if __name__=='__main__':main()
