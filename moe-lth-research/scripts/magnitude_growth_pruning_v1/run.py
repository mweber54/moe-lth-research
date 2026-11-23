"""One registered shared-reset continuation, with exact interruption state."""
from common import *
import argparse,signal

def run(seed,branch,attempt,resume=False):
 reg=frozen();cfg=reg['source_configs'][str(seed)];device=torch.device('cuda');torch.set_num_threads(4);R.configure_device(device)
 train,val=data(seed);identity=reg['data_identity'][str(seed)];assert R._batch_sequence_hash(train[1000:2500])==identity['continuation_train_hash'] and R._batch_sequence_hash(val)==identity['validation_hash'];train=train[1000:2500]
 model,opt,scaler,indices,handles=assemble(seed,branch,device);start=time.monotonic();applied=skips=0;begin=0;curves=[];perbatch=[];baseline=None;stop={'requested':False}
 def interrupt(*_):stop['requested']=True
 old_signals={s:signal.signal(s,interrupt) for s in (signal.SIGTERM,signal.SIGINT)}
 checkpoint=attempt/'interruption.pt';log=attempt/'training.jsonl';jobid=f'seed{seed}_{branch}';idn=dict(registration_sha256=sha(D/'registration.json'),seed=seed,branch=branch,source_checkpoint_sha256=reg['checkpoints'][str(seed)]['1000']['sha256'],data_identity=identity,mask_file_sha256=reg['mask_files'][str(seed)]['sha256'],budget=1500)
 def save_interrupt(step):
  atomic_tensor(checkpoint,dict(identity=idn,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),step=step,source_training_step=1000+step,applied=applied,skips=skips,curves=curves,per_batch=perbatch,baseline=baseline,training_log_bytes=log.stat().st_size if log.exists() else 0,wall_seconds_prior=prior_wall+time.monotonic()-start,**rng()))
 prior_wall=0.
 try:
  if resume:
   saved=decode_numpy(torch.load(checkpoint,map_location='cpu',weights_only=True));assert saved['identity']==idn;model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);begin=saved['step'];applied=saved['applied'];skips=saved['skips'];curves=saved['curves'];perbatch=saved['per_batch'];baseline=saved['baseline'];prior_wall=saved['wall_seconds_prior']
   if log.exists():
    with log.open('r+b') as f:f.truncate(saved['training_log_bytes'])
   save(attempt/'curve.json',curves)
   set_rng(saved);check_zero(model,indices)
  else:
   assert not checkpoint.exists() and not log.exists()
   # Unmasked baseline is evaluated from a separate exact copy; no control substitution.
   unmasked=load_model_from_checkpoint(cfg['model'],reg['checkpoints'][str(seed)]['1000']['path'],device);baseline,_=evaluate(unmasked,val,device);del unmasked;torch.cuda.empty_cache()
   initial_ce,batches=evaluate(model,val,device)
   if branch=='control':assert initial_ce==baseline
   curves.append(dict(continuation_step=0,source_training_step=1000,ce=initial_ce));perbatch.extend(dict(continuation_step=0,**r) for r in batches)
   save(attempt/'initial_integrity.json',dict(initial_checkpoint_hash=idn['source_checkpoint_sha256'],unmasked_model_hash=reg['checkpoints'][str(seed)]['1000']['model_hash'],masked_model_hash=state_dict_hash(model.state_dict()),baseline_ce=baseline,post_mask_ce=initial_ce,fresh_optimizer_entries=len(opt.state),scaler=scaler.state_dict(),all_parameters_trainable=all(p.requires_grad for p in model.parameters()),tied_weights=model.token_embedding.weight is model.lm_head.weight,removed_coordinates=sum(len(i) for i in indices.values()),rng_reset_seed=seed,source_data_offset=1000))
   seed_everything(seed) # production stream starts from the common registered RNG reset
   save_interrupt(0)
  for step in range(begin+1,1501):
   row=update(model,opt,scaler,train[step-1],cfg,indices,device);applied+=row['applied'];skips+=row['amp_skip'];append(log,dict(continuation_step=step,source_training_step=1000+step,**row))
   if skips>100:raise FloatingPointError('Registered AMP-skip engineering ceiling exceeded')
   if step in EVAL_STEPS:
    ce,batches=evaluate(model,val,device);curves.append(dict(continuation_step=step,source_training_step=1000+step,ce=ce));perbatch.extend(dict(continuation_step=step,**r) for r in batches);save(attempt/'curve.json',curves);print(jobid,step,'CE',ce,'wall',round(time.monotonic()-start),flush=True)
   if step%25==0:save(D/'status'/f'{jobid}.json',dict(status='RUNNING',seed=seed,branch=branch,attempt=str(attempt),host=socket.getfqdn(),pid=os.getpid(),continuation_step=step,source_training_step=1000+step,applied=applied,amp_skips=skips,heartbeat=time.time()))
   if step%250==0 or stop['requested']:save_interrupt(step)
   if stop['requested']:
    save(D/'status'/f'{jobid}.json',dict(status='INTERRUPTED',attempt=str(attempt),seed=seed,branch=branch,last_committed_step=step,checkpoint=str(checkpoint)));return None
  assert applied+skips==1500 and [r['continuation_step'] for r in curves]==list(EVAL_STEPS);check_zero(model,indices)
  assert all(torch.isfinite(p).all() for p in model.parameters())
  params=dict(model.named_parameters());moment_checks=True
  for n,idx in indices.items():
   for field in ('exp_avg','exp_avg_sq'):
    if field in opt.state[params[n]]:assert torch.count_nonzero(opt.state[params[n]][field].view(-1)[idx]).item()==0
  out=attempt/'final.pt';atomic_tensor(out,dict(model={n:t.detach().cpu() for n,t in model.state_dict().items()},step=2500,continuation_step=1500,config=cfg,registration_sha256=sha(D/'registration.json')))
  save(attempt/'curve.json',curves);table(attempt/'per_batch_validation.csv',perbatch)
  result=dict(seed=seed,branch=branch,attempt=str(attempt),checkpoint=str(out),checkpoint_sha256=sha(out),final_model_hash=state_dict_hash(model.state_dict()),initial_checkpoint_hash=idn['source_checkpoint_sha256'],baseline_ce=baseline,immediate_ce=curves[0]['ce'],final_ce=curves[-1]['ce'],attempted_updates=1500,applied_updates=applied,amp_skips=skips,source_training_step=2500,precision='FP16 train; FP32 TF32-off task-only validation',mask_persistence=True,removed_optimizer_moments_zero=moment_checks,all_parameters_trainable=True,train_sequence_hash=identity['continuation_train_hash'],validation_sequence_hash=identity['validation_hash'],wall_seconds=prior_wall+time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),registration_sha256=sha(D/'registration.json'))
  save(attempt/'result.json',result);save(D/'status'/f'{jobid}.json',dict(status='COMPLETE',result_path=str(attempt/'result.json'),attempt=str(attempt),host=socket.getfqdn(),finished=time.time()));return result
 finally:
  for h in handles:h.remove()
  for s,old in old_signals.items():signal.signal(s,old)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--branch',choices=BRANCHES,required=True);p.add_argument('--attempt',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 with lock(Path(a.attempt)/'execution.lock',False):run(a.seed,a.branch,Path(a.attempt),a.resume)
