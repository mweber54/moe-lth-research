"""Use the native recovery trainer twice; keep full instrumentation isolated."""
from common import *
from moe_lth.data import build_dataloaders
from moe_lth.pruning.router_age import load_model_from_checkpoint
import argparse,fcntl,socket,traceback


def batches(job,total=5000):
 c=job['config'];R.seed_everything(job['seed']);tr,va=build_dataloaders(c['data'],int(c['training']['batch_size']),job['seed']);train=R._materialize_batches(tr,total);val=R._materialize_validation_batches(va,int(c['data']['validation_blocks']));return train,val

def run_segment(job,initial,masks,router_path,train,val,dest,offset,steps,instrument=True):
 c=job['config'];eh=state_dict_hash({n:t for n,t in initial.items() if parameter_group(n)=='expert'});sh=state_dict_hash({n:t for n,t in initial.items() if parameter_group(n)=='shared'});dev=torch.device('cuda');reference=model(initial,c);refs=selected_experts_per_batch(reference,R._calibration_batches(val),dev);del reference
 previous={};before_updates={};snapshot_dir=dest.parent/'snapshots';update_rows=[]
 def forward(m,t,step):
  if instrument and (offset+step in STEPS or offset+step in (1,2499,2501,2526,2601)):
   before_updates.clear();before_updates.update({n:p.detach().clone() for n,p in m.named_parameters() if parameter_group(n) in ('router','expert')})
  return m(t)
 def callback(m,opt,scaler,step):
  absolute=offset+step
  save(dest.parent/'heartbeat.json',dict(host=socket.gethostname(),pid=os.getpid(),step=absolute,unix=time.time()))
  if not instrument:return
  if before_updates and step:
   rows=[]
   for n,p in m.named_parameters():
    if n not in before_updates:continue
    change=p.detach()-before_updates[n];grad=p.grad;den=change.norm()*grad.norm() if grad is not None else 0
    rows.append(dict(step=absolute,parameter=n,group=parameter_group(n),update_norm=float(change.norm()),clipped_gradient_norm=float(grad.norm()) if grad is not None else None,gradient_update_cosine=float((change*grad).sum()/den) if den>0 else None))
   table(dest.parent/'updates'/f'step_{absolute}.csv',rows);before_updates.clear()
  if absolute in STEPS or step in (0,steps):
   snapshot_dir.mkdir(parents=True,exist_ok=True);label=f'step_{absolute:04d}'+('_after_reset' if offset and step==0 else '')
   for n,mask in masks.items():assert torch.count_nonzero(m.state_dict()[n][~mask.to(dev)]).item()==0
   path=snapshot_dir/(label+'.pt');tmp=path.with_suffix('.tmp');torch.save(dict(model={n:t.detach().cpu().clone() for n,t in m.state_dict().items()},step=absolute,config=c,resumable=False),tmp);tmp.replace(path)
   save(snapshot_dir/(label+'.json'),dict(step=absolute,model_hash=state_dict_hash(m.state_dict()),expert_hash=state_dict_hash(component_state_dict(m,'expert')),router_hash=state_dict_hash(component_state_dict(m,'router')),shared_hash=state_dict_hash(component_state_dict(m,'shared')),snapshot_sha256=sha(path),optimizer_entries=len(opt.state),scaler=scaler.state_dict(),segment_offset=offset))
   if step in (0,steps):
    p=dest.parent/f'optimizer_{offset}_{step}.pt';torch.save(dict(optimizer=opt.state_dict(),scaler=scaler.state_dict(),model_hash=state_dict_hash(m.state_dict()),step=absolute),p)
 segment_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 record=R._run_recovery_condition(config=c,condition_name=job['id'],pruned_base_state=initial,router_checkpoint=str(router_path),router_age_percent=0,router_step=offset,masks=masks,expert_hash=eh,shared_hash=sh,mask_hash=R._mask_hash(masks),reference_selected=refs,calibration_batches=R._calibration_batches(val),train_batches=train,validation_batches=val,train_batch_hash=R._batch_sequence_hash(train),validation_batch_hash=R._batch_sequence_hash(val),device=dev,recovery_steps=steps,dense_loss=float(job['canonical_pair']['dense_final_validation_loss']),output_dir=dest,confidence_control=False,target_confidence=None,seed=job['seed'],sparsity=job['sparsity'] or 0.,router_mode='trainable',state_snapshot_callback=callback if instrument else None,training_forward_callback=forward if instrument else None,interruption_checkpoint=dest/'interruption.pt')
 record.update(wall_seconds=time.monotonic()-segment_started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),instrumented=instrument)
 save(dest/'record.json',record);return record

def condition(job):
 base=OUT/'reruns'/job['id'];base.mkdir(parents=True,exist_ok=True)
 with (base/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  done=base/'completed.json'
  if done.exists():
   d=read(done);assert d['cohort_sha256']==sha(OUT/'cohort.json') and d['protocol_sha256']==sha(OUT/'protocol.md')
   assert d['implementation']==read(OUT/'launch.json')['implementation']
   verify_file(d['checkpoint'],d['sha256']);return
  attempt=base/f'attempt_{time.time_ns()}';attempt.mkdir();save(base/'status.json',dict(status='RUNNING',attempt=str(attempt),unix=time.time(),host=socket.gethostname()))
  try:
   registered={x['path']:x['sha256'] for x in frozen()['files']}
   for input_path in (job['initial'],job['mask']):
    if input_path:verify_file(input_path,registered[input_path])
   train,val=batches(job);pair=job['canonical_pair'];assert R._batch_sequence_hash(train[:2500])==pair['data_hash'];assert R._batch_sequence_hash(val)==pair['validation_hash'];assert R._batch_sequence_hash(train[2500:])==job['extension_identity']['training_batch_hash']
   masks=load_masks(job['mask']) if job['mask'] else {};initial=masked(state(job['initial']),masks)
   first=run_segment(job,initial,masks,job['initial'],train[:2500],val,attempt/'segment_0',0,2500)
   cp=attempt/'segment_0/checkpoints/final_recovered.pt';boundary=state(cp)
   second=run_segment(job,boundary,masks,cp,train[2500:],val,attempt/'segment_2500',2500,2500)
   a=read(attempt/'snapshots/step_2500.json');b=read(attempt/'snapshots/step_2500_after_reset.json');assert a['model_hash']==b['model_hash'] and b['optimizer_entries']==0
   final=attempt/'segment_2500/checkpoints/final_recovered.pt';save(done,dict(cohort_sha256=sha(OUT/'cohort.json'),protocol_sha256=sha(OUT/'protocol.md'),implementation=read(OUT/'launch.json')['implementation'],checkpoint=str(final),sha256=sha(final),attempt=str(attempt),final_validation_ce=second['final_validation_loss'],first_applied=first['optimizer_steps_applied'],second_applied=second['optimizer_steps_applied'],reset_weights_unchanged=True));save(base/'status.json',dict(status='COMPLETE',attempt=str(attempt),unix=time.time()))
  except BaseException as e:
   save(base/'status.json',dict(status='FAILED',attempt=str(attempt),unix=time.time(),error=repr(e),traceback=traceback.format_exc()));raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,required=True);a=p.parse_args();torch.set_num_threads(2);R.configure_device(torch.device('cuda'))
 assert torch.__version__.startswith('2.7.0') and torch.cuda.is_available()
 f=frozen()
 for j in f['conditions']:
  if j['seed'] in a.seeds:condition(j)
if __name__=='__main__':main()
