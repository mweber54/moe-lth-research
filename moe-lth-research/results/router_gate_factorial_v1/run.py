"""Bounded 24-recovery worker. Reuses immutable archived controls; no donors trained."""
from support import *
from moe_lth.utils import configure_device
from moe_lth.experiments.fixed_schedule_intervention import Schedule
import time,gc,fcntl,traceback,shutil,os

def cosine(a,b):
 a=a.detach().double().flatten();b=b.detach().double().flatten();d=a.norm()*b.norm()
 return float(a.dot(b)/d) if d>0 else None

def main():
 seeds=tuple(int(v) for v in os.environ.get('FACTORIAL_SEEDS','7,17,29').split(','));assert set(seeds)<=set((7,17,29))
 worker_id='_'.join(map(str,seeds));status_path=OUT/f'status_worker_{worker_id}.json'
 lock=(OUT/f'worker_{worker_id}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 valid=read(OUT/'validation_complete.json');assert valid['passed']
 ready=read(OUT/'execution_ready.json');assert ready['passed']
 for p,h in ready['code_hashes'].items():assert file_hash(p)==h,p
 for p,h in read(OUT/'execution_inputs.json').items():assert file_hash(p)==h,p
 for p,h in valid['code_hashes'].items():
  if Path(p).name!='run.py':assert file_hash(p)==h,('Changed validated code',p)
 for p,h in read(OUT/'protected_inputs.json').items():assert file_hash(p)==h,p
 assert shutil.disk_usage(OUT).free>70*1024**3
 torch.set_num_threads(4);device=torch.device('cuda');configure_device(device)
 completed=[];start=time.time();new_count=0;limit=int(os.environ.get('FACTORIAL_MAX_NEW','24'))
 for seed in seeds:
  selected=sorted([x for x in rows() if x['seed']==seed],key=lambda r:-r['age'])
  cfg,initial,ticket,masks=states(selected[0],device);tr,va,th,vh=data(cfg,seed);cal=R._calibration_batches(va)
  dm=read(Path(selected[0]['dense'])/'metadata.json');final=R.load_model_from_checkpoint(cfg['model'],str(Path(dm['router_checkpoint']).parent/'step_2500.pt'),device)
  reference=selected_experts_per_batch(final,cal,device);del final;torch.cuda.empty_cache()
  for arm in ('C','D'):
   for row in selected:
    for kind,base in [('dense',initial),('sparse',ticket)]:
     age=row['age'];tag=f'{arm}/seed{seed}/R{age}/{kind}';job=OUT/'recoveries'/tag
     cell_lock=(OUT/('cell_'+tag.replace('/','_')+'.lock')).open('w');fcntl.flock(cell_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     if (job/'complete.json').exists():completed.append(tag);continue
     if job.exists():raise RuntimeError(f'Incomplete recovery preserved; refusing overwrite: {job}')
     meta=read(Path(row[kind])/'metadata.json');reader=FactorialSchedule(BASE/f'schedules/e8/seed{seed}',arm)
     extras=OUT/'measurements'/tag;extras.mkdir(parents=True,exist_ok=False)
     prefix=OUT/'snapshots'/tag;prefix.mkdir(parents=True,exist_ok=False)
     watch=set(range(1,2501,50))|set(range(50,2501,50))|set(STEPS)|{t+1 for t in STEPS if t<2500}
     context={};diag={};identity=dict(arm=arm,seed=seed,age=age,kind=kind);jobstart=time.time()
     write(status_path,dict(status='running',current=tag,step=0,completed=completed,started=start))
     def train(model,x,step):
      if step in watch:context['before']=model.token_embedding.weight.detach().clone();context['step']=step
      return reader.train(model,x,step)
     def snapshot(model,opt,scaler,step):
      E=model.token_embedding.weight
      if step in watch and step>0:
       assert context['step']==step
       delta=E.detach()-context.pop('before');grad=E.grad.detach();op=opt.state.get(E,{})
       task=diag.get('task');aux=diag.get('aux')
       rec=dict(step=step,actual_embedding_update_norm=float(delta.norm()),clipped_embedding_gradient_norm=float(grad.norm()) if torch.isfinite(grad).all() else None,cosine_update_negative_task=cosine(delta,-task.to(device)) if task is not None else None,cosine_update_negative_task_plus_aux=cosine(delta,-(task+aux).to(device)) if task is not None else None,diagnostic_source_step=diag.get('step'),adam_step=int(op.get('step',0)),adam_first_moment_norm=float(op['exp_avg'].norm()) if 'exp_avg' in op else None,adam_second_moment_norm=float(op['exp_avg_sq'].norm()) if 'exp_avg_sq' in op else None,lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay'],elapsed_seconds=time.time()-jobstart)
       for groupname in ('expert','router','shared'):
        vals=[p.grad.detach() for n,p in model.named_parameters() if R.parameter_group(n)==groupname and p.grad is not None]
        rec[groupname+'_postclip_norm']=sum(float(g.double().square().sum()) for g in vals)**.5 if all(torch.isfinite(g).all() for g in vals) else None
       dg.append(extras/'updates.csv',[rec])
       if step in STEPS or step-1 in STEPS:np.savez_compressed(extras/f'update_{step:04d}.npz',actual_update=delta.cpu().numpy(),clipped_gradient=grad.cpu().numpy())
      if step%50==0:
       write(status_path,dict(status='running',current=tag,step=step,completed=completed,condition_seconds=time.time()-jobstart,total_seconds=time.time()-start));print(tag,step,round(time.time()-jobstart,1),flush=True)
      if step not in STEPS:return
      rng=dg.rng_state();mode=[m.training for m in model.modules()];mh=dg.digest(model.state_dict());oh=dg.digest(opt.state_dict());sh=dg.digest(scaler.state_dict());gh=dg.digest([p.grad for p in model.parameters()])
      path=prefix/f'step_{step:04d}.pt';assert not path.exists()
      torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),grad_scaler=scaler.state_dict(),rng=rng,step=step,config=cfg,mask=masks,mask_applied=kind=='sparse',historical_metadata=meta,identity=identity,training_hash=th,validation_hash=vh),path)
      native=R._evaluate_condition_loss(model,va,device,False,1.)
      assignment=R._evaluate_condition_loss(model,va,device,False,1.,Schedule(reader.path,'replay').validation,step) if step%50==0 else None
      # Missing archived primary occurrence at25: fixed step0 probe is diagnostic only.
      x,y=(v.to(device) for v in va[0]);_,_,payload=reader.payload(x,0,0,'validation')
      n,e=diagnostics(model,x,y,payload,arm,extras/f'E_{step:04d}.npz');diag.clear();diag.update(e,step=step)
      dg.append(extras/'diagnostics.csv',[dict(step=step,**n)])
      dg.append(extras/'readouts.csv',[dict(step=step,native_loss=native,assignment_loss=assignment,primary_readout_available=step%50==0)])
      for m,training in zip(model.modules(),mode):m.training=training
      dg.restore_rng(rng)
      assert mh==dg.digest(model.state_dict()) and oh==dg.digest(opt.state_dict()) and sh==dg.digest(scaler.state_dict()) and gh==dg.digest([p.grad for p in model.parameters()]) and dg.digest(rng)==dg.digest(dg.rng_state())
      dg.append(extras/'mutation_audit.csv',[dict(step=step,model_unchanged=True,optimizer_unchanged=True,scaler_unchanged=True,gradients_unchanged=True,rng_restored=True)])
      with (OUT/'snapshot_manifest.lock').open('w') as manifest_lock:
       fcntl.flock(manifest_lock,fcntl.LOCK_EX)
       dg.append(OUT/'snapshot_manifest.csv',[dict(**identity,step=step,path=str(path),sha256=file_hash(path),model_hash=mh,optimizer_hash=oh)])
     record=R._run_recovery_condition(config=cfg,condition_name='factorial_'+tag.replace('/','_'),pruned_base_state=base,router_checkpoint=meta['router_checkpoint'],router_age_percent=age,router_step=meta['router_step'],masks={} if kind=='dense' else masks,expert_hash=meta['expert_state_hash'],shared_hash=meta['shared_state_hash'],mask_hash=meta['mask_hash'],reference_selected=reference,calibration_batches=cal,train_batches=tr,validation_batches=va,train_batch_hash=th,validation_batch_hash=vh,device=device,recovery_steps=2500,dense_loss=read(Path(row['dense'])/'summary.json')['final_validation_loss'],output_dir=job,confidence_control=False,target_confidence=None,seed=seed,sparsity=.85 if kind=='sparse' else 0.,router_mode='trainable',state_snapshot_callback=snapshot,training_forward_callback=train,validation_forward_callback=reader.validation)
     replay=reader.verify_complete_replay();dg.append(extras/'gates.csv',reader.gate_records)
     native=read(job/'summary.json')
     write(job/'complete.json',dict(**identity,record=record,replay=replay,elapsed_seconds=time.time()-jobstart,validation_manifest_sha256=file_hash(OUT/'validation_complete.json')))
     new_count+=1
     completed.append(tag);write(status_path,dict(status='running',completed=completed,total_seconds=time.time()-start));gc.collect();torch.cuda.empty_cache()
     if new_count>=limit:
      write(status_path,dict(status='bounded_segment_complete',completed=completed,new_count=new_count,total_seconds=time.time()-start));return
  del tr,va,initial,ticket,masks;gc.collect()
 assert len(completed)==8*len(seeds)
 for p,h in read(OUT/'protected_inputs.json').items():assert file_hash(p)==h,p
 write(status_path,dict(status='complete',completed=completed,total_seconds=time.time()-start,prior_artifacts_unchanged=True))
if __name__=='__main__':
 try:main()
 except Exception:
  (OUT/('worker_failure_'+os.environ.get('FACTORIAL_SEEDS','all').replace(',','_')+'.txt')).write_text(traceback.format_exc());raise
