"""Native training bytecode with isolated dynamic-mask/event/readout adapters."""
from common import *
import math

@contextlib.contextmanager
def readout_mode(model):
 flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
 before=state_dict_hash(model.state_dict())
 with protected(model):
  try:
   torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model.eval()
   with torch.inference_mode(),torch.autocast(device_type=next(model.parameters()).device.type,enabled=False):yield
   assert state_dict_hash(model.state_dict())==before,'Readout modified parameters/buffers'
  finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags

def evaluate(model,val,*unused,**kw):
 device=next(model.parameters()).device;total=0.;count=0
 with readout_mode(model):
  for x,y in val:
   out=model(x.to(device));loss=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.to(device).flatten(),reduction='none')
   total+=float(loss.double().sum());count+=y.numel()
 return dict(ce=total/count,loss_sum=total,target_tokens=count,batches=len(val),precision='FP32_TF32off')

def diagnostic(model,probe):
 device=next(model.parameters()).device;rows=[];traces=[];loss=aux=0.;count=0
 with readout_mode(model):
  for batch,(x,y) in enumerate(probe):
   out=model(x.to(device));loss+=float(torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.to(device).flatten(),reduction='none').double().sum());count+=y.numel();aux+=float(out.auxiliary_loss)
   layers=[]
   for layer,tr in enumerate(out.routes):
    ids=tr.selected_expert_indices.detach().flatten();acc=tr.accepted_mask.detach().flatten();gate=tr.selected_probabilities.detach().flatten();E=tr.probabilities.shape[-1]
    layers.append((ids.cpu(),acc.cpu(),gate.float().cpu()))
    for e in range(E):
     proposed=ids==e;accepted=proposed&acc
     rows.append(dict(batch=batch,layer=layer,expert=e,proposed=int(proposed.sum()),accepted=int(accepted.sum()),
        selected_gate_sum=float(gate[proposed].double().sum()),accepted_gate_sum=float(gate[accepted].double().sum()),
        selected_gate_min=float(gate[proposed].min()) if proposed.any() else None,selected_gate_max=float(gate[proposed].max()) if proposed.any() else None,
        dropped_assignment_fraction=float((~acc).float().mean())))
   traces.append(layers)
  norms={g:math.sqrt(sum(float(p.detach().double().square().sum()) for n,p in model.named_parameters() if group(n)==g)) for g in ('E','R','S')}
 return dict(task_ce=loss/count,target_tokens=count,auxiliary_layer_mean=aux/len(probe),routing=rows,parameter_norms=norms),traces

def overlap(before,after):
 result=[]
 for b,(aa,bb) in enumerate(zip(before,after)):
  for layer,((i,a,g),(j,c,h)) in enumerate(zip(aa,bb)):
   for e in range(32):
    p=i==e;q=j==e;pa=p&a;qa=q&c;u=int((p|q).sum());au=int((pa|qa).sum())
    result.append(dict(batch=b,layer=layer,expert=e,proposed_set_jaccard=float((p&q).sum()/u) if u else None,
        accepted_set_jaccard=float((pa&qa).sum()/au) if au else None,assignment_agreement=float((i==j).float().mean()),
        acceptance_agreement=float((a==c).float().mean()),selected_gate_rms_change=float((g-h).double().square().mean().sqrt())))
 return result

def run_segment(v,condition,initial,router,train,val,dest,offset,assignment,steps=None,events=None,interrupt_after=None,resume_interval=100,diagnostic_probe=None):
 steps=steps or len(train);manager=MaskManager(assignment,condition,offset)
 if events is not None:
  # Engineering-only compressed event clock; production always uses the registered attempts.
  manager_clock=lambda absolute:events.get(absolute,manager.event)
 else:manager_clock=lambda absolute:event_count(condition,absolute)
 dest=Path(dest);resuming=(dest/'interruption.pt').exists();times={'diagnostics':0.,'evaluation':0.};before_update={};last_step={'value':offset}
 records=[];applied={'count':0};last_resume={'event':manager.event};device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 config=copy.deepcopy(v['config']);initial=masked(initial,manager.masks);probe=diagnostic_probe or train[:2]
 if events is not None:config['engineering_compressed_event_clock']=events
 def safe_save(payload,path,*args,**kwargs):
  if isinstance(payload,dict) and 'python_rng' in payload and 'optimizer' in payload:
   payload=dict(payload,transition=manager.serial(),absolute_step=offset+payload['step'],offset=offset,sampler_next_batch=offset+payload['step'])
  return torch.save(encode_numpy(payload),path,*args,**kwargs)
 def safe_load(path,*args,**kwargs):
  kwargs['weights_only']=True;saved=decode_numpy(torch.load(path,*args,**kwargs))
  if isinstance(saved,dict) and 'transition' in saved:
   manager.restore(saved['transition']['event']);assert R._mask_hash(manager.masks)==saved['transition']['mask_hash'];last_resume['event']=manager.event
   assert saved['sampler_next_batch']==saved['absolute_step']
  return saved
 proxy=Proxy(torch,save=safe_save,load=safe_load)
 def native_eval(model,batches,*a,**kw):
  start=time.monotonic();value=evaluate(model,batches);times['evaluation']+=time.monotonic()-start;return value['ce']
 original=R._run_recovery_condition
 globals_=dict(original.__globals__,torch=proxy,register_mask_gradient_hooks=manager.register,
       _evaluate_condition_loss=native_eval,RECOVERY_EVAL_INTERVAL=1000000)
 fn=types.FunctionType(original.__code__,globals_,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__
 def forward(m,x,step):
  absolute=offset+step;last_step['value']=absolute
  if absolute in GRID or step==1:
   before_update.clear();before_update.update({n:p.detach().clone() for n,p in m.named_parameters()})
  return R._condition_forward(m,x,False,1.)
 def callback(model,opt,scaler,step):
  absolute=offset+step
  if step==0 and resuming:return
  if step==0:
   assert not opt.state
   save(dest.parent/f'reset_{offset}.json',dict(step=absolute,model_hash=state_dict_hash(model.state_dict()),optimizer_entries=0,scaler=scaler.state_dict(),seed=v['seed']))
   assert state_dict_hash(model.state_dict())==state_dict_hash(initial)
  # Update norms precede pruning, so zeroing displacement cannot masquerade as an optimizer update.
  if before_update and step:
   norms={g:math.sqrt(sum(float((p.detach()-before_update[n]).double().square().sum()) for n,p in model.named_parameters() if group(n)==g)) for g in ('E','R','S')}
   append(dest/'updates.jsonl',dict(step=absolute,actual_update_l2=norms,definition='actual one-attempt parameter change before pruning; tied parameters counted once'))
   before_update.clear()
  target=manager_clock(absolute)
  if target>manager.event:
   start=time.monotonic();pre,traces=diagnostic(model,probe);times['diagnostics']+=time.monotonic()-start
   if absolute==1000:
    start=time.monotonic();preval=evaluate(model,val);times['evaluation']+=time.monotonic()-start
    save(dest/'grid_1000_pre.json',dict(step=absolute,phase='pre_pruning',**preval))
   # Normal production uses the exact attempted-update clock, regardless of AMP application.
   if events is not None:
    saved_condition=manager.condition;manager.condition='gradual';event=manager.prune(model,opt,target*100);manager.condition=saved_condition
   else:event=manager.prune(model,opt,absolute)
   start=time.monotonic();post,posttraces=diagnostic(model,probe);times['diagnostics']+=time.monotonic()-start
   save(dest/f'event_{absolute:04d}.json',dict(event=event,pre=pre,post=post,damage=post['task_ce']-pre['task_ce'],overlap=overlap(traces,posttraces),mask_hash=R._mask_hash(manager.masks),state_preservation_verified=True))
  if absolute in GRID and not(offset and step==0):
   start=time.monotonic();metrics=evaluate(model,val);times['evaluation']+=time.monotonic()-start
   save(dest/f'grid_{absolute:04d}.json',dict(step=absolute,phase='post_event' if absolute<=1000 else 'pre_reset' if absolute==2500 else 'native',**metrics))
   start=time.monotonic();d,_=diagnostic(model,probe);times['diagnostics']+=time.monotonic()-start;save(dest/f'routing_{absolute:04d}.json',dict(step=absolute,**d))
  if absolute in (0,1000,2500,5000) and not(offset and step==0):atomic_torch(dest.parent/f'snapshots/step_{absolute:04d}.pt',dict(model=state(model),config=config,step=absolute))
  save(dest.parent/'heartbeat.json',dict(host=socket.getfqdn(),pid=os.getpid(),attempted_step=absolute,event=manager.event,unix=time.time()))
  if interrupt_after is not None and step==interrupt_after:raise RuntimeError('synthetic interruption')
 # Calibration forwards preserve RNG. They are native engineering setup, not validation selection.
 with protected():
  reference_model=TinyMoELanguageModel(config['model']).to(device);reference_model.load_state_dict(initial);reference_model.eval();reference=selected_experts_per_batch(reference_model,R._calibration_batches(val),device);del reference_model
 hashes={g:state_dict_hash({n:t for n,t in initial.items() if parameter_group(n)==g}) for g in ('expert','shared')}
 started=time.monotonic()
 r=fn(config=config,condition_name=f'{condition}_{offset}',pruned_base_state=initial,router_checkpoint=str(router),router_age_percent=0,router_step=offset,
    masks=manager.masks,expert_hash=hashes['expert'],shared_hash=hashes['shared'],mask_hash=R._mask_hash(manager.masks),reference_selected=reference,
    calibration_batches=R._calibration_batches(val),train_batches=train,validation_batches=val,train_batch_hash=R._batch_sequence_hash(train),validation_batch_hash=R._batch_sequence_hash(val),
    device=device,recovery_steps=steps,dense_loss=1.,output_dir=dest,confidence_control=False,target_confidence=None,seed=v['seed'],sparsity=0. if condition=='dense' else .8,
    router_mode='trainable',state_snapshot_callback=callback,training_forward_callback=forward,interruption_checkpoint=dest/'interruption.pt',interruption_interval=resume_interval,diagnostic_steps=(0,steps))
 r.update(wall_seconds=time.monotonic()-started,diagnostic_seconds=times['diagnostics'],common_evaluation_seconds=times['evaluation'],transition=manager.serial(),active_coordinates_final=manager.active())
 save(dest/'record.json',r)
 return r

def execute(job,v,attempt):
 train,val=batches(v);assignment=load(v['schedule']);records=[]
 if job['reuse']:
  ref=v['references'][job['condition']];seen=set()
  for step,path in ref['snapshots'].items():
   assert sha(path)==read(OUT/'input_hashes.json')[path]
   m=load_model_from_checkpoint(v['config']['model'],path,torch.device('cuda'))
   if job['condition']=='fixed':
    masks=load_masks(v['mask']);params=dict(m.named_parameters())
    assert all(torch.count_nonzero(params[n][~mask.cuda()])==0 for n,mask in masks.items())
   r=evaluate(m,val)
   save(attempt/f'grid_{int(step):04d}.json',dict(step=int(step),phase='reference_native',reused=True,source=path,**r));seen.add(int(step));del m
  if 5000 not in seen:
   assert sha(ref['checkpoint'])==read(OUT/'input_hashes.json')[ref['checkpoint']]
   m=load_model_from_checkpoint(v['config']['model'],ref['checkpoint'],torch.device('cuda'));save(attempt/'grid_5000.json',dict(step=5000,reused=True,source=ref['checkpoint'],**evaluate(m,val)));del m
  return dict(checkpoint=ref['checkpoint'],reused=True,attempted_updates=5000,applied_updates=ref['applied'],amp_skips=5000-ref['applied'],final_mask_hash=v['mask_hash'] if job['condition']=='fixed' else None,source=ref)
 initial=state(load_model_from_checkpoint(v['config']['model'],v['initial'],torch.device('cpu')));router=v['initial'];boundary_hash=None
 for offset in (0,2500):
  dest=attempt/f'segment_{offset}'
  if (dest/'record.json').exists():r=read(dest/'record.json')
  else:r=run_segment(v,job['condition'],initial,router,train[offset:offset+2500],val,dest,offset,assignment,diagnostic_probe=train[:2])
  records.append(r);router=str(dest/'checkpoints/final_recovered.pt');initial=load(router)['model']
  if offset==0:boundary_hash=state_dict_hash(initial)
  else:assert read(attempt/'reset_2500.json')['model_hash']==boundary_hash
  if offset==0:assert r['transition']['mask_hash']==v['mask_hash'],'Not the exact final IMP mask'
 assert records[-1]['transition']['mask_hash']==v['mask_hash']
 final=load(router)['model'];masks=load_masks(v['mask'])
 assert all(torch.count_nonzero(final[n][~m]).item()==0 for n,m in masks.items())
 return dict(checkpoint=router,reused=False,attempted_updates=5000,applied_updates=sum(r['optimizer_steps_applied'] for r in records),amp_skips=sum(r['amp_overflow_step_count'] for r in records),final_mask_hash=v['mask_hash'],reset_weights_preserved=True,records=records)
