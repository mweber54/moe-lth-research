"""Short canonical-shape equivalence and AMP resume; never production outcomes."""
from af_common import *
import af_native as native
import socket,copy
from af_diagnostics import diagnose,model_state
from moe_lth.data import build_dataloaders

def main():
 torch.set_num_threads(4);device=torch.device('cuda');R.configure_device(device)
 if not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Canonical CUDA environment required')
 host=socket.getfqdn();base=OUT/'validation'/host/f'attempt_{time.time_ns()}';base.mkdir(parents=True)
 c=next(c for c in read(HIST/'cohort.json')['conditions'] if c['seed']==7 and c['sparsity']==.8);cfg=c['config'];masks=load_masks(c['mask']);m=load_model_from_checkpoint(cfg['model'],c['initial'],torch.device('cpu'));apply_masks_(m,masks);initial=model_state(m);del m
 R.seed_everything(7);tr,va=build_dataloaders(cfg['data'],128,7);train=R._materialize_batches(tr,25);val=R._materialize_validation_batches(va,32)
 if R._batch_sequence_hash(val)!=c['canonical_pair']['validation_hash']:raise RuntimeError('Validation mismatch')
 timing=[];records=[]
 for name in ('canonical','adapter'):
  dest=base/name/'segment';fn=native.native_function
  if name=='canonical':native.native_function=lambda *a,**kw:R._run_recovery_condition
  def callback(model,opt,scaler,step):
   if step not in (0,25):return
   p=base/name/f'snapshots/step_{step:04d}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(model=model_state(model),config=cfg,step=step),p)
   diagnose(model,val[0],masks,'full',cfg,step,base/name,p,base/name/'snapshots/step_0000.pt' if step else None,base/name/'snapshots/step_0000.pt')
  started=time.monotonic();torch.cuda.reset_peak_memory_stats()
  try:r=native.segment(cfg,initial,c['initial'],masks,train,val,dest,'full',7,callback=callback if name=='adapter' else None,device=device)
  finally:native.native_function=fn
  records.append(r);timing.append(dict(track=name,seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated()))
 a=load(base/'canonical/segment/checkpoints/final_recovered.pt')['model'];b=load(base/'adapter/segment/checkpoints/final_recovered.pt')['model'];maxdiff=max(float((a[n]-b[n]).abs().max()) for n in a)
 if not all(torch.allclose(a[n],b[n],atol=1e-6,rtol=1e-5) for n in a):raise RuntimeError('GPU parameter equivalence failed')
 if abs(records[0]['final_validation_loss']-records[1]['final_validation_loss'])>=1e-6:raise RuntimeError('GPU CE equivalence failed')
 for k in ('optimizer_steps_applied','amp_overflow_step_count','loss_scale_final'):
  if records[0][k]!=records[1][k]:raise RuntimeError('AMP/update equivalence failed '+k)
 # Reproduce the native second-call reset on the same weights, not an init restart.
 cp=base/'adapter/segment/checkpoints/final_recovered.pt';seen=[]
 native.segment(cfg,b,cp,masks,train[:1],val,base/'reset','full',7,offset=2500,callback=lambda model,opt,scaler,step:seen.append(dict(step=step,model_hash=state_dict_hash(model.state_dict()),optimizer_entries=len(opt.state),scaler=scaler.state_dict())) if step==0 else None)
 if seen[0]['model_hash']!=state_dict_hash(b) or seen[0]['optimizer_entries']!=0:raise RuntimeError('Reset mismatch')
 # Real CUDA-FP16 tiny resume exercises scaler state and all RNG/data position.
 tiny=copy.deepcopy(cfg);tiny['model'].update(num_layers=2,num_experts=2,d_model=8,num_heads=2,expert_hidden_size=16,vocab_size=32,max_seq_len=8,dropout=.1);tiny['data'].update(seq_len=8,validation_blocks=1);tiny['training'].update(batch_size=2)
 R.seed_everything(19);tm=TinyMoELanguageModel(tiny['model']);tp=base/'tiny_initial.pt';torch.save(dict(model=tm.state_dict(),config=tiny,step=0,validation_loss=None),tp);ts=model_state(tm);tb=[(torch.randint(0,32,(2,8)),torch.randint(0,32,(2,8))) for _ in range(6)]
 kw=dict(config=tiny,initial=ts,router_path=tp,masks={},train=tb,val=tb[:1],condition='expert_only',seed=19,device=device,resume_interval=2)
 native.segment(dest=base/'tiny_full',**kw)
 def interrupt(model,opt,scaler,step):
  if step==3:raise RuntimeError('simulated validation interruption')
 try:native.segment(dest=base/'tiny_resumed',callback=interrupt,**kw)
 except RuntimeError as e:
  if 'simulated validation' not in str(e):raise
 native.segment(dest=base/'tiny_resumed',**kw)
 aa=load(base/'tiny_full/interruption.pt');bb=load(base/'tiny_resumed/interruption.pt')
 def equal(x,y):
  if torch.is_tensor(x):return torch.equal(x,y)
  if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
  if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
  return x==y
 if not all(equal(aa[k],bb[k]) for k in ('model','optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','step')):raise RuntimeError('GPU AMP/RNG/optimizer resume mismatch')
 result=dict(passed=True,host=host,steps=25,max_parameter_absolute_difference=maxdiff,validation_ce_difference=abs(records[0]['final_validation_loss']-records[1]['final_validation_loss']),atol=1e-6,rtol=1e-5,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,reset=seen[0],cuda_fp16_resume_exact=True,diagnostics_enabled_native_equivalence=True,timing=timing,attempt=str(base),unix=time.time())
 save(OUT/'validation'/f'{host}.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
