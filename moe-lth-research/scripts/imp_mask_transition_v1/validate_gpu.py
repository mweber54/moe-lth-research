"""Seed17 native equivalence, real CUDA resume/event/reset and resource benchmark."""
from common import *
from engine import run_segment,evaluate
from af_native import segment as native_segment,native_function

def tree_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 torch.set_num_threads(4);R.configure_device(torch.device('cuda'));host=socket.getfqdn()
 assert host in HOSTS and torch.__version__.startswith('2.7.0') and torch.cuda.is_available()
 assert torch.cuda.mem_get_info()[0]>=24*1024**3
 manifest=read(OUT/'manifest.json');v=next(v for v in manifest['variants'] if v['seed']==17)
 attempt=OUT/'validation'/host/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True)
 train,val=batches(v);assignment=load(v['schedule']);initial=state(load_model_from_checkpoint(v['config']['model'],v['initial'],torch.device('cpu')))
 timing=[];comparisons=[]
 for condition in ('dense','fixed'):
  masks=load_masks(v['mask']) if condition=='fixed' else {};base=masked(initial,masks)
  for mode in ('native','adapter'):
   torch.cuda.reset_peak_memory_stats();start=time.monotonic();dest=attempt/f'{condition}_{mode}'/'segment'
   if mode=='native':
    # Existing native adapter changes only safe serialization; all groups remain trainable.
    native_segment(v['config'],base,v['initial'],masks,train[:8],val[:2],dest,'full',17,device=torch.device('cuda'))
   else:run_segment(v,condition,base,v['initial'],train[:8],val[:2],dest,0,assignment)
   timing.append(dict(condition=condition,mode=mode,attempts=8,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
  a=load(attempt/f'{condition}_native/segment/checkpoints/final_recovered.pt')['model'];b=load(attempt/f'{condition}_adapter/segment/checkpoints/final_recovered.pt')['model']
  maxdiff=max(float((a[n]-b[n]).abs().max()) for n in a)
  assert all(torch.allclose(a[n],b[n],atol=1e-6,rtol=1e-5) for n in a),'Native equivalence failed'
  values=[]
  for st in (a,b):
   m=TinyMoELanguageModel(v['config']['model']).cuda();m.load_state_dict(st);values.append(evaluate(m,val[:2])['ce']);del m
  assert abs(values[0]-values[1])<1e-6
  comparisons.append(dict(condition=condition,max_parameter_absolute_difference=maxdiff,common_ce_difference=abs(values[0]-values[1])))
  del a,b;torch.cuda.empty_cache()
 # Canonical-shape event overhead and exact final-mask smoke check; compressed clock is engineering-only.
 torch.cuda.reset_peak_memory_stats();started=time.monotonic()
 event_record=run_segment(v,'gradual',initial,v['initial'],train[:8],val[:2],attempt/'canonical_events/segment',0,assignment,events={4:5,8:10})
 assert event_record['transition']['mask_hash']==v['mask_hash']
 final_state=load(attempt/'canonical_events/segment/checkpoints/final_recovered.pt')['model'];final_masks=load_masks(v['mask'])
 assert all(torch.count_nonzero(final_state[n][~mask])==0 for n,mask in final_masks.items())
 timing.append(dict(condition='engineering_gradual_events',mode='adapter_events',attempts=8,seconds=time.monotonic()-started,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),diagnostic_seconds=event_record['diagnostic_seconds']))
 del final_state;torch.cuda.empty_cache()
 # Tiny real CUDA-FP16/dropout run with two compressed events and an interruption.
 tiny=copy.deepcopy(v);tiny['config']=copy.deepcopy(v['config']);tiny['config']['model'].update(vocab_size=32,max_seq_len=8,num_layers=2,num_heads=2,d_model=8,num_experts=2,expert_hidden_size=16,dropout=.1)
 tiny['config']['training']['batch_size']=2;tiny['config']['data'].update(seq_len=8,validation_blocks=1)
 R.seed_everything(17);m=TinyMoELanguageModel(tiny['config']['model']);s=state(m);tp=attempt/'tiny_initial.pt';atomic_torch(tp,dict(model=s,config=tiny['config'],step=0));tiny['initial']=str(tp)
 ts=[(torch.randint(0,32,(2,8)),torch.randint(0,32,(2,8))) for _ in range(6)]
 final={n:(torch.arange(p.numel()).reshape_as(p)%5==0) for n,p in expert_weight_parameters(m).items()};schedule=assignments(final,17)
 kw=dict(v=tiny,condition='gradual',initial=s,router=tp,train=ts,val=ts[:1],offset=0,assignment=schedule,events={2:5,4:10},resume_interval=1)
 run_segment(dest=attempt/'tiny_full/segment',**kw)
 try:run_segment(dest=attempt/'tiny_resumed/segment',interrupt_after=3,**kw)
 except RuntimeError as e:
  if 'synthetic interruption' not in str(e):raise
 else:raise AssertionError('Interruption was not exercised')
 run_segment(dest=attempt/'tiny_resumed/segment',**kw)
 aa=load(attempt/'tiny_full/segment/interruption.pt');bb=load(attempt/'tiny_resumed/segment/interruption.pt')
 for key in ('model','optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','step','transition','sampler_next_batch'):
  assert tree_equal(aa[key],bb[key]),'Resume mismatch '+key
 assert aa['transition']['event']==10
 assert len(list((attempt/'tiny_resumed/segment').glob('event_*.json')))==2
 assert all(torch.count_nonzero(aa['model'][n][~mask])==0 for n,mask in final.items())
 # Second-call reset preserves weights and consumes the supplied suffix, not the prefix.
 checkpoint=attempt/'tiny_full/segment/checkpoints/final_recovered.pt'
 run_segment(tiny,'fixed',aa['model'],checkpoint,ts[4:5],ts[:1],attempt/'tiny_reset/segment',2500,schedule)
 reset=read(attempt/'tiny_reset/reset_2500.json');assert reset['model_hash']==state_dict_hash(aa['model']) and reset['optimizer_entries']==0
 result=dict(passed=True,host=host,source_seed=17,attempt=str(attempt),comparisons=comparisons,atol=1e-6,rtol=1e-5,ce_atol=1e-6,
   exact_cuda_dropout_optimizer_scaler_rng_resume=True,event_resume_no_duplicate=True,reset_weights_unchanged=True,timing=timing,
   torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),unix=time.time())
 save(OUT/'validation'/f'{host}.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
