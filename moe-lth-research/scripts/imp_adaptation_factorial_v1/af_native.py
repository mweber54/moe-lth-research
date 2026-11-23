"""Native function code reused unchanged; isolated globals adapt parameter flags/I/O."""
from af_common import *
import types,math,copy


def encode_numpy(payload):
 if not isinstance(payload,dict) or 'numpy_rng' not in payload:return payload
 p=dict(payload);rng=p['numpy_rng']
 p['numpy_rng']={'algorithm':rng[0],'keys':torch.tensor(rng[1].astype(np.int64)),'position':rng[2],'has_gauss':rng[3],'cached_gauss':rng[4]}
 return p


def decode_numpy(payload):
 if isinstance(payload,dict) and isinstance(payload.get('numpy_rng'),dict):
  p=dict(payload);r=p['numpy_rng'];p['numpy_rng']=(r['algorithm'],r['keys'].cpu().numpy().astype(np.uint32),r['position'],r['has_gauss'],r['cached_gauss']);return p
 return payload


class Proxy:
 def __init__(self,base,**overrides):self.base=base;self.overrides=overrides
 def __getattr__(self,n):return self.overrides[n] if n in self.overrides else getattr(self.base,n)


def native_function(condition,clip_callback=None,assembled_callback=None):
 original=R._run_recovery_condition
 original_assemble=R.assemble_router_age_model
 def assemble(*args,**kwargs):
  model=configure(original_assemble(*args,**kwargs),condition)
  if assembled_callback:assembled_callback(model)
  return model
 def clip(parameters,max_norm,*args,**kwargs):
  norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm,*args,**kwargs)
  if clip_callback:
   n=float(norm);coef=float(torch.clamp(max_norm/(norm+1e-6),max=1.))
   clip_callback(n if math.isfinite(n) else None,coef if math.isfinite(coef) else None)
  return norm
 def safe_save(payload,path,*args,**kwargs):return torch.save(encode_numpy(payload),path,*args,**kwargs)
 def safe_load(path,*args,**kwargs):kwargs['weights_only']=True;return decode_numpy(torch.load(path,*args,**kwargs))
 proxy=Proxy(torch,nn=Proxy(torch.nn,utils=Proxy(torch.nn.utils,clip_grad_norm_=clip)),save=safe_save,load=safe_load)
 globals_=dict(original.__globals__,assemble_router_age_model=assemble,torch=proxy)
 function=types.FunctionType(original.__code__,globals_,original.__name__,original.__defaults__,original.__closure__);function.__kwdefaults__=original.__kwdefaults__
 return function


def segment(config,initial,router_path,masks,train,val,dest,condition,seed,offset=0,callback=None,forward=None,clip_callback=None,assembled=None,device=None,reference=None,interruption=True,resume_interval=100):
 device=device or torch.device('cuda');cfg=copy.deepcopy(config);cfg['adaptation_factorial_condition']=condition
 if reference is None:
  with protected():
   m=TinyMoELanguageModel(cfg['model']).to(device);m.load_state_dict(initial);m.eval()
   reference=selected_experts_per_batch(m,R._calibration_batches(val),device);del m
 hashes={g:state_dict_hash({n:t for n,t in initial.items() if parameter_group(n)==g}) for g in ('expert','shared')}
 return native_function(condition,clip_callback,assembled)(config=cfg,condition_name=f'{condition}_{offset}',pruned_base_state=initial,router_checkpoint=str(router_path),router_age_percent=0,router_step=offset,masks=masks,expert_hash=hashes['expert'],shared_hash=hashes['shared'],mask_hash=R._mask_hash(masks),reference_selected=reference,calibration_batches=R._calibration_batches(val),train_batches=train,validation_batches=val,train_batch_hash=R._batch_sequence_hash(train),validation_batch_hash=R._batch_sequence_hash(val),device=device,recovery_steps=len(train),dense_loss=1.,output_dir=dest,confidence_control=False,target_confidence=None,seed=seed,sparsity=1-sum(int(x.sum()) for x in masks.values())/sum(x.numel() for x in masks.values()) if masks else 0.,router_mode='trainable' if 'R' in TRAINABLE[condition] else 'frozen',state_snapshot_callback=callback,training_forward_callback=forward,interruption_checkpoint=dest/'interruption.pt' if interruption else None,interruption_interval=resume_interval)
