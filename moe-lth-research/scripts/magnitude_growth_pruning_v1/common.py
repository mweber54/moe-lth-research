"""Isolated prospective magnitude-growth intervention; no historical writes."""
import sys,copy,math,socket,threading,traceback,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts/imp_adaptation_factorial_v1'))
from af_common import *
from af_native import encode_numpy,decode_numpy
sys.path.insert(0,str(HERE))
from moe_lth.data import build_dataloaders
from moe_lth.utils import create_grad_scaler,seed_everything,resolve_autocast_dtype
D=ROOT/'results/magnitude_growth_pruning_v1';SEEDS=(7,17,29,43,61);BRANCHES=('control','growing','stable');EVAL_STEPS=(0,25,100,250,500,1000,1500)

def atomic_tensor(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');torch.save(encode_numpy(x),q);q.replace(p)
def source(seed):
 return ROOT/(f'results/32e_router_age_localization/references/seed_{seed}' if seed in (7,17,29) else f'results/router_conditioned_fixed_budget_dataset/sources/e32/seed_{seed}')
def registry():return read(D/'registration.json')
def variant(seed):return next(v for v in read(ROOT/'results/imp_adaptation_factorial_v1/manifest.json')['variants'] if v['seed']==seed and not v['sparsity'])
def data(seed):
 v=variant(seed);c=v['config'];tr,va=build_dataloaders(c['data'],c['training']['batch_size'],seed,reshuffle_each_epoch=False)
 train=R._materialize_batches(tr,2500);val=R._materialize_validation_batches(va,32)
 assert R._batch_sequence_hash(train)==v['canonical_pair']['data_hash']
 assert R._batch_sequence_hash(val)==v['canonical_pair']['validation_hash']
 return train,val

def selected(seed,branch,device='cpu'):
 if branch=='control':return {}
 path=D/'masks'/f'seed{seed}.pt';assert sha(path)==registry()['mask_files'][str(seed)]['sha256']
 x=load(path);return {n:t.to(device) for n,t in x[branch].items()}
def zero_(model,indices):
 with torch.no_grad():
  params=dict(model.named_parameters())
  for n,idx in indices.items():params[n].view(-1).index_fill_(0,idx.to(params[n].device),0.)
def hooks(model,indices):
 params=dict(model.named_parameters());hs=[]
 for n,idx in indices.items():
  idx=idx.to(params[n].device)
  def hook(grad,idx=idx):
   result=grad.clone();result.view(-1).index_fill_(0,idx,0.);return result
  hs.append(params[n].register_hook(hook))
 return hs

def check_zero(model,indices):
 params=dict(model.named_parameters())
 if indices:assert torch.count_nonzero(torch.cat([params[n].view(-1)[idx.to(params[n].device)] for n,idx in indices.items()])).item()==0

def assemble(seed,branch,device):
 cfg=variant(seed)['config'];path=source(seed)/'checkpoints/step_1000.pt';expected=registry()['checkpoints'][str(seed)]['1000'];assert sha(path)==expected['sha256']
 cp=load(path);assert cp['step']==1000 and not any(k in cp for k in ('optimizer','scaler','torch_rng','rng'))
 model=TinyMoELanguageModel(cfg['model']).to(device);model.load_state_dict(cp['model']);assert state_dict_hash(model.state_dict())==expected['model_hash']
 for p in model.parameters():p.requires_grad_(True)
 assert model.token_embedding.weight is model.lm_head.weight
 indices=selected(seed,branch,device);zero_(model,indices);hs=hooks(model,indices)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']);scaler=create_grad_scaler(device.type=='cuda')
 assert not opt.state
 assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
 seed_everything(seed) # identical reset AFTER constructor and data/diagnostics
 return model,opt,scaler,indices,hs

def update(model,opt,scaler,batch,cfg,indices,device):
 model.train();x,y=(t.to(device) for t in batch);opt.zero_grad(set_to_none=True)
 with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
  out=model(x);ce=torch.nn.functional.cross_entropy(out.logits.reshape(-1,out.logits.shape[-1]),y.reshape(-1));loss=ce+cfg['routing']['aux_loss_weight']*out.auxiliary_loss
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
 scaler.scale(loss).backward();scaler.unscale_(opt)
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']);before=float(scaler.get_scale());scaler.step(opt);scaler.update();after=float(scaler.get_scale());applied=after>=before
 zero_(model,indices);check_zero(model,indices)
 return dict(task_ce=float(ce.detach()),auxiliary=float(out.auxiliary_loss.detach()),combined_loss=float(loss.detach()),applied=applied,amp_skip=not applied,scale_before=before,scale_after=after,gradient_norm=float(norm) if torch.isfinite(norm) else None)

def evaluate(model,val,device):
 # Explicit state-preserving, task-only FP32 complete-batch validation.
 statehash=state_dict_hash(model.state_dict());tf=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32);rows=[]
 with protected(model):
  try:
   torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model.eval()
   with torch.inference_mode():
    for i,(x,y) in enumerate(val):
     out=model(x.to(device));loss=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.to(device).flatten(),reduction='sum');rows.append(dict(validation_batch_index=i,target_tokens=y.numel(),task_ce_sum=float(loss)))
  finally:torch.backends.cuda.matmul.allow_tf32=tf[0];torch.backends.cudnn.allow_tf32=tf[1]
 assert state_dict_hash(model.state_dict())==statehash
 ce=sum(r['task_ce_sum'] for r in rows)/sum(r['target_tokens'] for r in rows);assert math.isfinite(ce)
 return ce,rows

def rng():
 return dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def set_rng(x):
 random.setstate(x['python_rng']);np.random.set_state(x['numpy_rng']);torch.set_rng_state(x['torch_rng'].cpu())
 if x['cuda_rng']:torch.cuda.set_rng_state_all([v.cpu() for v in x['cuda_rng']])
def frozen():
 r=registry();seal=read(D/'execution_seal.json');assert sha(D/'registration.json')==seal['registration_sha256'];assert sha(D/'protocol.md')==seal['protocol_sha256']
 for name,h in seal['implementation'].items():assert sha(ROOT/name)==h,'Implementation changed: '+name
 return r
