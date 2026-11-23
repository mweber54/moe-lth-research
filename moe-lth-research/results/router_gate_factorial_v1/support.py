from factorial import *
import torch.nn.functional as F
import numpy as np
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.utils import seed_everything,resolve_data_seed
from moe_lth.pruning.router_age import state_dict_hash,selected_experts_per_batch
from moe_lth.experiments import optimization_geometry_diagnostics as dg
R=trainer()
def rows():return [x for x in read(ROOT/'results/cross_arch_optimization_geometry/diagnostic_rerun/run_manifest.json')['conditions'] if x['experts']==8 and x['age'] in (20,100)]
def data(cfg,seed):
 seed_everything(seed);a,b=build_dataloaders(cfg['data'],128,resolve_data_seed(cfg));tr=R._materialize_batches(a,2500);va=R._materialize_validation_batches(b,32)
 return tr,va,R._batch_sequence_hash(tr),R._batch_sequence_hash(va)
def states(row,device):
 cfg=load_config(row['config']);meta=read(Path(row['dense'])/'metadata.json');ref=Path(meta['router_checkpoint']).parent
 model=R.load_model_from_checkpoint(cfg['model'],str(ref/'step_0.pt'),torch.device('cpu'));initial={k:v.detach().clone() for k,v in model.state_dict().items()};del model
 masks=torch.load(ROOT/f"results/expert_count_replication/e8/masks/s0.85/seed_{row['seed']}/pruning_mask.pt",map_location='cpu',weights_only=False)
 ticket={k:v*masks[k] if k in masks else v for k,v in initial.items()}
 return cfg,initial,ticket,masks

def gradient(model,x,y,payload,arm,split=False):
 """FP32 diagnostic; autograd.grad does not accumulate into training .grad buffers."""
 params=list(model.parameters());names=[n for n,p in model.named_parameters()];hooks=[];leaf=None
 if split:
  leaf=model.token_embedding.weight.detach().clone().requires_grad_(True)
  hooks.append(model.token_embedding.register_forward_hook(lambda m,a,o:F.embedding(a[0],leaf)))
 try:
  cap=[];out=forward_payload(model,x,payload,arm,capture=cap)
  ce=F.cross_entropy(out.logits.flatten(0,1),y.flatten());ps=params+([leaf] if split else [])
  task=torch.autograd.grad(ce,ps,allow_unused=True,retain_graph=True)
  aux=torch.autograd.grad(.1*out.auxiliary_loss,ps,allow_unused=True)
 finally:
  for h in hooks:h.remove()
 norms={}
 for g in ('router','expert','shared'):
  ix=[i for i,n in enumerate(names) if R.parameter_group(n)==g]
  for label,gs in [('task',task),('aux',aux)]:norms[g+'_'+label+'_norm']=sum(float(gs[i].detach().double().square().sum()) for i in ix if gs[i] is not None)**.5
 e=names.index('token_embedding.weight');z=torch.zeros_like(params[e]);te=task[e] if task[e] is not None else z;ae=aux[e] if aux[e] is not None else z
 extra={'task':te.detach().cpu(),'aux':ae.detach().cpu()}
 if split:extra['input']=task[-1].detach().cpu();extra['output']=te.detach().cpu()
 norms.update(ce=float(ce.detach()),aux_loss=float(out.auxiliary_loss.detach()),gate_nondegenerate_fraction=float(torch.cat([(q['native'].detach().float().flatten()>0)&(q['native'].detach().float().flatten()<1) for q in cap]).float().mean()))
 return out,task,aux,norms,extra

def diagnostics(model,x,y,payload,arm,path):
 rng=dg.rng_state();mode=[m.training for m in model.modules()];tf=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
 before=dg.digest(model.state_dict());gradhash=dg.digest([p.grad for p in model.parameters()])
 try:
  model.eval();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  with torch.autocast(device_type=x.device.type,enabled=False):
   _,_,_,norms,e=gradient(model,x,y,payload,arm)
   _,_,_,_,sp=gradient(model,x,y,payload,arm,True)
  err=float((e['task']-sp['input']-sp['output']).norm()/(e['task'].norm()+1e-12));assert err<2e-5,err
  e.update(input=sp['input'],output=sp['output']);norms['tied_decomposition_relative_error']=err
  norms['embedding_task_norm']=float(e['task'].norm());norms['embedding_aux_norm']=float(e['aux'].norm())
  np.savez_compressed(path,**{k:v.numpy() for k,v in e.items()})
 finally:
  for m,b in zip(model.modules(),mode):m.training=b
  torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf;dg.restore_rng(rng)
 assert dg.digest(model.state_dict())==before and dg.digest([p.grad for p in model.parameters()])==gradhash
 return norms,e
