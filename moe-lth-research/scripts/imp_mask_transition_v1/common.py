"""Isolated known-mask experiment utilities; historical artifacts are read-only."""
import sys,os,json,time,hashlib,copy,types,contextlib,socket,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/imp_adaptation_factorial_v1')]
import numpy as np
import torch
from af_common import read,save,sha,table,append,lock,load,protected,inventory,group,masked
from af_native import Proxy,encode_numpy,decode_numpy
from moe_lth.experiments import run_router_age_recovery as R
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.router_age import load_model_from_checkpoint,state_dict_hash,parameter_group,selected_experts_per_batch
from moe_lth.pruning.masks import load_masks,apply_masks_,expert_weight_parameters
from moe_lth.data import build_dataloaders
OUT=ROOT/'results/imp_mask_transition_v1';HIST=ROOT/'results/high_sparsity_coupling_v1'
SEEDS=(7,17,29,43,61);CONDITIONS=('dense','fixed','abrupt','gradual')
GRID=(0,*range(100,1001,100),1250,1500,1750,2000,2250,2500,2525,2600,2750,3000,3500,4000,4500,5000)
HOSTS=('viper.cs.kent.edu','adder.cs.kent.edu')
def state(m):return {n:t.detach().cpu().clone() for n,t in m.state_dict().items()}
def atomic_torch(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(obj,temp);temp.replace(path)
def batches(v):
 R.seed_everything(v['seed']);tr,va=build_dataloaders(v['config']['data'],128,v['seed'])
 train=R._materialize_batches(tr,5000);val=list(va)
 assert R._batch_sequence_hash(train[:2500])==v['canonical_pair']['data_hash']
 assert R._batch_sequence_hash(train[2500:])==v['extension_identity']['training_batch_hash']
 assert R._batch_sequence_hash(val)==v['canonical_pair']['validation_hash']
 return train,val
def stable_seed(seed,name):return int.from_bytes(hashlib.sha256(f'imp-mask-transition-v1:{seed}:{name}'.encode()).digest()[:8],'big')%(2**63-1)
def assignments(final,seed):
 result={}
 for name,mask in final.items():
  indices=(~mask).flatten().nonzero().flatten();N=len(indices)
  order=torch.randperm(N,generator=torch.Generator().manual_seed(stable_seed(seed,name)))
  assigned=torch.zeros(mask.numel(),dtype=torch.uint8)
  for k in range(1,11):assigned[indices[order[(k-1)*N//10:k*N//10]]]=k
  result[name]=assigned.reshape_as(mask)
 return result
def event_count(condition,attempt):
 return 10 if condition=='fixed' else (10 if condition=='abrupt' and attempt>=1000 else (min(10,attempt//100) if condition=='gradual' else 0))
def mask_at(assignment,condition,attempt):
 k=event_count(condition,attempt)
 return {n:((a==0)|(a>k)) for n,a in assignment.items()} if condition!='dense' else {}
def implementation():
 files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src/moe_lth').rglob('*.py'))+[ROOT/'scripts/imp_adaptation_factorial_v1'/n for n in ('af_common.py','af_native.py')]+[ROOT/'tests/test_imp_mask_transition.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in files}
def frozen():
 seal=read(OUT/'seal.json')
 for n,h in seal['files'].items():assert sha(OUT/n)==h,n
 for p,h in seal['implementation'].items():assert sha(ROOT/p)==h,p
 return read(OUT/'manifest.json')

class MaskManager:
 def __init__(self,assignment,condition,attempt=0):
  self.assignment=assignment;self.condition=condition;self.event=event_count(condition,attempt)
  self.masks=mask_at(assignment,condition,attempt);self.device_masks={};self.handles=[]
 def register(self,model,masks):
  params=dict(model.named_parameters())
  for n,mask in self.masks.items():
   self.device_masks[n]=mask.to(params[n].device,dtype=params[n].dtype)
   self.handles.append(params[n].register_hook(lambda gradient,n=n:gradient*self.device_masks[n]))
  return self.handles
 def restore(self,event):
  self.event=event
  for n,a in self.assignment.items():
   if n not in self.masks:continue
   self.masks[n].copy_((a==0)|(a>event))
   if n in self.device_masks:self.device_masks[n].copy_(self.masks[n])
 @torch.no_grad()
 def prune(self,model,optimizer,attempt):
  target=event_count(self.condition,attempt)
  if target<=self.event:return None
  params=dict(model.named_parameters());removed=0;retained_exact=True
  for n,a in self.assignment.items():
   new=((a>self.event)&(a<=target)).to(params[n].device);p=params[n];removed+=int(new.sum())
   # In-place masked_fill changes precisely deleted entries; retained values and scalar steps are untouched.
   p.masked_fill_(new,0)
   for key,value in optimizer.state.get(p,{}).items():
    if torch.is_tensor(value) and value.shape==p.shape:value.masked_fill_(new,0)
  old=self.event;self.restore(target)
  return dict(attempt=attempt,event_before=old,event_after=target,newly_removed=removed,retained_coordinates_untouched=retained_exact)
 def serial(self):return dict(condition=self.condition,event=self.event,mask_hash=R._mask_hash(self.masks))
 def active(self):return sum(int(m.sum()) for m in self.masks.values()) if self.masks else sum(x.numel() for x in self.assignment.values())
