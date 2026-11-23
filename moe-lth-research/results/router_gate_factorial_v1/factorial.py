"""Exact gate-value/Jacobian factorial adapter; original production code is unchanged."""
from __future__ import annotations
import sys,json,hashlib,importlib.util
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from moe_lth.models.router import RouteOverride
from moe_lth.experiments.fixed_schedule_intervention import Schedule,file_hash
from moe_lth.experiments.optimization_geometry_diagnostics import digest
ARMS={'A':('native','native'),'B':('donor','detached'),'C':('donor','native'),'D':('native','detached')}
STEPS=(0,25,100,250,500,1000,1500,2000,2500)
BASE=ROOT/'results/gate_amplitude_replay_intervention'
class DonorValueNativeJacobian(torch.autograd.Function):
 @staticmethod
 def forward(ctx,native,donor):
  if native.shape!=donor.shape or native.dtype!=donor.dtype:raise ValueError('Gate shape/dtype mismatch')
  return donor.clone()
 @staticmethod
 def backward(ctx,grad):return grad,None

def transform(native,donor,arm):
 if arm=='A':return native
 if arm=='B':return donor.detach()
 if arm=='C':return DonorValueNativeJacobian.apply(native,donor.detach())
 if arm=='D':return native.detach()
 raise ValueError(arm)

def forward_payload(model,x,payload,arm,*,capture=None):
 """Force IDs/priorities; replace only selected coefficients, retaining native probabilities for aux."""
 hooks=[]
 for layer,(block,v) in enumerate(zip(model.blocks,payload['layers'])):
  ids=v['ids'].to(x.device,torch.long);donor=v['capacity_scores'].to(x.device)
  def hook(module,args,output,layer=layer,ids=ids,donor=donor):
   assert torch.equal(output.selected_experts,ids.reshape_as(output.selected_experts))
   g=output.selected_probability;c=donor.reshape_as(g).to(dtype=g.dtype)
   new=transform(g,c,arm)
   if capture is not None:capture.append(dict(layer=layer,native=g,donor=c,used=new,probabilities=output.probabilities))
   return replace(output,selected_probability=new)
  hooks.append(block.moe.router.register_forward_hook(hook))
 try:
  overrides=[RouteOverride(v['ids'].to(x.device,torch.long),capacity_scores=v['capacity_scores'].to(x.device)) for v in payload['layers']]
  out=model(x,overrides)
 finally:
  for h in hooks:h.remove()
 for t,v in zip(out.routes,payload['layers']):
  assert torch.equal(t.selected_expert_indices,v['ids'].to(x.device,torch.long))
  assert torch.equal(t.accepted_mask,v['accepted'].to(x.device))
  if arm in ('B','C'):assert torch.equal(t.selected_probabilities,v['capacity_scores'].to(x.device,t.selected_probabilities.dtype))
 return out

class FactorialSchedule(Schedule):
 def __init__(self,path,arm):
  super().__init__(path,'replay');self.arm=arm;self.gate_records=[]
 def payload(self,x,step,batch,phase):
  key=self._key(phase,step,batch)
  if key not in self.entries:raise RuntimeError(f'Unavailable donor occurrence {key}')
  p=self.path/key;h=file_hash(p)
  if h!=self.entries[key]:raise RuntimeError(f'Donor changed {key}')
  d=torch.load(p,map_location='cpu',weights_only=False)
  assert d['input_hash']==digest(x) and d['shape']==list(x.shape) and d['step']==step and d['batch_index']==batch and d['phase']==phase
  return key,h,d
 def forward(self,model,x,step,batch,phase):
  key,h,payload=self.payload(x,step,batch,phase);cap=[];arm=self.arm if phase=='train' else 'B'
  out=forward_payload(model,x,payload,arm,capture=cap if phase=='train' and (step==1 or step%50==0) else None)
  self.counts[phase]+=1;self.rolling[phase].update(key.encode());self.rolling[phase].update(h.encode())
  if cap:
   with torch.no_grad():
    for q,t in zip(cap,out.routes):
     a=t.accepted_mask;g=q['native'].reshape_as(a)[a].detach().float();c=q['donor'].reshape_as(a)[a].detach().float();u=q['used'].reshape_as(a)[a].detach().float()
     self.gate_records.append(dict(step=step,layer=q['layer'],accepted=int(a.sum()),native_mean=float(g.mean()),donor_mean=float(c.mean()),used_mean=float(u.mean()),native_exact_zero_fraction=float((g==0).float().mean()),native_exact_one_fraction=float((g==1).float().mean()),maximum_forward_error=float((u-(c if arm=='C' else g)).abs().max()),gate_jacobian='native' if arm=='C' else 'zero'))
  return out

def trainer():
 p=OUT/'provenance/native_trainer.py'
 assert file_hash(p)=='082ff8495a27f160c1057613d273f75b53ea7d7644df5e1149e011ce24f3bdb8'
 spec=importlib.util.spec_from_file_location('factorial_frozen_trainer',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def group(s,k):
 from moe_lth.pruning.router_age import parameter_group
 return {n:v for n,v in s.items() if parameter_group(n)==k}
