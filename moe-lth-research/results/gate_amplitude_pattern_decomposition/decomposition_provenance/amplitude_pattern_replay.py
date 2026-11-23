"""Immutable compiled amplitude rules; primary validation uses a separate R100 reader."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from moe_lth.models.router import RouteOverride
from moe_lth.pruning.router_age import set_router_temperature
from moe_lth.experiments.fixed_schedule_intervention import Schedule,file_hash
from moe_lth.experiments.optimization_geometry_diagnostics import digest

def coefficient_hash(layers):
 h=hashlib.sha256()
 for x in layers:
  x=np.ascontiguousarray(x);h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.tobytes())
 return h.hexdigest()

class PatternSchedule(Schedule):
 def __init__(self,donor_path,transform_path,condition):
  super().__init__(donor_path,'replay');self.transform_path=Path(transform_path);self.transform=json.loads((self.transform_path/'manifest.json').read_text());assert self.transform['status']=='validated'
  assert condition in self.transform['conditions'];self.condition=condition
  self.arrays={phase:np.load(self.transform_path/f'{condition}_{phase}.npy',mmap_mode='r') for phase in ['train','validation']};self.coefficient_sequence=hashlib.sha256()
 def forward(self,model,x,step,batch,phase):
  key=self._key(phase,step,batch);sha=file_hash(self.path/key);assert sha==self.entries[key],'Donor schedule changed'
  donor=torch.load(self.path/key,map_location='cpu',weights_only=False);entry=self.transform['phase_entries'][phase][key]
  assert entry['donor_file_hash']==sha and entry['input_hash']==donor['input_hash']==digest(x) and entry['shape']==list(x.shape)
  assert donor['phase']==phase and donor['step']==step and donor['batch_index']==batch
  b,s=x.shape;raw=self.arrays[phase][entry['index']];values=[np.array(a[:b,:s],copy=True) for a in raw]
  coefficient_digest=coefficient_hash(values);assert coefficient_digest==self.transform['amplitude_hashes'][phase][self.condition][entry['index']],'Transformed amplitude changed'
  overrides=[RouteOverride(v['ids'].to(x.device,torch.long),torch.from_numpy(a).to(x.device).detach(),v['capacity_scores'].to(x.device)) for a,v in zip(values,donor['layers'])]
  set_router_temperature(model,1.);output=model(x,overrides);assert len(output.routes)==len(values)
  for trace,v,a in zip(output.routes,donor['layers'],values):
   assert torch.equal(trace.selected_expert_indices,v['ids'].to(x.device,torch.long))
   assert torch.equal(trace.accepted_mask,v['accepted'].to(x.device))
   assert torch.equal(trace.selected_probabilities,torch.from_numpy(a).to(x.device)) and not trace.selected_probabilities.requires_grad
  self.counts[phase]+=1;self.rolling[phase].update(key.encode());self.rolling[phase].update(sha.encode());self.coefficient_sequence.update(key.encode());self.coefficient_sequence.update(coefficient_digest.encode())
  return output
 def verify_training(self):
  assert self.counts['train']==self.manifest['counts']['train']
  assert self.rolling['train'].hexdigest()==self.manifest['sequence_hashes']['train']
  return dict(training_occurrences=self.counts['train'],training_route_hash=self.rolling['train'].hexdigest(),transformed_amplitude_sequence_hash=self.coefficient_sequence.hexdigest(),max_amplitude_error=0.,detached=True,assignments_exact=True,acceptances_exact=True)
