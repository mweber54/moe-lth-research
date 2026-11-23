"""Exact detached coefficient replay using the existing complete donor archive."""
import hashlib
import torch
from moe_lth.models.router import RouteOverride
from moe_lth.pruning.router_age import set_router_temperature
from moe_lth.experiments.fixed_schedule_intervention import Schedule,file_hash
from moe_lth.experiments.optimization_geometry_diagnostics import digest,append

class GateSchedule(Schedule):
 def __init__(self,path,stats_path=None,identity=None):
  super().__init__(path,'replay');self.stats_path=stats_path;self.identity=identity or {};self.amplitude_hash=hashlib.sha256();self.max_amplitude_error=0.
 def forward(self,model,x,step,batch,phase):
  key=self._key(phase,step,batch);path=self.path/key
  if key not in self.entries:raise RuntimeError(f'Missing schedule {key}')
  sha=file_hash(path)
  if sha!=self.entries[key]:raise RuntimeError(f'Archive changed {key}')
  payload=torch.load(path,map_location='cpu',weights_only=False)
  if payload['input_hash']!=digest(x) or payload['shape']!=list(x.shape):raise RuntimeError(f'Input identity mismatch {key}')
  assert payload['step']==step and payload['batch_index']==batch and payload['phase']==phase
  set_router_temperature(model,1.)
  overrides=[RouteOverride(v['ids'].to(x.device,torch.long),v['capacity_scores'].to(x.device).detach(),v['capacity_scores'].to(x.device)) for v in payload['layers']]
  output=model(x,overrides);assert len(output.routes)==len(payload['layers'])
  for layer,(trace,v) in enumerate(zip(output.routes,payload['layers'])):
   assert torch.equal(trace.selected_expert_indices,v['ids'].to(x.device,torch.long)),key
   assert torch.equal(trace.accepted_mask,v['accepted'].to(x.device)),key
   assert torch.equal(trace.selected_probabilities,v['capacity_scores'].to(x.device)),key
   assert not trace.selected_probabilities.requires_grad,key
   self.amplitude_hash.update(key.encode());self.amplitude_hash.update(str(layer).encode());self.amplitude_hash.update(v['capacity_scores'].numpy().tobytes())
  if self.stats_path and phase=='train' and (step==1 or step%50==0):
   self.observe(output,payload,step)
  self.counts[phase]+=1;self.rolling[phase].update(key.encode());self.rolling[phase].update(sha.encode())
  return output
 def observe(self,output,payload,step):
  rows=[]
  with torch.no_grad():
   for layer,(t,v) in enumerate(zip(output.routes,payload['layers'])):
    p=t.probabilities.detach().float();native_amp,native_id=p.max(-1);donor=t.selected_probabilities.detach().float().squeeze(-1);ids=t.selected_expert_indices.squeeze(-1);accepted=t.accepted_mask.reshape_as(ids);proposal=p.gather(-1,ids.unsqueeze(-1)).squeeze(-1)
    for expert in [-1]+list(range(p.shape[-1])):
     selected=accepted if expert==-1 else accepted&(ids==expert)
     if not selected.any():continue
     a=proposal[selected];b=donor[selected];q=torch.quantile(a,torch.tensor([.05,.25,.5,.75,.95],device=a.device))
     rows.append({**self.identity,'step':step,'layer':layer,'expert':expert,'observations':int(selected.sum()),'donor_amplitude_mean':float(b.mean()),'native_recipient_proposal_amplitude_mean':float(a.mean()),'difference':float((a-b).mean()),'amplitude_sd':float(a.std(unbiased=False)),'donor_amplitude_sd':float(b.std(unbiased=False)),'q05':float(q[0]),'q25':float(q[1]),'q50':float(q[2]),'q75':float(q[3]),'q95':float(q[4]),'entropy':float(t.entropy.detach().float()[selected].mean()),'margin':float(t.margin.detach().float()[selected].mean()),'native_selected_gate_mean':float(native_amp[selected].mean()),'proposal_context':'native probabilities on intervention hidden states; donor-accepted edges'})
  append(self.stats_path,rows)
 def verify_complete_replay(self):
  result=super().verify_complete_replay();result.update(amplitude_schedule_hash=self.amplitude_hash.hexdigest(),maximum_absolute_amplitude_error=self.max_amplitude_error,detached_amplitudes=True);return result
