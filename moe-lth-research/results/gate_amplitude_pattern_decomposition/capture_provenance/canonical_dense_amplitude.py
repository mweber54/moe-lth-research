"""Capture native probabilities on donor-assigned edges without changing Level-2 training."""
import json
from pathlib import Path
import torch
from moe_lth.experiments.gate_amplitude_replay import GateSchedule
from moe_lth.experiments.fixed_schedule_intervention import file_hash
from moe_lth.experiments.optimization_geometry_diagnostics import digest

class DenseAmplitudeCapture(GateSchedule):
 def __init__(self,donor_path,output_path,identity,expected_statistics=None):
  super().__init__(donor_path);self.output_path=Path(output_path);self.output_path.mkdir(parents=True,exist_ok=False)
  self.identity=identity;self.capture_entries={};self.expected_statistics=expected_statistics;self.checked_summary_rows=0
 def forward(self,model,x,step,batch,phase):
  output=super().forward(model,x,step,batch,phase)
  key=self._key(phase,step,batch);layers=[]
  with torch.no_grad():
   for layer,t in enumerate(output.routes):
    proposal=t.probabilities.detach().gather(-1,t.selected_expert_indices).cpu()
    assert torch.isfinite(proposal).all()
    layers.append(dict(amplitudes=proposal,expert_ids=t.selected_expert_indices.detach().to('cpu',torch.uint8),accepted=t.accepted_mask.detach().cpu()))
    if self.expected_statistics is not None and phase=='train' and (step==1 or step%50==0):
     p=t.probabilities.detach().float().gather(-1,t.selected_expert_indices).squeeze(-1);ids=t.selected_expert_indices.squeeze(-1);ok=t.accepted_mask.reshape_as(ids)
     for expert in [-1]+list(range(t.probabilities.shape[-1])):
      mask=ok if expert==-1 else ok&(ids==expert)
      if not mask.any():continue
      expected=self.expected_statistics.loc[(step,layer,expert)]
      assert int(mask.sum())==int(expected['observations'])
      assert abs(float(p[mask].mean())-float(expected['native_recipient_proposal_amplitude_mean']))<1e-7,'Native proposal mean differs from Level 2'
      self.checked_summary_rows+=1
  payload={**self.identity,'phase':phase,'step':step,'batch_index':batch,'input_hash':digest(x),'shape':list(x.shape),'indexing':'row=sample within chronological materialized batch; column=token position; final axis=selected route slot','source_donor_file_hash':self.entries[key],'layers':layers,'forward_logits_hash':digest(output.logits),'auxiliary_loss':float(output.auxiliary_loss.detach())}
  path=self.output_path/key;path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(payload,path)
  self.capture_entries[key]=file_hash(path)
  return output
 def finish(self):
  replay=self.verify_complete_replay()
  m={**self.identity,'source':'canonical age/seed dense amplitude source tensor; native proposals along validated Level-2 dense recovery','entries':self.capture_entries,'counts':self.counts,'replay_audit':replay,'checked_historical_summary_rows':self.checked_summary_rows,'status':'captured_pending_trajectory_validation'}
  (self.output_path/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return m
