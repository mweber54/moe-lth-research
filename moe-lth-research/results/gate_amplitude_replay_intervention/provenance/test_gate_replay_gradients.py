"""Detached gate replay must preserve forward values, not the native backward pass."""
from copy import deepcopy
import torch
from moe_lth.config import DEFAULT_CONFIG
from moe_lth.models import TinyMoELanguageModel
from moe_lth.models.router import RouteOverride

def check_gate_gradients():
 torch.set_num_threads(1);torch.manual_seed(7)
 cfg=deepcopy(DEFAULT_CONFIG['model']);cfg.update(vocab_size=256,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=2,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=.6)
 native=TinyMoELanguageModel(cfg);fixed=deepcopy(native);x=torch.randint(0,256,(4,8));y=torch.randint(0,256,(4,8))
 out=native(x);overrides=[RouteOverride(t.selected_expert_indices.detach(),t.selected_probabilities.detach(),t.selected_probabilities.detach()) for t in out.routes];replayed=fixed(x,overrides)
 assert torch.equal(out.logits,replayed.logits)
 for a,b in zip(out.routes,replayed.routes):
  assert torch.equal(a.accepted_mask,b.accepted_mask) and torch.equal(a.selected_probabilities,b.selected_probabilities)
  assert not b.selected_probabilities.requires_grad
 def gradients(model,output,aux=False):
  loss=output.auxiliary_loss if aux else torch.nn.functional.cross_entropy(output.logits.flatten(0,1),y.flatten())
  return torch.autograd.grad(loss,list(model.parameters()),allow_unused=True,retain_graph=True)
 ng=gradients(native,out);fg=gradients(fixed,replayed);ag=gradients(fixed,replayed,True)
 result={}
 for group in ['router','expert','shared']:
  ids=[i for i,(n,p) in enumerate(fixed.named_parameters()) if ('router' if '.router.' in n else 'expert' if '.experts.' in n else 'shared')==group]
  norm=lambda gs:sum(float(gs[i].float().square().sum()) for i in ids if gs[i] is not None)**.5
  result[group]={'native_task_norm':norm(ng),'fixed_task_norm':norm(fg),'fixed_aux_norm':norm(ag)}
 assert result['router']['native_task_norm']>0 and result['router']['fixed_task_norm']==0 and result['router']['fixed_aux_norm']>0
 assert any(a is not None and b is not None and not torch.equal(a,b) for (n,p),a,b in zip(fixed.named_parameters(),ng,fg) if '.router.' not in n)
 return {'same_state_logits_exact':True,'amplitudes_exact':True,'accepted_tokens_exact':True,'detached_gate':True,'nonrouter_task_gradients_differ':True,'gradients':result}

def test_gate_gradient_semantics():check_gate_gradients()
if __name__=='__main__':
 import json
 from pathlib import Path
 result=check_gate_gradients();print(json.dumps(result,indent=2));Path('results/gate_amplitude_replay_intervention/gradient_semantics_check.json').write_text(json.dumps(result,indent=2)+'\n')

def test_gate_archive_replay(tmp_path):
 from moe_lth.experiments.fixed_schedule_intervention import Schedule
 from moe_lth.experiments.gate_amplitude_replay import GateSchedule
 torch.manual_seed(11)
 cfg=deepcopy(DEFAULT_CONFIG['model']);cfg.update(vocab_size=256,max_seq_len=8,num_layers=2,num_heads=2,d_model=16,num_experts=2,expert_hidden_size=32,dropout=0.,top_k=1,capacity_factor=.6)
 model=TinyMoELanguageModel(cfg);x=torch.randint(0,256,(4,8))
 capture=Schedule(tmp_path/'archive','capture');native=capture.train(model,x,1);capture.finish_capture({'fixture':True})
 replay=GateSchedule(tmp_path/'archive',tmp_path/'stats.csv',{'seed':11});forced=replay.train(model,x,1);audit=replay.verify_complete_replay()
 assert torch.equal(native.logits,forced.logits) and audit['maximum_absolute_amplitude_error']==0
 assert (tmp_path/'stats.csv').is_file()
 wrong=GateSchedule(tmp_path/'archive')
 try:wrong.train(model,x.roll(1,0),1)
 except RuntimeError as exc:assert 'Input identity mismatch' in str(exc)
 else:raise AssertionError('Wrong input accepted')
