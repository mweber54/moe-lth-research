import json,csv
from pathlib import Path
import torch
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments.optimization_geometry_diagnostics import digest
from moe_lth.models.router import RouteOverride
from moe_lth.experiments.gate_amplitude_replay import GateSchedule
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything
prior=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention');root=Path('results/gate_amplitude_replay_intervention');m=json.loads((prior/'run_manifest.json').read_text());device=torch.device('cuda');configure_device(device);rows=[]
for e in [8,12]:
 for seed in [7,17,29]:
  row=next(x for x in m['conditions'] if x['experts']==e and x['seed']==seed and x['age']==100);cfg=load_config(row['config']);seed_everything(seed)
  _,loader=build_dataloaders(cfg['data'],int(cfg['training']['batch_size']),resolve_data_seed(cfg));val=r._materialize_validation_batches(loader,32);x,y=val[0];x=x.to(device);y=y.to(device)
  payload=torch.load(prior/f'schedules/e{e}/seed{seed}/validation/step_2500_batch_000.pt',map_location='cpu',weights_only=False);assert digest(x)==payload['input_hash']
  model=r.load_model_from_checkpoint(cfg['model'],str(prior/f'donors/e{e}/seed{seed}/checkpoints/final_recovered.pt'),device);model.eval()
  with torch.no_grad():
   native=model(x);forced=model(x,[RouteOverride(v['ids'].to(device,torch.long),v['capacity_scores'].to(device).detach(),v['capacity_scores'].to(device)) for v in payload['layers']])
   assert torch.equal(native.logits,forced.logits)
   for a,b,v in zip(native.routes,forced.routes,payload['layers']):
    assert torch.equal(a.selected_expert_indices,b.selected_expert_indices)
    assert torch.equal(a.accepted_mask,b.accepted_mask) and torch.equal(b.accepted_mask,v['accepted'].to(device))
    assert torch.equal(a.selected_probabilities,b.selected_probabilities) and torch.equal(b.selected_probabilities,v['capacity_scores'].to(device))
   loss=torch.nn.functional.cross_entropy(forced.logits.flatten(0,1),y.flatten()).item()
  native_loss=r._evaluate_condition_loss(model,val,device,False,1.)
  gate_loss=r._evaluate_condition_loss(model,val,device,False,1.,GateSchedule(prior/f'schedules/e{e}/seed{seed}').validation,2500)
  assert native_loss==gate_loss
  rows.append(dict(experts=e,seed=seed,check='final_donor_same_state_validation_batch_0',logits_exact=True,assignments_exact=True,accepted_tokens_exact=True,amplitudes_exact=True,max_amplitude_error=0.,batch_loss=loss,training_final_identity_tested=False,full_validation_loss_exact=True,native_validation_loss=native_loss,gate_validation_loss=gate_loss));print('PASS same-state donor',e,seed,flush=True)
  del model,native,forced;torch.cuda.empty_cache()
with (root/'donor_same_state_forward_checks.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
