"""Same-state, paired-identity and gradient checks for all compiled sources/conditions."""
from pathlib import Path
import json,gc
import torch
import pandas as pd
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments.optimization_geometry_diagnostics import digest,append
from moe_lth.experiments.amplitude_pattern_replay import PatternSchedule
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything
ROOT=Path('results/gate_amplitude_pattern_decomposition');LEVEL2=Path('results/gate_amplitude_replay_intervention')

def main():
 m=json.loads((ROOT/'amplitude_transform_manifest.json').read_text());assert m['status']=='all_transforms_validated' and m['sources']==18
 capture=json.loads((ROOT/'capture_run_manifest.json').read_text());assert capture['validated_sources']==18
 device=torch.device('cuda');configure_device(device);data={}
 for row in capture['conditions']:
  e,seed,age=row['experts'],row['seed'],row['age'];cfg=load_config(row['config'])
  if seed not in data:
   seed_everything(seed);a,b=build_dataloaders(cfg['data'],int(cfg['training']['batch_size']),resolve_data_seed(cfg));data[seed]=(next(iter(a)),next(iter(b)))
  path=LEVEL2/f'snapshots/gate/e{e}/seed{seed}/R{age}/dense/step_0000.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
  dense=r.load_model_from_checkpoint(cfg['model'],str(path),device);sparse=r.load_model_from_checkpoint(cfg['model'],str(path),device);r.apply_masks_(sparse,saved['mask'])
  original=[digest(q.state_dict()) for q in [dense,sparse]]
  for condition in m['conditions']:
   for phase,(x,y) in zip(['train','validation'],data[seed]):
    x=x.to(device);y=y.to(device);outputs=[]
    for model in [dense,sparse]:
     model.train(phase=='train');reader=PatternSchedule(LEVEL2/f'schedules/e{e}/seed{seed}',ROOT/f'transformed/e{e}/seed{seed}/R{age}',condition)
     with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=phase=='train'):
      out=reader.train(model,x,1) if phase=='train' else reader.validation(model,x,0,0)
      task=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.flatten())
     routers=[p for n,p in model.named_parameters() if '.router.' in n]
     task_grads=torch.autograd.grad(task,routers,allow_unused=True,retain_graph=True);assert all(g is None or torch.count_nonzero(g)==0 for g in task_grads)
     aux_grads=torch.autograd.grad(out.auxiliary_loss,routers,allow_unused=True);assert any(g is not None and torch.count_nonzero(g)>0 for g in aux_grads)
     outputs.append([t.selected_probabilities.detach().cpu() for t in out.routes])
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    append(ROOT/'compiled_forward_checks.csv',[dict(experts=e,seed=seed,age=age,condition=condition,phase=phase,paired_amplitudes_exact=True,accepted_tokens_exact=True,router_task_gradient_zero=True,router_auxiliary_gradient_available=True)])
   assert [digest(q.state_dict()) for q in [dense,sparse]]==original
  del dense,sparse,saved,out;gc.collect();torch.cuda.empty_cache();print(f'[compiled forward passed] e{e} seed{seed} R{age}',flush=True)
 (ROOT/'compiled_forward_integrity.json').write_text(json.dumps(dict(status='passed',sources=18,conditions=7,phase_checks=252,paired_forward_checks=504),indent=2)+'\n')
if __name__=='__main__':main()
