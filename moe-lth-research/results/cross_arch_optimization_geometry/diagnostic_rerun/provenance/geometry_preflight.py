import json,sys
from pathlib import Path
import torch
from moe_lth.config import load_config
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import state_dict_hash,component_state_dict
root=Path('results'); rows=[]
for e,ages in [(8,[0,20,100]),(12,[20,60,100])]:
 for seed in [7,17,29]:
  for age in ages:
   base=root/('router_development_12expert_retry4_20260902' if age==60 else f'expert_count_replication/e{e}')
   sp=base/f'sparse/s0.85/seed_{seed}/R{age}'
   dp=base/f'dense/seed_{seed}/R{age}'
   sm=json.loads((sp/'metadata.json').read_text());dm=json.loads((dp/'metadata.json').read_text())
   cp=Path(sm['router_checkpoint']);ref=cp.parent
   cfg=load_config(ref.parent/'resolved_config.yaml')
   init=r.load_model_from_checkpoint(cfg['model'],str(ref/'step_0.pt'),torch.device('cpu'))
   final=r.load_model_from_checkpoint(cfg['model'],str(ref/'step_2500.pt'),torch.device('cpu'))
   masks=expert_local_magnitude_masks(final,.85)
   router=r.load_model_from_checkpoint(cfg['model'],str(cp),torch.device('cpu'))
   ticket=r.build_fixed_pruned_base(cfg['model'],str(ref/'step_0.pt'),masks,torch.device('cpu'))
   checks={'E0':state_dict_hash(component_state_dict(init,'expert'))==dm['expert_state_hash'],'shared':state_dict_hash(component_state_dict(init,'shared'))==sm['shared_state_hash']==dm['shared_state_hash'],'mask':r._mask_hash(masks)==sm['mask_hash'],'ticket':state_dict_hash({k:v for k,v in ticket.items() if r.parameter_group(k)=='expert'})==sm['expert_state_hash'],'router':state_dict_hash(component_state_dict(router,'router'))==sm['initial_router_state_hash']==dm['initial_router_state_hash']}
   row={'experts':e,'seed':seed,'age':age,'checks':checks,'config':str(ref.parent/'resolved_config.yaml'),'sparse':str(sp),'dense':str(dp)}; rows.append(row)
   print(json.dumps(row),flush=True)
Path('/tmp/geometry_preflight.json').write_text(json.dumps(rows,indent=2))
