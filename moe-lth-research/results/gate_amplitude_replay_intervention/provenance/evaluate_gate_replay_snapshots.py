"""Read-only common-readout reevaluation of retained states; no baseline retraining."""
from pathlib import Path
import json,gc
import torch
import pandas as pd
from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as r
from moe_lth.experiments import optimization_geometry_diagnostics as d
from moe_lth.experiments.gate_amplitude_replay import GateSchedule
from moe_lth.utils import configure_device,resolve_data_seed,seed_everything
ROOT=Path('results/gate_amplitude_replay_intervention')
L0=Path('results/cross_arch_optimization_geometry/diagnostic_rerun');L1=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')

def main():
 manifest=json.loads((ROOT/'gate_replay_manifest.json').read_text());device=torch.device('cuda');configure_device(device)
 existing=set()
 if (ROOT/'common_readout_snapshot_losses.csv').exists():
  existing=set(pd.read_csv(ROOT/'common_readout_snapshot_losses.csv')[['level','experts','seed','age','kind','step']].itertuples(index=False,name=None))
 for level,source in [('native',L0),('assign',L1),('gate',ROOT)]:
  snaps=pd.read_csv(source/'snapshot_manifest.csv')
  if level=='assign':snaps=snaps[snaps.phase=='clamped']
  data={}
  for row in snaps.itertuples():
   key=(level,row.experts,row.seed,row.age,row.kind,row.step)
   if key in existing:continue
   cfg=load_config(next(x['config'] for x in manifest['conditions'] if x['experts']==row.experts and x['seed']==row.seed and x['age']==row.age))
   if row.seed not in data:
    seed_everything(row.seed);_,loader=build_dataloaders(cfg['data'],int(cfg['training']['batch_size']),resolve_data_seed(cfg));data[row.seed]=r._materialize_validation_batches(loader,32)
   val=data[row.seed];schedule=ROOT/f'schedules/e{row.experts}/seed{row.seed}'
   state=torch.load(row.path,map_location='cpu',weights_only=False);model=r.load_model_from_checkpoint(cfg['model'],row.path,device);before=d.digest(model.state_dict());assert before==row.model_hash; rng=d.rng_state()
   loss=r._evaluate_condition_loss(model,val,device,False,1.,GateSchedule(schedule).validation,row.step)
   model.eval();x,y=[v.to(device) for v in val[0]];out=GateSchedule(schedule).validation(model,x,row.step,0)
   task_loss=torch.nn.functional.cross_entropy(out.logits.flatten(0,1),y.flatten());aux_loss=out.auxiliary_loss
   objective=task_loss+float(cfg['routing']['aux_loss_weight'])*aux_loss
   model.zero_grad(set_to_none=True);objective.backward()
   diag=[]
   for group in ['expert','router','shared']:
    sums=0.;count=0;bad=0
    for name,p in model.named_parameters():
     if r.parameter_group(name)!=group or p.grad is None:continue
     g=p.grad.detach().float()
     if row.kind=='sparse' and name in state.get('mask',{}):g=g*state['mask'][name].to(device)
     finite=torch.isfinite(g);count+=g.numel();bad+=int((~finite).sum());sums+=float(g[finite].double().square().sum())
    diag.append(dict(level=level,experts=row.experts,seed=row.seed,age=row.age,kind=row.kind,step=row.step,group=group,gradient_norm=sums**.5,nonfinite_fraction=bad/max(count,1),precision='fp32 common fixedgate validation batch 0 diagnostic',task_loss=float(task_loss.detach()),auxiliary_loss=float(aux_loss.detach()),auxiliary_weight=float(cfg['routing']['aux_loss_weight']),grad_scaler_scale=state.get('grad_scaler',{}).get('scale',None)))
   d.append(ROOT/'common_readout_gradient_diagnostics.csv',diag)
   model.zero_grad(set_to_none=True);d.restore_rng(rng);assert before==d.digest(model.state_dict())
   d.append(ROOT/'readout_mutation_audit.csv',[dict(level=level,experts=row.experts,seed=row.seed,age=row.age,kind=row.kind,step=row.step,model_unchanged=True,rng_restored=True)])
   d.append(ROOT/'common_readout_snapshot_losses.csv',[dict(level=level,experts=row.experts,seed=row.seed,age=row.age,kind=row.kind,step=row.step,loss=loss)])
   del model,state,out;gc.collect();torch.cuda.empty_cache()
   print(f'[common readout] {level} e{row.experts} seed{row.seed} R{row.age} {row.kind} step{row.step}',flush=True)
if __name__=='__main__':main()
