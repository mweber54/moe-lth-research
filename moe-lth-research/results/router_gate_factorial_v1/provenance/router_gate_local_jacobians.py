"""Observed local native/used gate Jacobians on fixed archived validation batch0."""
import sys
sys.path.insert(0,'results/router_gate_factorial_v1')
from support import *
from moe_lth.models import TinyMoELanguageModel
from moe_lth.utils import configure_device
import pandas as pd

def main():
 torch.set_num_threads(4);device=torch.device('cuda');configure_device(device);records=[]
 for seed in (7,17,29):
  row=next(r for r in rows() if r['seed']==seed);cfg=load_config(row['config']);_,va,_,_=data(cfg,seed);x=va[0][0].to(device);sch=FactorialSchedule(BASE/f'schedules/e8/seed{seed}','C');_,_,payload=sch.payload(x,0,0,'validation')
  for arm in ('C','D'):
   for age in (20,100):
    for kind in ('dense','sparse'):
     tag=f'{arm}/seed{seed}/R{age}/{kind}';model=TinyMoELanguageModel(cfg['model']).to(device);model.eval()
     for step in STEPS:
      state=torch.load(OUT/'snapshots'/tag/f'step_{step:04d}.pt',map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(state['model']);proj=[];cap=[];handles=[]
      for block in model.blocks:handles.append(block.moe.router.projection.register_forward_hook(lambda m,a,o:proj.append(o)))
      try:
       with torch.autocast('cuda',dtype=torch.float16):output=forward_payload(model,x,payload,arm,capture=cap)
      finally:
       for h in handles:h.remove()
      for layer,(q,z,t) in enumerate(zip(cap,proj,output.routes)):
       native=torch.autograd.grad(q['native'].sum(),z,retain_graph=True)[0]
       used=torch.autograd.grad(q['used'].sum(),z,retain_graph=True)[0] if q['used'].requires_grad else torch.zeros_like(native)
       if arm=='C':assert torch.equal(native,used)
       else:assert not used.any()
       a=t.accepted_mask.flatten();norm=native.detach().float().norm(dim=-1)[a];g=q['native'].detach().flatten()[a]
       records.append(dict(arm=arm,seed=seed,age=age,kind=kind,step=step,layer=layer,accepted=int(a.sum()),native_local_J_exact_zero_fraction=float((norm==0).float().mean()),native_local_J_mean_norm=float(norm.mean()),native_local_J_max_norm=float(norm.max()),selected_gate_exact_zero_fraction=float((g==0).float().mean()),selected_gate_exact_one_fraction=float((g==1).float().mean()),used_J_matches_requested_policy=True))
      del state,output,cap,proj
     del model;torch.cuda.empty_cache()
 pd.DataFrame(records).to_csv(OUT/'fixed_probe_gate_jacobians.csv',index=False);print('Local actual FP16 Jacobians checked:',len(records))
if __name__=='__main__':main()
