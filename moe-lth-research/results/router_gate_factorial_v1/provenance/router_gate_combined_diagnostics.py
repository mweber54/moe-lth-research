import sys
sys.path.insert(0,'results/router_gate_factorial_v1')
from support import *
from moe_lth.utils import configure_device
import pandas as pd

def main():
 torch.set_num_threads(4);device=torch.device('cuda');configure_device(device);records=[];inputs=[];readouts=[];old=pd.read_csv(BASE/'outcomes.csv')
 for seed in (7,17,29):
  cfg=load_config(next(r for r in rows() if r['seed']==seed)['config']);_,va,_,_=data(cfg,seed);x,y=(z.to(device) for z in va[0]);sch=FactorialSchedule(BASE/f'schedules/e8/seed{seed}','B');_,_,p=sch.payload(x,0,0,'validation')
  masks=torch.load(ROOT/f'results/expert_count_replication/e8/masks/s0.85/seed_{seed}/pruning_mask.pt',map_location='cpu',weights_only=False)
  for age in (20,100):
   for kind in ('dense','sparse'):
    src=BASE/f'recovery/e8/seed{seed}/R{age}/{kind}/checkpoints/final_recovered.pt';h=file_hash(src);m=R.load_model_from_checkpoint(cfg['model'],str(src),device);handles=R.register_mask_gradient_hooks(m,{} if kind=='dense' else masks)
    target=OUT/f'measurements/B_reused/seed{seed}/R{age}/{kind}';target.mkdir(parents=True,exist_ok=False)
    n,e=diagnostics(m,x,y,p,'B',target/'E_2500.npz');assert n['router_task_norm']==0
    expected=old[(old.experts==8)&(old.seed==seed)&(old.age==age)].iloc[0]
    for policy in ('fixedgate','native','assign'):
     from moe_lth.experiments.fixed_schedule_intervention import Schedule
     reader=FactorialSchedule(sch.path,'B') if policy=='fixedgate' else Schedule(sch.path,'replay') if policy=='assign' else None
     loss=R._evaluate_condition_loss(m,va,device,False,1.,reader.validation if reader else None,2500);ref=float(expected[f'gate_{policy}_{kind}_loss']);err=abs(loss-ref);assert err<1e-12,(seed,age,kind,policy,loss,ref)
     readouts.append(dict(seed=seed,age=age,kind=kind,readout=policy,observed_loss=loss,audited_loss=ref,absolute_error=err,passed=True))
    records.append(dict(arm='B',seed=seed,age=age,kind=kind,step=2500,**n,measurement='new fixed-step0-probe diagnostic on retained historical final checkpoint'))
    assert file_hash(src)==h;inputs.append(dict(path=str(src),sha256=h))
    for handle in handles:handle.remove()
    del m;torch.cuda.empty_cache()
 pd.DataFrame(readouts).to_csv(OUT/'combined_endpoint_readout_recheck.csv',index=False)
 pd.DataFrame(records).to_csv(OUT/'combined_endpoint_gradient_diagnostics.csv',index=False);write(OUT/'combined_endpoint_diagnostic_provenance.json',dict(checkpoints=inputs,original_files_unchanged=True,original_training_gradients=False,diagnostic_payload='validation/step_0000_batch_000.pt'))
 print('Retained B endpoint diagnostics:',len(records))
if __name__=='__main__':main()
