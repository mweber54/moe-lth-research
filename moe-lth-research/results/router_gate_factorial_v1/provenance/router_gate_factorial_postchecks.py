import sys,json,pathlib,math
sys.path.insert(0,'results/router_gate_factorial_v1')
from support import *
import pandas as pd,time
def await_complete(path):
 while not (path/'complete.json').exists():time.sleep(5)

def main():
 torch.set_num_threads(2);records=[];selfchecks=[];finalchecks=[]
 for seed in (7,17,29):
  a=OUT/f'recoveries/C/seed{seed}/R100/dense';await_complete(a)
  actual=[json.loads(x) for x in (a/'metrics.jsonl').read_text().splitlines()]
  orig=ROOT/f'results/expert_count_replication/e8/dense/seed_{seed}/R100'
  original=[json.loads(x) for x in (orig/'metrics.jsonl').read_text().splitlines()]
  m1=torch.load(a/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False,mmap=True)['model'];m2=torch.load(orig/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False,mmap=True)['model']
  same=dg.digest(m1)==dg.digest(m2);assert actual==original and same
  selfchecks.append(dict(seed=seed,validation_observations=len(actual),trace_bit_exact=True,final_model_bit_exact=same))
  for arm in ('C','D'):
   for age in (20,100):
    for kind in ('dense','sparse'):
     tag=f'{arm}/seed{seed}/R{age}/{kind}';job=OUT/'recoveries'/tag;extra=OUT/'measurements'/tag;await_complete(job)
     endpoint=torch.load(job/'checkpoints/final_recovered.pt',map_location='cpu',weights_only=False,mmap=True)['model'];last=torch.load(OUT/'snapshots'/tag/'step_2500.pt',map_location='cpu',weights_only=False,mmap=True)
     assert dg.digest(endpoint)==dg.digest(last['model'])
     assert all(torch.isfinite(v).all() for v in endpoint.values())
     assert torch.equal(endpoint['token_embedding.weight'],endpoint['lm_head.weight'])
     if kind=='sparse':
      for name,mask in last['mask'].items():assert not endpoint[name][~mask.bool()].any(),name
     finalchecks.append(dict(arm=arm,seed=seed,age=age,kind=kind,final_matches_snapshot=True,all_weights_finite=True,embedding_tie_preserved=True,mask_zeros_preserved=True))
     del endpoint,last
     print('Final-state checks passed',tag,flush=True)
     applied={r['step']:r['optimizer_step_applied'] for r in [json.loads(x) for x in (job/'gradient_stats/gradient_stats.jsonl').read_text().splitlines()]}
     for t in STEPS[:-1]:
      s=torch.load(OUT/'snapshots'/tag/f'step_{t:04d}.pt',map_location='cpu',weights_only=False,mmap=True);e=s['model']['token_embedding.weight'];pg=s['optimizer']['param_groups'][0];idx=pg['params'][0];op=s['optimizer']['state'].get(idx,{})
      z=np.load(extra/f'update_{t+1:04d}.npz');g=torch.from_numpy(z['clipped_gradient']);observed=torch.from_numpy(z['actual_update'])
      if not applied[t+1]:
       assert not observed.any();records.append(dict(arm=arm,seed=seed,age=age,kind=kind,source_step=t,update_step=t+1,skipped=True,max_absolute_error=0,relative_update_error=0,max_error_in_rounding_bounds=0));continue
      b1,b2=pg['betas'];step=int(op.get('step',0))+1;m=op.get('exp_avg',torch.zeros_like(e)).clone();v=op.get('exp_avg_sq',torch.zeros_like(e)).clone()
      m.lerp_(g,1-b1);v.mul_(b2).addcmul_(g,g,value=1-b2)
      den=v.sqrt().div_(math.sqrt(1-b2**step)).add_(pg['eps']);expected=e.clone().mul_(1-pg['lr']*pg['weight_decay']);expected.addcdiv_(m,den,value=-pg['lr']/(1-b1**step));predicted=expected-e;err=(predicted-observed).abs();bound=8*torch.finfo(e.dtype).eps*(e.abs()+expected.abs()+1)
      assert bool((err<=bound).all()),(tag,t,float(err.max()))
      records.append(dict(arm=arm,seed=seed,age=age,kind=kind,source_step=t,update_step=t+1,skipped=False,max_absolute_error=float(err.max()),relative_update_error=float(err.norm()/(observed.norm()+1e-12)),max_error_in_rounding_bounds=float((err/bound).max())))
 pd.DataFrame(records).to_csv(OUT/'adamw_update_reconstruction.csv',index=False);write(OUT/'postrun_validation.json',dict(C_dense_R100_self_replay=selfchecks,final_state_checks=finalchecks,adamw_observations=len(records),AdamW_reconstruction_passed=True,rounding_bound='8*float32_eps*(abs(before)+abs(after)+1); no optimization-outcome threshold',auxiliary_graph='Same native auxiliary objective; earlier gate detaches also affect later auxiliary paths.'))
 print('All 3 C dense R100 self-replays bit exact; AdamW reconstructions:',len(records))
if __name__=='__main__':main()
