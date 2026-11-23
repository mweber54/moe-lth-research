"""Re-evaluate existing endpoints with TF32 on/off; never updates parameters."""
from common import *
from runner import batches
from review_audit import REVIEW
import argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();torch.set_num_threads(2);f=frozen()
 for seed in a.seeds:
  jobs=[j for j in f['conditions'] if j['seed']==seed];_,val=batches(jobs[0],total=1);rows=[]
  for j in jobs:
   historical=read(Path(j['historical_checkpoints']['5000']).parents[1]/'extension_manifest.json');rerun=read(OUT/'reruns'/j['id']/'completed.json');m=model(state(j['historical_checkpoints']['5000']),j['config']);values={}
   for tf32 in (False,True):
    torch.backends.cuda.matmul.allow_tf32=tf32;torch.backends.cudnn.allow_tf32=tf32
    with protected(m,strict_fp32=False):values[tf32]=R._evaluate_condition_loss(m,val,torch.device('cuda'),False,1.)
   rows.append(dict(condition=j['id'],seed=seed,sparsity=j['sparsity'],historical_logged_ce=historical['extension_final_validation_cross_entropy'],rerun_logged_ce=rerun['final_validation_ce'],reevaluated_tf32_off=values[False],reevaluated_tf32_on=values[True],historical_residual_off=values[False]-historical['extension_final_validation_cross_entropy'],rerun_residual_on=values[True]-rerun['final_validation_ce']));print(rows[-1],flush=True);del m
  table(REVIEW/'ce'/f'seed_{seed}.csv',rows)
if __name__=='__main__':main()
