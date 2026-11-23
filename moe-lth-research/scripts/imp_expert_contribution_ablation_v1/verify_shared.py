"""Read-only supplemental pair verification; no model forward or training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.imp_capacity_diagnostic_v1 import run as c
OUT=ROOT/'results/imp_expert_contribution_ablation_v1'
def main():
    variants=c.adaptation_variants();rows=[]
    for seed in c.SEEDS:
        for m in c.MODEL_TYPES:
            start,_,_,_=c._state_for(seed,m,0,variants[seed]);end,_,_,_=c._state_for(seed,m,500,variants[seed])
            assert set(start)==set(end)
            groups={}
            for group in ('router','experts','shared'):
                names=[n for n in start if c.common.group_for_parameter(n)==group]
                changed=[n for n in names if not c.torch.equal(start[n],end[n])]
                bias=[n for n in names if n.endswith('bias')]
                if group=='shared':assert changed
                else:assert not changed
                groups[group]=dict(start_sha256=c._group_hash(start,group),end_sha256=c._group_hash(end,group),tensor_count=len(names),bias_tensor_count=len(bias),changed_tensor_count=len(changed),changed_names=changed,delta_l2=sum(float((end[n].double()-start[n].double()).square().sum()) for n in names)**.5)
            rows.append(dict(seed=seed,model_type=m,groups=groups,status='PASS'))
            print(seed,m,'shared movement and frozen groups verified',flush=True)
    c._json_write(OUT/'shared_parameter_verification.json',dict(status='PASS',pairs=rows,source_sha256=c._sha256(Path(__file__)),training_updates=0))
if __name__=='__main__':main()
