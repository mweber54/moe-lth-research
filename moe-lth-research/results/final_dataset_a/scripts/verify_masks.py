"""Check existing masks against R100 magnitude order; never create or save a mask."""
from collections import defaultdict
from pathlib import Path
import pandas as pd,torch
from build import OUT,relocated,csv,C
from moe_lth.pruning.magnitude_prune import _expert_key

def main():
 torch.set_num_threads(2);m=pd.read_csv(OUT/'inputs/mask_manifest.csv');cp=pd.read_csv(OUT/'inputs/source_checkpoint_manifest.csv');rows=[]
 for (e,seed),g in m.groupby(['num_experts','seed']):
  p=relocated(cp[(cp.num_experts==e)&(cp.seed==seed)&(cp.router_age==100)].iloc[0]['path']);ck=C.load(p)
  for r in g.itertuples():
   mask=torch.load(relocated(r.path),map_location='cpu',weights_only=True);groups=defaultdict(list)
   for n in mask:groups[_expert_key(n)].append(n)
   bad=[];keep_total=0;total=0;min_margin=float('inf')
   for expert,names in groups.items():
    counts=[mask[n].numel() for n in names];kept=sum(int(mask[n].sum()) for n in names);count=sum(counts);expected=max(1,int(round(count*(1-r.target_sparsity))));keep_total+=kept;total+=count
    retained_min=min(float(ck['model'][n].abs()[mask[n].bool()].min()) for n in names if mask[n].any())
    pruned_max=max(float(ck['model'][n].abs()[~mask[n].bool()].max()) for n in names if (~mask[n].bool()).any());margin=retained_min-pruned_max;min_margin=min(min_margin,margin)
    if len(names)!=2 or any(not n.endswith('.weight') for n in names) or kept!=expected or margin<0:bad.append(expert)
   ok=not bad and len(groups)==4*e and keep_total==r.keep_count and total-keep_total==r.pruned_count and abs((1-keep_total/total)-r.actual_sparsity)<1e-12
   rows.append({'num_experts':e,'seed':seed,'sparsity':r.target_sparsity,'source_r100_checkpoint':str(p),'mask_path':str(relocated(r.path)),'expert_groups':len(groups),'keep_count':keep_total,'pruned_count':total-keep_total,'actual_sparsity':1-keep_total/total,'minimum_retained_minus_maximum_pruned_magnitude':min_margin,'invalid_expert_groups':';'.join(bad),'passed':ok})
   del mask
  del ck
  print('R100 mask-order check',e,seed,flush=True)
 result=csv(OUT/'audits/dataset_a_mask_magnitude_verification.csv',rows);assert result.passed.all();print('All 150 stored masks satisfy expert-local R100 magnitude order and exact rounded keep counts; no masks generated.',flush=True)
if __name__=='__main__':main()
