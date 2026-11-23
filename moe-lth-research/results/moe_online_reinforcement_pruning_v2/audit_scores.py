"""Check archived score exclusions and original-sign tensors against initial model."""
import json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;V1=HERE.parent/'moe_online_reinforcement_pruning_v1'
results={}
for seed in (7,17,29):
 for order in ('A','B'):
  p=V1/'prefixes'/f'seed{seed}_{order}';v=json.loads((p/'verification.json').read_text());initial=torch.load(v['initial_checkpoint'],map_location='cpu',weights_only=False)['model'];payload=torch.load(p/'scores.pt',map_location='cpu',weights_only=True)
  assert payload['attempts']==250 and payload['applied_updates_1_250']==249
  totals={'zero_denominator':0,'zero_start_sign':0,'eligible':0}
  assert len(payload['scores'])==256
  for name,record in payload['scores'].items():
   sign=record['initial_sign'];den=record['absolute_update'];score=record['score'];valid=(sign!=0)&(den>0)
   assert torch.equal(sign,initial[name].sign())
   assert torch.equal(torch.isfinite(score),valid)
   assert bool((den>=0).all()) and bool((score[valid].abs()<=1.00000001).all())
   totals['zero_denominator']+=int((den==0).sum());totals['zero_start_sign']+=int((sign==0).sum());totals['eligible']+=int(valid.sum())
  assert totals==v['excluded_totals'];results[f'seed{seed}_{order}']=totals
  print(seed,order,totals,flush=True)
path=HERE/'score_validation.json';path.write_text(json.dumps(results,indent=2)+'\n')
