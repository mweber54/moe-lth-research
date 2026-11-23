"""Verify repeated pilot accumulators, original aggregate and permutation exposure."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
stem='split_half_gradients_seed17_B_250_h'
def load(h,label,suffix=''):
 return torch.load(P/f'{stem}{h}_{label}{suffix}.pt',map_location='cpu',weights_only=False)
def same(a,b):
 for key in ('sign_sum','resolved_batches','mean_gradient_magnitude_sum'):
  for da,db in zip(a[key],b[key]):
   for x,y in zip(da,db): assert torch.equal(x,y),key
 for key in ('accepted_occurrences','contributing_batches'):
  assert np.array_equal(a[key],b[key]),key
 assert a['batch_ids']==b['batch_ids'] and a['model_unchanged'] and b['model_unchanged'] and a['rng_unchanged'] and b['rng_unchanged']
original=[load(h,'original') for h in (1,2)]
permuted=[load(h,'perm0') for h in (1,2)]
for h in (1,2):
 same(original[h-1],load(h,'original','_repeat'))
 same(permuted[h-1],load(h,'perm0','_repeat'))
old=torch.load(P/'gradients_seed17_B_250.pt',map_location='cpu',weights_only=False)
assert original[0]['batch_ids']+original[1]['batch_ids']==old['batch_ids']
for key in ('sign_sum','resolved_batches'):
 for d in range(2):
  for a,b,c in zip(original[0][key][d],original[1][key][d],old[key][d]):assert torch.equal(a+b,c),key
for key in ('accepted_occurrences','contributing_batches'):
 assert np.array_equal(original[0][key]+original[1][key],old[key]),key
exposure=[]
for h in range(2):
 a,b=original[h],permuted[h]
 assert np.array_equal(a['accepted_occurrences'].sum(axis=0),b['accepted_occurrences'].sum(axis=0)), 'total eligible expert exposure changed'
 exposure.append({'half':h+1,'original_accepted_occurrences':a['accepted_occurrences'].tolist(),'permuted_accepted_occurrences':b['accepted_occurrences'].tolist(),'original_contributing_batches':a['contributing_batches'].tolist(),'permuted_contributing_batches':b['contributing_batches'].tolist(),'max_absolute_occurrence_difference':int(np.max(np.abs(a['accepted_occurrences']-b['accepted_occurrences']))),'max_absolute_batch_difference':int(np.max(np.abs(a['contributing_batches']-b['contributing_batches'])))})
result={'cell':'seed17_B_250','bitwise_repeat_passed':True,'original_integer_accumulators_match_archived':True,'exposure':exposure,'exposure_exactly_matched':all(x['max_absolute_occurrence_difference']==0 and x['max_absolute_batch_difference']==0 for x in exposure),'model_and_rng_unchanged':True}
(P/'split_half_pilot_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
