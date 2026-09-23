import json,csv,hashlib
from pathlib import Path
import torch
from moe_lth.experiments.fixed_schedule_intervention import file_hash
root=Path('results/gate_amplitude_replay_intervention');prior=Path('results/cross_arch_optimization_geometry/fixed_schedule_intervention')
rows=[]
for e in [8,12]:
 for seed in [7,17,29]:
  p=prior/f'schedules/e{e}/seed{seed}';m=json.loads((p/'manifest.json').read_text());roll={'train':hashlib.sha256(),'validation':hashlib.sha256()};amp=hashlib.sha256();counts={'train':0,'validation':0};dtypes=set()
  for key,sha in m['entries'].items():
   f=p/key;assert file_hash(f)==sha
   x=torch.load(f,map_location='cpu',weights_only=False);assert x['input_hash'] and x['shape'];assert len(x['layers'])==4
   for layer in x['layers']:
    v=layer['capacity_scores'];assert v.shape==layer['ids'].shape and v.shape[-1]==1;assert v.numel()==layer['accepted'].numel();assert torch.isfinite(v).all() and ((v>=0)&(v<=1)).all()
    amp.update(v.numpy().tobytes());dtypes.add(str(v.dtype))
   phase=x['phase'];counts[phase]+=1;roll[phase].update(key.encode());roll[phase].update(sha.encode())
  assert counts==m['counts'];assert {k:v.hexdigest() for k,v in roll.items()}==m['sequence_hashes']
  rows.append(dict(experts=e,seed=seed,path=str(p.resolve()),training_occurrences=counts['train'],validation_occurrences=counts['validation'],gate_source='capacity_scores equals exact unnormalized selected branch coefficient',amplitude_hash=amp.hexdigest(),training_schedule_hash=m['sequence_hashes']['train'],validation_schedule_hash=m['sequence_hashes']['validation'],dtypes=';'.join(sorted(dtypes)),all_files_verified=True))
  print('PASS donor',e,seed,counts,flush=True)
with (root/'donor_amplitude_manifest.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print('All six amplitude archives complete and hash verified')
