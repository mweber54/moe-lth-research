"""Compare replay's CPU accumulators on two batches against independent one-batch runs."""
import os, sys, tempfile
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
src=(P/'replay.py').read_text()
src=src.replace("model.cuda();model.train()", "model.cpu();model.train()")
src=src.replace("device='cuda'", "device='cpu'")
src=src.replace("data_path=ROOT/'data/multidomain_balanced/multidomain_train.txt'", "selected = selected[int(os.environ['SPLIT_START']):int(os.environ['SPLIT_END'])]\n data_path=ROOT/'data/multidomain_balanced/multidomain_train.txt'")
src=src.replace("dest=OUT/f'gradients_{cell}_{step}.pt'", "dest=Path(os.environ['SPLIT_DEST'])")
ns={'__file__':str(P/'replay.py'),'__name__':'split_cpu_replay'}
exec(compile(src,str(P/'replay.py'),'exec'),ns)
with tempfile.TemporaryDirectory() as tmp:
 results=[]
 for start,end in [(0,1),(1,2),(0,2)]:
  dest=Path(tmp)/f'{start}_{end}.pt'
  os.environ.update(SPLIT_START=str(start),SPLIT_END=str(end),SPLIT_DEST=str(dest))
  ns['main'](17,'B',250)
  results.append(torch.load(dest,map_location='cpu',weights_only=False))
 a,b,ab=results
 for field in ('sign_sum','resolved_batches','mean_gradient_magnitude_sum'):
  for d in range(2):
   for x,y,z in zip(a[field][d],b[field][d],ab[field][d]):
    assert torch.equal(x+y,z),field
 for field in ('accepted_occurrences','contributing_batches'):
  assert ((a[field]+b[field])==ab[field]).all(),field
 print('CPU actual-gradient accumulator check passed',flush=True)
