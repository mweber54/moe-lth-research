"""Construct byte-source labels only after exact corpus verification."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
m=json.loads((ROOT/'data/multidomain_balanced/metadata.json').read_text())['train']; n=m['chunk_chars']; tiny=(ROOT/m['tiny_path']).read_text(); wiki=(ROOT/m['wiki_path']).read_text(); target=min(len(tiny),len(wiki))
def sample(s):
 if len(s)<=target:return s
 k=max(1,target//n); mx=len(s)-n
 return ''.join(s[round(i*mx/max(1,k-1)):round(i*mx/max(1,k-1))+n] for i in range(k))[:k*n]
tiny,wiki=sample(tiny),sample(wiki)
if (len(tiny),len(wiki))!=(m['tiny_chars'],m['wiki_chars']):raise RuntimeError('source lengths mismatch')
pieces=[(j,s[i:i+n].encode()) for i in range(0,max(len(tiny),len(wiki)),n) for j,s in ((1,tiny),(2,wiki)) if i<len(s)]
data=b'\n\n'.join(x for _,x in pieces); expected=(ROOT/m['output_path']).read_bytes()
if hashlib.sha256(data).digest()!=hashlib.sha256(expected).digest():raise RuntimeError('corpus mismatch')
labels=np.concatenate([np.concatenate(([0,0] if i else [],np.full(len(chunk),label,dtype=np.uint8))).astype(np.uint8) for i,(label,chunk) in enumerate(pieces)])
if len(labels)!=len(data):raise RuntimeError('label length mismatch')
np.save(OUT/'byte_domain_labels.npy',labels,allow_pickle=False)
records=[]; seq=128
for seed in (7,17,29):
 for order in ('A','B'):
  indices=torch.load(OUT/f'indices_seed{seed}_{order}.pt',weights_only=True).numpy()
  for step in (250,500):
   starts=indices[:step].reshape(-1,1)*seq+np.arange(seq); inputs=labels[starts]; targets=labels[starts+1]
   for name,value in (('unknown',0),('tinystories',1),('wikitext103',2)):
    mask=targets==value
    # Every input position at or before the prediction is part of its causal context.
    crossing=(np.cumsum(inputs!=value,axis=1)>0)&mask if value else np.zeros_like(mask)
    records.append(dict(seed=seed,order=order,step=step,domain=name,target_positions=int(mask.sum()),context_crossing_positions=int(crossing.sum()) if value else '',scope='historical target positions; not expert or gradient coverage'))
with (OUT/'group_coverage.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
