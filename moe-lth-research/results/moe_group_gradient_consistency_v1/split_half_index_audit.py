"""Audit batch selection using only recorded index artifacts and batch IDs."""
import hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
out=[]
for seed in (7,17,29):
 for order in ('A','B'):
  ip=P/f'indices_seed{seed}_{order}.pt';ix=torch.load(ip,map_location='cpu',weights_only=True)
  for step in (250,500):
   old=torch.load(P/f'gradients_seed{seed}_{order}_{step}.pt',map_location='cpu',weights_only=False); ids=old['batch_ids']; seen=set(); selected=[]
   for i,row in enumerate(ix[:step].tolist()):
    seq=set(row)
    if not seen.intersection(seq):selected.append(i);seen.update(seq)
   assert selected==ids and len(ids)>=180
   splits={}
   for name,chosen in [('first_180',ids[:180]),('all_retained',ids)]:
    mid=len(chosen)//2;a,b=chosen[:mid],chosen[mid:]
    left=set(ix[a].flatten().tolist());right=set(ix[b].flatten().tolist());assert not left&right
    assert a+b==chosen
    splits[name]={'batch_ids':[a,b],'counts':[len(a),len(b)],'sequence_index_overlap':len(left&right),'reconstructs_recorded_selection':True}
   out.append({'cell':f'seed{seed}_{order}_{step}','checkpoint_sha256':old['checkpoint_sha256'],'indices_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'retained_count':len(ids),'splits':splits})
(P/'split_half_index_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('index audit passed for',len(out),'cells; primary halves reconstruct first 180 and have no cross-half sequence overlap')
