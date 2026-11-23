"""CPU validation of pilot-cell indexing and signed accumulator arithmetic."""
import hashlib, json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
r=torch.load(P/'gradients_seed17_B_250.pt',map_location='cpu',weights_only=False)
idx=torch.load(P/'indices_seed17_B.pt',map_location='cpu',weights_only=True)
ids=r['batch_ids']
assert len(ids)==180 and ids==sorted(ids) and len(set(ids))==180
halves=[ids[:90],ids[90:]]
a,b=(set(idx[h].flatten().tolist()) for h in halves)
assert not a.intersection(b)
# Exercise per-weight resolution, signed sums, and magnitudes, including zero and nonfinite values.
g=torch.tensor([[1.,-2.,0.,1e-13,float('nan')],[-3.,-4.,2.,-1e-13,1.]])
ok=torch.isfinite(g)&(g.abs()>1e-12)
s=torch.where(ok,g.sign(),0).to(torch.int16).sum(0)
c=ok.to(torch.int16).sum(0)
m=torch.where(ok,g.abs(),0).sum(0)
assert s.tolist()==[0,-2,1,0,1] and c.tolist()==[2,2,1,0,1] and m.tolist()==[4.,6.,2.,0.,1.]
result={'cell':'seed17_B_250','batch_ids':halves,'half_sizes':[len(h) for h in halves], 'sequence_index_overlap':len(a&b),'synthetic_accumulator_check':'passed','actual_gradient_accumulator_check':'not performed','checkpoint_sha256':r['checkpoint_sha256'],'indices_sha256':hashlib.sha256((P/'indices_seed17_B.pt').read_bytes()).hexdigest()}
(P/'split_half_cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('pilot indexing and synthetic accumulator arithmetic passed; actual gradients not checked')
