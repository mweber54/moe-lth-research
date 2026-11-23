"""Deterministic frozen-checkpoint task-gradient pilot; no optimizer construction."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy, hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from moe_lth.models import TinyMoELanguageModel

def digest(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 meta=json.loads((ROOT/'results/moe_connection_lifetime_v1/prefixes/seed7_A/state250.json').read_text());cp=ROOT/'results/moe_connection_lifetime_v1/prefixes/seed7_A/state250.pt'
 h=hashlib.sha256()
 with cp.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 if h.hexdigest()!=meta['sha256']: raise RuntimeError('checkpoint hash mismatch')
 config=json.loads((ROOT/'results/imp_recovery_order_stability_v1/canonical_stream_manifest.json').read_text())['seeds']['7']['config']
 saved=torch.load(cp,map_location='cpu',mmap=True,weights_only=False); model=TinyMoELanguageModel(copy.deepcopy(config['model']));model.load_state_dict(saved['model'],strict=True);model.cuda();model.train()
 for p in model.parameters():p.requires_grad_(False)
 names=sorted(n for n,p in model.named_parameters() if n.startswith('blocks.3.moe.experts.') and n.endswith('.fc2.weight'))
 if len(names)!=32:raise RuntimeError('expert scope mismatch')
 params=dict(model.named_parameters());weights=[params[n] for n in names]
 for p in weights:p.requires_grad_(True)
 indices=torch.load(OUT/'indices_seed7_A.pt',weights_only=True)[:2]; alignment=json.loads((OUT/'batch_alignment.json').read_text())[0]
 if hashlib.sha256((OUT/'indices_seed7_A.pt').read_bytes()).hexdigest()!=alignment['indices_sha256']:raise RuntimeError('indices hash mismatch')
 data=np.frombuffer((ROOT/'data/multidomain_balanced/multidomain_train.txt').read_bytes(),dtype=np.uint8); labels=np.load(OUT/'byte_domain_labels.npy'); prov=json.loads((OUT/'provenance.json').read_text())
 if hashlib.sha256(data.tobytes()).hexdigest()!=prov['corpus']['training_sha256'] or hashlib.sha256((OUT/'byte_domain_labels.npy').read_bytes()).hexdigest()!=prov['domain_labels_sha256']:raise RuntimeError('data provenance mismatch')
 cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all();py_rng=random.getstate();numpy_rng=np.random.get_state()
 before={n:digest(p) for n,p in model.state_dict().items()}
 passes=[]
 for repeat in range(2):
  rows=[]
  for batch_id,ix in enumerate(indices):
   offsets=ix.numpy()[:,None]*128+np.arange(128); x=torch.tensor(data[offsets].astype(np.int64),device='cuda'); y=torch.tensor(data[offsets+1].astype(np.int64),device='cuda'); domain=torch.tensor(labels[offsets+1].astype(np.int64),device='cuda')
   out=model(x);losses=F.cross_entropy(out.logits.reshape(-1,256),y.reshape(-1),reduction='none').reshape_as(y);trace=out.routes[-1];selected=trace.selected_expert_indices.reshape_as(domain);accepted=trace.accepted_mask.reshape_as(domain)
   grads={};counts={}
   for d in (1,2):
    mask=domain==d;counts[str(d)]=[int(((selected==i)&accepted&mask).sum().item()) for i in range(32)]
    gradients=torch.autograd.grad(losses[mask].mean(),weights,retain_graph=d==1,allow_unused=True)
    grads[str(d)]={name:(g.detach().cpu() if g is not None else None) for name,g in zip(names,gradients)}
   rows.append({'batch':batch_id,'routes_sha256':digest(selected),'accepted_sha256':digest(accepted),'counts':counts,'gradients':grads})
   del out,losses,gradients,grads
  passes.append(rows)
 for a,b in zip(*passes):
  if a['routes_sha256']!=b['routes_sha256'] or a['accepted_sha256']!=b['accepted_sha256'] or a['counts']!=b['counts']:raise RuntimeError('route replay not deterministic')
  for d in ('1','2'):
   for n in names:
    x,y=a['gradients'][d][n],b['gradients'][d][n]
    if (x is None)!=(y is None) or (x is not None and not torch.equal(x,y)):raise RuntimeError('gradient replay not bitwise deterministic '+n)
 if any(digest(p)!=before[n] for n,p in model.state_dict().items()):raise RuntimeError('model mutated')
 if not torch.equal(cpu_rng,torch.get_rng_state()) or any(not torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all())) or py_rng!=random.getstate():raise RuntimeError('RNG mutated')
 z=np.random.get_state()
 if numpy_rng[0]!=z[0] or not np.array_equal(numpy_rng[1],z[1]) or numpy_rng[2:]!=z[2:]:raise RuntimeError('NumPy RNG mutated')
 result={'seed':7,'order':'A','step':250,'batch_ids':[0,1],'precision':'float32, TF32 disabled, deterministic algorithms enabled','checkpoint_sha256':meta['sha256'],'deterministic_bitwise':True,'model_unchanged':True,'rng_unchanged':True,'batches':passes[0]}
 torch.save(result,OUT/'gradient_pilot.pt');print('PILOT PASSED',flush=True)
if __name__=='__main__':main()
