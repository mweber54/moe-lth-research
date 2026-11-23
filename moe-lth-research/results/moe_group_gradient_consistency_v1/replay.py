"""Frozen historical task gradients. No optimizer, updates, or pruning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from moe_lth.models import TinyMoELanguageModel

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def main(seed,order,step):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 protocol=json.loads((OUT/'protocol.json').read_text());provenance=json.loads((OUT/'provenance.json').read_text());cell=f'seed{seed}_{order}'
 native=json.loads((ROOT/'results/moe_connection_lifetime_v1/native_provenance.json').read_text())
 for entry in native['model'].values():
  if sha(Path(entry['path']))!=entry['loaded_sha256']:raise RuntimeError('native model source mismatch: '+entry['path'])
 if not (OUT/'gradient_pilot.pt').exists() or not torch.load(OUT/'gradient_pilot.pt',weights_only=False)['deterministic_bitwise']:raise RuntimeError('pilot not passed')
 cp=ROOT/'results/moe_connection_lifetime_v1/prefixes'/cell/f'state{step}.pt'
 if not cp.exists() and step==250: cp=ROOT/'results/moe_online_reinforcement_pruning_v1/prefixes'/cell/'native/interruption.pt'
 if sha(cp)!=provenance['cells'][f'{cell}_{step}']['state_sha256']:raise RuntimeError('checkpoint mismatch')
 config=json.loads((ROOT/'results/imp_recovery_order_stability_v1/canonical_stream_manifest.json').read_text())['seeds'][str(seed)]['config'];saved=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
 cpu_rng=torch.get_rng_state().clone(); cuda_rng=torch.cuda.get_rng_state_all();py_rng=random.getstate();np_rng=np.random.get_state()
 model=TinyMoELanguageModel(copy.deepcopy(config['model']));model.load_state_dict(saved['model'],strict=True);model.cuda();model.train();torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
 for p in model.parameters():p.requires_grad_(False)
 params=dict(model.named_parameters());names=sorted((n for n in params if n.startswith('blocks.3.moe.experts.') and n.endswith('.fc2.weight')),key=lambda n:int(n.split('.')[4]));weights=[params[n] for n in names]
 if len(weights)!=32:raise RuntimeError('scope mismatch')
 for p in weights:p.requires_grad_(True)
 before={n:tensor_hash(p) for n,p in model.state_dict().items()}
 ip=OUT/f'indices_{cell}.pt';rec=next(x for x in provenance['batch_alignment'] if x['seed']==seed and x['order']==order)
 if sha(ip)!=rec['indices_sha256']:raise RuntimeError('batch index mismatch')
 idx=torch.load(ip,weights_only=True)[:step];seen=set();selected=[]
 for i,row in enumerate(idx.tolist()):
  s=set(row)
  if not seen.intersection(s):seen.update(s);selected.append(i)
 data_path=ROOT/'data/multidomain_balanced/multidomain_train.txt';label_path=OUT/'byte_domain_labels.npy'
 if sha(data_path)!=provenance['corpus']['training_sha256'] or sha(label_path)!=provenance['domain_labels_sha256']:raise RuntimeError('corpus/label mismatch')
 data=np.frombuffer(data_path.read_bytes(),dtype=np.uint8);labels=np.load(label_path)
 sums=[[torch.zeros_like(p,dtype=torch.int16) for p in weights] for _ in range(2)];resolved=[[torch.zeros_like(p,dtype=torch.int16) for p in weights] for _ in range(2)];magnitude=[[torch.zeros_like(p,dtype=torch.float32) for p in weights] for _ in range(2)]
 occurrences=np.zeros((2,32),dtype=np.int64);contrib=np.zeros((2,32),dtype=np.int64);start=time.monotonic()
 for position,batch_id in enumerate(selected):
  offsets=idx[batch_id].numpy()[:,None]*128+np.arange(128);x=torch.tensor(data[offsets].astype(np.int64),device='cuda');y=torch.tensor(data[offsets+1].astype(np.int64),device='cuda');domain=torch.tensor(labels[offsets+1].astype(np.int64),device='cuda')
  out=model(x); losses=F.cross_entropy(out.logits.reshape(-1,256),y.reshape(-1),reduction='none').reshape_as(y);trace=out.routes[-1];route=trace.selected_expert_indices.reshape_as(domain);accepted=trace.accepted_mask.reshape_as(domain)
  for d in (1,2):
   mask=domain==d;ntarget=int(mask.sum());counts=torch.bincount(route[mask&accepted],minlength=32).cpu().numpy();occurrences[d-1]+=counts;contrib[d-1]+=(counts>0)
   if not ntarget:continue
   gradients=torch.autograd.grad(losses[mask].mean(),weights,retain_graph=d==1,allow_unused=True)
   for j,g in enumerate(gradients):
    if counts[j] and g is not None:
     value=g.detach()*(ntarget/int(counts[j])); ok=torch.isfinite(value)&(value.abs()>1e-12)
     sums[d-1][j].add_(torch.where(ok,value.sign(),0).to(torch.int16));resolved[d-1][j].add_(ok.to(torch.int16));magnitude[d-1][j].add_(torch.where(ok,value.abs(),0))
  if position%10==0 or position+1==len(selected):print(cell,step,position+1,'/',len(selected),'elapsed',round(time.monotonic()-start,1),flush=True)
  del out,losses,gradients
 if any(tensor_hash(p)!=before[n] for n,p in model.state_dict().items()):raise RuntimeError('model mutated')
 if not torch.equal(cpu_rng,torch.get_rng_state()) or any(not torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all())) or py_rng!=random.getstate():raise RuntimeError('RNG mutated')
 now=np.random.get_state()
 if np_rng[0]!=now[0] or not np.array_equal(np_rng[1],now[1]) or np_rng[2:]!=now[2:]:raise RuntimeError('NumPy RNG mutated')
 result={'seed':seed,'order':order,'step':step,'protocol_sha256':sha(OUT/'protocol.json'),'checkpoint_sha256':sha(cp),'batch_ids':selected,'accepted_occurrences':occurrences,'contributing_batches':contrib,'names':names,'sign_sum':[[x.cpu() for x in a] for a in sums],'resolved_batches':[[x.cpu() for x in a] for a in resolved],'mean_gradient_magnitude_sum':[[x.cpu() for x in a] for a in magnitude],'model_unchanged':True,'rng_unchanged':True,'elapsed_seconds':time.monotonic()-start}
 dest=OUT/f'gradients_{cell}_{step}.pt';tmp=dest.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(dest);print('SAVED',dest,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('seed',type=int,choices=(7,17,29));ap.add_argument('order',choices=('A','B'));ap.add_argument('step',type=int,choices=(250,500));args=ap.parse_args();main(args.seed,args.order,args.step)
