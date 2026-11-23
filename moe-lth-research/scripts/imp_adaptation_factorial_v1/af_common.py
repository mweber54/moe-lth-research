"""Isolated adaptation experiment, atomic shared-filesystem operations."""
import os,sys,json,csv,hashlib,time,uuid,fcntl,random,contextlib
from pathlib import Path
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from moe_lth.experiments import run_router_age_recovery as R
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.router_age import parameter_group,state_dict_hash,load_model_from_checkpoint,selected_experts_per_batch
from moe_lth.pruning.masks import load_masks,apply_masks_,expert_weight_parameters
OUT=ROOT/'results/imp_adaptation_factorial_v1';HIST=ROOT/'results/high_sparsity_coupling_v1'
SEEDS=(7,17,29,43,61);CONDITIONS=('full','expert_only','expert_router','expert_shared');STEPS=(0,100,2500,2600,5000)
TRAINABLE={'full':{'E','R','S'},'expert_only':{'E'},'expert_router':{'E','R'},'expert_shared':{'E','S'}}
HOSTS=('viper.cs.kent.edu','adder.cs.kent.edu')
def group(n):return {'expert':'E','router':'R','shared':'S'}[parameter_group(n)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
 with q.open('w') as f:json.dump(v,f,indent=2,allow_nan=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
 q.replace(p)
def table(p,rs,fields=None):
 rs=list(rs);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{uuid.uuid4().hex}.tmp');fields=fields or list(dict.fromkeys(k for r in rs for k in r))
 with q.open('w') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rs)
 q.replace(p)
def append(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a') as f:f.write(json.dumps(v,allow_nan=False,default=str)+'\n')
def lock(p,blocking=True):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));return f
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def masked(s,m):return {n:t.detach().cpu().clone()*m[n] if n in m else t.detach().cpu().clone() for n,t in s.items()}
def inventory(model,masks=None):
 masks=masks or {};unique=dict(model.named_parameters());aliases={}
 for n,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(n)
 if model.token_embedding.weight is not model.lm_head.weight:raise RuntimeError('Embedding tying broken')
 rows=[]
 for n,p in unique.items():
  names=aliases[id(p)];groups={group(a) for a in names}
  if len(groups)!=1:raise RuntimeError('Tied parameter crosses E/R/S')
  count=int(masks[n].sum()) if n in masks else p.numel()
  rows.append(dict(parameter=n,group=group(n),aliases=';'.join(names),parameter_count=p.numel(),eligible_gradient_coordinates=count,prunable=n in masks))
 if len({id(p) for p in unique.values()})!=len(rows):raise RuntimeError('Duplicate parameter counted')
 if sum(r['parameter_count'] for r in rows)!=sum(p.numel() for p in model.parameters()):raise RuntimeError('Nonexhaustive grouping')
 return rows
def configure(model,condition):
 for n,p in model.named_parameters():p.requires_grad_(group(n) in TRAINABLE[condition])
 inventory(model)
 return model
@contextlib.contextmanager
def protected(model=None):
 py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
 modes=[(m,m.training) for m in model.modules()] if model is not None else []
 grads={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()} if model is not None else {}
 try:yield
 finally:
  random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
  if cuda is not None:torch.cuda.set_rng_state_all(cuda)
  for m,flag in modes:m.training=flag
  if model is not None:
   for n,p in model.named_parameters():
    old=grads[n]
    if (p.grad is None)!=(old is None) or (old is not None and not torch.equal(p.grad.contiguous().view(torch.uint8),old.contiguous().view(torch.uint8))):raise RuntimeError('Diagnostic changed existing gradients')
def implementation():
 files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src/moe_lth').rglob('*.py'))+[ROOT/'scripts/imp_80_85/run.py',ROOT/'scripts/imp_extended_recovery/pipeline.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in files}
def frozen():
 s=read(OUT/'seal.json')
 if sha(OUT/'input_hashes.json')!=s['input_hashes_sha256']:raise RuntimeError('Frozen input inventory changed')
 if sha(OUT/'protocol.md')!=s['protocol_sha256'] or sha(OUT/'manifest.json')!=s['manifest_sha256']:raise RuntimeError('Frozen design changed')
 for n,h in s['implementation'].items():
  if sha(ROOT/n)!=h:raise RuntimeError('Sealed implementation changed '+n)
 return read(OUT/'manifest.json')
def status(j):return read(OUT/'status'/f"{j['id']}.json")
