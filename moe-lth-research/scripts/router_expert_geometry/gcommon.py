"""Isolated geometry experiment; historical packages are imported read-only."""
import sys, json, csv, hashlib, os, contextlib, random, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/nominal_exposure_dataset')]
import pilot_common as P
import torch
OUT=ROOT/'results/router_expert_geometry_v1'
SEEDS=(7,17,29,43,61); AGES=(0,50,100); SP=(.5,.65); STEPS=(0,25,100,250,500,1250,2500)
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_name(p.name+f'.{os.getpid()}.tmp'); q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n'); q.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def table(p,rs):
 rs=list(rs); p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rs:return
 q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
 q.replace(p)
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def grid():
 result=[]
 for seed in SEEDS:
  for age in AGES:
   dense=f'G_e32_seed{seed}_R{age}_dense'
   for kind,sp in [('dense',0.)]+[(kind,sp) for sp in SP for kind in ('learned','random')]:
    result.append(dict(condition_id=dense if kind=='dense' else f'G_e32_seed{seed}_R{age}_{kind}_s{sp:.2f}',seed=seed,router_age=age,source_id=f'source_e32_seed{seed}_T2500',job_type='dense' if kind=='dense' else 'sparse',mask_type=kind,sparsity=sp,dense_condition_id=dense,status='PENDING',host='',pid='',attempt='',heartbeat='',error=''))
 return result
@contextlib.contextmanager
def protected(model,strict_fp32=True):
 modes=[(m,m.training) for m in model.modules()];flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
 if strict_fp32:torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 with P.preserve_rng(),torch.no_grad():
  try:
   model.eval()
   with torch.autocast(device_type=next(model.parameters()).device.type,enabled=False):yield
  finally:
   for m,mode in modes:m.training=mode
   torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
