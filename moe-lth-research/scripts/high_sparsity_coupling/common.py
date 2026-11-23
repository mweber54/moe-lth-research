"""Isolated exploratory IMP cohort; no historical outputs are mutated."""
import os, sys, json, csv, hashlib, time, contextlib, random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/fixed_budget_dataset'),str(ROOT/'scripts/router_expert_geometry'),str(ROOT/'scripts/ticket_geometry_endpoint')]
import numpy as np
import torch
from moe_lth.experiments import run_router_age_recovery as R
from moe_lth.pruning.router_age import state_dict_hash,parameter_group,component_state_dict,selected_experts_per_batch
from moe_lth.pruning.masks import load_masks
from moe_lth.models import TinyMoELanguageModel
from gcommon import protected
from numerics import cka
OUT=ROOT/'results/high_sparsity_coupling_v1'
SEEDS=(7,17,29,43,61)
STEPS=(0,25,100,250,500,1000,1500,2000,2500,2525,2600,3000,3500,4000,4500,5000)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def table(p,rows):
 rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r));q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('w') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
 q.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def state(p):return load(p)['model']
def masked(s,m):return {n:t*m[n] if n in m else t.clone() for n,t in s.items()}
def model(s,c,device='cuda'):
 with R.preserve_rng() if hasattr(R,'preserve_rng') else rng():m=TinyMoELanguageModel(c['model']).to(device)
 m.load_state_dict(s);m.eval();return m
@contextlib.contextmanager
def rng():
 a=random.getstate();b=np.random.get_state();c=torch.get_rng_state();d=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
 try:yield
 finally:
  random.setstate(a);np.random.set_state(b);torch.set_rng_state(c)
  if d is not None:torch.cuda.set_rng_state_all(d)
def verify_file(p,expected=None):
 p=Path(p);h=sha(p)
 if expected and h!=expected:raise RuntimeError(f'SHA mismatch {p}: {h} != {expected}')
 return dict(path=str(p),sha256=h,bytes=p.stat().st_size)
def frozen():
 m=read(OUT/'cohort.json');assert sha(OUT/'cohort.json')==read(OUT/'freeze.json')['cohort_sha256'];assert sha(OUT/'protocol.md')==read(OUT/'freeze.json')['protocol_sha256'];return m
