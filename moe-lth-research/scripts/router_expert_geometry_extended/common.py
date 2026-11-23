"""Inference-only, post hoc extension. Never import a training entry point."""
import sys, json, hashlib, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/router_expert_geometry')]
from gcommon import protected
BASE = ROOT/'results/router_expert_geometry_v1'
OUT = BASE/'analysis_extended/v1'
SEEDS = (7,17,29,43,61)
STEPS = (0,25,100,250,500,1250,2500)
CONFIG = dict(version=1, minimum_samples=8, variance_floor=1e-12,
              counterfactual_tokens=128, normalization='within layer/expert and selection, population z-score',
              correlation='Pearson primary, Spearman sensitivity', shuffle='all 31 nonidentity cyclic expert pairings',
              selection='existing 8 sequences/1024 positions, unchanged order for capacity',
              gram_dtype='float64', ranks=[8,16,32], primary_rank=16, post_hoc=True)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def frame(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    (d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)).to_csv(q,index=False);q.replace(p)
def code_hash():
    paths=[ROOT/'scripts/router_expert_geometry_extended'/n for n in ('common.py','measurements.py','run.py')]
    paths += sorted((ROOT/'src/moe_lth/models').glob('*.py'))
    paths += [ROOT/'scripts/router_expert_geometry/gcommon.py', ROOT/'scripts/router_expert_geometry/metrics.py']
    return hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in paths},sort_keys=True).encode()).hexdigest()
def tensor_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        x=t.detach().cpu().contiguous();h.update(n.encode());h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()
def bootstrap(v):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    if not len(v): return (None,None,None)
    g=np.random.default_rng(8103)
    lo,hi=np.quantile(g.choice(v,(10000,len(v)),replace=True).mean(1),[.025,.975])
    return float(v.mean()),float(lo),float(hi)
