from __future__ import annotations
import hashlib, importlib.util, json, os, random, socket, sys, uuid
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
V2=OUT.with_name("imp_early_sign_reinforcement_neuron_matched_v2")
sys.path[:0]=[str(ROOT/"src"),str(V2)]
import torch, numpy as np
from common_v2 import pilot as p
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.router_age import component_state_dict, parameter_group, selected_experts_per_batch, state_dict_hash
N=p.N
SEEDS=(7,17,29); ORDERS=("A","B")
PROTOCOL_SHA="403caae38a808e39571bcd1526a48cf44e06801e57e239f18800789bb690e2fc"

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def save_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2,sort_keys=True,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)

def save_torch(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)

def entry(seed,order):
 archive=json.loads((V2/"inputs.json").read_text())
 return next(x for x in archive['inputs'] if x['seed']==seed and x['order']==order)

def material(seed,order):
 m=N._load_training_material(seed);config,batches,validation=p.ordered(seed,order,m);return config,batches,validation

def expert_names(state):
 return tuple(n for n,t in state.items() if '.moe.experts.' in n and n.endswith('.weight') and t.ndim==2)

def nested_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a.detach().cpu(),b.detach().cpu())
 if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
 return a==b

def admission():
 import subprocess,time
 q=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 rows=[]
 for line in q.stdout.splitlines():
  i,name,free,total,util=[x.strip() for x in line.split(',',4)];rows.append(dict(index=int(i),name=name,free_bytes=int(free)*1024**2,total_bytes=int(total)*1024**2,utilization=int(util)))
 ok=[r for r in rows if r['free_bytes']>=24*1024**3 and r['utilization']<=90]
 if not ok:raise RuntimeError(f'GPU admission failed without killing processes: {rows}')
 return dict(selected=max(ok,key=lambda r:r['free_bytes']),all=rows,host=socket.getfqdn(),timestamp=time.time(),minimum_free=24*1024**3,maximum_utilization=90)
