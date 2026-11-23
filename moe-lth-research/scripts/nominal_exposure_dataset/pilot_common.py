"""Dataset B nominal-exposure pilot definitions. Dataset A is read-only."""
import sys,json,hashlib,os,random,contextlib
from pathlib import Path
from collections import Counter
import numpy as np,torch,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/fixed_budget_dataset')]
import common as A
from moe_lth.pruning.router_age import state_dict_hash,parameter_group
OUT=ROOT/'results/dataset_b_pilot';AD=ROOT/'results/final_dataset_a';SEEDS=(7,17,29,43,61);AGES=(0,50,100);SP=(.75,.95);ARMS={'A':(2500,2500),'B':(10000,2500),'C':(2500,10000),'D':(10000,10000)}
def atomic(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_bytes(b if isinstance(b,bytes) else b.encode());q.replace(p)
def jsave(p,x):atomic(p,json.dumps(x,indent=2,default=str)+'\n')
def csv(p,x):atomic(p,(x if isinstance(x,pd.DataFrame) else pd.DataFrame(x)).to_csv(index=False,na_rep='NA'))
def sha(p):return A.sha(p)
def read(p):return json.loads(Path(p).read_text())
def attempts(e):
 if e not in (8,12,16,24,32):raise ValueError('Architecture outside nominal schedule')
 return 2500*e//8
def router_step(budget,age):
 if age not in range(0,101,10):raise ValueError('Unexpected relative age')
 return budget*age//100
def jobid(arm,seed,age,kind,sp=0.,mask='dense'):return f'{arm}_e32_seed{seed}_R{age}_{kind}'+(f'_{mask}_s{sp:.2f}' if kind=='sparse' else '')
def grid():
 rows=[]
 for arm,(sb,rb) in ARMS.items():
  for seed in SEEDS:
   for age in AGES:
    dense=jobid(arm,seed,age,'dense')
    base=dict(arm=arm,num_experts=32,seed=seed,router_age=age,router_checkpoint_step=router_step(sb,age),source_attempts=sb,recovery_attempts=rb,source_id=f'source_e32_seed{seed}_T{sb}',source_mask_family=f'e32_seed{seed}_T{sb}')
    rows.append(dict(**base,condition_id=dense,job_type='dense',mask_type='dense',sparsity=0.,dense_condition_id=dense,stage={'A':1,'C':2,'B':4,'D':4}[arm]))
    for sp in SP:
     rows.append(dict(**base,condition_id=jobid(arm,seed,age,'sparse',sp,'learned'),job_type='sparse',mask_type='learned',sparsity=sp,dense_condition_id=dense,stage={'A':1,'C':2,'B':4,'D':4}[arm]))
     if arm in ('A','D') and age in (0,100):rows.append(dict(**base,condition_id=jobid(arm,seed,age,'sparse',sp,'random'),job_type='sparse',mask_type='random',sparsity=sp,dense_condition_id=dense,stage=1 if arm=='A' else 4))
 for seed in SEEDS:rows.append(dict(condition_id=f'source_e32_seed{seed}_T10000',job_type='source',num_experts=32,seed=seed,source_attempts=10000,stage=3))
 return rows
def mask_seed(source_id,sp):return int(hashlib.sha256(f'dataset-b-random-v1|{source_id}|{sp:.2f}'.encode()).hexdigest()[:16],16)%(2**63-1)
def random_mask(learned,source_id,sp):
 g=torch.Generator(device='cpu').manual_seed(mask_seed(source_id,sp));result={}
 for n,v in sorted(learned.items()):
  flat=torch.zeros(v.numel(),dtype=torch.bool);flat[torch.randperm(v.numel(),generator=g)[:int(v.sum())]]=True;result[n]=flat.reshape(v.shape)
 return result
def check_masks(learned,randomized):
 return set(learned)==set(randomized) and all(learned[n].shape==randomized[n].shape and int(learned[n].sum())==int(randomized[n].sum()) for n in learned)
@contextlib.contextmanager
def preserve_rng():
 py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
 try:yield
 finally:
  random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
  if cuda is not None:torch.cuda.set_rng_state_all(cuda)
def safe_attempt(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 for i in range(1,100000):
  p=root/f'attempt_{i:04d}'
  if not p.exists():return p
 raise RuntimeError('Too many attempts')
def reject_model_only_resume(payload):
 required={'model','optimizer','scheduler','scaler','rng','data_stream'}
 if not required.issubset(payload):raise ValueError('No exact continuation: optimizer/scaler/RNG/data-stream state incomplete; restart in a new attempt')
 # Even a complete payload must be supported by a separately tested resume engine.
 raise NotImplementedError('This pilot resumes the queue, not a partially trained trajectory; new attempts restart from specified initialization')
def exposure_at(path,steps,source=False):
 proposed=Counter();accepted=Counter();active=Counter();applied_active=Counter();updates=0;available=True;rows=[];seen=[]
 for line in Path(path).open():
  r=json.loads(line);step=r['step']
  if step==0:continue
  seen.append(step);p=r.get('proposed' if source else 'expert_token_counts');a=r.get('accepted' if source else 'accepted_expert_token_counts')
  if p is None or a is None:raise ValueError('Missing raw exposure counters')
  proposed.update(p);accepted.update(a);active.update({k:int(v>0) for k,v in a.items()});flag=r.get('optimizer_step_applied');available &= flag is not None
  if flag:updates+=1;applied_active.update({k:int(v>0) for k,v in a.items()})
  if step in steps:
   if seen!=list(range(1,step+1)):raise ValueError('Non-contiguous checkpoint exposure')
   vals=np.array([accepted[k] for k in sorted(proposed)],float);rows.append(dict(step=step,attempted_updates=step,applied_updates=updates if available else None,amp_skipped_updates=step-updates if available else None,proposed=dict(proposed),accepted=dict(accepted),dropped={k:proposed[k]-accepted[k] for k in proposed},active_attempt_steps=dict(active),active_applied_steps=dict(applied_active) if available else None,mean_accepted_per_layer_expert=float(vals.mean()),accepted_load_cv=float(vals.std()/vals.mean()) if vals.mean() else 0.))
 if 0 in steps:
  zero={k:0 for k in proposed};rows.insert(0,dict(step=0,attempted_updates=0,applied_updates=0,amp_skipped_updates=0,proposed=zero,accepted=zero.copy(),dropped=zero.copy(),active_attempt_steps=zero.copy(),active_applied_steps=zero.copy(),mean_accepted_per_layer_expert=0.,accepted_load_cv=None))
 if not set(steps)<={r['step'] for r in rows}:raise ValueError('Missing checkpoint exposure; never substitute whole-run totals')
 return rows
