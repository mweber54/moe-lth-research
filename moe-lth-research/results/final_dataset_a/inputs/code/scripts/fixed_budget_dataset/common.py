"""Canonical Dataset A definitions. No exposure matching or mechanism interventions."""
import sys,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'results/router_conditioned_fixed_budget_dataset'
ES=(8,12,16,24,32);SEEDS=(7,17,29,43,61);AGES=tuple(range(0,101,10));SP=(.70,.75,.80,.85,.90,.95)
TITLE='Router-Conditioned Sparse Recoverability Under Equal Global Training/Recovery Budgets'
import torch
from moe_lth.config import load_config
from moe_lth.pruning.router_age import state_dict_hash,parameter_group

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def jread(p):return json.loads(Path(p).read_text())
def jwrite(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,default=str)+'\n');tmp.replace(p)
def writecsv(p,rows):
 import pandas as pd
 f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.csv.tmp');f.to_csv(tmp,index=False,na_rep='NA');tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def group(s,g):return {n:v for n,v in s.items() if parameter_group(n)==g}
def config(e,seed,dest=None):
 c=load_config(ROOT/'configs/expert_count_replication/e12_seed7.yaml');c['seed']=seed;c['model']['num_experts']=e;c['training']['checkpoint_steps']=[0,500,2500];c['training']['checkpoint_only_steps']=list(range(250,2500,250));c['training']['record_exposure']=True;c['output_dir']=str(dest or OUT/f'sources/e{e}/seed_{seed}');return c

def protocol_errors(c,e,seed):
 target=config(e,seed);errors=[]
 for groupname,keys in {'model':['vocab_size','max_seq_len','num_layers','num_heads','d_model','num_experts','expert_hidden_size','dropout','top_k','capacity_factor'],'training':['steps','batch_size','learning_rate','weight_decay','grad_clip','precision','eval_interval'],'routing':['mode','aux_loss_weight'],'data':['seq_len','validation_blocks','tokenizer','reshuffle_each_epoch']}.items():
  for k in keys:
   actual=c.get(groupname,{}).get(k,target[groupname].get(k))
   if actual!=target[groupname].get(k):errors.append(f'{groupname}.{k}={actual}')
 if c.get('seed')!=seed:errors.append('seed')
 for k in ('train_path','validation_path'):
  p=Path(c.get('data',{}).get(k) or '')
  if not p.is_absolute():p=ROOT/p
  expected=ROOT/target['data'][k]
  if p.resolve()!=expected.resolve():errors.append('data.'+k)
 if c.get('training',{}).get('data_seed') not in (None,seed):errors.append('data_seed')
 return errors

def primary_ref(e,seed):
 if seed not in (7,17,29):return None
 if e==8:return ROOT/f'results/expert_count_replication/e8/reference/seed_{seed}'
 if e==12:return ROOT/f'results/router_development_12expert_retry4_20260902/references/reference/seed_{seed}'
 return ROOT/f'results/{e}e_router_age_localization/references/seed_{seed}'
def cid(kind,e,s,a=None,sp=None):return f'{kind}_e{e}_seed{s}'+(f'_R{a}' if a is not None else '')+(f'_s{sp:.2f}' if sp else '')
