"""Shared immutable scope and atomic filesystem operations for the bounded v2 run."""
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import uuid
from pathlib import Path

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/imp_mask_information')]
import torch
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.masks import load_masks
from moe_lth.pruning.imp import validate_masks
from moe_lth.pruning.router_age import parameter_group,state_dict_hash
OUT=ROOT/'results/imp_mask_information_v2'
OLD=ROOT/'results/imp_mask_information_v1'
SEEDS=(7,17,29,43,61)
HOSTS=('viper.cs.kent.edu','adder.cs.kent.edu')


def read(p): return json.loads(Path(p).read_text())


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    with tmp.open('w') as f:
        json.dump(d,f,indent=2,allow_nan=False,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def table(p,rs,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{uuid.uuid4().hex}.tmp')
    fields=fields or list(dict.fromkeys(k for r in rs for k in r))
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
    os.replace(tmp,p)


def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def lock(p,blocking=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB));return f


def canonical():
    # The old runner imports a module named common; isolate that legacy name.
    budget_spec=importlib.util.spec_from_file_location('canonical_budget_v2',ROOT/'scripts/fixed_budget_dataset/common.py')
    budget=importlib.util.module_from_spec(budget_spec);budget_spec.loader.exec_module(budget)
    previous=sys.modules.get('common');sys.modules['common']=budget
    try:
        spec=importlib.util.spec_from_file_location('canonical_imp_v2',ROOT/'scripts/imp_80_85/run.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        if previous is not None:sys.modules['common']=previous
        else:sys.modules.pop('common',None)
    return module


def initial_hashes(state,masks):
    masked={n:t.detach().cpu().clone()*masks[n] if n in masks else t.detach().cpu().clone() for n,t in state.items()}
    return {g:state_dict_hash({n:t for n,t in masked.items() if parameter_group(n)==g}) for g in ('expert','shared','router')}


def validate_coordinates(state,masks):
    for n,m in masks.items():
        expected=state[n].cpu()*m
        if not torch.equal(expected[m],state[n].cpu()[m]):raise ValueError('Initialization coordinates changed')
        if torch.count_nonzero(expected[~m]):raise ValueError('Masked initialization nonzero')
    return initial_hashes(state,masks)


def sources():
    files=list((ROOT/'scripts/imp_mask_information_v2').glob('*.py'))+list((ROOT/'scripts/imp_mask_information_v2').glob('*.cpp'))
    files += [ROOT/p for p in read(OLD/'pilot_plan.json')['source_hashes']]
    files += list((ROOT/'src/moe_lth').rglob('*.py'))+[ROOT/'scripts/fixed_budget_dataset/common.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}


def frozen():
    seal=read(OUT/'seal.json')
    if sha(OUT/'protocol.md')!=read(OUT/'registration.json')['protocol_sha256']:raise RuntimeError('Protocol changed')
    if sha(OUT/'manifest.json')!=seal['manifest_sha256']:raise RuntimeError('Manifest changed')
    for name,h in seal['sources'].items():
        if sha(ROOT/name)!=h:raise RuntimeError('Implementation changed: '+name)
    return read(OUT/'manifest.json')


def job_status(job):return read(OUT/'status'/f"{job['id']}.json")


def result(job):return read(Path(job_status(job)['result_path']))


def matrix_gate(effects):
    return set(effects)==set(SEEDS) and all(__import__('math').isfinite(v) and v>.02 for v in effects.values())
