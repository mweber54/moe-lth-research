"""Read-only reuse of verified native continuation primitives."""
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('exposure_native_reference',ROOT/'scripts/magnitude_growth_pruning_v1/common.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
from af_common import *
from af_native import encode_numpy,decode_numpy
import copy,math,socket,threading,traceback,shlex,subprocess
from moe_lth.utils import create_grad_scaler,seed_everything
D=ROOT/'results/exposure_stratified_pruning_v1';OLD=ROOT/'results/gradient_sensitivity_pruning_v1'
SEEDS=(7,17,29,43,61);BRANCHES=('C','L','H');EVAL_STEPS=(0,25,100,250,500,1000,1500)
source=ref.source;variant=ref.variant;data=ref.data;atomic_tensor=ref.atomic_tensor
zero_=ref.zero_;hooks=ref.hooks;check_zero=ref.check_zero;evaluate=ref.evaluate;rng=ref.rng;set_rng=ref.set_rng

def registry():return read(D/'registration.json')
def names(state):return sorted(n for n in state if '.experts.' in n and n.endswith(('.fc1.weight','.fc2.weight')))
def selected(seed,branch,device='cpu'):
 if branch=='C':return {}
 p=D/'masks'/f'seed{seed}.pt';assert sha(p)==registry()['mask_files'][str(seed)]['sha256']
 return {n:t.to(device) for n,t in load(p)[branch].items()}
