"""Isolated gradient measurement; prior experiment helpers are read-only dependencies."""
import importlib.util
import math
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('growth_reference_common', ROOT/'scripts/magnitude_growth_pruning_v1/common.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
from af_common import *
from af_native import encode_numpy
D = ROOT/'results/gradient_sensitivity_pruning_v1'
SEEDS = (7,17,29,43,61)

def atomic_tensor(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(encode_numpy(value),tmp); tmp.replace(path)

def tensor_hash(t):
    a=t.detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

def batch_hash(batch):
    return R._batch_sequence_hash([batch])

def names(state):
    return sorted(n for n,t in state.items() if '.experts.' in n and n.endswith(('.fc1.weight','.fc2.weight')))

def metadata(seed,name):
    p=name.split('.')
    return dict(seed=seed,parameter=name,layer=int(p[1]),expert=int(p[4]),projection=p[5])

def frozen():
    seal=read(D/'analysis_seal.json')
    amendment_path=D/'pre_scoring_code_amendment.json'
    amendment=read(amendment_path) if amendment_path.exists() else None
    if amendment:
        assert sha(D/'analysis_seal.json')==amendment['original_analysis_seal_sha256']
        for p,h in amendment['files'].items(): assert sha(ROOT/p)==h, 'Amended file changed: '+p
    for p,h in seal['files'].items():
        if amendment and p in amendment['files']: continue
        assert sha(ROOT/p)==h, 'Sealed file changed: '+p
    return read(D/'registration.json')
