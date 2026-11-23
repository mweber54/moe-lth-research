"""Shared paths, state resolution and I/O for the checkpoint-only functional geometry extension.

All inputs are read-only. Outputs go only under OUT. No training is performed here.
"""
import os, sys, json, csv, hashlib, glob, time, contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src')]
import numpy as np
import torch

HSC = ROOT / 'results/high_sparsity_coupling_v1'
OUT = HSC / 'functional_geometry_extension_v1'
SEEDS = (7, 17, 29, 43, 61)
SPARSITIES = (0.80, 0.85)
STEPS = (0, 25, 100, 250, 500, 1000, 1500, 2000, 2500, 2525, 2600, 3000, 3500, 4000, 4500, 5000)
RESET_STEP = 2500
SCRIPT_DIR = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(v, indent=2, allow_nan=False, default=str) + '\n')
    q.replace(p)


def table(p, rows):
    rows = list(rows)
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    q = p.with_name(p.name + f'.{os.getpid()}.tmp')
    with q.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    q.replace(p)


def implementation_hash(*names):
    h = hashlib.sha256()
    for n in names:
        h.update((SCRIPT_DIR / n).read_bytes())
    return h.hexdigest()


_COHORT = None


def cohort():
    global _COHORT
    if _COHORT is None:
        _COHORT = read(HSC / 'cohort.json')
    return _COHORT


def condition(cid):
    return next(c for c in cohort()['conditions'] if c['id'] == cid)


def cond_id(seed, sparsity=None):
    return f'e32_seed{seed}_dense' if sparsity is None else f'e32_seed{seed}_final_s{sparsity:.2f}'


def attempt_dir(cid):
    a = sorted(glob.glob(str(HSC / 'reruns' / cid / 'attempt_*')))
    done = [x for x in a if (Path(x) / 'segment_2500/checkpoints/final_recovered.pt').exists()]
    if len(done) != 1:
        raise RuntimeError(f'{cid}: expected one completed attempt, found {done}')
    return Path(done[0])


def snapshot_path(cid, step, after_reset=False):
    name = f'step_{step:04d}' + ('_after_reset' if after_reset else '')
    return attempt_dir(cid) / 'snapshots' / (name + '.pt')


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def mask_for(seed, sparsity):
    k = 8 if abs(sparsity - 0.80) < 1e-9 else 9
    return ROOT / f'results/imp_80_85/trajectories/e32/seed_{seed}/mask_{k:02d}.pt'


def selection_state_path(seed, sparsity):
    """Direct trained IMP state C that selected the mask (round k-1 final_recovered)."""
    k = 8 if abs(sparsity - 0.80) < 1e-9 else 9
    return ROOT / f'results/imp_80_85/trajectories/e32/seed_{seed}/round_{k - 1:02d}/checkpoints/final_recovered.pt'


def initial_path(seed):
    return Path(condition(cond_id(seed))['initial'])


def source_final_path(seed):
    return Path(condition(cond_id(seed))['source_reference'])


def build_model(state, seed, device='cuda'):
    from moe_lth.models import TinyMoELanguageModel
    cfg = condition(cond_id(seed))['config']
    m = TinyMoELanguageModel(cfg['model']).to(device)
    m.load_state_dict(state)
    m.eval()
    return m


def probe(seed, device='cuda'):
    p = load(HSC / f'probes/seed_{seed}.pt')
    return dict(tokens=p['tokens'].to(device), targets=p['targets'].to(device), positions=p['positions'].to(device),
                source_full_inputs=[x.to(device) for x in p['source_full_inputs']],
                validation_hash=p.get('validation_hash'), batch_index=p.get('batch_index'))


def configure_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextlib.contextmanager
def heartbeat(path, **info):
    path = Path(path)
    save(path, dict(status='RUNNING', unix=time.time(), pid=os.getpid(), **info))
    try:
        yield lambda **kw: save(path, dict(status='RUNNING', unix=time.time(), pid=os.getpid(), **info, **kw))
    except BaseException as e:
        save(path, dict(status='FAILED', unix=time.time(), pid=os.getpid(), error=repr(e), **info))
        raise
    else:
        save(path, dict(status='COMPLETE', unix=time.time(), pid=os.getpid(), **info))
