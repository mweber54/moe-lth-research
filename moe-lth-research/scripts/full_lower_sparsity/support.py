"""Standard-library guards shared by preparation and the handoff."""
import csv, hashlib, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'results/dataset_b_pilot'
OUT = ROOT/'results/dataset_a_full_lower_sparsity'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as h: return hashlib.file_digest(h, 'sha256').hexdigest()
def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(value,indent=2)+'\n'); q.replace(p)
def rows(p):
    with Path(p).open() as h: return list(csv.DictReader(h))
def parent_complete(records):
    return (len(records)==225 and len({r['condition_id'] for r in records})==225
            and all(r['status'] in ('COMPLETE','REUSED') for r in records))
def expected_ids():
    return {f'{arm}_e32_seed{seed}_R{age}_sparse_{kind}_s{s:.2f}'
        for arm in 'ABCD' for seed in (7,17,29,43,61) for age in (0,50,100)
        for s in (.50,.55,.60,.65)
        for kind in (('learned','random') if arm in 'AD' and age in (0,100) else ('learned',))}
def verify_map(base, records):
    for rel,digest in records.items():
        if sha(base/rel)!=digest: raise RuntimeError('Hash mismatch: '+str(base/rel))
def validate_seal():
    seal=read(OUT/'extension_seal.json')
    verify_map(ROOT,seal['code']); verify_map(OUT,seal['immutable'])
    verify_map(Path('/'),seal['inputs'])
    return seal
