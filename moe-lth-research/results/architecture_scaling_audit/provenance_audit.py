from pathlib import Path
import json,hashlib,csv,torch,sys
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(R/'src'))
from moe_lth.pruning.router_age import state_dict_hash
rows=[]
for s in (7,17,29):
 for step in (0,500,2500):
  a=R/f'results/expert_count_replication/e12/reference/seed_{s}/checkpoints/step_{step}.pt';b=R/f'results/router_development_12expert_retry4_20260902/references/reference/seed_{s}/checkpoints/step_{step}.pt'
  aa=torch.load(a,map_location='cpu',weights_only=False);bb=torch.load(b,map_location='cpu',weights_only=False)
  rows.append(dict(seed=s,step=step,original=str(a),expanded=str(b),model_hash_original=state_dict_hash(aa['model']),model_hash_expanded=state_dict_hash(bb['model']),states_equal=state_dict_hash(aa['model'])==state_dict_hash(bb['model'])))
with (O/'12e_reference_provenance.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
files=[]
for p in (R/'src').rglob('*.py'):
 files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
for p in (R/'scripts').glob('*.py'):
 files.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(O/'code_hashes.json').write_text(json.dumps(files,indent=2))
print(rows)
