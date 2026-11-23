"""Independently verify saved pairs against raw weights, scores and audits."""
import hashlib,json,math,os,time,uuid
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;V1=HERE.parent/'moe_online_reinforcement_pruning_v1'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 report={'created_unix':time.time(),'cells':{}}
 manifest=json.loads((HERE/'input_manifest.json').read_text())
 for seed in (7,17,29):
  for order in ('A','B'):
   cell=f'seed{seed}_{order}';folder=HERE/'matching'/cell;file=folder/'selections.pt';audit=json.loads((folder/'audit.json').read_text())
   for artifact in ('checkpoint','scores','selections','matrix_audit'):
    item=manifest['cells'][cell]['files'][artifact]
    if sha(Path(item['path']))!=item['sha256']:raise RuntimeError(f'input hash mismatch: {cell} {artifact}')
   assert audit['complete'] and audit['energy_pass'] and audit['matrices_audited']==256
   selections=torch.load(file,map_location='cpu',weights_only=True);p=V1/'prefixes'/cell
   state=torch.load(p/'native/interruption.pt',map_location='cpu',weights_only=False)['model'];scores=torch.load(p/'scores.pt',map_location='cpu',weights_only=True)['scores']
   assert set(selections)=={'random','high'} and set(selections['random'])==set(selections['high'])==set(scores)
   audits={a['name']:a for a in audit['matrices']}
   archived=torch.load(V1/'matching'/cell/'selections.pt',map_location='cpu',weights_only=True)
   archived_audit={a['tensor']:a for a in json.loads((V1/'matching'/cell/'matrix_audit.json').read_text())}
   count=0;overlap=0;score_sum=[0.,0.];maximum_energy_gap=0.
   for name in sorted(scores):
    w=state[name].numpy().ravel().astype(np.float64);nr,nc=state[name].shape;sc=scores[name]['score'].numpy().ravel();sgn=scores[name]['initial_sign'].numpy().ravel();den=scores[name]['absolute_update'].numpy().ravel()
    rank=np.lexsort((np.arange(len(w)),np.abs(w)));pool=rank[:len(w)//4];pool=pool[(sgn[pool]!=0)&(den[pool]>0)];bins=np.array_split(pool,5)
    r=selections['random'][name].numpy();h=selections['high'][name].numpy()
    for idx in (r,h):
     assert len(idx)==13107 and len(np.unique(idx))==len(idx) and np.isin(idx,pool).all()
    for b in bins:
     rb=r[np.isin(r,b)];hb=h[np.isin(h,b)]
     assert np.array_equal(np.bincount(rb//nc,minlength=nr),np.bincount(hb//nc,minlength=nr))
     assert np.array_equal(np.bincount(rb%nc,minlength=nc),np.bincount(hb%nc,minlength=nc))
    er=float(np.square(w[r]).sum());eh=float(np.square(w[h]).sum());assert abs(eh-er)<=.05*er if er else eh==er
    maximum_energy_gap=max(maximum_energy_gap,abs(er-eh)/er if er else 0.)
    a=audits[name];assert abs(float(sc[h].sum())-a['objective'])<=1e-5
    # The archived MILP is a separately executed optimum for the same five-bin
    # degree constraints; independently check its primal and objective too.
    ah=archived['high'][name].numpy(); aa=archived_audit[name]
    assert aa['solver']['magnitude_strata']==5 and aa['solver']['mip_gap']==0.0
    assert len(ah)==13107 and len(np.unique(ah))==13107 and np.isin(ah,pool).all()
    assert abs(float(sc[h].sum())-float(sc[ah].sum()))<=1e-5
    for b in bins:
     rb=r[np.isin(r,b)];hb=ah[np.isin(ah,b)]
     assert np.array_equal(np.bincount(rb//nc,minlength=nr),np.bincount(hb//nc,minlength=nr)) and np.array_equal(np.bincount(rb%nc,minlength=nc),np.bincount(hb%nc,minlength=nc))
    assert all(abs(c['min_objective']-c['dual_min_objective'])<=1e-5 for c in a['certificates'])
    overlap+=len(np.intersect1d(r,h,assume_unique=True));score_sum[0]+=float(sc[r].sum());score_sum[1]+=float(sc[h].sum());count+=len(r)
   sep=(score_sum[1]-score_sum[0])/count;over=overlap/count;assert count==3355392 and sep>=.2 and over<=.3
   report['cells'][cell]={'selection_sha256':sha(file),'audit_sha256':sha(folder/'audit.json'),'deletions':count,'overlap_fraction':over,'score_separation':sep,'max_relative_energy_gap':maximum_energy_gap}
   print(cell,report['cells'][cell],flush=True)
 tmp=HERE/('.sealed.'+uuid.uuid4().hex);tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,HERE/'sealed_inputs.json')
if __name__=='__main__':main()
