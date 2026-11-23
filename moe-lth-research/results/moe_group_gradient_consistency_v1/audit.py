"""CPU-only provenance and descriptive baseline; never modifies the archive."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
OLD=ROOT/'results/moe_connection_lifetime_v1'
sys.path.insert(0,str(OLD))
from groups import scope

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(8<<20),b''): h.update(block)
 return h.hexdigest()

def corpus():
 p=ROOT/'data/multidomain_balanced/metadata.json'; m=json.loads(p.read_text())['train']; n=m['chunk_chars']
 tiny=(ROOT/m['tiny_path']).read_text(encoding='utf-8'); wiki=(ROOT/m['wiki_path']).read_text(encoding='utf-8'); target=min(len(tiny),len(wiki))
 def sample(s):
  if target>=len(s): return s
  count=max(1,target//n); maximum=len(s)-n
  return ''.join(s[round(i*maximum/max(1,count-1)):round(i*maximum/max(1,count-1))+n] for i in range(count))[:count*n]
 tiny,wiki=sample(tiny),sample(wiki)
 if (len(tiny),len(wiki))!=(m['tiny_chars'],m['wiki_chars']): raise RuntimeError('source length mismatch')
 pieces=[s[i:i+n] for i in range(0,max(len(tiny),len(wiki)),n) for s in (tiny,wiki) if i<len(s)]
 data='\n\n'.join(pieces).encode(); actual=ROOT/m['output_path']; digest=hashlib.sha256(data).hexdigest()
 if digest!=sha(actual): raise RuntimeError('training corpus mismatch')
 return dict(training_sha256=digest,training_bytes=len(data),tiny_source_sha256=sha(ROOT/m['tiny_path']),wiki_source_sha256=sha(ROOT/m['wiki_path']),metadata_sha256=sha(p),reconstruction='sample evenly using original _sample_evenly, alternate 16384-character chunks, join with two newline bytes',tokenizer='byte, 256 IDs',labels='source chunk; separator bytes unknown')

def main():
 provenance={'status':'incomplete','corpus':corpus(),'cells':{},'aggregation_prespecified':'For each eligible weight, compute abs(mean of nonzero resolved batch-mean gradient signs) separately in each domain. Aggregate as the minimum of the two domain consistencies; both domains must meet coverage and numerical-resolution thresholds. Unknowns are never imputed.','limitations':['Native batch/index reconstruction, gradient pilot and selection analysis not yet completed.']}
 (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 fields=['seed','order','step','matrix','eligible','measure_a','measure_b','spearman','top_k','top_k_overlap','archived_random_vs_reinforcement_symmetric_difference']
 with (OUT/'baseline_disagreement.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
  for seed in (7,17,29):
   for order in ('A','B'):
    cell=f'seed{seed}_{order}'
    for step in (250,500):
     directory=OLD/'prefixes'/cell
     if step==250 and not (directory/'state250.json').exists():
      files=json.loads((ROOT/'results/moe_online_reinforcement_pruning_v2/input_manifest.json').read_text())['cells'][cell]['files']
      statefile=Path(files['checkpoint']['path']); scorefile=Path(files['scores']['path'])
      meta={'sha256':files['checkpoint']['sha256'],'scores_sha256':files['scores']['sha256'],'prefix_batch_hash':None}
     else:
      meta=json.loads((directory/f'state{step}.json').read_text()); statefile=directory/f'state{step}.pt'; scorefile=directory/f'scores{step}.pt'
     if sha(statefile)!=meta['sha256'] or sha(scorefile)!=meta['scores_sha256']: raise RuntimeError(f'archive digest mismatch: {cell} {step}')
     saved=torch.load(statefile,map_location='cpu',mmap=True,weights_only=False)
     if saved['step']!=step or not all(k in saved for k in ('model','optimizer','scaler','torch_rng','cuda_rng')): raise RuntimeError('invalid full state')
     scores=torch.load(scorefile,map_location='cpu',mmap=True,weights_only=True); scores=scores.get('scores',scores); names=scope(saved['model'])
     sets=torch.load(OLD/'groups'/cell/'sets.pt',map_location='cpu',weights_only=True) if step==250 else None
     for name in names:
      w=saved['model'][name].numpy().ravel(); d=scores[name]; reinforcement=d['score'].numpy().ravel(); movement=d['absolute_update'].numpy().ravel(); valid=(d['initial_sign'].numpy().ravel()!=0)&(movement>0)
      if not np.array_equal(np.isfinite(reinforcement),valid): raise RuntimeError('score validity mismatch '+name)
      pool=np.lexsort((np.arange(w.size),np.abs(w)))[:w.size//4]; pool=pool[valid[pool]]; k=13107
      if len(pool)<k: raise RuntimeError('insufficient candidates')
      vals={'magnitude':np.abs(w[pool]),'absolute_movement':movement[pool],'reinforcement':reinforcement[pool]}
      disagreement=''
      if sets is not None:
       r=sets[name]['R'].numpy(); h=sets[name]['H'].numpy()
       if not (np.isin(r,pool).all() and np.isin(h,pool).all()): raise RuntimeError('archived selection outside pool')
       disagreement=len(np.setxor1d(r,h))
      for a,b in [('magnitude','absolute_movement'),('magnitude','reinforcement'),('absolute_movement','reinforcement')]:
       # Highest scores are compared consistently; overlap is independent of rank correlation direction.
       x,y=vals[a],vals[b]; ix=np.lexsort((pool,-x))[:k]; iy=np.lexsort((pool,-y))[:k]
       writer.writerow(dict(seed=seed,order=order,step=step,matrix=name,eligible=len(pool),measure_a=a,measure_b=b,spearman=float(spearmanr(x,y).statistic),top_k=k,top_k_overlap=len(np.intersect1d(ix,iy)),archived_random_vs_reinforcement_symmetric_difference=disagreement))
     provenance['cells'][f'{cell}_{step}']={'state_sha256':meta['sha256'],'scores_sha256':meta['scores_sha256'],'archived_prefix_batch_sha256':meta['prefix_batch_hash'],'matrices':len(names)}
     (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');f.flush();print(cell,step,flush=True)
if __name__=='__main__': main()
