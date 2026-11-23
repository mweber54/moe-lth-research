"""Compare source indices, native training batches, and sealed prefix hashes on CPU."""
import hashlib,json,sys
from itertools import cycle,islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'results/moe_online_reinforcement_pruning_v1'))
from study_common import material,N
from moe_lth.data import ByteTokenizer

def run(seed,order):
 cfg,batches,_=material(seed,order)
 if cfg['data'].get('tokenizer','byte')!='byte' or cfg['model']['vocab_size']!=256 or ByteTokenizer.vocab_size!=256: raise RuntimeError('tokenizer mismatch')
 data=Path(cfg['data']['train_path']).read_bytes(); seq=int(cfg['data']['seq_len']); count=(len(data)-1)//seq; batchsize=int(cfg['training']['batch_size'])
 loader=DataLoader(torch.arange(count),batch_size=batchsize,shuffle=True,drop_last=True,generator=torch.Generator().manual_seed(seed))
 canonical=list(islice(cycle(loader),2500)); permutation=N._permutations()[order]['segment_0']; indices=[canonical[int(i)] for i in permutation[:500]]
 raw=torch.tensor(list(data),dtype=torch.long); offsets=torch.arange(seq)
 for i,ix in enumerate(indices):
  starts=ix[:,None]*seq+offsets
  if not torch.equal(raw[starts],batches[i][0]) or not torch.equal(raw[starts+1],batches[i][1]): raise RuntimeError(f'batch alignment mismatch {seed} {order} {i}')
 checks={}
 for step in (250,500):
  p=ROOT/'results/moe_connection_lifetime_v1/prefixes'/f'seed{seed}_{order}'/f'state{step}.json'
  if p.exists():
   actual=N.recovery._batch_sequence_hash(batches[:step]); expected=json.loads(p.read_text())['prefix_batch_hash']
   if actual!=expected: raise RuntimeError('prefix hash mismatch')
   checks[str(step)]=actual
 v=ROOT/'results/moe_online_reinforcement_pruning_v1/prefixes'/f'seed{seed}_{order}'/'verification.json'
 expected=json.loads(v.read_text())['train_hash_1_251']; actual=N.recovery._batch_sequence_hash(batches[:251])
 if actual!=expected: raise RuntimeError('native 251 prefix hash mismatch')
 checks['251']=actual
 dest=OUT/f'indices_seed{seed}_{order}.pt';torch.save(torch.stack(indices),dest)
 return {'seed':seed,'order':order,'batches_verified':500,'prefix_hashes':checks,'indices_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
if __name__=='__main__':
 records=[]
 for seed in (7,17,29):
  for order in ('A','B'):
   records.append(run(seed,order)); (OUT/'batch_alignment.json').write_text(json.dumps(records,indent=2)+'\n');print(seed,order,'verified',flush=True)
