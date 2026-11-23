"""Resumable, serialized study runner. Each GPU replay uses the original admission rule."""
import csv,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
P=Path(__file__).resolve().parent;ROOT=P.parents[1];PY=sys.executable
admit=ROOT/'results/moe_connection_lifetime_v1/admitted.py';replay=P/'split_half_replay.py'
audit=json.loads((P/'split_half_index_audit.json').read_text());status=P/'split_half_replay_status.json'
def save(x):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(status)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def fname(cell,h,label,mode):return P/f'split_half_gradients_{cell}_h{h}_{label}{"_all_retained" if mode=="all_retained" else ""}.pt'
def run(entry,mode,h,perm=None):
 cell=entry['cell'];label='original' if perm is None else f'perm{perm}';dest=fname(cell,h,label,mode)
 if dest.exists():return
 seed=int(cell.split('_')[0][4:]);order=cell.split('_')[1];step=int(cell.split('_')[2])
 args=[PY,str(admit),str(replay),str(seed),order,str(step),str(h)]
 if mode=='all_retained':args+=['--all-retained']
 if perm is not None:args+=['--permutation',str(perm)]
 save({'status':'running','cell':cell,'analysis':mode,'half':h,'label':label,'log':str(P/'split_half_study.log')})
 with (P/'split_half_study.log').open('a') as log:
  print('RUN',args,file=log,flush=True);subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert dest.exists()
def verify(entry,mode):
 cell=entry['cell'];a,b=[load(fname(cell,h,'original',mode)) for h in (1,2)]
 split=entry['splits'][mode];assert [a['batch_ids'],b['batch_ids']]==split['batch_ids']
 if mode=='all_retained' or entry['retained_count']==180: full=load(P/f'gradients_{cell}.pt')
 else:
  run(entry,mode,0);full=load(fname(cell,0,'original',mode))
 assert a['batch_ids']+b['batch_ids']==full['batch_ids']
 for key in ('sign_sum','resolved_batches'):
  for d in range(2):
   for x,y,z in zip(a[key][d],b[key][d],full[key][d]):assert torch.equal(x+y,z),(cell,mode,key)
 for key in ('accepted_occurrences','contributing_batches'):assert np.array_equal(a[key]+b[key],full[key]),(cell,mode,key)
 assert a['model_unchanged'] and b['model_unchanged'] and a['rng_unchanged'] and b['rng_unchanged']
 return a,b
def merge():
 for name in ('split_half_reproducibility.csv','split_half_permutation_null.csv','split_half_coverage.csv','split_half_permutation_comparison.csv'):
  rows=[];keys=[]
  for f in sorted((P/'split_half_cell_analysis').glob('*/'+name)):
   with f.open(newline='') as g:
    rd=csv.DictReader(g)
    for k in rd.fieldnames or []:
     if k not in keys:keys.append(k)
    rows.extend(rd)
  if rows:
   tmp=P/(name+'.tmp')
   with tmp.open('w',newline='') as g:
    w=csv.DictWriter(g,fieldnames=keys);w.writeheader();w.writerows(rows)
   tmp.replace(P/name)
try:
 for entry in audit:
  modes=['first_180']+(['all_retained'] if entry['retained_count']>180 else [])
  for mode in modes:
   cell=entry['cell'];out=P/'split_half_cell_analysis'/f'{cell}_{mode}'
   if (out/'complete.json').exists():continue
   for h in (1,2):run(entry,mode,h)
   original=verify(entry,mode)
   for i in range(20):
    for h in (1,2):
     run(entry,mode,h,i);r=load(fname(cell,h,f'perm{i}',mode));o=original[h-1]
     assert r['batch_ids']==o['batch_ids'] and r['model_unchanged'] and r['rng_unchanged']
     assert np.array_equal(r['accepted_occurrences'].sum(axis=0),o['accepted_occurrences'].sum(axis=0)),(cell,mode,i,h,'total expert exposure')
   seed=int(cell.split('_')[0][4:]);order=cell.split('_')[1];step=int(cell.split('_')[2]);args=[PY,str(P/'split_half_analysis.py'),'--seed',str(seed),'--order',order,'--step',str(step),'--permutations','20','--output-dir',str(out)]
   if mode=='all_retained':args+=['--all-retained']
   save({'status':'analyzing','cell':cell,'analysis':mode,'permutations':20})
   subprocess.run(args,cwd=ROOT,check=True,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
   (out/'complete.json').write_text(json.dumps({'cell':cell,'analysis':mode,'primary_reconstruction_verified':True,'model_rng_unchanged':True,'permutations':20})+'\n');merge()
 save({'status':'all_replays_and_analyses_complete','primary_cells':12,'sensitivity_cells':9,'permutations_each':20,'note':'verdict and cross-cell interpretation require review'})
except Exception as e:
 save({'status':'failed','error':repr(e),'time':time.time()});raise
