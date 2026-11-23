"""Reproduce controls and independently solve the registered per-bin flow LP."""
import argparse,hashlib,json,math,os,sys,time,warnings
from pathlib import Path
import numpy as np, scipy, torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
HERE=Path(__file__).resolve().parent; V1=HERE.parent/'moe_online_reinforcement_pruning_v1'
def seed64(seed,order,name):
 return int.from_bytes(hashlib.sha256(f'moe_online_reinforcement_pruning_v1|random|{seed}|{order}|{name}'.encode()).digest()[:8],'big')
def flow(pool,score,random,nr,nc):
 pool=np.sort(pool); m=len(pool); ix=np.arange(m); rows=pool//nc; cols=pool%nc
 A=coo_matrix((np.ones(2*m),(np.r_[rows,nr+cols],np.r_[ix,ix])),shape=(nr+nc,m)).tocsr()
 b=np.r_[np.bincount(random//nc,minlength=nr),np.bincount(random%nc,minlength=nc)].astype(float)
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',message='Unrecognized options detected')
  res=linprog(-score[pool],A_eq=A,b_eq=b,bounds=(0,1),method='highs-ds',options={'threads':1,'presolve':True})
 if not res.success: raise RuntimeError(res.message)
 if np.max(np.abs(res.x-np.rint(res.x)),initial=0)>1e-7: raise RuntimeError('non-integral LP solution')
 chosen=pool[res.x>.5]
 if not np.array_equal(A@np.rint(res.x),b): raise RuntimeError('invalid flow')
 # HiGHS returns dual marginals; dual residuals are recorded for independent audit.
 dual=float(b@res.eqlin.marginals + res.lower.marginals@np.zeros(m) + res.upper.marginals@np.ones(m))
 if abs(float(res.fun)-dual)>1e-5: raise RuntimeError('dual objective gap')
 return chosen,{'min_objective':float(res.fun),'dual_min_objective':dual,'iterations':int(res.nit),'edges':m}
def main(seed,order,limit):
 p=V1/'prefixes'/f'seed{seed}_{order}'; s=V1/'matching'/f'seed{seed}_{order}'/'selections.pt'
 cp=torch.load(p/'native/interruption.pt',map_location='cpu',weights_only=False); scores=torch.load(p/'scores.pt',map_location='cpu',weights_only=True); old=torch.load(s,map_location='cpu',weights_only=True)
 assert scores['attempts']==250 and scores['applied_updates_1_250']==249
 names=sorted(old['random']); assert len(names)==256 and set(names)==set(old['high']); audits=[]; allrandom={}; allhigh={}; totaloverlap=0; totals=[0.,0.]; total=0
 for name in names[:limit]:
  weights=cp['model'][name].numpy(); nr,nc=weights.shape; w=weights.ravel().astype(np.float64); obj=scores['scores'][name]; R=obj['score'].numpy().ravel(); den=obj['absolute_update'].numpy().ravel(); sign=obj['initial_sign'].numpy().ravel(); valid=(den>0)&(sign!=0)
  if not np.array_equal(np.isfinite(R),valid): raise RuntimeError('eligibility mismatch '+name)
  if len(w)!=262144: raise RuntimeError('unexpected matrix shape')
  rank=np.lexsort((np.arange(len(w)),np.abs(w))); eligible=rank[:len(w)//4]; eligible=eligible[valid[eligible]]; bins=np.array_split(eligible,5)
  random=np.sort(np.random.default_rng(seed64(seed,order,name)).choice(eligible,13107,replace=False)); archived=old['random'][name].numpy()
  if not np.array_equal(random,archived): raise RuntimeError('random mismatch '+name)
  high=[]; proofs=[]
  for pool in bins:
   randombin=random[np.isin(random,pool)]; x,cert=flow(pool,R,randombin,nr,nc); high.append(x);proofs.append(cert)
  high=np.sort(np.concatenate(high)); assert len(high)==13107 and len(np.unique(high))==13107
  re=float(np.square(w[random]).sum()); he=float(np.square(w[high]).sum()); passed=abs(re-he)<=0.05*re if re else he==re
  old_high=old['high'][name].numpy(); overlap=int(np.intersect1d(random,high,assume_unique=True).size); totaloverlap+=overlap; totals[0]+=float(R[random].sum()); totals[1]+=float(R[high].sum());total+=len(random)
  row_delta=np.bincount(high//nc,weights=np.square(w[high]),minlength=nr)-np.bincount(random//nc,weights=np.square(w[random]),minlength=nr)
  col_delta=np.bincount(high%nc,weights=np.square(w[high]),minlength=nc)-np.bincount(random%nc,weights=np.square(w[random]),minlength=nc)
  audits.append({'name':name,'random_energy':re,'high_energy':he,'energy_pass':bool(passed),'objective':float(R[high].sum()),'v1_high_equal':bool(np.array_equal(high,old_high)),'v1_objective':float(R[old_high].sum()),'overlap':overlap,'row_energy_difference':row_delta.tolist(),'column_energy_difference':col_delta.tolist(),'certificates':proofs})
  allrandom[name]=torch.from_numpy(random);allhigh[name]=torch.from_numpy(high)
  print(seed,order,len(audits),name,'energy_pass',passed,'same_v1',np.array_equal(high,old_high),'gap',float(R[high].sum()-R[old_high].sum()),flush=True)
  if not passed:
   print('REGISTERED ENERGY GATE FAILED; stopping',flush=True);break
 dest=HERE/'matching'/f'seed{seed}_{order}';dest.mkdir(parents=True,exist_ok=True)
 result={'seed':seed,'order':order,'matrices_audited':len(audits),'complete':len(audits)==256,'energy_pass':all(a['energy_pass'] for a in audits),'score_separation':(totals[1]-totals[0])/total,'overlap_fraction':totaloverlap/total,'solver':f'scipy {scipy.__version__} highs-ds threads=1','matrices':audits}
 path=dest/'audit.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
 if result['complete'] and result['energy_pass'] and result['score_separation']>=.2 and result['overlap_fraction']<=.3:
  path=dest/'selections.pt';tmp=path.with_suffix('.tmp');torch.save({'random':allrandom,'high':allhigh},tmp);tmp.replace(path)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('seed',type=int,choices=(7,17,29));a.add_argument('order',choices=('A','B'));a.add_argument('--limit',type=int,default=256);z=a.parse_args();main(z.seed,z.order,z.limit)
