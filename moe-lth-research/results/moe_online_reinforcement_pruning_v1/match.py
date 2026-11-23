"""Frozen random draw and exact degree-constrained high-score selection."""
from __future__ import annotations
import argparse,hashlib,json,math,os,time
from pathlib import Path
from study_common import *
from scipy.optimize import milp,LinearConstraint,Bounds
from scipy.sparse import coo_matrix,vstack

def seed64(seed,order,name):
 return int.from_bytes(hashlib.sha256(f'moe_online_reinforcement_pruning_v1|random|{seed}|{order}|{name}'.encode()).digest()[:8],'big')

def solve(pool,R,rows,cols,row_target,col_target,bins,w2,random_bins,random_energy,k,random):
 m=len(pool);nr=len(row_target);nc=len(col_target);ix=np.arange(m)
 objective=-R[pool].astype(np.float64)+pool.astype(np.float64)*(1e-12/max(1,int(pool.max())))
 start=time.time()
 def gates(ch):
  counts=np.bincount(bins[ch],minlength=5);energy=float(w2[pool[ch]].sum())
  return counts,energy,bool(np.max(np.abs(counts-random_bins))/k<=.02+1e-15 and abs(energy-random_energy)<=.05*random_energy+1e-30)
 for subdivisions in (1,2,4,10,20):
  fine=np.full(m,-1,dtype=np.int16)
  for coarse in range(5):
   local=np.flatnonzero(bins==coarse)
   ordered=local[np.lexsort((pool[local],w2[pool[local]]))]
   for sub,part in enumerate(np.array_split(ordered,subdivisions)):fine[part]=coarse*subdivisions+sub
  groups=5*subdivisions
  A=coo_matrix((np.ones(2*m),(np.r_[fine*nr+rows,groups*nr+fine*nc+cols],np.r_[ix,ix])),shape=(groups*(nr+nc),m)).tocsr()
  random_local=np.searchsorted(pool,random)
  # pool is in stable magnitude order, not numerical order; map actual IDs.
  lookup={int(value):i for i,value in enumerate(pool)};random_local=np.array([lookup[int(value)] for value in random])
  target=np.r_[np.bincount(fine[random_local]*nr+rows[random_local],minlength=groups*nr),np.bincount(fine[random_local]*nc+cols[random_local],minlength=groups*nc)]
  res=milp(objective,integrality=np.ones(m),bounds=Bounds(0,1),constraints=LinearConstraint(A,target,target),options={'time_limit':180,'mip_rel_gap':0.0})
  if not res.success:raise RuntimeError(f'Stratified exact flow failed at {groups} bins: {res.message}')
  chosen=np.flatnonzero(res.x>.5);assert len(chosen)==k
  counts,energy,passed=gates(chosen)
  if passed:
   return np.sort(pool[chosen]),dict(seconds=time.time()-start,magnitude_strata=groups,objective=float(-R[pool[chosen]].sum()),mip_gap=float(res.mip_gap),mip_node_count=int(res.mip_node_count))
 raise RuntimeError('No frozen magnitude-stratification level passed removed-energy matching')

def run(seed,order):
 root=OUT/'prefixes'/f'seed{seed}_{order}';verification=json.loads((root/'verification.json').read_text());assert verification['status']=='VERIFIED'
 destination=OUT/'matching'/f'seed{seed}_{order}';destination.mkdir(parents=True,exist_ok=True)
 if (destination/'complete.json').exists():return
 payload=torch.load(root/'native/interruption.pt',map_location='cpu',weights_only=False);assert payload['step']==250
 score_payload=torch.load(root/'scores.pt',map_location='cpu',weights_only=True);scores=score_payload['scores']
 random_sel={};high_sel={};audits=[];cell_random=[];cell_high=[]
 for index,name in enumerate(expert_names(payload['model'])):
  tensor=payload['model'][name].detach().cpu();w=tensor.numpy().ravel().astype(np.float64);R=scores[name]['score'].numpy().ravel().astype(np.float64)
  nr,nc=tensor.shape;N=len(w);k=math.floor(.05*N+.5);rank=np.lexsort((np.arange(N),np.abs(w)));quartile=rank[:N//4];pool=quartile[np.isfinite(R[quartile])]
  if len(pool)<k:raise RuntimeError(f'Insufficient eligible small pool {name}: {len(pool)} < {k}')
  rng=np.random.default_rng(seed64(seed,order,name));random=np.sort(rng.choice(pool,k,replace=False));row_target=np.bincount(random//nc,minlength=nr);col_target=np.bincount(random%nc,minlength=nc)
  pool_order=pool[np.lexsort((pool,np.abs(w[pool])))];parts=np.array_split(pool_order,5);bin_of=np.full(N,-1,dtype=np.int8)
  for b,part in enumerate(parts):bin_of[part]=b
  pool_bins=bin_of[pool];random_bins=np.bincount(bin_of[random],minlength=5);random_energy=float(np.square(w[random]).sum())
  high,solver=solve(pool,R,pool//nc,pool%nc,row_target,col_target,pool_bins,np.square(w),random_bins,random_energy,k,random)
  random_sel[name]=torch.from_numpy(random);high_sel[name]=torch.from_numpy(high);cell_random.append(R[random]);cell_high.append(R[high])
  rc=np.bincount(high//nc,minlength=nr);cc=np.bincount(high%nc,minlength=nc);assert np.array_equal(rc,row_target) and np.array_equal(cc,col_target)
  hb=np.bincount(bin_of[high],minlength=5);he=float(np.square(w[high]).sum());overlap=len(np.intersect1d(random,high,assume_unique=True))
  re=np.bincount(random//nc,weights=np.square(w[random]),minlength=nr);he_row=np.bincount(high//nc,weights=np.square(w[high]),minlength=nr)
  ce=np.bincount(random%nc,weights=np.square(w[random]),minlength=nc);he_col=np.bincount(high%nc,weights=np.square(w[high]),minlength=nc)
  audits.append(dict(seed=seed,order=order,tensor=name,shape=list(tensor.shape),matrix_count=N,delete_count=k,quartile_count=N//4,eligible_small_count=len(pool),random_seed=seed64(seed,order,name),
   random_mean_score=float(R[random].mean()),high_mean_score=float(R[high].mean()),score_gap=float(R[high].mean()-R[random].mean()),overlap=overlap,overlap_fraction=overlap/k,
   random_mean_abs_magnitude=float(np.abs(w[random]).mean()),high_mean_abs_magnitude=float(np.abs(w[high]).mean()),random_energy=random_energy,high_energy=he,energy_relative_gap=abs(he-random_energy)/max(random_energy,1e-300),
   random_bin_counts=random_bins.tolist(),high_bin_counts=hb.tolist(),maximum_bin_fraction_gap=float(np.max(np.abs(hb-random_bins))/k),
   row_energy_random=re.tolist(),row_energy_high=he_row.tolist(),column_energy_random=ce.tolist(),column_energy_high=he_col.tolist(),solver=solver,
   checks=dict(count=len(high)==len(random)==k,row_counts=bool(np.array_equal(rc,row_target)),column_counts=bool(np.array_equal(cc,col_target)),magnitude_bins=bool(np.max(np.abs(hb-random_bins))/k<=.02+1e-15),energy=bool(abs(he-random_energy)<=.05*random_energy+1e-30))))
  print(seed,order,index+1,len(scores),name,solver['magnitude_strata'],flush=True)
 selections=destination/'selections.pt';save_torch(selections,dict(random=random_sel,high=high_sel))
 flat_r=np.concatenate(cell_random);flat_h=np.concatenate(cell_high);total=sum(len(x) for x in cell_random);overlap=sum(x['overlap'] for x in audits)
 cell=dict(seed=seed,order=order,matrices=len(audits),deletions_per_arm=total,mean_score_random=float(flat_r.mean()),mean_score_high=float(flat_h.mean()),score_separation=float(flat_h.mean()-flat_r.mean()),overlap=overlap,overlap_fraction=overlap/total,
  matrix_checks_pass=all(all(x['checks'].values()) for x in audits),score_separation_pass=bool(flat_h.mean()-flat_r.mean()>=.20),overlap_pass=bool(overlap/total<=.30),magnitude_strata_counts={str(level):sum(x['solver']['magnitude_strata']==level for x in audits) for level in (5,10,20,50,100)})
 cell['passed']=cell['matrix_checks_pass'] and cell['score_separation_pass'] and cell['overlap_pass']
 save_json(destination/'matrix_audit.json',audits);save_json(destination/'complete.json',dict(**cell,selections=str(selections),selections_sha256=sha(selections),matrix_audit_sha256=sha(destination/'matrix_audit.json'),prefix_checkpoint_sha256=verification['checkpoint_sha256'],scores_sha256=verification['scores_sha256'],protocol_sha256=PROTOCOL_SHA))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('seed',type=int,choices=SEEDS);a.add_argument('order',choices=ORDERS);z=a.parse_args();run(z.seed,z.order)
