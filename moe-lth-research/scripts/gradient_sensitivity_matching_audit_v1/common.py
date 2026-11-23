"""Array-only audit dependencies; original experiment is read-only."""
import importlib.util,sys,types,csv,math,time,socket,subprocess,os,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=module('frozen_gradient_common',ROOT/'scripts/gradient_sensitivity_pruning_v1/common.py')
# The legacy scripts use bare common imports; bind only while importing them.
current=sys.modules.get('common');sys.modules['common']=g
original=module('frozen_gradient_match',ROOT/'scripts/gradient_sensitivity_pruning_v1/match.py')
measurement=module('frozen_gradient_measure',ROOT/'scripts/gradient_sensitivity_pruning_v1/measure.py')
if current is not None:sys.modules['common']=current
else:sys.modules.pop('common',None)
import numpy as np
import torch
from scipy.stats import ks_2samp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
D=ROOT/'results/gradient_sensitivity_matching_audit_v1';OLD=g.D;SEEDS=g.SEEDS
read=g.read;save=g.save;sha=g.sha;table=g.table;load=g.load;metadata=g.metadata

def rows(path):
 with Path(path).open() as f:return list(csv.DictReader(f))
def deny(*a,**k):raise RuntimeError('Model/gradient/optimizer operation prohibited in feasibility audit')
g.TinyMoELanguageModel.forward=deny
torch.autograd.grad=deny;torch.Tensor.backward=deny;torch.optim.AdamW.step=deny

def features(values):
 shape=values.shape[1:];v=values.reshape(3,-1).astype(np.float64);m=np.abs(v);rms=float(np.sqrt(np.mean(m[2]**2)))
 t=np.array([250.,500.,1000.]);t-=t.mean()
 f=np.stack([m[2]/rms,(t[:,None]*m).sum(0)/np.dot(t,t)*750/rms,np.abs(np.diff(m,axis=0)).sum(0)/rms],axis=1)
 small=np.argsort(m[2],kind='stable')[:v.shape[1]//2];strata=np.full(v.shape[1],-1,dtype=np.int16)
 for i,idx in enumerate(np.array_split(small,50)):strata[idx]=i
 return shape,v,f,rms,small,strata,np.sign(v[2]).astype(np.int8)

def balance(seed,name,h,l,f,A,shape,design):
 h=np.asarray(h,dtype=np.int64);l=np.asarray(l,dtype=np.int64);k=len(h);N=int(np.prod(shape));meta=metadata(seed,name)
 assert len(l)==k and len(np.unique(h))==k and len(np.unique(l))==k and not np.intersect1d(h,l).size
 rh=np.bincount(h//shape[1],minlength=shape[0]);rl=np.bincount(l//shape[1],minlength=shape[0]);ch=np.bincount(h%shape[1],minlength=shape[1]);cl=np.bincount(l%shape[1],minlength=shape[1])
 r=dict(**meta,design=design,pairs=k,removed_coordinates_per_branch=k,distinct_coordinates_both_branches=2*k,dose=k/N,row_max_abs_difference=int(np.abs(rh-rl).max()),column_max_abs_difference=int(np.abs(ch-cl).max()),column_TV=float(np.abs(ch-cl).sum()/(2*k)) if k else None,
  row_occupancy_H=float(np.mean(rh>0)),row_occupancy_L=float(np.mean(rl>0)),column_occupancy_H=float(np.mean(ch>0)),column_occupancy_L=float(np.mean(cl>0)),
  column_HHI_H=float(np.sum((ch/k)**2)) if k else None,column_HHI_L=float(np.sum((cl/k)**2)) if k else None,
  column_max_share_H=float(ch.max()/k) if k else None,column_max_share_L=float(cl.max()/k) if k else None)
 reasons=[]
 for j,key in enumerate(('magnitude','slope','variation')):
  if k:
   a=f[h,j];b=f[l,j];sd=float(np.sqrt((a.var()+b.var())/2));delta=float(a.mean()-b.mean());smd=delta/sd if sd else (0. if delta==0 else None)
   ks=float(ks_2samp(a,b,method='asymp').statistic);r.update({key+'_H_mean':float(a.mean()),key+'_L_mean':float(b.mean()),key+'_SMD':smd,key+'_KS':ks})
   if smd is None or abs(smd)>.05 or ks>.05:reasons.append(key+'_balance')
  else:r.update({key+'_H_mean':None,key+'_L_mean':None,key+'_SMD':None,key+'_KS':None})
 if k:
  eh=float(np.dot(f[h,0],f[h,0]));el=float(np.dot(f[l,0],f[l,0]));energy=abs(eh-el)/max(eh,el,1e-300)
  ah=float(A[h].mean());al=float(A[l].mean());contrast=ah>0 and ah>=2*al
  r.update(removed_energy_H_over_rms_squared=eh,removed_energy_L_over_rms_squared=el,energy_relative_imbalance=energy,mean_A_H=ah,mean_A_L=al,sensitivity_ratio=ah/al if al>0 else None,sensitivity_pass=contrast)
  if energy>.005:reasons.append('energy_balance')
  if not contrast:reasons.append('sensitivity_contrast')
 else:r.update(energy_relative_imbalance=None,sensitivity_ratio=None,sensitivity_pass=False);reasons.append('no_pairs')
 if r['row_max_abs_difference']:reasons.append('row_balance')
 if k and ((design=='C' and r['column_TV']>.05+1e-10) or (design!='C' and r['column_max_abs_difference'])):reasons.append('column_balance')
 r['non_dose_checks_pass']=not reasons
 if k<656:reasons.append('minimum_dose')
 if k>2621:reasons.append('target_cap')
 r['scientifically_included']=not reasons;r['exclusion_reasons']=';'.join(reasons)
 return r

def frozen():
 config=read(D/'audit_config.json');seal=read(D/'audit_seal.json')
 assert sha(D/'audit_config.json')==seal['config_sha256']
 for p,h in seal['implementation'].items():assert sha(ROOT/p)==h,p
 for p,h in config['original_inputs'].items():assert sha(ROOT/p)==h,p
 assert read(OLD/'completion.json')['status']=='STOPPED_MATCHING_GATE'
 return config
