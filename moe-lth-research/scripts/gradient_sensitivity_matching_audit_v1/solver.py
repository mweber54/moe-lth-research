"""Isolated pure SciPy MILP subprocess, hard address-space and parent wall limits."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import resource,time,sys,json,warnings
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix,vstack,hstack,csr_matrix,eye
path,design,output=sys.argv[1:];a=np.load(path);h=a['h'];l=a['l'];eh=a['eh'];el=a['el'];cols=int(a['columns']);n=len(h)
start=time.monotonic()
try:
 unique=np.unique(np.r_[h,l]);ih=np.searchsorted(unique,h);il=np.searchsorted(unique,l);ii=np.arange(n)
 capacity=coo_matrix((np.ones(2*n),(np.r_[ih,il],np.r_[ii,ii])),shape=(len(unique),n)).tocsr()
 B=coo_matrix((np.r_[np.ones(n),-np.ones(n)],(np.r_[h%cols,l%cols],np.r_[ii,ii])),shape=(cols,n)).tocsr()
 energy=csr_matrix(np.vstack([.995*eh-el,.995*el-eh]));card=csr_matrix(np.ones((1,n)))
 if design=='B':
  M=vstack([capacity,B,energy,card],format='csc');lower=np.r_[np.full(len(unique),-np.inf),np.zeros(cols),[-np.inf]*3];upper=np.r_[np.ones(len(unique)),np.zeros(cols),[0.,0.,2621.]];objective=-np.ones(n);integrality=np.ones(n);bounds=Bounds(np.zeros(n),np.ones(n))
 else:
  z=eye(cols,format='csr');padding=csr_matrix((len(unique),cols));ep=csr_matrix((2,cols));cp=csr_matrix((1,cols))
  M=vstack([hstack([capacity,padding]),hstack([B,-z]),hstack([-B,-z]),csr_matrix(np.r_[np.full(n,-.1),np.ones(cols)][None,:]),hstack([energy,ep]),hstack([card,cp])],format='csc')
  lower=np.full(M.shape[0],-np.inf);upper=np.r_[np.ones(len(unique)),np.zeros(2*cols+3),2621.];objective=np.r_[-np.ones(n),np.zeros(cols)];integrality=np.r_[np.ones(n),np.zeros(cols)];bounds=Bounds(np.zeros(n+cols),np.r_[np.ones(n),np.full(cols,np.inf)])
 with warnings.catch_warnings(record=True) as ws:
  result=milp(objective,integrality=integrality,bounds=bounds,constraints=LinearConstraint(M,lower,upper),options={'time_limit':30,'node_limit':20000,'mip_rel_gap':0.,'threads':1})
 chosen=np.flatnonzero(result.x[:n]>.5) if result.x is not None else np.array([],dtype=np.int64)
 np.savez_compressed(output+'.npz',selected_edges=chosen)
 def finite(z):return float(z) if z is not None and np.isfinite(z) else None
 record={'status':int(result.status),'message':result.message,'incumbent_pairs':int(len(chosen)) if result.x is not None else None,'upper_bound_pairs':finite(-result.mip_dual_bound if getattr(result,'mip_dual_bound',None) is not None else None),'mip_gap':finite(getattr(result,'mip_gap',np.nan)),'nodes':finite(getattr(result,'mip_node_count',np.nan)),'warnings':[str(w.message) for w in ws], 'seconds':time.monotonic()-start,'maximum_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'graph_edges':n,'graph_coordinates':len(unique),'binary_variables':n,'encoded':'coordinate capacity, row/sign/calipers via eligible edges, columns exact/TV, energy bound, target cap; SMD/KS independently evaluated'}
except BaseException as e:
 record={'status':'EXCEPTION','message':repr(e),'seconds':time.monotonic()-start,'maximum_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'incumbent_pairs':None,'upper_bound_pairs':None}
with open(output+'.json','w') as f:json.dump(record,f,indent=2,allow_nan=False)
