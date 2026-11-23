import torch,numpy as np,math
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
p='prefixes/seed7_A/'
a=torch.load(p+'native/interruption.pt',map_location='cpu',weights_only=False)
s=torch.load(p+'scores.pt',map_location='cpu',weights_only=True)['scores']
z=torch.load('matching/seed7_A/selections.pt',map_location='cpu',weights_only=True)
n=next(iter(z['random'])); w=a['model'][n].numpy().ravel(); r=s[n]['score'].numpy().ravel(); N=len(w); rank=np.lexsort((np.arange(N),np.abs(w))); pool=rank[:N//4]; pool=pool[np.isfinite(r[pool])]; nr,nc=a['model'][n].shape; rnd=z['random'][n].numpy(); high=z['high'][n].numpy(); m=len(pool); ii=np.arange(m)
A=coo_matrix((np.ones(2*m),(np.r_[pool//nc,nr+pool%nc],np.r_[ii,ii])),shape=(nr+nc,m)).tocsr(); b=np.r_[np.bincount(rnd//nc,minlength=nr),np.bincount(rnd%nc,minlength=nc)]
res=linprog(-r[pool],A_eq=A,b_eq=b,bounds=(0,1),method='highs'); print('matrix',n,'success',res.success,'registered_score',float(r[high].sum()),'unconstrained_optimum',float(-res.fun) if res.success else res.message,'max_fractionality',float(np.max(np.abs(res.x-np.round(res.x)))) if res.success else '',flush=True)
