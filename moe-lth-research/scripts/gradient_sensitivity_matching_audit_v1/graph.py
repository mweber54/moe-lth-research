from common import *
import ctypes
from scipy.optimize import linprog
from scipy.sparse import vstack,eye,csr_matrix,hstack

_lib=None
def range_counts(points):
 global _lib
 if _lib is None:
  _lib=ctypes.CDLL(str(D/'bin/ranges.so'));_lib.count_ranges.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
 points=np.ascontiguousarray(points,dtype=np.float64);hi=np.zeros((len(points),3),dtype=np.uint64);lo=np.zeros((len(points),3),dtype=np.uint8)
 _lib.count_ranges(points.ctypes.data,len(points),hi.ctypes.data,lo.ctypes.data);return hi,lo

def score_counts(A):
 s=np.sort(A);hc=np.where(A>0,np.searchsorted(s,A/2,side='right'),0);lc=np.where(A>0,len(s)-np.searchsorted(s,2*A,side='left'),len(s)-np.searchsorted(s,0,side='right'))
 return int(hc.sum()),int(np.count_nonzero(hc)),int(np.count_nonzero(lc)),int(np.count_nonzero((hc>0)|(lc>0)))

def attrition_and_graph(seed,name,f,A,shape,small,strata,sign):
 meta=metadata(seed,name);out=[]
 def emit(stage,edge,hc,lc,uc,kind='possible_edges',**extra):
  out.append(dict(**meta,stage=stage,possible_edges=edge,unique_high_coordinates=hc,unique_low_coordinates=lc,unique_union_coordinates=uc,count_kind=kind,**extra))
 emit('currently_small',None,None,None,len(small));emit('sensitivity',*score_counts(A[small]))
 totals=np.zeros((5,4),dtype=np.int64)
 for si,idx in enumerate(np.array_split(small,50)):
  totals[0]+=score_counts(A[idx])
  for sg in (-1,0,1):
   ids=idx[sign[idx]==sg]
   if not len(ids):continue
   totals[1]+=score_counts(A[ids]);hi,lo=range_counts(np.column_stack([f[ids],A[ids]]))
   for j in range(3):totals[j+2]+=np.array([hi[:,j].sum(),np.count_nonzero(hi[:,j]),np.count_nonzero(lo[:,j]),np.count_nonzero((hi[:,j]>0)|(lo[:,j]>0))],dtype=np.int64)
 for stage,v in zip(('stratum','sign','magnitude_caliper','slope_caliper','variation_caliper'),totals):emit(stage,*[int(x) for x in v])
 key=(small//shape[1])*150+strata[small]*3+(sign[small]+1);order=np.argsort(key,kind='stable');ids=small[order];keys=key[order]
 hs=[];ls=[];cost=[]
 for group in np.split(ids,np.flatnonzero(np.diff(keys))+1):
  if len(group)<2:continue
  aa=A[group];h,l=np.nonzero((aa[:,None]>0)&(aa[:,None]>aa[None,:])&(aa[:,None]>=2*aa[None,:]))
  if not len(h):continue
  delta=np.abs(f[group[h]]-f[group[l]]);valid=np.all(delta<=np.array([.005,.1,.1]),axis=1);h=h[valid];l=l[valid];delta=delta[valid]
  if len(h):hs.append(group[h]);ls.append(group[l]);cost.append(np.sum((delta/np.array([.005,.1,.1]))**2,axis=1))
 h=np.concatenate(hs) if hs else np.array([],dtype=np.int64);l=np.concatenate(ls) if ls else h.copy();c=np.concatenate(cost) if cost else np.array([],dtype=float)
 emit('exact_row',len(h),len(np.unique(h)),len(np.unique(l)),len(np.union1d(h,l)))
 assert all(out[i]['possible_edges']>=out[i+1]['possible_edges'] for i in range(1,len(out)-1))
 return out,h,l,c

def component_diagnostics(h,l,columns):
 if not len(h):return dict(column_SCCs=columns,cyclic_edges=0,cyclic_unique_coordinates=0,exact_column_coordinate_upper_bound=0)
 hc=h%columns;lc=l%columns
 _,labels=connected_components(coo_matrix((np.ones(len(h)),(hc,lc)),shape=(columns,columns)).tocsr(),directed=True,connection='strong')
 cyclic=labels[hc]==labels[lc];nh=len(np.unique(h[cyclic]));nl=len(np.unique(l[cyclic]));union=len(np.union1d(h[cyclic],l[cyclic]))
 return dict(column_SCCs=len(np.unique(labels)),cyclic_edges=int(cyclic.sum()),cyclic_unique_coordinates=union,exact_column_coordinate_upper_bound=min(nh,nl,union//2,2621))

def restricted_graph(h,l,c,greedy_h,greedy_l):
 if len(h)<=300000:return h,l,c,False
 # Retain every original disjoint pair, then cheapest alternative edges.
 base=set(zip(greedy_h.tolist(),greedy_l.tolist()));mandatory=np.array([i for i,p in enumerate(zip(h.tolist(),l.tolist())) if p in base],dtype=np.int64)
 remaining=np.lexsort((l,h,c));keep=set(mandatory.tolist());chosen=mandatory.tolist()
 for i in remaining:
  if int(i) not in keep:chosen.append(int(i));keep.add(int(i))
  if len(chosen)>=300000:break
 chosen=np.array(chosen,dtype=np.int64);return h[chosen],l[chosen],c[chosen],True

def lp_C(h,l,f,shape):
 # Original greedy pairs are already disjoint; continuous relaxation is only a heuristic generator.
 n=len(h);cols=shape[1]
 if not n:return h,l,{'status':'EMPTY_GRAPH','upper_bound':0.,'fractional_variables':0,'seconds':0.}
 idx=np.arange(n);B=coo_matrix((np.r_[np.ones(n),-np.ones(n)],(np.r_[h%cols,l%cols],np.r_[idx,idx])),shape=(cols,n)).tocsr()
 z=eye(cols,format='csr');zero=csr_matrix((1,cols));eh=f[h,0]**2;el=f[l,0]**2
 constraints=vstack([hstack([B,-z]),hstack([-B,-z]),csr_matrix(np.r_[np.full(n,-.1),np.ones(cols)][None,:]),hstack([csr_matrix((.995*eh-el)[None,:]),zero]),hstack([csr_matrix((.995*el-eh)[None,:]),zero]),csr_matrix(np.r_[np.ones(n),np.zeros(cols)][None,:])],format='csr')
 rhs=np.r_[np.zeros(2*cols+3),2621.];start=time.monotonic()
 result=linprog(np.r_[-np.ones(n),np.zeros(cols)],A_ub=constraints,b_ub=rhs,bounds=[(0,1)]*n+[(0,None)]*cols,method='highs',options={'time_limit':3,'threads':1})
 if result.x is None:return h[:0],l[:0],{'status':str(result.status),'upper_bound':None,'fractional_variables':None,'seconds':time.monotonic()-start,'message':result.message}
 x=result.x[:n];selected=x>=1-1e-7
 return h[selected],l[selected],{'status':str(result.status),'upper_bound':float(-result.fun) if result.status==0 else None,'fractional_variables':int(np.sum((x>1e-7)&(x<1-1e-7))),'seconds':time.monotonic()-start,'message':result.message}
