"""Detached amplitude transformations; acceptance is never an input to capacity selection here."""
from collections import defaultdict
import hashlib
import numpy as np

def fixed_seed(seed,*labels):
 return int.from_bytes(hashlib.sha256(('|'.join(map(str,(seed,)+labels))).encode()).digest()[:8],'little')

def block_permutation(counts,seed):
 """One seed-level rank permutation, restricted to equal-length expert/layer blocks.

Singleton count strata stay fixed. No padding, truncation, or interpolation.
The returned mapping is target step -> source step, separately by layer/expert.
"""
 counts=np.asarray(counts);steps,layers,experts=counts.shape
 ranks=np.random.default_rng(fixed_seed(seed,'time_blocks')).permutation(steps)
 result=np.broadcast_to(np.arange(steps)[:,None,None],counts.shape).copy()
 for layer in range(layers):
  for expert in range(experts):
   for count in np.unique(counts[:,layer,expert]):
    ids=np.flatnonzero(counts[:,layer,expert]==count)
    if count==0 or len(ids)<2:continue
    ordered=ids[np.argsort(ranks[ids])]
    result[ordered,layer,expert]=np.roll(ordered,1)
 return result

def token_permutation(count,seed,step,layer,expert):
 """Independent of age and model kind: matched members receive identical indices."""
 return np.random.default_rng(fixed_seed(seed,'tokens',step,layer,expert)).permutation(count)

def accepted_means(amplitudes,ids,accepted,experts):
 """Exact edge-weighted sums/counts; means are rounded once when materialized."""
 sums=np.zeros((len(amplitudes),experts),dtype=np.float64);counts=np.zeros_like(sums,dtype=np.int64)
 for layer,(a,i,ok) in enumerate(zip(amplitudes,ids,accepted)):
  a=np.asarray(a).reshape(-1);i=np.asarray(i).reshape(-1);ok=np.asarray(ok).reshape(-1)
  assert np.isfinite(a[ok]).all()
  sums[layer]=np.bincount(i[ok],weights=a[ok].astype(np.float64),minlength=experts)
  counts[layer]=np.bincount(i[ok],minlength=experts)
 return sums,counts

def materialize_static(ids,accepted,sums,counts,condition,dtype=np.float16):
 sums=np.asarray(sums);counts=np.asarray(counts)
 if condition=='global_mean':
  assert counts.sum()>0;means=np.full(sums.shape,sums.sum()/counts.sum())
 elif condition=='layer_mean':
  assert (counts.sum(1)>0).all();means=np.broadcast_to((sums.sum(1)/counts.sum(1))[:,None],sums.shape)
 elif condition=='expert_layer_mean':
  means=np.divide(sums,counts,out=np.full(sums.shape,np.nan),where=counts>0)
 else:raise ValueError(condition)
 result=[]
 for layer,(i,ok) in enumerate(zip(ids,accepted)):
  i=np.asarray(i);ok=np.asarray(ok).reshape(i.shape);values=means[layer][i]
  assert np.isfinite(values[ok]).all(),'Missing observed mean for an accepted edge'
  result.append(np.where(ok,values,0).astype(dtype))
 return result


def source_coefficient_dtype(arrays):
 """Preserve coefficient precision; training autocast does not determine router dtype."""
 dtypes={np.asarray(a).dtype for a in arrays}
 if not dtypes or any(d not in (np.dtype('float16'),np.dtype('float32'),np.dtype('float64')) for d in dtypes):
  raise ValueError(f'Unsupported source coefficient dtypes: {dtypes}')
 return np.result_type(*dtypes)
