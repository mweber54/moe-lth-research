import numpy as np
from moe_lth.experiments.amplitude_pattern_transforms import block_permutation,token_permutation,accepted_means,materialize_static

def test_transform_invariants():
 ids=np.array([[0,0,1,1],[0,1,1,1]]);ok=np.array([[1,1,1,0],[1,1,1,1]],dtype=bool);a=np.array([[.2,.4,.8,.7],[.1,.3,.5,.9]],dtype=np.float16)
 sums,counts=accepted_means(a,ids,ok,2)
 for condition in ['global_mean','layer_mean','expert_layer_mean']:
  x=np.array(materialize_static(ids,ok,sums,counts,condition));y=np.array(materialize_static(ids,ok,sums,counts,condition))
  assert np.array_equal(x,y) and np.isfinite(x).all();assert (x[~ok]==0).all()
 assert float(np.array(materialize_static(ids,ok,sums,counts,'global_mean'))[0,0])==float(np.float16(a[ok].astype(np.float64).mean()))
 counts=np.array([[[3,2]],[[3,4]],[[1,2]],[[3,5]]]);p=block_permutation(counts,7)
 assert np.array_equal(p,block_permutation(counts,7))
 for l in range(counts.shape[1]):
  for e in range(counts.shape[2]):
   assert sorted(p[:,l,e])==list(range(4));assert np.array_equal(counts[:,l,e],counts[p[:,l,e],l,e])
 values=np.arange(17,dtype=np.float16);perm=token_permutation(17,7,100,0,1)
 assert np.array_equal(perm,token_permutation(17,7,100,0,1)) and np.array_equal(np.sort(values[perm]),values)
if __name__=='__main__':test_transform_invariants();print('PASS weighted means, deterministic paired indexing, exact block lengths and token-shuffle marginals')


def test_fp32_source_precision_is_preserved():
 from moe_lth.experiments.amplitude_pattern_transforms import source_coefficient_dtype
 source=np.array([0.4580853,0.43471524,0.4972526,0.5013486],dtype=np.float32)
 assert not np.array_equal(source,source.astype(np.float16))
 dtype=source_coefficient_dtype([source]);assert dtype==np.dtype('float32')
 perm=token_permutation(len(source),7,1,0,0)
 target=np.zeros(source.shape,dtype=dtype);target[:]=source[perm]
 assert np.array_equal(np.sort(source),np.sort(target))
 assert source_coefficient_dtype([source,np.array([.125],dtype=np.float16)])==np.dtype('float32')
 assert source_coefficient_dtype([source.astype(np.float16)])==np.dtype('float16')
