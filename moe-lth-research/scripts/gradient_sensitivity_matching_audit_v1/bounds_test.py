from common import *
from graph import component_diagnostics
rng=np.random.default_rng(9167)
for trial in range(100):
 cols=4;hc=rng.integers(0,cols,8);lc=(hc+rng.integers(1,cols,8))%cols
 h=np.arange(8)*cols+hc;l=np.arange(8)*cols+lc
 bound=component_diagnostics(h,l,cols)['exact_column_coordinate_upper_bound'];maximum=0
 for mask in range(256):
  idx=np.flatnonzero([(mask>>i)&1 for i in range(8)])
  if np.array_equal(np.bincount(h[idx]%cols,minlength=cols),np.bincount(l[idx]%cols,minlength=cols)):maximum=max(maximum,len(idx))
 assert maximum<=bound
save(D/'bound_proof_checks.json',{'passed':True,'random_graphs':100,'subsets_exhaustively_checked_per_graph':256,'certified_SCC_coordinate_upper_bound_never_below_exact_enumerated_maximum':True,'new_model_operations':0})
print('SCC bound independently verified on100 exhaustive small graphs')
