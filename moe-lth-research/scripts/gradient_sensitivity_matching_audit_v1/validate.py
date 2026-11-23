from common import *
from graph import range_counts

def run_solver(edges,columns,design,name):
 p=D/'validation'/f'{name}.npz';p.parent.mkdir(parents=True,exist_ok=True)
 h=np.array([e[0] for e in edges],dtype=np.int64);l=np.array([e[1] for e in edges],dtype=np.int64)
 np.savez(p,h=h,l=l,eh=np.ones(len(h)),el=np.ones(len(h)),columns=columns)
 out=D/'validation'/name;subprocess.run([str(ROOT/'.venv-highres/bin/python'),str(HERE/'solver.py'),str(p),design,str(out)],check=True,timeout=45)
 r=read(str(out)+'.json');idx=np.load(str(out)+'.npz')['selected_edges'];hh=h[idx];ll=l[idx]
 assert len(np.union1d(hh,ll))==2*len(idx)
 ch=np.bincount(hh%columns,minlength=columns);cl=np.bincount(ll%columns,minlength=columns)
 if design=='B':assert np.array_equal(ch,cl)
 else:assert not len(idx) or np.abs(ch-cl).sum()/(2*len(idx))<=.05
 assert r['status']==0 and r['upper_bound_pairs']==len(idx)
 return len(idx)

def main():
 frozen();rng=np.random.default_rng(191792)
 for n in (1,7,40,100):
  points=np.column_stack([rng.uniform(0,.03,n),rng.normal(0,.15,n),rng.uniform(0,.3,n),rng.choice([0.,1.,2.,4.,8.],n)])
  high,low=range_counts(points)
  for d in (1,2,3):
   valid=(points[:,3,None]>0)&(points[:,3,None]>=2*points[None,:,3])&(points[:,3,None]>points[None,:,3])
   for k in range(d):valid &= np.abs(points[:,k,None]-points[None,:,k])<=[.005,.1,.1][k]
   assert np.array_equal(high[:,d-1],valid.sum(1));assert np.array_equal(low[:,d-1]>0,valid.any(0))
 # Row0 a(col0) has cheap edge to b(col1), or alternative c(col2); row1 d(col2)->e(col0).
 assert run_solver([(0,1),(0,2),(5,3)],3,'B','alternative_pairing')==2
 assert run_solver([(0,1),(5,3)],3,'B','greedy_restricted_infeasible')==0
 # Greedy cycle decomposition takes length2 and blocks an available length3 cycle.
 assert run_solver([(0,1),(4,3),(7,8),(11,9)],3,'B','maximum_cycle_packing')==3
 pairs=[dict(h=h,l=l,hc=h%3,lc=l%3,cost=float(i)) for i,(h,l) in enumerate([(0,1),(4,3),(7,8),(11,9)])]
 greedy,lengths=original.cycles_from_pairs(pairs,3,2621);assert len(greedy)==2
 # Twenty-edge path meets C's TV=1/20 but violates exact columns; each pair uses a distinct row.
 edges=[(i*21+i,i*21+i+1) for i in range(20)]
 assert run_solver(edges,21,'B','acyclic_exact_columns')==0
 assert run_solver(edges,21,'C','path_TV_point05')==20
 # Count denominators and signed-vs-absolute growth.
 assert round(262144*.01)==2621 and math.ceil(262144*.0025)==656
 values=np.array([[[-.1,-.2]], [[-.2,-.5]], [[-.3,-.2]]]);shape,v,f,rms,small,strata,sign=features(values)
 assert f[0,1]>0 and f[1,2]>0 and np.all(sign==-1)
 save(D/'numerical_checks.json',{'passed':True,'range_counts_equal_bruteforce_including_zero_and_ratio_boundaries':True,'binary_coordinate_disjointness_reconstructed':True,'rows_and_columns_reconstructed':True,'alternative_pairing_known_optimum':2,'greedy_restricted_known_optimum':0,'greedy_cycle_count':2,'alternative_cycle_optimum':3,'acyclic_exact_columns_optimum':0,'twenty_edge_path_C_TV':.05,'dose_denominators_verified':True,'absolute_growth_not_signed_change':True,'model_operations':0})
 print('ARRAY/GRAPH/MILP SYNTHETIC CHECKS PASSED',flush=True)
if __name__=='__main__':main()
