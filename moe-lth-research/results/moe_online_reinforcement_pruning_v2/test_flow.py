import itertools
import numpy as np
from audit_match import flow
pool=np.arange(12); score=np.array([.1,.9,.4,.2,.8,.3,.6,.7,.5,.15,.25,.35]); random=np.array([0,2,5,7,8,10]);nr,nc=3,4
chosen,cert=flow(pool,score,random,nr,nc)
row=np.bincount(random//nc,minlength=nr);col=np.bincount(random%nc,minlength=nc)
feasible=[np.array(x) for x in itertools.combinations(pool,len(random)) if np.array_equal(np.bincount(np.array(x)//nc,minlength=nr),row) and np.array_equal(np.bincount(np.array(x)%nc,minlength=nc),col)]
best=max(float(score[x].sum()) for x in feasible)
assert abs(float(score[chosen].sum())-best)<1e-10
assert abs(cert['min_objective']-cert['dual_min_objective'])<1e-10
assert np.array_equal(chosen,flow(pool,score,random,nr,nc)[0])
print('exhaustive flow test passed:',len(feasible),'feasible solutions, optimum',best)
