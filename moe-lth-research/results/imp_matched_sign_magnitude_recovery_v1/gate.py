import math,statistics
MARGIN=0.02486047
TCRIT=4.30265272974946
SCOPE='conditional on tested source seed, three fixed masks and paired A/B orders'
def interval(values):
 assert len(values)==3 and all(math.isfinite(x) for x in values)
 mean=statistics.fmean(values);sd=statistics.stdev(values);h=TCRIT*sd/math.sqrt(3)
 return dict(mean=mean,sd=sd,CI_low=mean-h,CI_high=mean+h)
def seed_result(c,m,b):
 result={}
 for label,values in [('C',c),('R_M',m),('R_Bm',b)]:
  result.update({label+'_'+k:v for k,v in interval(values).items()})
 result['direct_pass']=result['C_mean']>MARGIN and result['C_CI_low']>0
 for a in ('M','Bm'):result[a+'_reopens']=result['R_'+a+'_mean']>MARGIN and result['R_'+a+'_CI_low']>0
 result['M_near_zero']=result['R_M_CI_low']>=-MARGIN and result['R_M_CI_high']<=MARGIN
 result['arm_outcome']='both_reopen' if result['M_reopens'] and result['Bm_reopens'] else 'Bm_only' if result['Bm_reopens'] else 'M_only' if result['M_reopens'] else 'neither_reopens'
 result.update(draws=3,interval_scope=SCOPE,practical_margin=MARGIN)
 return result
def consistency(results):
 assert set(results)=={7,17,29}
 passed=[s for s in (7,17,29) if results[s]['direct_pass']]
 positive=all(results[s]['C_mean']>0 for s in results)
 return dict(consistency_pass=len(passed)>=2 and positive,seeds_direct_pass=passed,all_three_point_estimates_positive=positive,required_seeds=2,continuation_authorized=False)
