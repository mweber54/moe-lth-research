"""Exact prospective definitions from the committed Stage1 amendment."""
import math,statistics
LEVEL_THRESHOLD=0.04972095
REOPEN_THRESHOLD=0.02486047
TCRIT=math.sqrt(2*.95**2/(1-.95**2))
def interval(values):
 assert len(values)==3 and all(math.isfinite(x) for x in values)
 mean=statistics.fmean(values);sd=statistics.stdev(values);half=TCRIT*sd/math.sqrt(3)
 return mean,sd,mean-half,mean+half
def seed_result(level_draws,reopening_draws):
 L,Lsd,Llo,Lhi=interval(level_draws);R,Rsd,Rlo,Rhi=interval(reopening_draws)
 practical_L=L>LEVEL_THRESHOLD;practical_R=R>REOPEN_THRESHOLD;direction_L=Llo>0;direction_R=Rlo>0
 lp=practical_L and direction_L;rp=practical_R and direction_R;joint=lp and rp
 return dict(L=L,L_sd=Lsd,L_CI_low=Llo,L_CI_high=Lhi,R=R,R_sd=Rsd,R_CI_low=Rlo,R_CI_high=Rhi,level_practical=practical_L,reopening_practical=practical_R,level_direction=direction_L,reopening_direction=direction_R,level_pass=lp,increase_pass=rp,joint_pass=joint,outcome='joint_pass' if joint else 'level_only' if lp else 'increase_only' if rp else 'neither',draws=3,interval_scope='conditional on source seed, two tested masks and paired fixed orders')
def interim_gate(results):
 assert set(results)=={7,17,29}
 passed=[s for s in (7,17,29) if results[s]['joint_pass']]
 return dict(stage1_gate_pass=len(passed)>=2,same_seeds_passing_both=passed,required_joint_seeds=2,tested_seeds=[7,17,29],training_continuation_authorized=False,stop_after_stage1=True)
def allowed_attempt(previous,next_step):return next_step==previous+1 and 1<=next_step<=2500
