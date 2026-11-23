"""Meaningful paired statistics, matching, scope/cap checks; no recovery/training."""
from pathlib import Path
import ast,csv,math
import run
from gate import seed_result,consistency,interval
checks=[]
def check(name,condition):
 assert condition,name;checks.append(dict(check=name,passed=True,training_updates=0))
x=seed_result([.04,.041,.042],[0,0,0],[.04,.041,.042]);zero=seed_result([0,0,0],[0,0,0],[0,0,0]);negative=seed_result([-.01]*3,[0]*3,[-.01]*3)
check('positive_direct_contrast',x['direct_pass'])
check('two_positive_and_third_zero_does_not_pass',not consistency({7:x,17:x,29:zero})['consistency_pass'])
check('all_positive_two_pass',consistency({7:x,17:x,29:seed_result([.001]*3,[0]*3,[.001]*3)})['consistency_pass'])
check('mixed_signs_fail_consistency',not consistency({7:x,17:x,29:negative})['consistency_pass'])
check('exact_zero_equivalence',zero['M_near_zero'])
# Two significant arms do not establish an arm difference.
check('direct_not_separate_arm_thresholds',not seed_result([.001]*3,[.1]*3,[.101]*3)['direct_pass'])
late=[.2,.3,.4];early=[.19,.29,.39];r=interval([a-b for a,b in zip(late,early)])
check('paired_covariance_retained',abs(r['mean']-.01)<1e-12 and r['sd']<1e-12)
# Exercise exact matching implementation on a realistic large vector including fixed zero survivors on CPU.
t=run.torch;v=t.cat([t.linspace(.001,.1,100000),-t.linspace(.001,.1,100000),t.zeros(3)])
pair,seeds,hashes,k=run.perturb_pair(v,7,'imp',1,'synthetic.fc1.weight')
en=v.double().square().sum();dist={a:float((z.double()-v.double()).norm()/en.sqrt()) for a,z in pair.items()}
check('nearest_prefix_distance_match',abs(dist['Bm']-dist['M'])<=.001)
check('sign_counts_and_zero_positions_fixed',t.equal(pair['Bm'].sign().sort().values,v.sign().sort().values) and t.equal(pair['Bm'][v==0],v[v==0]))
check('M_preserves_sign_and_Bm_preserves_magnitude',t.equal(pair['M'].sign(),v.sign()) and t.equal(pair['Bm'].abs(),v.abs()))
check('RNG_reproducible',all(t.equal(pair[a],run.perturb_pair(v,7,'imp',1,'synthetic.fc1.weight')[0][a]) for a in pair))
tree=ast.parse(Path(__file__).with_name('run.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_run_segment']
check('only_segment0_native_call',len(calls)==1 and ast.literal_eval(calls[0].args[8])==0 and ast.literal_eval(calls[0].args[9])=='segment_0')
code=Path(__file__).with_name('run.py').read_text();check('strict_no_replay_attempt_guard','step==count+1 and step<=2500' in code and "attempt.mkdir(parents=True,exist_ok=False)" in code)
with Path(__file__).with_name('policy_checks.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
print(len(checks),'checks passed; zero training updates')
