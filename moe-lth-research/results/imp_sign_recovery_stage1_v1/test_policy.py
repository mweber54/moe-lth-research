from pathlib import Path
import ast,math,csv
from gate import seed_result,interim_gate,interval,allowed_attempt
checks=[]
def check(name,condition):
 assert condition,name;checks.append(dict(check=name,passed=True,training_updates=0))
joint=seed_result([.11,.12,.13],[.04,.041,.042]);level=seed_result([.1,.1,.1],[0.,0.,0.]);increase=seed_result([0.,0.,0.],[.04,.04,.04]);neither=seed_result([0.,0.,0.],[0.,0.,0.])
check('joint_criteria',joint['joint_pass']);check('level_only',level['outcome']=='level_only');check('increase_only',increase['outcome']=='increase_only');check('neither',neither['outcome']=='neither')
check('same_seed_intersection_required',not interim_gate({7:joint,17:level,29:increase})['stage1_gate_pass'])
check('two_joint_seeds_pass',interim_gate({7:joint,17:joint,29:neither})['stage1_gate_pass'])
late=[.2,.3,.4];early=[.19,.29,.39];paired=[a-b for a,b in zip(late,early)];r=interval(paired)
check('paired_difference_interval_preserves_covariance',abs(r[0]-.01)<1e-12 and r[1]<1e-12)
check('cap_2500_and_no_replay',allowed_attempt(2499,2500) and not allowed_attempt(2500,2501) and not allowed_attempt(10,10) and not allowed_attempt(10,12))
tree=ast.parse(Path(__file__).with_name('run.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_run_segment']
check('only_native_segment0_call',len(calls)==1 and ast.literal_eval(calls[0].args[8])==0 and ast.literal_eval(calls[0].args[9])=='segment_0')
with Path(__file__).with_name('policy_checks.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
print(len(checks),'policy checks passed; zero training updates')
