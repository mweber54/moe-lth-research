"""Single dispatcher; scientific code/seal unchanged; no non-study process control."""
import os,sys,time,json,subprocess,shutil,shlex,fcntl
from pathlib import Path
root=Path('/users/kent/student/mweber54/moe-lth-research/moe-lth-research');out=root/'results/imp_first_moe_direction_v1'
ssh=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new']
query=['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits']
lock=(out/'resource_dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
started=time.monotonic();last_notice=0
while time.monotonic()-started<43200:
    observed={};candidate=None
    for host in ('viper.cs.kent.edu','adder.cs.kent.edu'):
        try:
            result=subprocess.run(query if host.startswith('viper') else ssh+[host,shlex.join(query)],capture_output=True,text=True,timeout=25,check=True)
            free,util=(int(v.strip()) for v in result.stdout.strip().splitlines()[0].split(','));observed[host]=dict(free_mib=free,utilization_percent=util)
            if free>=24*1024 and util<=90:candidate=host;break
        except Exception as exc:observed[host]=dict(error=repr(exc))
    if candidate is None:
        if time.monotonic()-last_notice>=50:
            print(json.dumps(dict(status='WAITING_FOR_EXISTING_ADMISSION',hosts=observed)),flush=True);last_notice=time.monotonic()
            p=out/'resource_wait_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status='WAITING_FOR_EXISTING_ADMISSION',scientifically_complete=False,hosts=observed,specification_sha256=__import__('hashlib').sha256((out/'specification_seal.json').read_bytes()).hexdigest()))+'\n');tmp.replace(p)
        time.sleep(5);continue
    print(json.dumps(dict(status='CANDIDATE_HOST',host=candidate,observed=observed)),flush=True)
    with (out/'execution_revision2.log').open('a') as log:
        if candidate.startswith('viper'):
            command=[str(root/'.venv/bin/python'),'-u',str(root/'scripts/imp_first_moe_direction_v1/run.py'),'--evaluate'];env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        else:
            command=ssh+[candidate,'cd '+shlex.quote(str(root))+' && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u scripts/imp_first_moe_direction_v1/run.py --evaluate'];env=None
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    status=json.loads((out/'completion.json').read_text())
    if status.get('status')=='RESOURCE_BLOCKED':
        shutil.copy2(out/'completion.json',Path(status['attempt'])/'completion.json');print('ADMISSION_RACE_PRESERVED',flush=True);time.sleep(5);continue
    print(json.dumps(dict(status='WORKER_FINISHED',worker_status=status.get('status'),returncode=result.returncode)),flush=True);sys.exit(result.returncode)
print('RESOURCE_WAIT_LIMIT_REACHED',flush=True);sys.exit(3)
