"""Seal and detach one worker on each established host; no automatic retries."""
from common import *
from af_cluster import SSH

def main():
 with lock(D/'launch.lock',False):
  assert not (D/'launch.json').exists() and not (D/'STOP.json').exists()
  reg=registry();assert read(D/'preparation_complete.json')['registration_sha256']==sha(D/'registration.json')
  for host in HOSTS:assert read(D/'validation'/f'{host}.json')['passed']
  save(D/'numerical_checks.json',dict(passed=True,synthetic_and_resume_checks={h:read(D/'validation'/f'{h}.json') for h in HOSTS},matching_counts_equal=True,all_masks_before_loss=True,registration_sha256=sha(D/'registration.json'),production_engineering_checks='per-attempt initial_integrity.json, result.json, pilot_engineering.json'))
  implementation={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')}
  implementation.update({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/moe_lth').rglob('*.py')})
  save(D/'execution_seal.json',dict(registration_sha256=sha(D/'registration.json'),protocol_sha256=sha(D/'protocol.md'),implementation=implementation,unix=time.time()))
  launches=[]
  for host in HOSTS:
   program="""import pathlib,subprocess,os,json,socket
root=pathlib.Path(%r)
out=root/'results/magnitude_growth_pruning_v1'
(out/'workers').mkdir(parents=True,exist_ok=True)
log=(out/'workers'/f'{socket.getfqdn()}.log').open('a')
p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(root/'scripts/magnitude_growth_pruning_v1/worker.py')],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid)))
""" % str(ROOT)
   command='cd '+shlex.quote(str(ROOT))+' && .venv-highres/bin/python -c '+shlex.quote(program);answer=subprocess.run([*SSH,host,command],capture_output=True,text=True,timeout=40)
   if answer.returncode:
    save(D/'launch.json',dict(partial=True,workers=launches,error=answer.stderr));raise RuntimeError('Partial launch; preserve existing workers')
   launches.append(json.loads(answer.stdout.strip().splitlines()[-1]));save(D/'launch.json',dict(workers=launches,maximum_branches=15,pilot_seed=17,registration_sha256=sha(D/'registration.json')))
  print(json.dumps(launches),flush=True)
if __name__=='__main__':main()
