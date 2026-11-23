"""Single manifest writer, two persistent direct workers; fail closed, no retries."""
from gcommon import *
import subprocess,fcntl,socket,argparse,shutil
HOSTS=('viper.cs.kent.edu','adder.cs.kent.edu')
def ssh(host,command):
 return subprocess.run(['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new',host,command],capture_output=True,text=True)
def check_inputs():
 expected=read(OUT/'registration.json')
 if (OUT/'launch_seal.json').exists():
  for rel,h in read(OUT/'launch_seal.json')['probe_hashes'].items():assert sha(OUT/rel)==h,'Probe hash changed'
  for rel,h in read(OUT/'launch_seal.json')['code'].items():assert sha(ROOT/rel)==h,'Code hash changed'
 assert sha(OUT/'initial_manifest.csv')==expected['initial_manifest_sha256']
 for r in rows(OUT/'input_inventory.csv'):
  if not Path(r['path']).is_file() or sha(r['path'])!=r['expected_sha256']:raise RuntimeError('Missing/hash-mismatched input '+r['path'])
 assert torch.cuda.is_available()
 return dict(host=socket.getfqdn(),verified=True,unix=time.time(),gpu_free_bytes=torch.cuda.mem_get_info()[0])
def coordinator():
 lock=(OUT/'experiment.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(OUT/'validation/source_protocol.json')['passed']
 assert read(OUT/'validation/focused.json')['passed']
 assert read(OUT/'validation/equivalence.json')['passed']
 assert read(OUT/'analysis/initial_analysis_status.json')['complete']
 seal=read(OUT/'launch_seal.json')
 assert shutil.disk_usage(OUT).free>seal['estimated_storage_bytes']+10*1024**3,'Insufficient disk for registered snapshots'
 for rel,h in seal['code'].items():assert sha(ROOT/rel)==h,'Code changed: '+rel
 for host in HOSTS:
  result=ssh(host,f'cd {ROOT} && OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry/cluster.py --check-local')
  save(OUT/'host_checks'/f'{host}.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
  if result.returncode:raise RuntimeError('Host inputs unavailable '+host)
 jobs=rows(OUT/'run_manifest.csv')
 registered={r['condition_id']:r for r in rows(OUT/'initial_manifest.csv')}
 assert set(registered)=={j['condition_id'] for j in jobs} and len(jobs)==75
 for j in jobs:
  assert all(j[k]==registered[j['condition_id']][k] for k in ('seed','router_age','source_id','job_type','mask_type','sparsity','dense_condition_id')),'Mutable manifest changed scientific scope'
 if any(j['status'] in ('RUNNING','FAILED_UNCERTAIN') for j in jobs):raise RuntimeError('Existing uncertain/running attempts require manual remote PID verification; no duplicate workers launched')
 for j in jobs:
  j.setdefault('artifact_path','');j.setdefault('artifact_sha256','')
 active={};run_started=time.monotonic()
 while True:
  for host,(job,process,log) in list(active.items()):
   heartbeat=Path(job['attempt'])/'heartbeat.json'
   if heartbeat.exists():
    hb=read(heartbeat);job['heartbeat']=hb['unix'];job['pid']=hb['pid']
   if process.poll() is not None:
    log.close();done=Path(job['attempt'])/'completed.json'
    if process.returncode==0 and done.exists():job['status']='COMPLETE';job['artifact_path']=job['attempt'];job['artifact_sha256']=read(done)['result']['final_checkpoint_sha256']
    else:job['status']='FAILED';job['error']=f'Worker exit {process.returncode}; preserve partial artifacts; no automatic retry'
    del active[host]
  table(OUT/'run_manifest.csv',jobs)
  for host in HOSTS:
   if host in active:continue
   candidates=[j for j in jobs if j['status']=='PENDING' and (j['job_type']=='dense' or next(x for x in jobs if x['condition_id']==j['dense_condition_id'])['status']=='COMPLETE')]
   if not candidates:continue
   mem=ssh(host,'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits')
   if mem.returncode or int(mem.stdout.strip().splitlines()[0])*1024**2<seal['admission_bytes']:continue
   j=candidates[0];attempt=OUT/'recoveries'/j['condition_id']/'attempt_0001'
   if attempt.exists():j['status']='FAILED_UNCERTAIN';j['error']='Existing attempt; manual review required';continue
   attempt.parent.mkdir(parents=True,exist_ok=True);logpath=OUT/'logs'/f"{j['condition_id']}.log";logpath.parent.mkdir(exist_ok=True);log=logpath.open('a')
   cmd=f'cd {ROOT} && OMP_NUM_THREADS=2 .venv-highres/bin/python -u scripts/router_expert_geometry/worker.py --condition {j["condition_id"]} --attempt {attempt}'
   process=subprocess.Popen(['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new',host,cmd],stdout=log,stderr=subprocess.STDOUT)
   j.update(status='RUNNING',host=host,pid=process.pid,attempt=str(attempt),heartbeat=time.time());active[host]=(j,process,log)
  table(OUT/'run_manifest.csv',jobs);save(OUT/'queue_status.json',dict(counts={s:sum(j['status']==s for j in jobs) for s in sorted({j['status'] for j in jobs})},active={h:dict(condition=j['condition_id'],ssh_pid=p.pid,attempt=j['attempt'],heartbeat=j['heartbeat']) for h,(j,p,l) in active.items()},unix=time.time(),elapsed_seconds=time.monotonic()-run_started))
  if not active and not any(j['status']=='PENDING' for j in jobs):break
  if not active and any(j['status'].startswith('FAILED') for j in jobs):raise RuntimeError('Queue blocked by failed dependency; no retries')
  time.sleep(15)
 subprocess.run([str(ROOT/'.venv-highres/bin/python'),str(Path(__file__).parent/'analyze_trajectories.py')],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check-local',action='store_true');a=p.parse_args()
 if a.check_local:print(json.dumps(check_inputs()))
 else:coordinator()
