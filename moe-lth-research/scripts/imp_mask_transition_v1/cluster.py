"""Existing SSH transport, exact-cohort sealing and detached single-host workers."""
from common import *
import argparse,subprocess,shlex,shutil
SSH=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new']

def seal():
 if (OUT/'seal.json').exists():raise RuntimeError('Preserve prior seal; engineering correction needs a versioned record')
 assert read(OUT/'cpu_validation.json')['passed']
 assert read(OUT/'audit_supplement.json')['passed']
 for host in HOSTS:
  assert read(OUT/'validation'/f'{host}.json')['passed']
  assert read(OUT/'benchmarks'/f'{host}.json')['attempts']==100
 save(OUT/'seal.json',dict(files={n:sha(OUT/n) for n in ('registration.json','protocol.md','manifest.json','input_hashes.json','supplemental_input_hashes.json','event_schedule.csv','batch_manifest.csv','artifact_inventory.csv','condition_manifest.csv','reference_compatibility.csv','available_states.csv','historical_aliases.csv','cpu_validation.json','audit_supplement.json')},implementation=implementation(),unix=time.time()))

def launch():
 manifest=frozen()
 if (OUT/'launch.json').exists():raise RuntimeError('Launch exists; inspect workers rather than duplicate them')
 assert not (OUT/'STOP.json').exists()
 assert shutil.disk_usage(OUT).free>read(OUT/'resource_estimate.json')['estimated_storage_bytes']
 short_bench=[t for host in HOSTS for t in read(OUT/'validation'/f'{host}.json')['timing']]
 bench=[read(OUT/'benchmarks'/f'{host}.json') for host in HOSTS]
 admission=max(24*1024**3,int(1.25*max(t['peak_allocated_bytes'] for t in bench)))
 estimate=read(OUT/'resource_estimate.json');estimate.update(benchmark=bench,admission_bytes=admission,
   estimated_gpu_hours=10*5000*np.median([t['wall_seconds']/t['attempts'] for t in bench])*1.2/3600,
   canonical_equivalence_and_event_benchmarks=short_bench,estimate_includes_startup_and_checkpoint_overhead=True,diagnostic_overhead_allowance=.2)
 save(OUT/'resource_estimate.json',estimate)
 launches=[]
 # Launch manifest exists before the first worker can claim a job.
 save(OUT/'launch.json',dict(workers=launches,admission_bytes=admission,new_training_jobs=10,reused_controls=10,registered_endpoints=20,manifest_sha256=sha(OUT/'manifest.json'),unix=time.time()))
 for host in HOSTS:
  program="""import pathlib,subprocess,os,json,socket,time
root=pathlib.Path(%r)
out=root/'results/imp_mask_transition_v1/workers';out.mkdir(parents=True,exist_ok=True)
log=(out/(socket.getfqdn()+'.log')).open('a')
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(root/'scripts/imp_mask_transition_v1/worker.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid,unix=time.time())))
"""%str(ROOT)
  command=f'cd {shlex.quote(str(ROOT))} && .venv-highres/bin/python -c {shlex.quote(program)}'
  answer=subprocess.run([*SSH,host,command],capture_output=True,text=True,timeout=30);assert answer.returncode==0,answer.stderr
  launches.append(json.loads(answer.stdout.strip().splitlines()[-1]));launch=read(OUT/'launch.json');launch['workers']=launches;save(OUT/'launch.json',launch)
 print(json.dumps(launches),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
 with lock(OUT/'launch.lock',False):
  if a.seal:seal()
  if a.launch:launch()
