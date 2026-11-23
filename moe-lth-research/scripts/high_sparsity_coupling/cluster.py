"""Detached, lock-protected disjoint seed workers and both-host artifact verification."""
from common import *
import subprocess,argparse,fcntl,socket,traceback
HOSTS={'viper.cs.kent.edu':[7,17,29],'adder.cs.kent.edu':[43,61]}
SSH=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new']
PYTHON=str(ROOT/'.venv-highres/bin/python')
def implementation_hashes():
 paths=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src/moe_lth/models').glob('*.py'))+list((ROOT/'src/moe_lth/pruning').glob('*.py'))+[ROOT/'src/moe_lth/data.py',ROOT/'src/moe_lth/utils.py',ROOT/'src/moe_lth/experiments/run_router_age_recovery.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def run_script(script,args):
 if (OUT/'launch.json').exists():
  expected=read(OUT/'launch.json').get('implementation',{})
  assert all(sha(ROOT/name)==value for name,value in expected.items()),'Production dependency changed'
 subprocess.run([PYTHON,str(ROOT/'scripts/high_sparsity_coupling'/script),*args],cwd=ROOT,check=True)
def local_worker(seeds):
 import signal,threading
 stop=threading.Event()
 def heart():
  while not stop.is_set():
   save(OUT/'workers'/f'{socket.gethostname()}.heartbeat.json',dict(pid=os.getpid(),seeds=seeds,unix=time.time(),host=socket.gethostname()));stop.wait(15)
 d=OUT/'workers';d.mkdir(exist_ok=True)
 with (d/f'{socket.gethostname()}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);threading.Thread(target=heart,daemon=True).start()
  try:
   launch=read(OUT/'launch.json')
   assert all(sha(Path(__file__).parent/name)==value for name,value in launch['code'].items()),'Implementation changed since launch'
   assert read(OUT/'validation/equivalence.json')['passed'];assert (OUT/'probe_config.json').exists();frozen()
   for seed in seeds:
    # Source and historical analysis is executed before rerunning each seed.
    run_script('analyze.py',['--seeds',str(seed)])
    with (OUT/'report.lock').open('a') as report_lock:
     fcntl.flock(report_lock,fcntl.LOCK_EX);run_script('report.py',[])
    run_script('runner.py',['--seeds',str(seed)])
    run_script('analyze.py',['--seeds',str(seed),'--track','rerun'])
    # Shared report mutation is serialized independently of condition dispatch.
    with (OUT/'report.lock').open('a') as report_lock:
     fcntl.flock(report_lock,fcntl.LOCK_EX);run_script('report.py',[])
   save(d/f'{socket.gethostname()}.completed.json',dict(seeds=seeds,unix=time.time()))
  except BaseException as e:
   save(d/f'{socket.gethostname()}.failed.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));raise
  finally:stop.set()
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--seeds',nargs='+',type=int);a=p.parse_args()
 if a.worker:local_worker(a.seeds);return
 if a.verify:
  processes=[]
  for host in HOSTS:
   log=(OUT/f'verify_{host}.log').open('a');cmd=f'cd {ROOT} && OMP_NUM_THREADS=2 {PYTHON} scripts/high_sparsity_coupling/audit.py --verify-local';processes.append((host,subprocess.Popen([*SSH,host,cmd],stdout=log,stderr=subprocess.STDOUT),log))
  for host,process,log in processes:
   code=process.wait();log.close()
   if code:raise RuntimeError(f'Host verification failed {host}: {code}')
  return
 if a.launch:
  frozen();assert read(OUT/'validation/equivalence.json')['passed']
  save(OUT/'launch.json',dict(hosts=HOSTS,unix=time.time(),protocol=sha(OUT/'protocol.md'),cohort=sha(OUT/'cohort.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},implementation=implementation_hashes()))
  for host,seeds in HOSTS.items():
   check=read(OUT/'hosts'/f'{host}.json');assert check['verified'] and check['torch'].startswith('2.7.0') and check['cuda']
   eq=read(OUT/'validation'/f'equivalence_{host}.json');assert eq['passed']
   memory=subprocess.run([*SSH,host,'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
   free=int(memory.stdout.strip().splitlines()[0])*1024**2;required=max(x['peak_cuda_allocated_bytes'] for x in eq['benchmark_segments'])*1.25
   assert free>required,f'Insufficient current memory on {host}: {free} <= {required}'
   log=OUT/'workers'/f'{host}.log';log.parent.mkdir(exist_ok=True);cmd=f'cd {ROOT}\nnohup env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 {PYTHON} scripts/high_sparsity_coupling/cluster.py --worker --seeds '+ ' '.join(map(str,seeds))+f' > {log} 2>&1 < /dev/null &\nexit 0'
   r=subprocess.run([*SSH,host,cmd],capture_output=True,text=True,timeout=30)
   if r.returncode:raise RuntimeError(r.stderr)
  save(OUT/'launch.json',dict(hosts=HOSTS,unix=time.time(),protocol=sha(OUT/'protocol.md'),cohort=sha(OUT/'cohort.json'),code={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},implementation=implementation_hashes()))
if __name__=='__main__':main()
