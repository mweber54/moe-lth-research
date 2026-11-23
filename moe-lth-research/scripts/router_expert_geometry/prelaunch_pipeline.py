"""Complete fixed pre-launch analysis then launch gated production, no scope adaptation."""
from gcommon import *
from cluster import ssh,HOSTS
import subprocess,fcntl

def main():
 lock=(OUT/'prelaunch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Separate per-host analysis chains, each strictly sequential. No source training.
 commands={HOSTS[0]:(7,29,61),HOSTS[1]:(17,43)};processes=[]
 for host,seeds in commands.items():
  code='\n'.join(f'OMP_NUM_THREADS=2 .venv-highres/bin/python -u scripts/router_expert_geometry/analyze_existing.py --seed {seed} || exit $?' for seed in seeds)
  cmd=f'cd {ROOT} || exit $?\n{code}'
  log=(OUT/f'prelaunch_{host}.log').open('a')
  proc=subprocess.Popen(['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new',host,cmd],stdout=log,stderr=subprocess.STDOUT);processes.append((host,proc,log))
 save(OUT/'prelaunch_status.json',dict(state='ANALYZING',processes={h:p.pid for h,p,l in processes},unix=time.time()))
 for host,proc,log in processes:
  if proc.wait():raise RuntimeError('Initial analysis failed on '+host)
  log.close()
 subprocess.run([str(ROOT/'.venv-highres/bin/python'),str(Path(__file__).parent/'summarize.py')],check=True)
 if not read(OUT/'validation/equivalence.json')['passed']:raise RuntimeError('GPU validation missing or failed')
 benchmark=read(OUT/'validation/equivalence.json');margin=4*1024**3
 code={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
 for folder in (ROOT/'src/moe_lth',ROOT/'scripts/nominal_exposure_dataset'):
  code.update({str(p.relative_to(ROOT)):sha(p) for p in folder.rglob('*.py')})
 code['scripts/fixed_budget_dataset/common.py']=sha(ROOT/'scripts/fixed_budget_dataset/common.py')
 save(OUT/'launch_seal.json',dict(code=code,admission_bytes=benchmark['peak_reserved_bytes']+margin,measured_peak_reserved_bytes=benchmark['peak_reserved_bytes'],memory_margin_bytes=margin,registered_input_hashes=sha(OUT/'input_inventory.csv'),probe_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'probes').glob('*.pt')},estimated_storage_bytes=benchmark['estimated_snapshot_storage_bytes'],diagnostic_seconds_estimate=benchmark['estimated_total_diagnostic_seconds']))
 save(OUT/'prelaunch_status.json',dict(state='LAUNCHING',unix=time.time()))
 subprocess.run([str(ROOT/'.venv-highres/bin/python'),'-u',str(Path(__file__).parent/'cluster.py')],check=True)
if __name__=='__main__':main()
