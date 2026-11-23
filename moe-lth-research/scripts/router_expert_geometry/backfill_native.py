"""Fixed five-seed historical native probes; reuse raw weight/frozen tables."""
from gcommon import *
from cluster import HOSTS
import subprocess
if __name__=='__main__':
 processes=[]
 for seed,host in zip(SEEDS,(HOSTS[0],HOSTS[1],HOSTS[0],HOSTS[1],HOSTS[0])):
  log=(OUT/f'native_backfill_seed{seed}.log').open('a');cmd=f'cd {ROOT} && OMP_NUM_THREADS=2 .venv-highres/bin/python -u scripts/router_expert_geometry/analyze_existing.py --seed {seed}'
  p=subprocess.Popen(['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','UserKnownHostsFile=/tmp/geometry_known_hosts',host,cmd],stdout=log,stderr=subprocess.STDOUT);processes.append((seed,p,log))
 for seed,p,log in processes:
  if p.wait():raise RuntimeError('Native historical probe failure seed '+str(seed))
  log.close()
 subprocess.run([str(ROOT/'.venv-highres/bin/python'),str(Path(__file__).parent/'summarize.py')],check=True)
 verified=read(OUT/'validation/equivalence_full_callback.json');assert verified['passed'];save(OUT/'validation/equivalence.json',verified)
 save(OUT/'analysis/native_backfill_completed.json',dict(seeds=SEEDS,unix=time.time()))
