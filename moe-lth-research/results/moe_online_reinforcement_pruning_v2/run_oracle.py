"""Admission and single-device lock for the untouched engineering oracle."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
with (HERE/'gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[tuple(int(x.strip()) for x in line.split(',')) for line in raw.splitlines()]
 device=next(x for x in rows if x[0]==0)
 if device[1]<24576 or device[2]>90:raise RuntimeError(f'GPU admission denied: {device}')
 print('admission',device,'time',time.time(),flush=True)
 seed,order=(sys.argv[1:3] if len(sys.argv)==3 else ('7','A'))
 cmd=[sys.executable,str(HERE/'branch.py'),seed,order,'untouched','--mode','oracle']
 p=subprocess.run(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'});sys.exit(p.returncode)
