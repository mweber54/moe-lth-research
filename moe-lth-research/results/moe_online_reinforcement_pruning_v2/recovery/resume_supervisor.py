"""Detached supervisor: resume the interrupted branch and finish remaining cells."""
import fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def main():
 with (ROOT/'recovery/supervisor.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
  for cmd in ([sys.executable,str(ROOT/'launcher.py'),'remaining'],[sys.executable,str(ROOT/'analyze.py')]):
   print('START',cmd,'unix',time.time(),flush=True)
   subprocess.run(cmd,cwd=ROOT,env=env,check=True)
   print('PASS',cmd,'unix',time.time(),flush=True)
  print('ALL STUDY BRANCHES VERIFIED AND ANALYSIS COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();sys.exit(1)
