"""Continue authorized work only while every frozen gate succeeds."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PIDS=(3800653,3800904,3800905,3800906,3800907,3800908)
def alive(pid):
 try:
  stat=(Path('/proc')/str(pid)/'stat').read_text().split();cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  return stat[2]!='Z' and 'audit_match.py' in cmd
 except (OSError,IndexError):return False
def run(script,*args):
 print('START',script,*args,'time',time.time(),flush=True)
 subprocess.run([sys.executable,str(HERE/script),*args],cwd=HERE,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
 print('PASS',script,*args,'time',time.time(),flush=True)
def main():
 print('waiting for matching audit PIDs',PIDS,flush=True)
 while any(alive(pid) for pid in PIDS):time.sleep(30)
 for seed in (7,17,29):
  for order in ('A','B'):
   root=HERE/'matching'/f'seed{seed}_{order}';audit=json.loads((root/'audit.json').read_text())
   if not audit.get('complete') or not (root/'selections.pt').exists():raise RuntimeError(f'Incomplete or failed matching audit: {root}')
 run('verify_selections.py');run('engineering.py');run('launcher.py','pilot');run('verify_pilot.py');run('launcher.py','remaining');run('analyze.py')
if __name__=='__main__':main()
