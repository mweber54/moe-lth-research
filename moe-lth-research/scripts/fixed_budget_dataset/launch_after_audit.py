"""Wait for the completed historical audit, then run Dataset A only."""
from common import *
import os,time,subprocess

def main():
 pid=int(os.environ['DATASET_AUDIT_PID']);gate=OUT/'audit_complete.json'
 while not gate.exists():
  state=Path(f'/proc/{pid}/stat')
  if not state.exists() or state.read_text().split()[2]=='Z':raise RuntimeError('Historical audit exited without its completion gate; no Dataset A training launched')
  print('WAITING: historical artifact audit must complete before training',flush=True);time.sleep(30)
 record=jread(gate);assert record['status']=='COMPLETE' and record['grid_rows']==1650 and record['seed_check_passed']
 assert sha(OUT/'canonical_grid_audit.csv')==record['grid_sha256']
 print('AUDIT GATE PASSED; executing only missing Dataset A jobs',flush=True)
 subprocess.run([sys.executable,'-u',str(ROOT/'scripts/fixed_budget_dataset/run.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
