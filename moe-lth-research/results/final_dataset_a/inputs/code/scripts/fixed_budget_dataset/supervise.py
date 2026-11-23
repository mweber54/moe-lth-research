"""Keep the audited Dataset A runner alive across ordinary process interruptions."""
from common import *
import argparse,fcntl,os,subprocess,time,traceback
import pandas as pd

def is_process(pid,suffix):
 try:
  p=Path(f'/proc/{pid}')
  return p.joinpath('stat').read_text().split()[2]!='Z' and any(x.endswith(suffix.encode()) for x in p.joinpath('cmdline').read_bytes().split(b'\0'))
 except (FileNotFoundError,ProcessLookupError):return False

def frozen_valid():
 p=OUT/'dataset_manifest.json'
 if not p.exists() or not jread(p).get('frozen'):return False
 d=jread(p);assert d['row_count']==1650 and sha(OUT/'router_conditioned_sparse_recoverability_equal_global_budget.csv')==d['sha256']
 j=pd.read_csv(OUT/'run_manifest.csv');assert len(j)==1950 and j.status.isin(['REUSED','COMPLETE']).all()
 assert j.groupby('job_type').size().to_dict()=={'dense':275,'source':25,'sparse':1650}
 assert not j.condition_id.duplicated().any()
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--runner-pid',type=int,required=True);p.add_argument('--progress-pid',type=int,default=0);args=p.parse_args()
 lock=(OUT/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 pid=args.runner_pid;progress_pid=args.progress_pid;child=None;progress_child=None;restarts=0
 while True:
  if frozen_valid():
   jwrite(OUT/'supervisor_status.json',{'status':'COMPLETE','dataset_frozen':True,'restarts':restarts});print('DATASET A FROZEN AND VERIFIED; supervisor stopping',flush=True);return
  jobs_complete=pd.read_csv(OUT/'run_manifest.csv').status.isin(['REUSED','COMPLETE']).all()
  if not jobs_complete and not is_process(progress_pid,'fixed_budget_dataset/progress.py'):
   if progress_child is not None:progress_child.wait()
   failure=OUT/'incremental_manifest_failure.json'
   if failure.exists():
    print('EXPORTER REQUIRES DIAGNOSIS',jread(failure)['error'],flush=True)
   else:
    with (OUT/'progress_execution.log').open('a') as log:
     progress_child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/fixed_budget_dataset/progress.py'),'--watch'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','PYTHONPATH':str(ROOT/'src')})
    progress_pid=progress_child.pid
  if is_process(pid,'fixed_budget_dataset/run.py'):
   jwrite(OUT/'supervisor_status.json',{'status':'RUNNING','runner_pid':pid,'restarts':restarts,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});time.sleep(30);continue
  if child is not None:child.wait()
  failure=OUT/'build_failure.json';record=jread(failure) if failure.exists() else {}
  error=record.get('error','Process interrupted without a failure record')
  # Scientific identity/protocol assertions require diagnosis, never altered settings or blind retries.
  if any(x in error for x in ('AssertionError','Storage reserve','Historical overlap','Reconstructed source differs','Integrity violation','Checkpoint/model mismatch')):
   jwrite(OUT/'supervisor_status.json',{'status':'NEEDS_DIAGNOSIS','error':error,'restarts':restarts,'dataset_frozen':False});raise RuntimeError(error)
  if failure.exists():
   archive=OUT/'failure_history';archive.mkdir(exist_ok=True);failure.rename(archive/f'failure_{time.time_ns()}.json')
  restarts+=1
  print('RESUME FROM EXISTING MANIFEST',restarts,error,flush=True)
  # No audit call. Complete/reused artifacts remain untouched; only incomplete jobs restart.
  with (OUT/'execution.log').open('a') as log:
   child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/fixed_budget_dataset/run.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','PYTHONPATH':str(ROOT/'src')})
  pid=child.pid;time.sleep(30)
if __name__=='__main__':main()
