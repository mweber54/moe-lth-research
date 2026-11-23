"""Persistent non-training supervisor; no retry or continuation code."""
from pathlib import Path
import sys,os,json,csv,datetime,time,subprocess,socket,fcntl
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=Path.cwd();sys.path.insert(0,str(OUT));from monitor import progress
HOST=socket.getfqdn()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def fail_accounting(reason):
 records=[]
 for p in (OUT/'status').glob('*.json'):
  status=json.loads(p.read_text());job=status.get('job_id',p.stem);root=OUT/'jobs'/job;ledger=root/'attempt_ledger.jsonl';started=0
  if ledger.exists():
   for line in ledger.read_text().splitlines():
    try:started=max(started,int(json.loads(line)['attempt']))
    except (ValueError,KeyError,json.JSONDecodeError):pass
  native=root/'segment_0/gradient_stats/gradient_stats.jsonl';completed=[]
  if native.exists():
   for line in native.read_text().splitlines():
    try:completed.append(json.loads(line))
    except json.JSONDecodeError:pass
  records.append(dict(job_id=job,status=status['status'],started_attempts=started,logged_completed_attempts=len(completed),applied_updates=sum(bool(r['optimizer_step_applied']) for r in completed),amp_skipped=sum(r['measurement_status']=='amp_overflow_skipped' for r in completed),reason=status.get('reason',''),as_of_utc=now()))
 if records:
  with (OUT/'failure_accounting.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 (OUT/'findings.md').write_text('Stage 1 stopped before completing the registered cohort. No retry, extra arm, seed or continuation was dispatched. The gate is not evaluated from incomplete data.\n\nReason: '+str(reason)+'\n\nActual durable attempted-update records and logged completed/applied/skipped updates are in [failure_accounting.csv](failure_accounting.csv): `job_id`, `started_attempts`, `logged_completed_attempts`, `applied_updates`, `amp_skipped`, `reason`, `as_of_utc`. In-progress counts are explicitly timestamped.\n')

def main():
 with (OUT/'locks/supervisor.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  save(OUT/'supervisor.json',dict(pid=os.getpid(),host=HOST,started_utc=now(),role='monitor only; never trains or dispatches'))
  while True:
   result=progress();save(OUT/'progress.json',dict(**result,checked_utc=now()))
   if result['stop']:
    save(OUT/'supervisor_stop_record.json',dict(stop=result['stop'],observed_utc=now()))
    time.sleep(10);fail_accounting(result['stop']);return 1
   if result['completion']:
    launched=json.loads((OUT/'launch.json').read_text())['workers']
    if all((OUT/'workers'/f"{r['host']}_accounting.json").exists() for r in launched):
     # Rebuild once both workers' final slot accounting exists; no training calls.
     with (OUT/'locks/report.lock').open('a') as reportlock:
      fcntl.flock(reportlock.fileno(),fcntl.LOCK_EX)
      subprocess.run([sys.executable,str(OUT/'report.py')],cwd=ROOT,check=True)
     final=progress();save(OUT/'progress.json',dict(**final,checked_utc=now()));save(OUT/'supervisor_completion.json',dict(completed_utc=now(),completed_runs=54,attempts=135000,stop_after_stage1=True,continuation_authorized=False));return 0
   # Detect an unexpected local worker death without inventing a retry.
   for worker in json.loads((OUT/'launch.json').read_text())['workers']:
    if worker['host']!=HOST or (OUT/'workers'/f"{worker['host']}_accounting.json").exists():continue
    try:os.kill(int(worker['pid']),0)
    except ProcessLookupError:
     save(OUT/'STOP.json',dict(reason='Registered worker exited without final accounting; no automatic retry',worker=worker,utc=now()))
   time.sleep(50)
if __name__=='__main__':
 try:sys.exit(main())
 except BaseException as exc:
  if isinstance(exc,SystemExit):raise
  save(OUT/'STOP.json',dict(reason='Supervisor failure: '+repr(exc),utc=now()));raise
