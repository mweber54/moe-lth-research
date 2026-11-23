"""Read-only progress, plus mandatory stop on a sealed-input or storage violation."""
from pathlib import Path
import json,os,hashlib,datetime
OUT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def last_attempt(path):
 if not path.exists():return 0
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-4096));data=f.read()
 for line in reversed(data.splitlines()):
  try:return int(json.loads(line)['attempt'])
  except (ValueError,KeyError,json.JSONDecodeError):continue
 return 0
def progress():
 reg=read(OUT/'registration.json');complete=0;failed=0;active=[];attempts=0
 for path in (OUT/'status').glob('*.json'):
  r=read(path);job=r.get('job_id',path.stem);root=OUT/'jobs'/job;n=last_attempt(root/'attempt_ledger.jsonl');attempts+=n
  if r['status']=='COMPLETE':complete+=1
  elif r['status']=='FAILED':failed+=1
  else:
   heartbeat=root/'heartbeat.json';step=read(heartbeat)['global_step'] if heartbeat.exists() else 0
   active.append(dict(job_id=job,started_attempts=n,completed_step=step,host=r.get('host')))
 written=0
 for p in OUT.rglob('*'):
  try:
   if p.is_file():written+=p.stat().st_size
  except FileNotFoundError:pass
 s=os.statvfs(OUT);free=s.f_bavail*s.f_frsize;remaining=max(0,188628797420.25-written)
 reasons=[]
 if free<remaining:reasons.append('Registered outputs no longer fit with planned allowance')
 if attempts>135000:reasons.append('Attempt cap violation')
 for file,h in reg['code_inputs'].items():
  if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=h:reasons.append('Sealed code changed: '+file)
 for r in __import__('csv').DictReader((OUT/'dispatch_input_hashes.csv').open()):
  if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:reasons.append('Sealed data/order input changed: '+r['path'])
 if reasons and not (OUT/'STOP.json').exists():
  p=OUT/'STOP.monitor.tmp';p.write_text(json.dumps(dict(reason=reasons,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())));p.replace(OUT/'STOP.json')
 result=dict(completed=complete,registered=54,failed=failed,started_attempts=attempts,cap=135000,active=active,free_GB=round(free/1e9,2),remaining_budget_GB=round(remaining/1e9,2),written_GB=round(written/1e9,2),stop=read(OUT/'STOP.json') if (OUT/'STOP.json').exists() else None,completion=read(OUT/'completion.json') if (OUT/'completion.json').exists() else None)
 print(json.dumps(result));return result
if __name__=='__main__':progress()
