"""Run fresh-process restart tests under fresh GPU admission and save evidence."""
import fcntl,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 if not (HERE/'sealed_inputs.json').exists():raise RuntimeError('all six selection pairs must first be independently sealed')
 records=[]
 with (HERE/'gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for seed in (7,17,29):
   for order in ('A','B'):
    while True:
     out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
     gpu=next(tuple(int(x.strip()) for x in l.split(',')) for l in out.splitlines() if int(l.split(',')[0])==0)
     if gpu[1]>=24576 and gpu[2]<=90:break
     print('admission denied, retrying:',gpu,flush=True);time.sleep(30)
    log=HERE/'engineering'/f'oracle_{seed}_{order}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(HERE/'branch.py'),str(seed),order,'untouched','--mode','oracle']
    with log.open('w') as f:
     p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'})
    if p.returncode:raise RuntimeError(f'oracle failed: {log}')
    result=json.loads(log.read_text().splitlines()[-1]);assert all(result['oracle_checks'].values()) and result['batch_index']==250
    records.append({'arm':'untouched','phase':'oracle','seed':seed,'order':order,'log':str(log),'log_sha256':sha(log),'result':result,'admission':gpu})
    print(seed,order,'oracle passed',flush=True)
  for arm in ('untouched','random-small','high-reinforcement-small'):
   for phase in ('baseline','resume'):
    while True:
     out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
     gpu=next(tuple(int(x.strip()) for x in l.split(',')) for l in out.splitlines() if int(l.split(',')[0])==0)
     if gpu[1]>=24576 and gpu[2]<=90:break
     print('admission denied, retrying:',gpu,flush=True);time.sleep(30)
    log=HERE/'engineering'/f'{arm}_{phase}_{time.time_ns()}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(HERE/'branch.py'),'7','A',arm,'--mode','engineering','--phase',phase]
    with log.open('w') as f:
     p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'})
    if p.returncode:raise RuntimeError(f'engineering check failed: {log}')
    lines=log.read_text().splitlines();result=json.loads(lines[-1]);assert result['phase']==phase and result['global_step']==252
    if phase=='resume':assert all(result['checks'].values())
    records.append({'arm':arm,'phase':phase,'log':str(log),'log_sha256':sha(log),'result':result,'admission':gpu})
    print(arm,phase,'passed',flush=True)
 baseline=[r['result'] for r in records if r['phase']=='baseline']
 if len(baseline)!=3 or max(x['pre_mask_ce'] for x in baseline)-min(x['pre_mask_ce'] for x in baseline)>1e-12:raise RuntimeError('initial branch evaluations disagree')
 if any('step251_ce' not in x for x in baseline):raise RuntimeError('step 251 must be evaluated separately')
 proof={'branch_sha256':sha(HERE/'branch.py'),'sealed_inputs_sha256':sha(HERE/'sealed_inputs.json'),'checks':records,'created_unix':time.time()}
 tmp=HERE/('.engineering.'+uuid.uuid4().hex);tmp.write_text(json.dumps(proof,indent=2)+'\n');os.replace(tmp,HERE/'engineering_gates.json')
if __name__=='__main__':main()
