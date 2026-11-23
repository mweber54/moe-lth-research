"""Serial GPU launcher; will not run before explicit evidence gates exist."""
import argparse,fcntl,json,os,socket,subprocess,sys,time,uuid
from pathlib import Path
import status
HERE=Path(__file__).resolve().parent
ARMS=status.ARMS
def write(path,value):
 tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)
def admitted():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[tuple(int(x.strip()) for x in line.split(',')) for line in text.splitlines()];item=next(x for x in rows if x[0]==0)
 return item,item[1]>=24576 and item[2]<=90
def main():
 parser=argparse.ArgumentParser();parser.add_argument('scope',choices=('pilot','remaining'));args=parser.parse_args()
 if not (HERE/'sealed_inputs.json').exists() or not (HERE/'engineering_gates.json').exists():raise RuntimeError('selection seal and verified engineering gates are required')
 if args.scope=='remaining':
  if not (HERE/'pilot_verified.json').exists():raise RuntimeError('verified pilot comparison required')
  for arm in ARMS:
   if status.status(HERE/'branches'/'seed7_A'/arm)!='COMPLETED':raise RuntimeError('verified seed7_A pilot is required')
 cells=[(7,'A')] if args.scope=='pilot' else [(s,o) for s in (7,17,29) for o in ('A','B') if (s,o)!=(7,'A')]
 with (HERE/'gpu0.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for seed,order in cells:
   for arm in ARMS:
    root=HERE/'branches'/f'seed{seed}_{order}'/arm;state=status.status(root)
    if state=='COMPLETED':continue
    if state=='RUNNING':raise RuntimeError(f'branch already active: {root}')
    while True:
     gpu,ok=admitted();print('GPU admission',gpu,flush=True)
     if ok:break
     time.sleep(30)
    root.mkdir(parents=True,exist_ok=True);attempt=time.time_ns();log=root/f'attempt_{attempt}.log'
    with log.open('w') as stream:
     cmd=[sys.executable,str(HERE/'branch.py'),str(seed),order,arm,'--mode','production'];proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1'})
     write(root/'running.json',{'pid':proc.pid,'host':socket.getfqdn(),'started_unix':time.time(),'log':str(log),'gpu_admission':gpu})
     print(f'launched pid={proc.pid} seed={seed} order={order} arm={arm} log={log}',flush=True)
     code=proc.wait()
    (root/'running.json').unlink(missing_ok=True)
    if code or status.status(root)!='COMPLETED':
     write(root/'failed.json',{'exit_code':code,'attempt':attempt,'log':str(log),'time':time.time()});raise RuntimeError(f'branch failed or did not verify: {root}')
    (root/'failed.json').unlink(missing_ok=True)
if __name__=='__main__':main()
