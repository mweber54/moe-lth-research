"""Established Viper/Adder dispatch pattern, with numerical-only seed17 gate."""
from common import *
from run import run
import contextlib,shutil

def pilot_gate():
 p=D/'pilot_engineering.json'
 if p.exists():return read(p)['passed']
 states=[read(D/'status'/f'seed17_{b}.json') for b in BRANCHES]
 if not all(s['status']=='COMPLETE' for s in states):return False
 results=[read(s['result_path']) for s in states]
 assert all(r['attempted_updates']==1500 and r['applied_updates']+r['amp_skips']==1500 and r['amp_skips']<=100 and r['mask_persistence'] and r['removed_optimizer_moments_zero'] for r in results)
 save(p,dict(passed=True,pilot_seed=17,branches=3,effect_signs_used=False,numerical_checks_only=True,peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in results),maximum_wall_seconds=max(r['wall_seconds'] for r in results),updated=time.time()))
 return True

def refresh():
 from report import analyze
 with lock(D/'report.lock'):analyze()

def claim():
 with lock(D/'dispatch.lock'):
  if (D/'STOP.json').exists():return None
  if torch.cuda.mem_get_info()[0]<24*1024**3:return None
  gate=pilot_gate()
  for seed in (17,7,29,43,61):
   if seed!=17 and not gate:continue
   for b in BRANCHES:
    job=f'seed{seed}_{b}';st=read(D/'status'/f'{job}.json')
    if st['status']!='PENDING':continue
    attempt=D/'attempts'/job/'attempt_0001';attempt.mkdir(parents=True,exist_ok=False)
    save(D/'status'/f'{job}.json',dict(status='RUNNING',seed=seed,branch=b,attempt=str(attempt),host=socket.getfqdn(),pid=os.getpid(),claimed=time.time()));return seed,b,attempt
 return None

def main():
 host=socket.getfqdn();assert host in HOSTS and torch.__version__.startswith('2.7.0');torch.set_num_threads(4);R.configure_device(torch.device('cuda'));assert shutil.disk_usage(D).free>40*1024**3
 with lock(D/'workers'/f'{host}.lock',False):
  frozen();assert all(read(D/'validation'/f'{h}.json')['passed'] for h in HOSTS);event=threading.Event();active={'job':None,'stage':'dispatch'}
  def heartbeat():
   while not event.is_set():save(D/'workers'/f'{host}.heartbeat.json',dict(host=host,pid=os.getpid(),active=dict(active),unix=time.time()));event.wait(15)
  t=threading.Thread(target=heartbeat,daemon=True);t.start()
  try:
   while not (D/'STOP.json').exists():
    frozen();job=claim()
    if job is None:
     if all(read(D/'status'/f'seed{s}_{b}.json')['status']=='COMPLETE' for s in SEEDS for b in BRANCHES):break
     active.update(job=None,stage='pilot/admission/dependencies');event.wait(15);continue
    seed,b,attempt=job;active.update(job=f'seed{seed}_{b}',stage='production continuation');path=D/'logs'/f'seed{seed}_{b}.log';path.parent.mkdir(exist_ok=True)
    try:
     with lock(attempt/'execution.lock',False):
      with path.open('a',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):result=run(seed,b,attempt)
     if result is None:break
    except BaseException as e:
     save(D/'status'/f'seed{seed}_{b}.json',dict(status='FAILED',seed=seed,branch=b,attempt=str(attempt),error=repr(e),traceback=traceback.format_exc()));save(D/'STOP.json',dict(host=host,job=active['job'],error=repr(e),traceback=traceback.format_exc()));raise
    refresh();torch.cuda.empty_cache();active.update(job=None,stage='dispatch')
   refresh();save(D/'workers'/f'{host}.finished.json',dict(unix=time.time(),stopped=(D/'STOP.json').exists()))
  finally:event.set();t.join(timeout=2)
if __name__=='__main__':main()
