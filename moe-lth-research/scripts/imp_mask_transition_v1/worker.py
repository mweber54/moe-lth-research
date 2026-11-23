"""Detached exclusive worker: seed17 pilot first, then the exact remaining cohort."""
from common import *
from engine import execute
import threading,traceback,argparse

def check_host():
 torch.set_num_threads(4);R.configure_device(torch.device('cuda'));host=socket.getfqdn()
 assert host in HOSTS and torch.cuda.is_available() and torch.__version__.startswith('2.7.0')
 return dict(host=host,torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),free_bytes=torch.cuda.mem_get_info()[0])

def statuses(manifest):return {j['id']:read(OUT/'status'/f"{j['id']}.json") for j in manifest['jobs']}

def claim(manifest):
 with lock(OUT/'dispatch.lock'):
  if (OUT/'STOP.json').exists():return None
  admission=read(OUT/'launch.json')['admission_bytes']
  if torch.cuda.mem_get_info()[0]<admission:return None
  ss=statuses(manifest);pilot_done=all(ss[f'seed17_{c}']['status'] in ('COMPLETE','REUSED') for c in CONDITIONS)
  jobs=sorted(manifest['jobs'],key=lambda j:(j['seed']!=17,SEEDS.index(j['seed']),CONDITIONS.index(j['condition'])))
  for j in jobs:
   if ss[j['id']]['status']!='PENDING':continue
   if j['seed']!=17 and not pilot_done:continue
   attempt=OUT/'attempts'/j['id']/f'attempt_{time.time_ns()}';attempt.mkdir(parents=True,exist_ok=False)
   save(OUT/'status'/f"{j['id']}.json",dict(status='RUNNING',attempt=str(attempt),host=socket.getfqdn(),pid=os.getpid(),claimed=time.time(),reuse=j['reuse']))
   return j,attempt
 return None

def refresh():
 from report import analyze
 with lock(OUT/'report.lock'):analyze()

def main():
 host=socket.getfqdn();identity=check_host();manifest=frozen()
 with lock(OUT/'workers'/f'{host}.lock',False):
  event=threading.Event();active={'job':None}
  def heartbeat():
   while not event.is_set():
    save(OUT/'workers'/f'{host}.heartbeat.json',dict(identity,**active,pid=os.getpid(),unix=time.time()));event.wait(15)
  thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
  try:
   while not (OUT/'STOP.json').exists():
    frozen();item=claim(manifest)
    if item is None:
     if all(s['status'] in ('COMPLETE','REUSED') for s in statuses(manifest).values()):refresh();break
     event.wait(15);continue
    job,attempt=item;active['job']=job['id'];v=next(x for x in manifest['variants'] if x['seed']==job['seed']);refresh()
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for p in (v['initial'],v['mask'],v['schedule']):assert sha(p)==read(OUT/'input_hashes.json')[p]
    try:
     log=(attempt/'attempt.log').open('a',buffering=1)
     with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=execute(job,v,attempt)
     log.close()
     assert result['attempted_updates']==5000 and result['applied_updates']+result['amp_skips']==5000
     if job['condition']!='dense':assert result['final_mask_hash']==v['mask_hash']
     endpoint=list(attempt.glob('grid_5000.json'))+list(attempt.glob('segment_2500/grid_5000.json'));assert len(endpoint)==1
     result.update(job=job['id'],seed=job['seed'],condition=job['condition'],attempt=str(attempt),checkpoint_sha256=sha(result['checkpoint']),common_endpoint=read(endpoint[0]),
        wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),runtime=identity)
     assert math_isfinite(result['common_endpoint']['ce'])
     save(attempt/'result.json',result)
     files={str(p.relative_to(attempt)):sha(p) for p in attempt.rglob('*') if p.is_file() and p.suffix!='.log'}
     save(attempt/'completion.json',dict(verified=True,files=files,checkpoint_sha256=result['checkpoint_sha256'],final_mask_hash=result['final_mask_hash'],unix=time.time()))
     save(OUT/'status'/f"{job['id']}.json",dict(status='REUSED' if job['reuse'] else 'COMPLETE',attempt=str(attempt),result=str(attempt/'result.json'),completion_sha256=sha(attempt/'completion.json'),host=host,updated=time.time()))
    except BaseException as e:
     save(OUT/'status'/f"{job['id']}.json",dict(status='FAILED',attempt=str(attempt),error=repr(e),traceback=traceback.format_exc()));raise
    torch.cuda.empty_cache();refresh();active['job']=None
   refresh()
  except BaseException as e:
   save(OUT/'STOP.json',dict(host=host,active=active,error=repr(e),traceback=traceback.format_exc(),unix=time.time()));refresh();raise
  finally:
   event.set();thread.join(timeout=2);save(OUT/'workers'/f'{host}.finished.json',dict(unix=time.time(),stopped=(OUT/'STOP.json').exists()))

def math_isfinite(x):return bool(np.isfinite(x))
if __name__=='__main__':main()
