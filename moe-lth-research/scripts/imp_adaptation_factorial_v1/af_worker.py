"""One memory-admitted detached worker per host; exclusive claims, no retries."""
from af_common import *
import socket,threading,traceback,argparse,contextlib
from af_engine import run,batches

def check_host(manifest):
 torch.set_num_threads(4);R.configure_device(torch.device('cuda'))
 if socket.getfqdn() not in HOSTS or not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Canonical registered host/runtime required')
 if torch.cuda.mem_get_info()[0]<24*1024**3:raise RuntimeError('Insufficient GPU memory; unrelated workloads preserved')
 for p,h in read(OUT/'input_hashes.json').items():
  if sha(p)!=h:raise RuntimeError('Input identity mismatch '+p)
 checked=[]
 for seed in SEEDS:
  v=next(v for v in manifest['variants'] if v['seed']==seed);tr,va=batches(v);checked.append(dict(seed=seed,train_prefix=R._batch_sequence_hash(tr[:2500]),train_suffix=R._batch_sequence_hash(tr[2500:]),validation=R._batch_sequence_hash(va)));del tr,va
 save(OUT/'hosts'/f'{socket.getfqdn()}.json',dict(passed=True,host=socket.getfqdn(),torch=str(torch.__version__),free_gpu_bytes=torch.cuda.mem_get_info()[0],manifest_sha256=sha(OUT/'manifest.json'),input_hashes_sha256=sha(OUT/'input_hashes.json'),checked=checked,unix=time.time()))
def refresh(manifest):
 from af_report import analyze
 with lock(OUT/'report.lock'):analyze(manifest)
def claim(manifest):
 with lock(OUT/'dispatch.lock'):
  if (OUT/'STOP.json').exists():return None
  minimum=max(24*1024**3,read(OUT/'launch.json').get('admission_bytes',0))
  if torch.cuda.mem_get_info()[0]<minimum:return None
  for j in manifest['jobs']:
   if status(j)['status'] not in ('PENDING','REUSED_PENDING_DIAGNOSTICS'):continue
   if any(status(next(x for x in manifest['jobs'] if x['id']==d))['status'] not in ('COMPLETE','REUSED') for d in j['depends_on']):continue
   attempt=OUT/'attempts'/j['id']/'attempt_0001'
   if attempt.exists():raise RuntimeError('Existing attempt preserved; manual review required '+str(attempt))
   attempt.mkdir(parents=True);save(OUT/'status'/f"{j['id']}.json",dict(status='RUNNING',attempt=str(attempt),host=socket.getfqdn(),pid=os.getpid(),reuse=j['reuse'],claimed=time.time()));return j
 return None
def worker():
 host=socket.getfqdn();torch.set_num_threads(4);R.configure_device(torch.device('cuda'))
 with lock(OUT/'workers'/f'{host}.lock',False):
  manifest=frozen();event=threading.Event();active={'job':None,'stage':'preflight'}
  def heart():
   while not event.is_set():save(OUT/'workers'/f'{host}.heartbeat.json',dict(host=host,pid=os.getpid(),active=dict(active),unix=time.time()));event.wait(15)
  thread=threading.Thread(target=heart,daemon=True);thread.start()
  try:
   host_check=OUT/'hosts'/f'{host}.json'
   checked=read(host_check) if host_check.exists() else {}
   reusable_check=checked.get('passed') and checked.get('manifest_sha256')==sha(OUT/'manifest.json') and checked.get('input_hashes_sha256')==sha(OUT/'input_hashes.json') and time.time()-checked.get('unix',0)<1800
   if not reusable_check:check_host(manifest)
   if not torch.__version__.startswith('2.7.0') or not torch.cuda.is_available():raise RuntimeError('Canonical worker runtime changed')
   while not (OUT/'STOP.json').exists():
    frozen();job=claim(manifest)
    if job is None:
     refresh(manifest)
     if all(status(j)['status'] in ('COMPLETE','REUSED') for j in manifest['jobs']):break
     active.update(job=None,stage='admission/dependencies');time.sleep(15);continue
    active.update(job=job['id'],stage='reuse_diagnostics' if job['reuse'] else 'training');st=status(job);attempt=Path(st['attempt']);v=next(v for v in manifest['variants'] if v['id']==job['variant_id']);start=time.monotonic()
    for p in (v['initial'],v['mask']):
     if p and sha(p)!=read(OUT/'input_hashes.json')[p]:raise RuntimeError('Source changed before attempt')
    log=OUT/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
    try:
     with log.open('a',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):result=run(job,v,attempt)
     training=[]
     for segment_index,path in enumerate(result['native_training_logs']):
      for row in map(json.loads,Path(path).read_text().splitlines()):
       if row['step']==0:continue
       aux=row.get('auxiliary_routing_loss')
       training.append(dict(step=segment_index*2500+row['step'],training_ce_mean_all_batch_tokens=row.get('train_cross_entropy'),native_auxiliary_layer_mean=aux,weighted_auxiliary_loss=float(v['config']['routing']['aux_loss_weight'])*aux if aux is not None else None,combined_training_loss=row.get('train_loss'),optimizer_step_applied=row.get('optimizer_step_applied'),loss_scale_before=row.get('loss_scale_before'),loss_scale_after=row.get('loss_scale_after')))
     table(attempt/'training_curve.csv',training)
     result['training_curve_path']=str(attempt/'training_curve.csv')
     result['training_aggregation']='per-attempt CE mean over 16384 target tokens; native auxiliary mean across layers, weight .1; applied/AMP-skip flags retained'
     result.update(job=job['id'],seed=job['seed'],sparsity=job['sparsity'],condition=job['condition'],attempted_updates=5000,attempt=str(attempt),checkpoint_sha256=sha(result['checkpoint']),wall_seconds=time.monotonic()-start)
     if result['applied_updates']+result['amp_skips']!=5000:raise RuntimeError('AMP accounting mismatch')
     if job['condition']!='full':
      full=read(read(OUT/'status'/f"{v['id']}_full.json")['result_path'])
      if abs(full['initial_ce']-result['initial_ce'])>1e-6:raise RuntimeError('Initial validation equality failed')
     save(attempt/'result.json',result);save(OUT/'status'/f"{job['id']}.json",dict(status='REUSED' if job['reuse'] else 'COMPLETE',attempt=str(attempt),result_path=str(attempt/'result.json'),host=host,pid=os.getpid(),updated=time.time()))
    except BaseException as e:
     save(OUT/'status'/f"{job['id']}.json",dict(status='FAILED',attempt=str(attempt),error=repr(e),traceback=traceback.format_exc()));raise
    refresh(manifest);torch.cuda.empty_cache();active.update(job=None,stage='dispatch')
   save(OUT/'workers'/f'{host}.finished.json',dict(unix=time.time(),stopped=(OUT/'STOP.json').exists()))
  except BaseException as e:
   save(OUT/'STOP.json',dict(host=host,pid=os.getpid(),active=active,error=repr(e),traceback=traceback.format_exc(),unix=time.time()));refresh(manifest);raise
  finally:event.set();thread.join(timeout=2);save(OUT/'workers'/f'{host}.heartbeat.json',dict(host=host,pid=os.getpid(),active=active,unix=time.time(),finished=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check-host',action='store_true');a=p.parse_args()
 if a.check_host:check_host(frozen())
 else:worker()
