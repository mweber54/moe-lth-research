"""Durable sequential queue. Wait for idle GPU; never kill another process."""
from pilot_common import *
import subprocess,time,fcntl,traceback,socket

def dependencies(job,lookup):
 if job['job_type']=='source':return True
 if int(job['source_attempts'])==10000 and lookup[job['source_id']]['status']!='COMPLETE':return False
 return job['job_type']=='dense' or lookup[job['dense_condition_id']]['status'] in ('COMPLETE','REUSED')
def device_processes():
 text=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 return text.splitlines() if text else []
def live_worker(cid):
 for path in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   parts=path.read_bytes().split(b'\0')
   if any(x.endswith(b'nominal_exposure_dataset/worker.py') for x in parts) and cid.encode() in parts:return int(path.parent.name)
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 return None
def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
 checks=read(OUT/'audits/preflight.json')
 if checks['status']!='PASS':raise RuntimeError('Preflight has not passed')
 expected=read(OUT/'manifests/immutable_hashes.json')
 for p,digest in expected.items():
  if sha(OUT/p)!=digest:raise RuntimeError('Immutable pilot inputs changed: '+p)
 for p,digest in checks['code_hashes'].items():
  if sha(ROOT/p)!=digest:raise RuntimeError('Validated code changed: '+p)
 for rel,digest in checks['evidence_hashes'].items():
  if sha(OUT/rel)!=digest:raise RuntimeError('Sealed preflight evidence changed: '+rel)
 for name in ['input_provenance.json','reuse_raw_input_hashes.json']:
  for path,record in read(OUT/'manifests'/name).items():
   p=Path(path);st=p.stat()
   if st.st_size!=record['bytes'] or st.st_mtime_ns!=record['mtime_ns']:
    if sha(p)!=record['sha256']:raise RuntimeError('Reuse/source artifact changed: '+path)
 jsave(OUT/'queue_identity.json',{'pid':os.getpid(),'host':socket.gethostname(),'started_at':start,'gpu':0})
 while True:
  jobs=pd.read_csv(OUT/'manifests/run_manifest.csv');lookup={r['condition_id']:r for r in jobs.to_dict('records')}
  # Interrupted attempts are preserved; resuming the queue starts a fresh numbered attempt.
  stale=jobs.status.eq('RUNNING')
  if stale.any():
   alive={r.condition_id:live_worker(r.condition_id) for r in jobs[stale].itertuples()};alive={k:v for k,v in alive.items() if v is not None}
   if alive:
    jsave(OUT/'queue_status.json',{'state':'ATTACHED_TO_EXISTING_WORKER','workers':alive,'pid':os.getpid(),'updated_at':time.time()});time.sleep(30);continue
   for i,r in jobs[stale].iterrows():
    p=Path(r.artifact_path)
    if (p/'pilot_completed.json').exists():jobs.at[i,'status']='COMPLETE';jobs.at[i,'artifact_sha256']=read(p/'pilot_completed.json')['final_checkpoint_sha256']
    else:jobs.at[i,'status']='PENDING';jobs.at[i,'block_reason']='Interrupted prior attempt retained; restart from correct initialization'
   csv(OUT/'manifests/run_manifest.csv',jobs);continue
  ready=[r for r in jobs.to_dict('records') if r['status']=='PENDING' and dependencies(r,lookup)];ready.sort(key=lambda r:(r['stage'],r['condition_id']))
  if not ready:jsave(OUT/'queue_status.json',{'state':'DONE_WITH_BLOCKED_DEPENDENCIES' if jobs.status.str.startswith('BLOCKED').any() else 'COMPLETE','pid':os.getpid(),'counts':jobs.status.value_counts().to_dict(),'updated_at':time.time()});return
  external=device_processes()
  if external:
   jsave(OUT/'queue_status.json',{'state':'WAITING_GPU','pid':os.getpid(),'external_compute_processes':external,'next_condition':ready[0]['condition_id'],'pending_ready':len(ready),'updated_at':time.time()});print('WAITING_GPU',external,'next',ready[0]['condition_id'],flush=True);time.sleep(30);continue
  import shutil
  if shutil.disk_usage(OUT).free<30*1024**3:jsave(OUT/'queue_status.json',{'state':'WAITING_STORAGE','pid':os.getpid(),'updated_at':time.time()});time.sleep(30);continue
  job=ready[0];cid=job['condition_id'];dest=safe_attempt(OUT/('sources' if job['job_type']=='source' else 'recoveries')/cid);idx=jobs.index[jobs.condition_id==cid][0];jobs.at[idx,'status']='RUNNING';jobs.at[idx,'artifact_path']=str(dest);csv(OUT/'manifests/run_manifest.csv',jobs)
  log=OUT/'logs'/f'{cid}_{dest.name}.log'
  with log.open('ab') as h:child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/nominal_exposure_dataset/worker.py'),'--condition',cid,'--attempt',str(dest)],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4'})
  jsave(OUT/'queue_status.json',{'state':'RUNNING','pid':os.getpid(),'worker_pid':child.pid,'condition_id':cid,'log':str(log),'updated_at':time.time()});rc=child.wait();jobs=pd.read_csv(OUT/'manifests/run_manifest.csv');idx=jobs.index[jobs.condition_id==cid][0]
  if rc or not (dest/'pilot_completed.json').exists():
   jobs.at[idx,'status']='FAILED_NEEDS_DIAGNOSIS';jobs.at[idx,'block_reason']=f'Worker exit {rc}; inspect {log}';csv(OUT/'manifests/run_manifest.csv',jobs);jsave(OUT/'queue_status.json',{'state':'FAILED_NEEDS_DIAGNOSIS','condition_id':cid,'worker_exit':rc,'log':str(log),'updated_at':time.time()});return
  jobs.at[idx,'status']='COMPLETE';jobs.at[idx,'artifact_sha256']=read(dest/'pilot_completed.json')['final_checkpoint_sha256'];csv(OUT/'manifests/run_manifest.csv',jobs)
if __name__=='__main__':
 try:main()
 except Exception:traceback.print_exc();raise
