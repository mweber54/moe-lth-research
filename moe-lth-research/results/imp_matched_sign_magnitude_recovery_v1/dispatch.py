"""Launch exactly the registered 108-run queue after all engineering checks pass."""
from pathlib import Path
import sys,json,subprocess,shlex,os,socket
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=Path.cwd();sys.path.insert(0,str(OUT));import run as stage
import coordinator
import gpu_policy
SSH=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15']
def main():
 if (OUT/'STOP.json').exists():return
 assert not (OUT/'launch.json').exists() and not (OUT/'jobs').exists(),'No duplicate dispatch or training before seal'
 reg=stage.verify();manifest=stage.rows(OUT/'job_manifest.csv');expected={(s,o,m,d,a) for s in (7,17,29) for o in ('A','B') for m in ('imp','random1','random2') for d in (1,2,3) for a in ('M','Bm')}
 assert {(int(r['seed']),r['order'],r['mask_type'],int(r['draw']),r['arm']) for r in manifest}==expected
 assert len(manifest)==len({r['job_id'] for r in manifest})==108 and sum(int(r['cap_updates']) for r in manifest)==270000
 assert all(r['passed']=='True' for r in stage.rows(OUT/'engineering_checks.csv'))
 assert all(r['passed']=='True' for r in stage.rows(OUT/'policy_checks.csv'))
 assert stage.rows(OUT/'report_check.csv')[0]['actual_training_updates']=='0'
 assert all(abs(float(r['realized_distance'])-float(r['target_distance']))<=.001 for r in manifest)
 invariants=stage.rows(OUT/'matrix_invariants.csv');assert len(invariants)==13824 and all(float(r['absolute_distance_mismatch'])<=.015 for r in invariants)
 inputs=[]
 for relative in ['canonical_stream_manifest.json','recovery_order_manifest.json']+[str(p.relative_to(OUT)) for p in (OUT/'permutations').glob('*.npy')]:
  actual=stage.sha(OUT/relative);assert actual==stage.sha(stage.OLD/relative)
  inputs.append(dict(path=str(OUT/relative),sha256=actual,matches_original_registered_input=True))
 # Original controls remain the exact files validated in the prior audit, and Stage 1 is finished.
 prior=stage.read(ROOT/'results/imp_sign_recovery_stage1_v1/completion.json');assert prior['completed_runs']==54 and prior['attempted_updates']==135000
 stage.table(OUT/'dispatch_input_hashes.csv',inputs)
 # Recheck storage immediately before the first worker dispatch, without deleting artifacts.
 space=stage.storage_check('immediately_before_108_run_dispatch')
 stage.save(OUT/'dispatch_seal.json',dict(registration_sha256=stage.sha(OUT/'registration.json'),input_hashes_csv_sha256=stage.sha(OUT/'dispatch_input_hashes.csv'),registration_commit=reg['amendment_commit'],pre_dispatch_new_training_runs=0,pre_dispatch_new_attempts=0,storage_check=space,scope='108 M/Bm runs only; <=2500 attempts each; stop regardless of result',sealed_utc=stage.utc()))
 launched=[]
 for host in ('adder.cs.kent.edu','viper.cs.kent.edu'):
  if (OUT/'STOP.json').exists():return
  # A host that cannot be contacted is skipped; a verified worker already launched
  # on the other host continues. Integrity failures still stop the entire study.
  probe=f"import sys,torch;sys.path.insert(0,{str(OUT)!r});import run;run.verify();assert torch.__version__.startswith('2.7.0');print('verified')"
  args=[str(ROOT/'.venv-highres/bin/python'),'-c',probe]
  try:
   r=coordinator.on_host(host,args,capture_output=True,text=True,timeout=180)
  except (OSError,subprocess.TimeoutExpired) as exc:
   gpu_policy.unavailable(host,'worker_dispatch',repr(exc))
   stage.save(OUT/f'dispatch_unavailable_{host}.json',dict(host=host,utc=stage.utc(),reason=repr(exc)));continue
  if r.returncode==255:
   gpu_policy.unavailable(host,'worker_dispatch',r.stderr)
   stage.save(OUT/f'dispatch_unavailable_{host}.json',dict(host=host,utc=stage.utc(),reason=r.stderr));continue
  if r.returncode:raise RuntimeError(f'Host integrity probe failed on {host}: {r.stderr}')
  if (OUT/'STOP.json').exists():return
  program=f'''import subprocess,os,json,socket,pathlib,time
root=pathlib.Path({str(ROOT)!r});out=pathlib.Path({str(OUT)!r});(out/'workers').mkdir(exist_ok=True)
with (out/'workers'/(socket.getfqdn()+'.log')).open('a') as f:
 p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(out/'run.py'),'worker'],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid,launched_unix=time.time())))
'''
  args=[str(ROOT/'.venv-highres/bin/python'),'-c',program]
  r=coordinator.on_host(host,args,capture_output=True,text=True,timeout=120,check=True)
  worker=json.loads(r.stdout.strip().splitlines()[-1])
  if worker['host']!=host:raise RuntimeError(f'Worker host mismatch: {worker}')
  launched.append(worker);stage.save(OUT/'launch.json',dict(workers=launched,registered_jobs=108,total_attempt_cap=270000,continuation_authorized=False,registration_commit=reg['amendment_commit']))
 if not launched:raise RuntimeError('No worker host reachable; no automatic retry')
 if (OUT/'STOP.json').exists():return
 (OUT/'locks').mkdir(exist_ok=True)
 with (OUT/'supervisor.log').open('a') as f:
  supervisor=subprocess.Popen([sys.executable,'-u',str(OUT/'supervise.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 stage.save(OUT/'supervisor_launch.json',dict(host=socket.getfqdn(),pid=supervisor.pid))
 print(json.dumps(dict(workers=launched,supervisor_pid=supervisor.pid,storage=space)))
def entry():
 try:main()
 except BaseException as exc:
  if not (OUT/'STOP.json').exists():stage.save(OUT/'STOP.json',dict(reason='Dispatch failure: '+repr(exc),utc=stage.utc(),no_automatic_retry=True))
  raise
if __name__=='__main__':entry()
