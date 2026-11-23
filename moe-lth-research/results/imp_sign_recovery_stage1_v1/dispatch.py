"""Preflight and launch the sealed 54-run Stage1 queue, never a later stage."""
from pathlib import Path
import sys,json,subprocess,shlex,os,socket,csv,hashlib,datetime
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=Path.cwd();sys.path.insert(0,str(OUT));import run as stage
SSH=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15']
assert not (OUT/'launch.json').exists(),'Do not double-dispatch'
assert not (OUT/'jobs').exists(),'No registered training may precede this first dispatch'
reg=stage.verify();manifest=stage.rows(OUT/'job_manifest.csv');expected={(s,o,m,d) for s in (7,17,29) for o in ('A','B') for m in ('imp','random1','random2') for d in (1,2,3)}
assert {(int(r['seed']),r['order'],r['mask_type'],int(r['draw'])) for r in manifest}==expected
assert len(manifest)==len({r['job_id'] for r in manifest})==54
assert sum(int(r['cap_updates']) for r in manifest)==135000
inputs=[]
for relative in ['canonical_stream_manifest.json','recovery_order_manifest.json']+[str(p.relative_to(OUT)) for p in (OUT/'permutations').glob('*.npy')]:
 actual=stage.sha(OUT/relative);original=stage.sha(stage.OLD/relative);assert actual==original
 inputs.append(dict(path=str(OUT/relative),sha256=actual,matches_original_registered_input=True))
stage.table(OUT/'dispatch_input_hashes.csv',inputs)
assert all(r['passed']=='True' for r in stage.rows(OUT/'engineering_checks.csv'))
assert all(r['passed']=='True' for r in stage.rows(OUT/'policy_checks.csv'))
assert stage.rows(OUT/'report_check.csv')[0]['actual_training_updates']=='0'
probe=f'''import sys,json,socket,torch
from pathlib import Path
root=Path({str(ROOT)!r});sys.path.insert(0,str(root/'results/imp_sign_recovery_stage1_v1'));import run
reg=run.verify();assert torch.__version__.startswith('2.7.0') and torch.cuda.is_available()
print(json.dumps(dict(host=socket.getfqdn(),torch=torch.__version__,cuda_build=torch.version.cuda,amendment_commit=reg['amendment_commit'],verified=True)))
'''
remote_command='cd '+shlex.quote(str(ROOT))+' && '+shlex.join([str(ROOT/'.venv-highres/bin/python'),'-c',probe])
r=subprocess.run([*SSH,'adder.cs.kent.edu',remote_command],capture_output=True,text=True,timeout=180,check=True)
remote_probe=json.loads(r.stdout.strip().splitlines()[-1]);assert remote_probe['amendment_commit']==reg['amendment_commit'];stage.save(OUT/'adder_preflight.json',remote_probe)
assert stage.torch.__version__.startswith('2.7.0') and stage.torch.cuda.is_available()
# Last prelaunch operation: a new filesystem query against the planned total allowance.
space=stage.storage_check('immediately_before_stage1_worker_launch');assert space['fits']
stage.save(OUT/'dispatch_seal.json',dict(registration_sha256=stage.sha(OUT/'registration.json'),input_hashes_csv_sha256=stage.sha(OUT/'dispatch_input_hashes.csv'),amendment_commit=reg['amendment_commit'],pre_dispatch_training_runs=0,pre_dispatch_attempts=0,storage_check=space,scope='Stage1 sign only, 54 jobs, <=2500 attempts/job; stop regardless of gate',sealed_utc=stage.utc()))
launched=[]
for host in (socket.getfqdn(),'adder.cs.kent.edu'):
 program=f'''import subprocess,os,json,socket,pathlib,time
root=pathlib.Path({str(ROOT)!r});out=root/'results/imp_sign_recovery_stage1_v1';(out/'workers').mkdir(exist_ok=True)
log=out/'workers'/(socket.getfqdn()+'.log')
with log.open('a') as f:
 p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(out/'run.py'),'worker'],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid,log=str(log),launched_unix=time.time())))
'''
 args=[str(ROOT/'.venv-highres/bin/python'),'-c',program]
 try:
  if host==socket.getfqdn():r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True)
  else:r=subprocess.run([*SSH,host,'cd '+shlex.quote(str(ROOT))+' && '+shlex.join(args)],capture_output=True,text=True,timeout=120,check=True)
  launched.append(json.loads(r.stdout.strip().splitlines()[-1]));stage.save(OUT/'launch.json',dict(workers=launched,stage=1,registered_jobs=54,total_attempt_cap=135000,continuation_authorized=False,amendment_commit=reg['amendment_commit']))
 except BaseException as exc:
  stage.save(OUT/'STOP.json',dict(reason='Dispatch failure: '+repr(exc),utc=stage.utc()));raise
print(json.dumps(dict(workers=launched,storage=space,amendment_commit=reg['amendment_commit'])))
