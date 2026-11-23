import json,os,socket,tempfile,subprocess,sys,time
from pathlib import Path
import status
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'seed7_A'/'untouched'
 assert status.status(p)=='NOT_STARTED';p.mkdir(parents=True);assert status.status(p)=='NOT_STARTED'
 (p/'checkpoint.pt').write_text('partial');assert status.status(p)=='STALE_OR_INTERRUPTED'
 (p/'failed.json').write_text('{}');assert status.status(p)=='FAILED';(p/'failed.json').unlink()
 script='import fcntl,sys,time; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX); print("ready",flush=True); time.sleep(15)'
 proc=subprocess.Popen([sys.executable,'-c',script,str(p/'lock'),'branch.py','7','A','untouched'],stdout=subprocess.PIPE,text=True)
 try:
  assert proc.stdout.readline().strip()=='ready'
  (p/'running.json').write_text(json.dumps({'pid':proc.pid,'host':socket.getfqdn()}));assert status.status(p)=='RUNNING'
 finally:proc.terminate();proc.wait()
 assert status.status(p)=='STALE_OR_INTERRUPTED'
 import torch
 identity={'seed':7,'order':'A','arm':'untouched'};losses={k:1. for k in ('pre_mask_250','post_mask_250','251','275','500','1000','2500')}
 torch.save({'identity':identity,'global_step':2500,'next_batch_index':2500,'losses':losses,'model':{'x':torch.tensor(1)},'optimizer':{'state':{0:{'x':1}}}},p/'checkpoint.pt')
 (p/'complete.json').write_text(json.dumps({'identity':identity,'global_step':2500,'losses':losses,'checkpoint_sha256':status.digest(p/'checkpoint.pt')}));assert status.status(p)=='COMPLETED'
 (p/'checkpoint.pt').write_bytes(b'corrupt');assert status.status(p)=='STALE_OR_INTERRUPTED'
 print('status transitions passed')
