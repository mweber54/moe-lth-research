"""Inspect branch evidence; a directory alone never establishes a running job."""
import fcntl,hashlib,json,math,os,socket
from pathlib import Path
HERE=Path(__file__).resolve().parent
ARMS=('untouched','random-small','high-reinforcement-small')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def status(root):
 if not root.exists():return 'NOT_STARTED'
 done=root/'complete.json';cp=root/'checkpoint.pt'
 if done.exists() and cp.exists():
  try:
   import torch
   d=json.loads(done.read_text()); saved=torch.load(cp,map_location='cpu',weights_only=False)
   if d['global_step']==saved['global_step']==2500 and saved['next_batch_index']==2500 and d['identity']==saved['identity'] and f"seed{d['identity']['seed']}_{d['identity']['order']}"==root.parent.name and d['identity']['arm']==root.name and d['losses']==saved['losses'] and set(('pre_mask_250','post_mask_250','251','275','500','1000','2500'))<=d['losses'].keys() and all(isinstance(v,(int,float)) and math.isfinite(v) for v in d['losses'].values()) and isinstance(saved['optimizer'],dict) and saved['optimizer']['state'] and saved['model'] and digest(cp)==d['checkpoint_sha256']:return 'COMPLETED'
  except Exception:pass
 meta=root/'running.json';lock=root/'lock'
 if meta.exists() and lock.exists():
  try:
   d=json.loads(meta.read_text());pid=int(d['pid']);host=d['host']
   if host==socket.getfqdn():
    os.kill(pid,0)
    argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
    if not any(x.endswith(b'/branch.py') or x==b'branch.py' for x in argv) or root.name.encode() not in argv:raise ValueError('PID is not this branch')
    cell=root.parent.name.removeprefix('seed').split('_',1)
    if len(cell)!=2 or any(x.encode() not in argv for x in cell):raise ValueError('branch identity mismatch')
    with lock.open('a') as f:
     try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
     except BlockingIOError:return 'RUNNING'
  except (OSError,ValueError,KeyError,json.JSONDecodeError):pass
 if (root/'failed.json').exists():return 'FAILED'
 return 'STALE_OR_INTERRUPTED' if any(root.iterdir()) else 'NOT_STARTED'
def main():
 counts={k:0 for k in ('NOT_STARTED','RUNNING','FAILED','COMPLETED','STALE_OR_INTERRUPTED')}
 for seed in (7,17,29):
  for order in ('A','B'):
   for arm in ARMS:
    s=status(HERE/'branches'/f'seed{seed}_{order}'/arm);counts[s]+=1;print(f'seed{seed}_{order} {arm}: {s}')
 print(json.dumps(counts,sort_keys=True))
if __name__=='__main__':main()
