"""Serialize the remaining read-only replay cells; stop at the first failure."""
import json,os,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def write(value):
 p=OUT/'replay_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def live(pid):
 try:
  os.kill(pid,0)
  stat=Path(f'/proc/{pid}/stat')
  return not stat.exists() or stat.read_text().split()[2]!='Z'
 except (ProcessLookupError,FileNotFoundError):return False
pid=int(sys.argv[1]);first=OUT/'gradients_seed7_A_250.pt'
write({'status':'waiting_for_first_cell','pid':pid})
while not first.exists():
 if not live(pid):write({'status':'failed','cell':'seed7_A_250','reason':'first replay exited without output'});sys.exit(1)
 time.sleep(5)
for seed in (7,17,29):
 for order in ('A','B'):
  for step in (250,500):
   cell=f'seed{seed}_{order}_{step}';output=OUT/f'gradients_{cell}.pt'
   if output.exists():continue
   log=OUT/f'replay_{cell}.log';write({'status':'running','cell':cell,'log':str(log)})
   with log.open('w') as f:
    result=subprocess.run([sys.executable,str(OUT/'replay.py'),str(seed),order,str(step)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
   if result.returncode or not output.exists():write({'status':'failed','cell':cell,'exit_code':result.returncode,'log':str(log)});sys.exit(1)
write({'status':'gradient_replay_cells_complete','note':'Coverage, activation, and constrained-selection analysis remain to be completed.'})
