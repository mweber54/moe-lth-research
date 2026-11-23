"""Run only the pilot under admitted.py; stop immediately on a failed step."""
import subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
for label in ('original','perm0'):
 for h in (1,2):
  for suffix in ('','_repeat'):
   cmd=[sys.executable,str(P/'split_half_replay.py'),'17','B','250',str(h)]
   if label=='perm0':cmd+=['--permutation','0']
   if suffix:cmd+=['--suffix',suffix]
   print('RUN',cmd,flush=True)
   subprocess.run(cmd,check=True)
subprocess.run([sys.executable,str(P/'split_half_pilot_verify.py')],check=True)
