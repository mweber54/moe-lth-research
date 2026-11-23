import sys,time,subprocess,json,shutil
from pathlib import Path
root=Path('/users/kent/student/mweber54/moe-lth-research/moe-lth-research');sys.path.insert(0,str(root));from scripts.imp_first_moe_direction_v1 import engine as e
out=root/'results/imp_first_moe_direction_v1'
for retry in range(10):
 print('WAITING_FOR_EXISTING_ADMISSION',retry,flush=True)
 try:e.c.gpu_admission_check(wait_for_idle_seconds=600)
 except RuntimeError as exc:print('ADMISSION_WAIT_EXPIRED',str(exc),flush=True);sys.exit(3)
 print('ADMISSION_AVAILABLE',flush=True)
 with (out/'execution_adder_v2.log').open('a') as log:
  result=subprocess.run([str(root/'.venv/bin/python'),'-u',str(root/'scripts/imp_first_moe_direction_v1/run.py'),'--evaluate'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 status=e.c._json_read(out/'completion.json')
 if status.get('status')=='RESOURCE_BLOCKED':
  shutil.copy2(out/'completion.json',Path(status['attempt'])/'completion.json');print('ADMISSION_RACE_PRESERVED_RETRY',flush=True);time.sleep(5);continue
 print('WORKER_FINISHED',status.get('status'),result.returncode,flush=True);sys.exit(result.returncode)
sys.exit(4)
