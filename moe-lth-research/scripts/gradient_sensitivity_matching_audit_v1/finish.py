from common import *
hold=g.lock(D/'locks/finalization.lock',False);start=time.monotonic()
while True:
 states=[D/'workers'/f'{host}.finished.json' for host in ('viper.cs.kent.edu','adder.cs.kent.edu')]
 if all(p.exists() for p in states) and (D/'solver_queue_complete.json').exists():break
 assert time.monotonic()-start<10800,'Fixed audit deadline exceeded'
 time.sleep(15)
for name in ('report.py','diagnosis.py','final_audit.py'):
 print('FINALIZING',name,flush=True)
 subprocess.run([str(ROOT/'.venv-highres/bin/python'),'-u',str(HERE/name)],cwd=ROOT,check=True)
print('MATCHING FEASIBILITY AUDIT COMPLETE',flush=True)
