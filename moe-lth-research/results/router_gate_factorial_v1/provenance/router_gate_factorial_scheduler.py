import os,time,json,pathlib,signal
root=pathlib.Path('/users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/router_gate_factorial_v1');path=root/'execution_scheduling.json'
def alive(pid):
 try:
  s=pathlib.Path(f'/proc/{pid}/stat').read_text().split();return s[2]!='Z'
 except FileNotFoundError:return False
def event(e):
 d=json.loads(path.read_text());d['events'].append(dict(utc_seconds=time.time(),**e));path.write_text(json.dumps(d,indent=2)+'\n')
for pid,nextpid in [(3229181,3229179),(3229179,3229183)]:
 while alive(pid):time.sleep(5)
 if alive(nextpid):os.kill(nextpid,signal.SIGCONT);event(dict(completed_process_pid=pid,resumed_pid=nextpid))
while alive(3229183):time.sleep(5)
event(dict(all_scheduled_processes_exited=True))
