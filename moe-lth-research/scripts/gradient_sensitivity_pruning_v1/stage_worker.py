"""Execute fixed scientific gates, never production training or loss evaluation."""
from common import *
import subprocess,socket
hold=lock(D/'locks/stage_worker.lock',blocking=False)
started=time.time();host=socket.gethostname()
while True:
    states=[D/'status'/f'measure_seed{s}.json' for s in SEEDS]
    if all(p.exists() and read(p)['status']=='COMPLETE' for p in states):break
    assert time.time()-started<7200,'Measurement deadline exceeded'
    save(D/'stage_status.json',{'status':'WAITING_FOR_FIVE_MEASUREMENTS','host':host,'heartbeat':time.time()})
    time.sleep(15)
for script in ('review.py','match.py'):
    save(D/'stage_status.json',{'status':'RUNNING','script':script,'host':host,'started':time.time()})
    subprocess.run([str(ROOT/'.venv-highres/bin/python'),'-u',str(HERE/script)],cwd=ROOT,check=True)
    if script=='review.py' and read(D/'reliability_review.json')['status']!='PASS':
        save(D/'stage_status.json',{'status':'STOPPED_RELIABILITY_GATE','host':host,'time':time.time()});raise SystemExit(0)
if read(D/'matching_review.json')['status']=='STOP_MATCHING':
    subprocess.run([str(ROOT/'.venv-highres/bin/python'),'-u',str(HERE/'report_matching_stop.py')],cwd=ROOT,check=True)
    save(D/'stage_status.json',{'status':'STOPPED_MATCHING_GATE','host':host,'time':time.time()})
else:
    save(D/'stage_status.json',{'status':'MATCHING_PASS_READY_FOR_NUMERICAL_PILOT','host':host,'time':time.time()})
