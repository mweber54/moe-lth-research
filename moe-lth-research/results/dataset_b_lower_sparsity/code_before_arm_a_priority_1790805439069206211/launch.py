"""Wait for parent completion; retain its exclusive lock throughout extension."""
import argparse, fcntl, socket, subprocess, sys, time
from support import *

def main():
    a=argparse.ArgumentParser(); a.add_argument('--check',action='store_true'); args=a.parse_args()
    if socket.getfqdn()!='viper.cs.kent.edu': raise RuntimeError('Run on viper')
    # Read-only preparation check works while parent is still training.
    verify_map(OUT,read(OUT/'preparation_hashes.json'))
    plan=rows(OUT/'manifests/job_plan.csv')
    if len(plan)!=320 or {r['condition_id'] for r in plan}!=expected_ids(): raise RuntimeError('Invalid grid')
    if args.check:
        print('PREPARATION CHECK PASS; no training launched'); return
    singleton=(OUT/'handoff.lock').open('a')
    fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(PARENT/'queue.lock').open('a')
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
        except BlockingIOError:
            save(OUT/'handoff_status.json',{'state':'WAITING_PARENT_LOCK','updated_at':time.time()})
            print('WAITING_PARENT_LOCK',flush=True); time.sleep(30)
    if not parent_complete(rows(PARENT/'manifests/run_manifest.csv')):
        save(OUT/'handoff_status.json',{'state':'BLOCKED_PARENT_INCOMPLETE','updated_at':time.time()})
        raise RuntimeError('Parent stopped without completing all jobs. No extension jobs launched.')
    save(OUT/'handoff_status.json',{'state':'PREPARING_RUNTIME','updated_at':time.time()})
    from prepare_runtime import prepare
    prepare()
    import coordinator as C
    for host in C.HOSTS:
        subprocess.run(C.command(host,[str(C.PYTHON),'-u',str(Path(__file__).parent/'worker_adapter.py'),'--validate']),check=True)
    save(OUT/'handoff_status.json',{'state':'COORDINATOR_STARTED','updated_at':time.time()})
    C.main(parent_lock_fd=lock.fileno())
if __name__=='__main__': main()
