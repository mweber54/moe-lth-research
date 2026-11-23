import argparse,fcntl,subprocess,sys,socket
from support import *
def main():
    a=argparse.ArgumentParser();a.add_argument('--check',action='store_true');args=a.parse_args()
    if socket.getfqdn()!='viper.cs.kent.edu': raise RuntimeError('Run on viper')
    # Retain both predecessor locks: no overlapping training supervisors.
    locks=[]
    for root in (PARENT,ROOT/'results/dataset_b_lower_sparsity'):
        lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    prior=rows(ROOT/'results/dataset_b_lower_sparsity/manifests/run_manifest.csv')
    if len(prior)!=115 or not all(r['status'] in ('COMPLETE','REUSED') for r in prior):
        raise RuntimeError('32E subset not complete')
    parent=rows(PARENT/'manifests/run_manifest.csv')
    if any(r['status'] in ('RUNNING','FAILED_UNCERTAIN') for r in parent):
        raise RuntimeError('Resolve uncertain parent workers before launch')
    validate_seal()
    import coordinator as C
    for host in C.HOSTS:
        subprocess.run(C.command(host,[str(C.PYTHON),'-u',str(Path(__file__).parent/'worker_adapter.py'),'--validate']),check=True)
    C.main(parent_lock_fd=locks[0].fileno())
if __name__=='__main__': main()
