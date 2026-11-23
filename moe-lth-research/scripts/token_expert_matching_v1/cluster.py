from run import *
from af_cluster import SSH
import subprocess
import shlex


def launch():
    with lock(DEST/'launch.lock',False):
        if (DEST/'launch.json').exists():raise RuntimeError('Launch already recorded; inspect preserved workers')
        validations=[read(DEST/'validation'/f'{host}.json') for host in HOSTS]
        if not all(x['passed'] for x in validations):raise RuntimeError('Both-host numerical checks required')
        if shutil.disk_usage(DEST).free<read(DEST/'resource_estimate.json')['suggested_free_storage_gib']*1024**3:raise RuntimeError('Storage admission failed')
        save(DEST/'validation.json',dict(production_ready=True,hosts=validations,cpu_tests='7 passed; test_token_expert_matching.py',limitations='Short canonical GPU equivalence; full donor trajectory validation remains mandatory per donor'))
        # Seal once all implementation files exist, before any pilot training outcome.
        seal=read(DEST/'seal.json');seal['implementation']={str(p.relative_to(ROOT)):sha(p) for p in HERE.glob('*.py')};save(DEST/'seal.json',seal)
        workers=[]
        for host in HOSTS:
            program="""import subprocess,pathlib,os,json,socket
root=pathlib.Path(%r)
out=root/'results/token_expert_matching_v1'
(out/'workers').mkdir(exist_ok=True)
log=(out/'workers'/f'{socket.getfqdn()}.log').open('a')
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(root/'scripts/token_expert_matching_v1/worker.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid)))
""" % str(ROOT)
            command='cd '+shlex.quote(str(ROOT))+' && .venv-highres/bin/python -c '+shlex.quote(program)
            answer=subprocess.run([*SSH,host,command],capture_output=True,text=True,timeout=45)
            if answer.returncode:
                save(DEST/'launch.json',dict(partial=True,workers=workers,error=answer.stderr));raise RuntimeError('Partial launch; existing workers preserved')
            workers.append(json.loads(answer.stdout.strip().splitlines()[-1]));save(DEST/'launch.json',dict(workers=workers,pilot_gated=True))
        print(json.dumps(workers))


if __name__=='__main__':
    import shutil
    launch()
