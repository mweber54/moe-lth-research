"""Existing two-host SSH transport; detached single workers, never kill jobs."""
from v2_common import *
import argparse
import shlex
import subprocess

SSH=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','UserKnownHostsFile=/tmp/geometry_known_hosts','-o','StrictHostKeyChecking=accept-new']


def detach_program(root):
    return """import subprocess,pathlib,json,time,socket,os
root=pathlib.Path(%r)
out=root/'results/imp_mask_information_v2'
(out/'workers').mkdir(parents=True,exist_ok=True)
log=(out/'workers'/f'{socket.getfqdn()}.log').open('a')
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_FORCE_WEIGHTS_ONLY_LOAD='1')
p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(root/'scripts/imp_mask_information_v2/worker.py')],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid,log=str(out/'workers'/f'{socket.getfqdn()}.log'),unix=time.time())))
""" % str(root)


def launch():
    manifest=frozen()
    if (OUT/'launch.json').exists():raise RuntimeError('Launch already recorded; inspect PIDs/heartbeats instead of duplicating workers')
    if not read(OUT/'validation.json')['passed']:raise RuntimeError('Tests not passed')
    if (OUT/'STOP.json').exists():raise RuntimeError('Failure stop exists')
    launches=[]
    for host in HOSTS:
        # Run host verification independently before starting either worker.
        command=f'cd {shlex.quote(str(ROOT))} && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-highres/bin/python scripts/imp_mask_information_v2/worker.py --check-host'
        logpath=OUT/'host_checks'/f'{host}.preflight.log';logpath.parent.mkdir(exist_ok=True)
        with logpath.open('a') as log:
            process=subprocess.Popen([*SSH,host,command],stdout=log,stderr=subprocess.STDOUT)
            launches.append((host,process))
    failed=[]
    for host,process in launches:
        if process.wait()!=0:failed.append(host)
    if failed:raise RuntimeError('Host checks failed; no workers launched: '+','.join(failed))
    launch_records=[]
    for host in HOSTS:
        check=read(OUT/'host_checks'/f'{host}.json')
        if not check['passed'] or check['free_gpu_bytes']<24*1024**3:raise RuntimeError('GPU admission failed')
        # subprocess with a new session detaches from SSH; argument lists avoid
        # shell interpolation of configuration paths or scientific parameters.
        program=detach_program(ROOT)
        command='.venv-highres/bin/python -c '+shlex.quote(program)
        command=f'cd {shlex.quote(str(ROOT))} && '+command
        answer=subprocess.run([*SSH,host,command],capture_output=True,text=True,timeout=30)
        if answer.returncode:
            save(OUT/'launch.json',dict(partial=True,workers=launch_records,error=answer.stderr));raise RuntimeError('Partial launch; preserve existing worker')
        launch_records.append(json.loads(answer.stdout.strip().splitlines()[-1]))
        save(OUT/'launch.json',dict(workers=launch_records,phase1_new=24,reused=6,phase2_conditional=25,manifest_sha256=sha(OUT/'manifest.json'),unix=time.time()))
    print(json.dumps(launch_records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--seal',action='store_true');a=p.parse_args()
    if a.seal:
        if (OUT/'seal.json').exists():raise RuntimeError('Existing seal; do not silently replace')
        save(OUT/'seal.json',dict(protocol_sha256=sha(OUT/'protocol.md'),manifest_sha256=sha(OUT/'manifest.json'),sources=sources(),unix=time.time()))
    if a.launch:
        with lock(OUT/'launch.lock',False):launch()
