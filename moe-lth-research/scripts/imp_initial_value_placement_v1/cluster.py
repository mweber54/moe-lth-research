"""Hash-verified detached single workers on the two registered hosts."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from run import ROOT, OUT, read, save, sha, verify, native

SSH=['ssh','-i',str(Path.home()/'.ssh/dataset_b_cluster'),'-o','BatchMode=yes','-o','ConnectTimeout=15']

def remote(host, args):
    command='cd '+shlex.quote(str(ROOT))+' && '+shlex.join(args)
    return subprocess.run([*SSH,host,command],capture_output=True,text=True,timeout=600,check=True)

def check_hosts():
    args=[str(ROOT/'.venv-highres/bin/python'),str(ROOT/'scripts/imp_initial_value_placement_v1/run.py'),'verify-host']
    # Sequential: avoid saturating shared storage with duplicate input reads.
    subprocess.run(args,cwd=ROOT,check=True)
    result=remote('adder.cs.kent.edu',args)
    (OUT/'adder_preflight.log').write_text(result.stdout+result.stderr)

def launch():
    reg=verify();assert reg['production_ready']
    seal=read(OUT/'production_seal.json');assert sha(OUT/'registration.json')==seal['registration_sha256']
    if (OUT/'launch.json').exists():raise RuntimeError('Launch exists; inspect existing workers before resuming')
    launched=[]
    for host in native.HOST_SEEDS:
        program='''import subprocess,os,json,socket,pathlib,time
root=pathlib.Path(%r)
out=root/'results/imp_initial_value_placement_v1'
(out/'workers').mkdir(exist_ok=True)
log=out/'workers'/(socket.getfqdn()+'.log')
with log.open('a') as f:
 p=subprocess.Popen([str(root/'.venv-highres/bin/python'),'-u',str(root/'scripts/imp_initial_value_placement_v1/run.py'),'worker'],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(host=socket.getfqdn(),pid=p.pid,log=str(log),launched_unix=time.time())))
'''%str(ROOT)
        args=[str(ROOT/'.venv-highres/bin/python'),'-c',program]
        if host==socket.getfqdn():result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,check=True)
        else:result=remote(host,args)
        launched.append(json.loads(result.stdout.strip().splitlines()[-1]))
        save(OUT/'launch.json',dict(workers=launched,registration_sha256=seal['registration_sha256'],partial=len(launched)!=2))
    print(json.dumps(launched,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check-hosts','launch']);a=p.parse_args()
    if a.command=='check-hosts':check_hosts()
    else:launch()
