"""Two-host persistent stage scheduler. A completed artifact verifies each job."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PYTHON=ROOT/".venv-highres/bin/python"
DEV=(7,17,29)
FRESH=(37,43,53)
ORDERS=("A","B")


def run_host(host,cmd,log):
    wrapper="adder.py" if host=="adder" else "admitted.py"
    argv=[str(PYTHON),str(HERE/wrapper),*map(str,cmd)]
    if host=="adder":
        remote="cd "+shlex.quote(str(ROOT))+" && "+shlex.join(argv)
        argv=["ssh","-i",str(Path.home()/".ssh/viper_to_adder"),"-o","BatchMode=yes",
              "-o","ConnectTimeout=10","adder.cs.kent.edu",remote]
    with log.open("x") as f:
        return subprocess.call(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)


def stage(name,jobs):
    work=queue.Queue()
    for job in jobs:
        work.put(job)
    failed=[]
    def worker(host):
        while True:
            try:
                key,cmd,proof=work.get_nowait()
            except queue.Empty:
                return
            if proof.exists():
                continue
            log=HERE/"jobs"/f"{name}_{key}_{time.time_ns()}.log"
            print(time.time(),"START",host,name,key,log,flush=True)
            rc=run_host(host,cmd,log)
            if rc or not proof.exists():
                failed.append((key,rc,str(log)))
            print(time.time(),"END",host,name,key,rc,proof.exists(),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(worker,["viper","adder"]))
    if failed:
        raise RuntimeError(f"{name} failed: {failed}")


def scoring(roster,t):
    return [(f"{s}{o}",[HERE/"stage.py","forecast","--seed",s,"--order",o,"--t",t],
             HERE/"seals"/f"seed{s}_{o}_{t}.json") for s in roster for o in ORDERS]


def main():
    (HERE/"jobs").mkdir(exist_ok=True)
    with (HERE/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # The initial standalone 7/A corrected run and first random-control
        # endpoint have their own logs. The scheduler shares the host locks.
        for phase,target in (("forward","expected501.pt"),("resume","verified.json")):
            stage("smoke_"+phase,[("7A",[HERE/"lifetime_engineering.py",phase],HERE/"smoke_7_A_isolated"/target)])
        for t in (250,500):
            jobs=scoring(DEV,t)
            stage(f"dev{t}",jobs[1:]+jobs[:1])
        stage("engineering_control",[(a,[HERE/"continuation.py",a],HERE/"random_control"/a/"verified.json") for a in ("H","R")])
        if not (HERE/"freeze.json").exists():
            subprocess.run([PYTHON,HERE/"stage.py","freeze"],check=True,cwd=ROOT)
        for t in (250,500):
            stage(f"fresh{t}",scoring(FRESH,t))
        jobs=[]
        for s in FRESH:
            for o in ORDERS:
                decision=json.loads((HERE/"seals"/f"seed{s}_{o}_250.json").read_text())
                if decision.get("ineligible"):
                    continue
                for t,arms in ((250,("H","R","0","dense")),(500,("H","R","0"))):
                    for a in arms:
                        jobs.append((f"{s}{o}_{t}_{a}",[HERE/"stage.py","branch","--seed",s,"--order",o,"--t",t,"--arm",a],
                                     HERE/"branches"/f"seed{s}_{o}"/str(t)/a/"verified.json"))
        jobs.extend((f"control_{a}",[HERE/"continuation.py",a],HERE/"random_control"/a/"verified.json") for a in ("H","R"))
        stage("continuation",jobs)
        subprocess.run([PYTHON,HERE/"report.py"],check=True,cwd=ROOT)

if __name__=="__main__":
    main()
