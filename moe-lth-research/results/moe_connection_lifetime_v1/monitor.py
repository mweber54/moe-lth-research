"""Maintain a live cluster ledger, with bounded recovery for ordinary interruptions."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def busy():
    with (HERE/"supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def read(path):
    return json.loads(path.read_text())


def update(data):
    path=HERE/"live.json"
    tmp=HERE/".live_monitor.tmp"
    tmp.write_text(json.dumps(data,indent=2)+"\n")
    os.replace(tmp,path)


def main():
    with (HERE/"monitor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        recoveries=0
        log=HERE/"supervisor_attempt1.log"
        while True:
            active=busy()
            status=read(HERE/"status.json") if (HERE/"status.json").exists() else {}
            data=dict(updated_unix=time.time(),supervisor_active=active,
                freeze=(HERE/"freeze.sha256").exists(),forecast_receipts=len(list((HERE/"seals").glob("*.json"))),
                control_endpoint_receipts=len(list((HERE/"random_control").glob("*/verified.json"))),
                continuation_endpoint_receipts=len(list((HERE/"branches").glob("*/*/*/verified.json"))),
                independently_verified_at_last_report=status.get("counts"),recoveries=recoveries)
            if active:
                update(data)
                time.sleep(60)
                continue
            if status.get("counts")==dict(held_out=36,dense=6,random_control=2) and not status.get("missing"):
                update(dict(data,complete=True))
                return
            tail=log.read_text()[-8000:] if log.exists() else ""
            # Scientific checks need diagnosis. Admission rejection is handled
            # by each host wrapper; overflows do not cause process failures.
            if "Traceback" in tail:
                failed_logs=sorted((HERE/"jobs").glob("*.log"),key=lambda p:p.stat().st_mtime,reverse=True)
                latest="\n".join(p.read_text()[-1200:] for p in failed_logs[:3])
                transient=any(s in latest for s in ("Connection reset","Broken pipe","Connection timed out",
                              "CUDA out of memory","Input/output error","Stale file handle"))
                if not transient:
                    update(dict(data,needs_inspection=str(log)))
                    return
            if recoveries>=2:
                update(dict(data,needs_inspection=str(log)))
                return
            recoveries+=1
            log=HERE/f"supervisor_recovery_{time.time_ns()}.log"
            with log.open("x") as output:
                subprocess.Popen([sys.executable,"-u",str(HERE/"supervise.py")],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            update(dict(data,relaunch_log=str(log)))
            time.sleep(60)


if __name__=="__main__":
    main()
