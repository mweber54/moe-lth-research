import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[1]/"scripts/imp_recovery_order_stability_v1"))
from shared_gpu_worker import GpuPolicy, decide_admission

assert socket.gethostname().split(".")[0] == "adder"
with (here/"adder_gpu0.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    while True:
        admitted, record = decide_admission(GpuPolicy())
        print(json.dumps(record),flush=True)
        if admitted:
            break
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    sys.exit(subprocess.call([sys.executable,*sys.argv[1:]],env=env))
