"""Run a study engineering command under the V2 device lock and admission rule."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
v2 = here.with_name("moe_online_reinforcement_pruning_v2")
with (v2 / "gpu0.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    while True:
        sample = subprocess.check_output(["nvidia-smi", "-i", "0",
            "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        free, utilization = map(int, sample.strip().split(","))
        print(time.time(), free, utilization, flush=True)
        if free >= 24576 and utilization <= 90:
            break
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    sys.exit(subprocess.call([sys.executable, *sys.argv[1:]], env=env))
