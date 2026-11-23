"""Run the sealed jobs with GPU admission and publish incremental results."""
import argparse
import fcntl
import socket
import time

from run import OUT, N, STEPS, DEVICE, digest, read, write, run_job, job_id, reuse_baseline
import gpu_policy
from report import main as report


def admit():
    while True:
        samples = []
        for index in range(5):
            if index:
                time.sleep(2)
            samples.append(gpu_policy.query_gpu())
        usage = [x["utilization_percent"] for x in samples]
        okay = (min(x["available_vram_mib"] for x in samples) >= 53248
                and max(usage) <= 90 and sum(usage)/len(usage) <= 60)
        write(OUT/f"admission_{socket.gethostname()}.json",dict(accepted=okay,samples=samples))
        if okay:
            return
        print("Waiting for shared GPU capacity",flush=True)
        time.sleep(30)


def worker(shard, count):
    assert 0 <= shard < count
    hostname = socket.gethostname()
    with (OUT/f"worker_{hostname}.lock").open("a") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_shard(shard,count)


def run_shard(shard, count):
    manifest = read(OUT/"inputs.json")
    seal = read(OUT/"matching_seal.json")
    assert seal["input_manifest"] == digest(OUT/"inputs.json")
    assert seal["matrix_audit"] == digest(OUT/"matrix_matching.json")
    for path, expected in manifest["hashes"].items():
        assert digest(path) == expected, f"Input changed: {path}"
    jobs = [job for index,job in enumerate(manifest["jobs"]) if (index//3)%count == shard]
    entries = {(x["seed"],x["order"],x["mask"]):x for x in manifest["inputs"]}
    native_segment = N._run_segment
    active = None

    def segment(job, attempt, config, state, masks, checkpoint, batches, validation, offset, name, device):
        path = attempt/"initial_evaluation.json"
        if not path.exists():
            snapshot = [(0,attempt/"snapshots/step_0000.pt","sign_reinforcement_initial")]
            evaluation = N._common_evaluate(snapshot,config,validation,active["seed"],active["order"],
                                            "imp80",job["job_id"],active["validation_hash"],DEVICE,
                                            allowed_steps=STEPS)
            write(path,evaluation)
        return native_segment(job,attempt,config,state,masks,checkpoint,batches,validation,offset,name,device)

    N._run_segment = segment
    progress = OUT/f"progress_{socket.gethostname()}.json"
    material, last_seed = None, None
    completed = 0
    for job in jobs:
        name = job_id(job)
        result = OUT/"jobs"/name/"result.json"
        seed,order,mask = job["seed"],job["order"],job["mask"]
        active = entries[seed,order,mask]
        if seed != last_seed:
            del material
            material = N._load_training_material(seed)
            last_seed = seed
        baseline_path = OUT/f"baseline_{seed}_{order}_{mask}.json"
        if not baseline_path.exists():
            write(baseline_path,reuse_baseline(active,material))
        if not result.exists():
            write(progress,dict(status="waiting_for_gpu",job=name,completed=completed,total=len(jobs)))
            admit()
            write(progress,dict(status="running",job=name,completed=completed,total=len(jobs)))
            print("START",name,flush=True)
        run_job(job,active,material,seal)
        if result.exists():
            completed += 1
            saved = read(result)
            immediate, = read(OUT/"jobs"/name/"attempt/initial_evaluation.json")
            assert immediate["model_sha256"] == saved["records"][0]["model_sha256"]
            assert abs(immediate["cross_entropy"] - saved["records"][0]["cross_entropy"]) <= 1e-6
            print("VERIFIED",name,flush=True)
        with (OUT/"report.lock").open("a") as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            report()
    write(progress,dict(status="complete",completed=completed,total=len(jobs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=1)
    args = parser.parse_args()
    worker(args.shard,args.shards)
