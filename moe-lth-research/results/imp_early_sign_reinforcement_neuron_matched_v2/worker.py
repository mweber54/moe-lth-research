"""Resume one shard after the matching seal passes and admit jobs on shared GPUs."""
import argparse
import fcntl
import socket
import time

from common_v2 import OUT, load, pilot as p

import gpu_policy


def admission(host):
    while True:
        samples = []
        for index in range(5):
            if index:
                time.sleep(2)
            samples.append(gpu_policy.query_gpu())
        usage = [sample["utilization_percent"] for sample in samples]
        okay = (min(sample["available_vram_mib"] for sample in samples) >= 53248
                and max(usage) <= 90 and sum(usage)/len(usage) <= 60)
        p.write(OUT / f"admission_{host}.json", dict(accepted=okay,samples=samples))
        if okay:
            return
        print("Waiting for shared GPU capacity", flush=True)
        time.sleep(30)


def run(shard):
    manifest = p.read(OUT / "inputs.json")
    seal = p.read(OUT / "matching_seal.json")
    assert seal["input_manifest"] == p.digest(OUT / "inputs.json")
    assert seal["rules"] == manifest["rules"]
    assert p.digest(seal["summary"]) == seal["summary_digest"]
    summary = p.read(seal["summary"])
    assert len(summary) == 18 and all(row["passed"] for row in summary)
    assert len(manifest["jobs"]) == 36 and len(seal["selections"]) == 18 and len(seal["matrix_audits"]) == 18
    for path, expected in (manifest["hashes"] | seal["matrix_audits"]).items():
        assert p.digest(path) == expected, path
    report = load("neuron_matched_report", OUT / "report.py")
    assert report.OUT == OUT
    print(f"Verified reporter {report.__file__} with output directory {report.OUT}", flush=True)
    p.OUT = OUT
    entries = {(row["seed"],row["order"]):row for row in manifest["inputs"]}
    hostname = socket.gethostname()
    progress = OUT / f"progress_{hostname}.json"
    jobs = [job for job in manifest["jobs"] if (0 if job["order"] == "A" else 1) == shard]
    completed, last_seed, material = 0, None, None
    for job in jobs:
        name = p.job_id(job)
        if (OUT / "stop.request").exists():
            p.write(progress, dict(status="drained",completed=completed,total=len(jobs)))
            return
        path = OUT / "jobs" / name / "result.json"
        if not path.exists():
            p.write(progress,dict(status="waiting_for_gpu",job=name,completed=completed,total=len(jobs)))
            admission(hostname)
            if (OUT / "stop.request").exists():
                p.write(progress, dict(status="drained",completed=completed,total=len(jobs)))
                return
            if job["seed"] != last_seed:
                del material
                material = p.N._load_training_material(job["seed"])
                last_seed = job["seed"]
            p.write(progress,dict(status="running",job=name,completed=completed,total=len(jobs)))
            print("START",name,flush=True)
            p.run_job(job, entries[job["seed"],job["order"]],material,seal)
        assert path.is_file(), path
        with (OUT / "report.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            report.main()
        print("VERIFIED",name,flush=True)
        completed += 1
    p.write(progress,dict(status="complete",completed=completed,total=len(jobs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    with (OUT / f"worker_{args.shard}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.shard)
