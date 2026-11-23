"""Single manifest writer; one worker per host; fail closed on uncertain jobs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/nominal_exposure_dataset"))
from pilot_common import *
from support import OUT, validate_seal
from supervisor import dependencies
import argparse
import fcntl
import shlex
import shutil
import socket
import subprocess
import time

HOSTS = ("viper.cs.kent.edu", "adder.cs.kent.edu")
SSH = [
    "ssh", "-i", str(Path.home()/".ssh/dataset_b_cluster"),
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
]
PYTHON = ROOT/".venv-highres/bin/python"


def command(host, args):
    text = "cd " + shlex.quote(str(ROOT)) + " && exec " + shlex.join(args)
    if host == socket.getfqdn():
        return ["bash", "-c", text]
    return SSH + [host, text]


def free_memory(host):
    output = subprocess.check_output(
        command(host, [
            "nvidia-smi", "--id=0", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ]),
        text=True, timeout=30,
    )
    return int(output.strip())


def required_memory(job):
    # Sparse threshold informed by one 2862 MiB reserved-memory measurement.
    # Dense/source requirements are not measured: retain conservative admission.
    return 24576  # Conservative headroom for the newly lower-sparsity conditions.


def validate_inputs():
    validate_seal()


def dependencies(job, lookup):
    return lookup[job['dense_condition_id']]['status'] == 'REUSED'


def save_status(active, hosts, draining=False):
    jobs = pd.read_csv(OUT/"manifests/run_manifest.csv", dtype={"block_reason": "object", "execution_host": "object", "artifact_path": "object", "artifact_sha256": "object"})
    jsave(OUT/"queue_status.json", {
        "state": "DRAINING_AFTER_FAILURE" if draining else
                 ("RUNNING" if active else "WAITING_RESOURCES_OR_DEPENDENCIES"),
        "pid": os.getpid(),
        "host": socket.getfqdn(),
        "workers": {
            host: {
                "condition_id": rec["cid"],
                "launcher_pid": rec["process"].pid,
                "log": str(rec["log"]),
                "attempt": str(rec["dest"]),
            } for host, rec in active.items()
        },
        "hosts": hosts,
        "counts": jobs.status.value_counts().to_dict(),
        "updated_at": time.time(),
    })


def main(parent_lock_fd=None):
    if parent_lock_fd is None:
        raise RuntimeError("Use launch.py; parent experiment lock required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if socket.getfqdn() != HOSTS[0]:
        raise RuntimeError("Run the coordinator on viper")

    lock = (OUT/"queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate_inputs()

    jobs = pd.read_csv(OUT/"manifests/run_manifest.csv", dtype={"block_reason": "object", "execution_host": "object", "artifact_path": "object", "artifact_sha256": "object"})
    uncertain = jobs.status.eq("RUNNING") | jobs.status.eq("FAILED_UNCERTAIN")
    if uncertain.any():
        raise RuntimeError(
            "Resolve existing RUNNING/FAILED_UNCERTAIN jobs before restart: "
            + ", ".join(jobs.loc[uncertain, "condition_id"])
        )

    for host in HOSTS:
        print(host, "free_mib", free_memory(host), flush=True)
        subprocess.run(command(host, [
            str(PYTHON), "-c",
            "import socket,torch; "
            "assert torch.cuda.is_available(); "
            "assert torch.__version__.startswith('2.7.0'); "
            "print(socket.getfqdn(),torch.__version__)",
        ]), check=True, timeout=60)

    if args.check:
        print("CLUSTER CHECK PASS; no jobs launched", flush=True)
        return

    jsave(OUT/"queue_identity.json", {
        "pid": os.getpid(), "host": socket.getfqdn(),
        "started_at": time.time(), "hosts": HOSTS,
        "mode": "single_writer_two_hosts",
    })
    active = {}
    draining = False

    while True:
        jobs = pd.read_csv(OUT/"manifests/run_manifest.csv", dtype={"block_reason": "object", "execution_host": "object", "artifact_path": "object", "artifact_sha256": "object"})

        for host, rec in list(active.items()):
            rc = rec["process"].poll()
            if rc is None:
                continue
            rec["handle"].close()
            idx = jobs.index[jobs.condition_id.eq(rec["cid"])][0]
            try:
                if rc != 0:
                    raise RuntimeError(f"Launcher/worker exit {rc}")
                done = read(rec["dest"]/"pilot_completed.json")
                if done["condition_id"] != rec["cid"]:
                    raise RuntimeError("Completion identity mismatch")
                checkpoint = rec["dest"]/(
                    "checkpoints/step_10000.pt" if rec["kind"] == "source"
                    else "checkpoints/final_recovered.pt"
                )
                if sha(checkpoint) != done["final_checkpoint_sha256"]:
                    raise RuntimeError("Completion checkpoint hash mismatch")
                jobs.at[idx, "status"] = "COMPLETE"
                jobs.at[idx, "artifact_sha256"] = done["final_checkpoint_sha256"]
                jobs.at[idx, "block_reason"] = ""
                print("COMPLETE", host, rec["cid"], flush=True)
            except Exception as exc:
                jobs.at[idx, "status"] = "FAILED_UNCERTAIN"
                jobs.at[idx, "block_reason"] = (
                    f"{host}: {exc}; inspect remote process and {rec['log']} "
                    "before retrying"
                )
                draining = True
                print("STOP DISPATCHING:", rec["cid"], str(exc), flush=True)
            csv(OUT/"manifests/run_manifest.csv", jobs)
            del active[host]

        if draining and not active:
            save_status(active, {}, True)
            return

        lookup = {r["condition_id"]: r for r in jobs.to_dict("records")}
        ready = [
            r for r in lookup.values()
            if r["status"] == "PENDING" and dependencies(r, lookup)
        ]
        ready.sort(key=lambda r: (r["stage"], r["condition_id"]))
        host_status = {}

        for host in HOSTS:
            if host in active:
                host_status[host] = {"state": "BUSY"}
                continue
            if draining:
                host_status[host] = {"state": "DRAINING"}
                continue
            try:
                available = free_memory(host)
            except Exception as exc:
                host_status[host] = {"state": "UNREACHABLE", "detail": str(exc)}
                continue
            host_status[host] = {"state": "IDLE", "free_mib": available}

            if shutil.disk_usage(OUT).free < 30*1024**3:
                host_status[host]["state"] = "WAITING_STORAGE"
                continue
            job = next(
                (r for r in ready if available >= required_memory(r)), None
            )
            if job is None:
                host_status[host]["state"] = (
                    "WAITING_MEMORY" if ready else "WAITING_DEPENDENCIES"
                )
                continue

            cid = job["condition_id"]
            dest = safe_attempt(
                OUT/("sources" if job["job_type"] == "source" else "recoveries")/cid
            )
            log = OUT/"logs"/f"{cid}_{dest.name}_{host.split('.')[0]}.log"
            idx = jobs.index[jobs.condition_id.eq(cid)][0]
            jobs.at[idx, "status"] = "RUNNING"
            jobs.at[idx, "artifact_path"] = str(dest)
            jobs.at[idx, "execution_host"] = host
            csv(OUT/"manifests/run_manifest.csv", jobs)

            launch = [
                "env", "CUDA_VISIBLE_DEVICES=0", "OMP_NUM_THREADS=4",
                str(PYTHON), "-u",
                str(ROOT/"scripts/full_lower_sparsity/worker_adapter.py"),
                "--condition", cid, "--attempt", str(dest),
            ]
            handle = log.open("ab")
            try:
                process = subprocess.Popen(
                    command(host, launch),
                    stdin=subprocess.DEVNULL,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
            except Exception:
                handle.close()
                # Leave RUNNING intact: restart must resolve the claim explicitly.
                raise

            active[host] = {
                "process": process, "handle": handle, "cid": cid,
                "dest": dest, "log": log, "kind": job["job_type"],
            }
            ready.remove(job)
            print("LAUNCHED", host, cid, "log", log, flush=True)

        save_status(active, host_status, draining)

        if not active and not jobs.status.eq("PENDING").any():
            jsave(OUT/"queue_status.json", {
                "state": "COMPLETE" if jobs.status.isin(
                    ["COMPLETE", "REUSED"]
                ).all() else "STOPPED_WITH_NONCOMPLETE_JOBS",
                "counts": jobs.status.value_counts().to_dict(),
                "updated_at": time.time(),
            })
            return
        if not active and not ready and jobs.status.eq("PENDING").any():
            jsave(OUT/"queue_status.json", {
                "state": "BLOCKED_DEPENDENCIES",
                "counts": jobs.status.value_counts().to_dict(),
                "updated_at": time.time(),
            })
            return
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit("Use launch.py")
