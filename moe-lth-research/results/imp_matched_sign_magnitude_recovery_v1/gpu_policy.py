"""Local, fail-closed GPU admission; importing this module never initializes CUDA."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = Path.cwd()
HOSTS = ("adder.cs.kent.edu", "viper.cs.kent.edu")
ENV = "MATCHED_GPU_POLICY"
SHARED_MIN_FREE_MIB = 53248
SHARED_SAMPLES = 5
SHARED_SAMPLE_INTERVAL_S = 2


def policy():
    raw = json.loads(os.environ.get(ENV, "{}"))
    allowed = {"min_free_mib", "exclusive_authorized_hosts", "allow_shared", "max_shared_utilization",
               "max_shared_mean_utilization",
               "authorization_path", "authorization_commit"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("Invalid GPU policy keys")
    minimum = raw.get("min_free_mib", {})
    if not isinstance(minimum, dict) or set(minimum) - set(HOSTS):
        raise ValueError("min_free_mib must map registered hostnames to MiB")
    if any(type(n) is not int or n < 0 for n in minimum.values()):
        raise ValueError("min_free_mib values must be nonnegative integers")
    exclusive = raw.get("exclusive_authorized_hosts", [])
    if not isinstance(exclusive, list) or set(exclusive) - set(HOSTS) or len(exclusive) != len(set(exclusive)):
        raise ValueError("exclusive_authorized_hosts must list registered hosts")
    shared = raw.get("allow_shared", False)
    maximum = raw.get("max_shared_utilization", 90)
    mean_maximum = raw.get("max_shared_mean_utilization", 60)
    if (type(shared) is not bool or type(maximum) is not int or not 0 <= maximum <= 90
            or type(mean_maximum) is not int or not 0 <= mean_maximum <= 60):
        raise ValueError("Invalid shared-GPU policy")
    if shared and any(n < SHARED_MIN_FREE_MIB for n in minimum.values()):
        raise ValueError("Shared minimum must meet the provisional 53248 MiB baseline")
    default_minimum = SHARED_MIN_FREE_MIB if shared else 0
    result = {"min_free_mib": {h: minimum.get(h, default_minimum) for h in HOSTS},
              "exclusive_authorized_hosts": sorted(exclusive),
              "allow_shared": shared, "max_shared_utilization": maximum,
              "max_shared_mean_utilization": mean_maximum}
    if shared:
        path, commit = raw.get("authorization_path"), raw.get("authorization_commit")
        if not isinstance(path, str) or not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("Shared policy requires a committed authorization and protocol amendment")
        repo = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip())
        file = (ROOT / path).resolve()
        relative = file.relative_to(repo).as_posix()
        data = file.read_bytes()
        registered = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
        if data != registered:
            raise ValueError("Shared authorization differs from its committed version")
        authorization = json.loads(data)
        if (authorization.get("experiment") != OUT.name
                or authorization.get("cluster_gpu_sharing_authorized") is not True
                or authorization.get("preregistration_amended_for_shared_gpu") is not True
                or authorization.get("preregistration_commit") != "b4e9451bd9de867bdca8e6f996bf93f62955de33"
                or not isinstance(authorization.get("cluster_authorization_reference"), str)
                or not authorization["cluster_authorization_reference"].strip()
                or not isinstance(authorization.get("protocol_amendment_text"), str)
                or not authorization["protocol_amendment_text"].strip()
                or not isinstance(authorization.get("hosts"), list)
                or not authorization["hosts"]
                or set(authorization["hosts"]) - set(HOSTS)):
            raise ValueError("Shared authorization lacks required scope or attestations")
        result["authorization"] = {
            "path": str(file), "commit": commit, "sha256": hashlib.sha256(data).hexdigest(),
            "hosts": authorization["hosts"],
        }
    elif "authorization_path" in raw or "authorization_commit" in raw:
        raise ValueError("Authorization fields require allow_shared=true")
    return result


def query_gpu():
    def query(fields):
        return subprocess.run(
            ["nvidia-smi", "--id=0", fields, "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15).stdout
    fields = query("--query-gpu=memory.total,memory.used,memory.free,utilization.gpu").strip().split(",")
    total, used, free, utilization = map(lambda x: int(x.strip()), fields)
    pids = [int(x.strip()) for x in query("--query-compute-apps=pid").splitlines()
            if x.strip() and int(x.strip()) != os.getpid()]
    if total <= 0 or not 0 <= used <= total or not 0 <= free <= total or not 0 <= utilization <= 100:
        raise ValueError("Invalid GPU telemetry")
    return {"memory_total_mib": total, "memory_used_mib": used,
            "available_vram_mib": free, "utilization_percent": utilization,
            "external_compute_pids": pids}


def decide(host, config, gpu):
    if host not in HOSTS:
        return False, "unregistered_host"
    if gpu["available_vram_mib"] < config["min_free_mib"][host]:
        return False, "insufficient_free_vram"
    if config["allow_shared"]:
        if host not in config["authorization"]["hosts"]:
            return False, "host_not_authorized_for_sharing"
        if gpu["utilization_percent"] > config["max_shared_utilization"]:
            return False, "shared_utilization_limit"
    else:
        if host not in config["exclusive_authorized_hosts"]:
            return False, "exclusive_allocation_not_authorized"
        if (gpu["external_compute_pids"]
                or gpu["memory_used_mib"] >= max(1024, gpu["memory_total_mib"] // 5)
                or gpu["utilization_percent"] > 10):
            return False, "exclusive_gpu_not_idle"
    return True, "accepted_shared_authorized" if config["allow_shared"] else "accepted_exclusive_idle"


def record_decision(record, out=OUT):
    directory = out / "gpu_admission"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.time_ns()}_{record['host']}_{os.getpid()}.json"
    with path.open("x") as f:
        json.dump(record, f, indent=2)
    return record


def unavailable(host, stage, reason):
    return record_decision({
        "host": host, "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": stage, "policy": policy(), "gpu": None, "accepted": False,
        "reason": "host_unavailable: " + reason,
    })


def admit(stage, *, out=OUT, host=None, config=None, query=query_gpu, sleep=time.sleep):
    host = host or socket.getfqdn()
    config = policy() if config is None else config
    record = {"host": host, "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "stage": stage, "policy": config, "gpu": None, "samples": [], "accepted": False}
    try:
        count = SHARED_SAMPLES if config["allow_shared"] else 1
        for index in range(count):
            if index:
                sleep(SHARED_SAMPLE_INTERVAL_S)
            record["gpu"] = query()
            record["samples"].append(record["gpu"])
        decisions = [decide(host, config, sample) for sample in record["samples"]]
        rejected = next((reason for accepted, reason in decisions if not accepted), None)
        if rejected is not None:
            record["reason"] = rejected
        elif config["allow_shared"] and (
                sum(s["utilization_percent"] for s in record["samples"]) / count
                > config["max_shared_mean_utilization"]):
            record["reason"] = "shared_mean_utilization_limit"
        else:
            record["accepted"] = True
            record["reason"] = decisions[-1][1]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["reason"] = "gpu_query_failed: " + repr(exc)
    return record_decision(record, out)


def dry_run():
    config = {"min_free_mib": {h: 20000 for h in HOSTS}, "exclusive_authorized_hosts": list(HOSTS),
              "allow_shared": False, "max_shared_utilization": 90, "max_shared_mean_utilization": 60}
    idle = {"memory_total_mib": 95830, "memory_used_mib": 4,
            "available_vram_mib": 95826, "utilization_percent": 0, "external_compute_pids": []}
    busy = dict(idle, memory_used_mib=1000, available_vram_mib=94830,
                external_compute_pids=[123])
    low = dict(idle, memory_used_mib=87830, available_vram_mib=8000)
    scenarios = [("adder_idle", HOSTS[0], idle), ("adder_other_user", HOSTS[0], busy),
                 ("viper_low_memory", HOSTS[1], low)]
    decisions = [{"scenario": name, "host": host, "accepted": decide(host, config, gpu)[0],
                  "reason": decide(host, config, gpu)[1]} for name, host, gpu in scenarios]
    return {"decisions": decisions, "training_attempts": 0}


if __name__ == "__main__":
    if sys.argv[1:] == ["dry-run"]:
        print(json.dumps(dry_run(), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "probe":
        print(json.dumps(admit(sys.argv[2])))
    else:
        raise SystemExit("usage: gpu_policy.py dry-run | probe STAGE")
