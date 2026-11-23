"""Operational shared-GPU worker for imp_recovery_order_stability_v1.

This module owns only resource admission and job scheduling. All scientific
execution (initialization, masks, batch orders, optimization, evaluation,
checkpoint semantics) is delegated unmodified to the sealed ``run.py`` through
``run._run_one_job``. ``run.py`` is imported, never edited, and its registered
SHA-256 is verified by ``run._verify_registration`` before anything runs.

The module never signals, suspends or otherwise touches other users' CUDA
processes. It only reads ``nvidia-smi`` and decides whether to launch.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
AMENDMENT_NAME = "execution_policy_amendment_v1.json"


def load_runner():
    name = "imp_recovery_order_stability_v1_run"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GpuPolicy:
    # Estimated peak GPU memory of one recovery job (MiB). Observed ~37 GB in
    # nvidia-smi on an H100 NVL during a production job; rounded up.
    job_memory_mib: int = 40960
    # Free memory that must remain above the job requirement for other users.
    safety_margin_mib: int = 12288
    utilization_samples: int = 5
    sample_interval_s: float = 2.0
    max_mean_utilization: float = 60.0
    max_sample_utilization: float = 90.0

    @property
    def required_free_mib(self) -> int:
        return self.job_memory_mib + self.safety_margin_mib


class GpuMonitorError(RuntimeError):
    pass


def query_gpu(device_index: int, run: Callable = subprocess.run) -> dict:
    """One nvidia-smi sample. Raises GpuMonitorError on any failure."""
    try:
        gpu = run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        total, used, util = (int(part.strip()) for part in gpu.stdout.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        raise GpuMonitorError(f"gpu query failed: {error!r}") from error
    if total <= 0 or used < 0 or used > total or not 0 <= util <= 100:
        raise GpuMonitorError(f"implausible gpu reading: {total},{used},{util}")
    own_mib = 0
    external = []
    try:
        apps = run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in apps.stdout.splitlines():
            if not line.strip():
                continue
            pid_text, memory_text = (part.strip() for part in line.split(",")[:2])
            if int(pid_text) == os.getpid():
                own_mib += int(memory_text) if memory_text.isdigit() else 0
            else:
                external.append(int(pid_text))
    except (OSError, subprocess.SubprocessError, ValueError):
        # Process attribution is informational; conservatively count all used
        # memory as external if it is unavailable.
        own_mib = 0
        external = None
    return {
        "memory_total_mib": total,
        "memory_used_mib": used,
        "own_memory_mib": own_mib,
        "utilization_percent": util,
        "external_compute_pids": external,
    }


def decide_admission(
    policy: GpuPolicy,
    device_index: int = 0,
    sampler: Callable[[int], dict] = query_gpu,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, dict]:
    """Admit only if memory fits (with margin) and utilization is not sustained-high.

    External CUDA processes are allowed. Any monitoring failure denies admission.
    """
    samples = []
    record = {"policy": asdict(policy), "required_free_mib": policy.required_free_mib}
    try:
        for index in range(policy.utilization_samples):
            if index:
                sleep(policy.sample_interval_s)
            samples.append(sampler(device_index))
    except GpuMonitorError as error:
        record.update(admit=False, reason=f"monitor_error: {error}", samples=samples)
        return False, record
    free = [s["memory_total_mib"] - (s["memory_used_mib"] - s["own_memory_mib"]) for s in samples]
    utils = [s["utilization_percent"] for s in samples]
    record.update(
        samples=samples,
        min_free_mib=min(free),
        mean_utilization=sum(utils) / len(utils),
        max_utilization=max(utils),
        external_compute_pids=samples[-1]["external_compute_pids"],
    )
    if min(free) < policy.required_free_mib:
        record.update(admit=False, reason="insufficient_free_memory")
    elif max(utils) > policy.max_sample_utilization or record["mean_utilization"] > policy.max_mean_utilization:
        record.update(admit=False, reason="sustained_high_utilization")
    else:
        record.update(admit=True, reason="shared_gpu_admitted")
    return record["admit"], record


@dataclass(frozen=True)
class RetryPolicy:
    # Number of additional attempts allowed for a FAILED job; 0 never retries.
    max_failed_retries: int = 1


def _attempt_dirs(out: Path, job_id: str) -> list[Path]:
    return sorted((out / "jobs" / job_id / "attempts").glob("attempt_*"))


def has_checkpoint(out: Path, job_id: str) -> bool:
    for attempt in _attempt_dirs(out, job_id):
        if (
            (attempt / "segment_0/checkpoints/final_recovered.pt").is_file()
            or (attempt / "segment_0/interruption.pt").is_file()
            or (attempt / "segment_2500/interruption.pt").is_file()
        ):
            return True
    return False


def job_lock_is_free(out: Path, job_id: str) -> bool:
    path = out / "locks" / f"{job_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True


def plan_jobs(
    out: Path,
    rows: list[dict],
    assigned: set[int],
    retry: RetryPolicy,
    lock_free: Callable[[Path, str], bool] = job_lock_is_free,
) -> tuple[list[tuple[dict, str]], list[dict]]:
    """Return (runnable [(row, action)], skipped [{job_id, reason}]).

    Actions: "fresh" (new attempt) or "resume" (existing valid checkpoint).
    COMPLETE jobs are never scheduled; other hosts' seeds are never scheduled.
    """
    runnable, skipped = [], []
    for row in rows:
        job_id = row["job_id"]
        if int(row["seed"]) not in assigned:
            skipped.append({"job_id": job_id, "reason": "assigned_to_other_host"})
            continue
        path = out / "status" / f"{job_id}.json"
        status = json.loads(path.read_text()).get("status") if path.is_file() else "PENDING"
        if status == "COMPLETE":
            skipped.append({"job_id": job_id, "reason": "complete"})
            continue
        if not lock_free(out, job_id):
            skipped.append({"job_id": job_id, "reason": "locked_by_active_worker"})
            continue
        attempts = len(_attempt_dirs(out, job_id))
        if status == "FAILED" and attempts > retry.max_failed_retries:
            skipped.append({"job_id": job_id, "reason": "failed_retries_exhausted"})
            continue
        # RUNNING with a free lock means the owning process died: interrupted.
        runnable.append((row, "resume" if has_checkpoint(out, job_id) else "fresh"))
    runnable.sort(key=lambda item: (int(item[0]["seed"]), item[0]["order"], item[0]["model_type"]))
    return runnable, skipped


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def verify_amendment(out: Path) -> dict:
    path = out / AMENDMENT_NAME
    if not path.is_file():
        raise RuntimeError(f"Missing execution-policy amendment {path}")
    amendment = json.loads(path.read_text())
    actual = sha256_file(Path(__file__).resolve())
    if amendment.get("shared_gpu_worker_sha256") != actual:
        raise RuntimeError("shared_gpu_worker.py does not match the recorded amendment hash")
    return amendment


class _GpuSampler(threading.Thread):
    """Logs GPU conditions during a job; observation only, never acts on others."""

    def __init__(self, log: Path, device_index: int, job_id: str, period_s: float = 60.0):
        super().__init__(daemon=True)
        self.log, self.device_index, self.job_id, self.period_s = log, device_index, job_id, period_s
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(self.period_s):
            try:
                sample = query_gpu(self.device_index)
            except GpuMonitorError as error:
                sample = {"error": str(error)}
            _append_jsonl(self.log, {"job_id": self.job_id, "unix": time.time(), "during_job": sample})


def _status_report(run, out: Path, runnable, skipped) -> dict:
    return {
        "runnable": [{"job_id": row["job_id"], "action": action} for row, action in runnable],
        "skipped": skipped,
    }


def worker_loop(args, run=None) -> None:
    run = run or load_runner()
    out = run.OUT
    host = socket.getfqdn()
    assigned = set(run._assigned_seeds(host))
    registration = run._verify_registration()
    verify_amendment(out)
    host_report = run._verify_artifacts_on_host()
    if not registration.get("production_ready") or host not in registration.get("host_input_verification", {}):
        raise RuntimeError(f"Host {host} is not covered by the production seal")
    if not host_report.get("verified"):
        raise RuntimeError(f"Host input verification failed for {host}")
    if not run._pilot_gate_passed():
        raise RuntimeError("Pilot gate has not passed; refusing to dispatch")
    import torch

    if not torch.cuda.is_available() or not torch.__version__.startswith("2.7.0"):
        raise RuntimeError("Verified CUDA/Torch 2.7.0 environment is required")
    policy = GpuPolicy(
        job_memory_mib=args.job_memory_mib,
        safety_margin_mib=args.safety_margin_mib,
        utilization_samples=args.utilization_samples,
        sample_interval_s=args.sample_interval_s,
        max_mean_utilization=args.max_mean_utilization,
        max_sample_utilization=args.max_sample_utilization,
    )
    retry = RetryPolicy(args.max_failed_retries)
    admission_log = out / "workers" / f"{host}_admission.jsonl"
    state_path = out / "workers" / f"{host}.json"
    while True:
        rows = run._read_job_manifest()
        runnable, skipped = plan_jobs(out, rows, assigned, retry)
        if not runnable:
            blocked = [s for s in skipped if s["reason"] == "locked_by_active_worker"]
            if blocked:
                time.sleep(args.idle_wait)
                continue
            run._atomic_json(state_path, {"status": "DONE", "skipped": skipped, "updated_unix": time.time()})
            return
        row, action = runnable[0]
        admit, record = decide_admission(policy, args.device_index)
        record.update(job_id=row["job_id"], action=action, unix=time.time())
        _append_jsonl(admission_log, record)
        if not admit:
            run._atomic_json(
                state_path,
                {"status": "WAITING_FOR_GPU", "reason": record["reason"], "next_job": row["job_id"],
                 "admission": record, "updated_unix": time.time()},
            )
            time.sleep(args.idle_wait)
            continue
        job_id = row["job_id"]
        if action == "fresh":
            status_path = run._job_status_path(job_id)
            if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "RUNNING":
                # Interrupted before any checkpoint: demote so run.py starts a new
                # preserved attempt instead of demanding a checkpoint.
                run._write_status(job_id, status="PENDING", note="interrupted without checkpoint; restarting")
        run._atomic_json(
            state_path,
            {"status": "RUNNING", "job_id": job_id, "action": action, "pid": os.getpid(),
             "admission": record, "updated_unix": time.time()},
        )
        sampler = _GpuSampler(out / "workers" / f"{host}_gpu_during_jobs.jsonl", args.device_index, job_id)
        sampler.start()
        torch.cuda.reset_peak_memory_stats()
        try:
            run._run_one_job(row, resume=(action == "resume"), device_index=args.device_index)
        except BaseException as error:  # run.py records FAILED; continue per retry policy
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            print(f"job {job_id} failed: {error!r}", flush=True)
        finally:
            sampler.stop_event.set()
            _append_jsonl(
                out / "workers" / f"{host}_job_memory.jsonl",
                {"job_id": job_id, "unix": time.time(),
                 "max_memory_allocated_mib": torch.cuda.max_memory_allocated() // (1 << 20),
                 "max_memory_reserved_mib": torch.cuda.max_memory_reserved() // (1 << 20)},
            )
            torch.cuda.empty_cache()


def acquire_and_run(args, run=None) -> None:
    run = run or load_runner()
    host = socket.getfqdn()
    lock_path = run.OUT / "locks" / f"worker_{host}_gpu{args.device_index}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A worker already owns {host} GPU {args.device_index}") from error
        worker_loop(args, run)


def build_parser() -> argparse.ArgumentParser:
    defaults = GpuPolicy()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "plan"))
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--idle-wait", type=int, default=60)
    parser.add_argument("--job-memory-mib", type=int, default=defaults.job_memory_mib)
    parser.add_argument("--safety-margin-mib", type=int, default=defaults.safety_margin_mib)
    parser.add_argument("--utilization-samples", type=int, default=defaults.utilization_samples)
    parser.add_argument("--sample-interval-s", type=float, default=defaults.sample_interval_s)
    parser.add_argument("--max-mean-utilization", type=float, default=defaults.max_mean_utilization)
    parser.add_argument("--max-sample-utilization", type=float, default=defaults.max_sample_utilization)
    parser.add_argument("--max-failed-retries", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run = load_runner()
    run.torch.set_num_threads(2)
    if args.command == "plan":
        run._verify_registration()
        verify_amendment(run.OUT)
        assigned = set(run._assigned_seeds(socket.getfqdn()))
        runnable, skipped = plan_jobs(run.OUT, run._read_job_manifest(), assigned, RetryPolicy(args.max_failed_retries))
        policy = GpuPolicy(
            args.job_memory_mib, args.safety_margin_mib, args.utilization_samples,
            args.sample_interval_s, args.max_mean_utilization, args.max_sample_utilization,
        )
        admit, record = decide_admission(policy, args.device_index)
        print(json.dumps({"plan": _status_report(run, run.OUT, runnable, skipped),
                          "admit": admit, "admission": record}, indent=2))
    else:
        acquire_and_run(args, run)


if __name__ == "__main__":
    main()
