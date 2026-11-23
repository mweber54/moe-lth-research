"""Exclusive, resumable worker for the registered adaptation experiment."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch

from common import CONDITIONS, MODEL_TYPES, OUT, SEEDS, json_read, json_save, sha256
from engine import reconstruct_prefix, run_condition


MINIMUM_FREE_GPU_BYTES = 24 * 1024**3
MINIMUM_FREE_DISK_BYTES = 200 * 1024**3
MAX_GPU_UTILIZATION = 90
GPU_IDLE_WAIT_SECONDS = 600


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def exclusive_claim(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    token = uuid.uuid4().hex
    payload = {
        "pid": os.getpid(),
        "host": host,
        "token": token,
        "created_unix": time.time(),
    }
    while True:
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                existing = json_read(path)
            except (OSError, json.JSONDecodeError):
                raise RuntimeError(f"Claim exists but cannot be read; inspect it manually: {path}")
            if existing.get("host") != host:
                raise RuntimeError(f"Claim is held on another host; refusing to steal it: {path}")
            if _pid_alive(int(existing.get("pid", -1))):
                raise RuntimeError(f"Claim is held by a live process: {path}")
            path.unlink()
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            break
    try:
        yield
    finally:
        try:
            existing = json_read(path)
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing.get("token") == token:
            path.unlink()


def gpu_admission_check(wait_for_idle_seconds: int = 0) -> dict:
    deadline = time.monotonic() + wait_for_idle_seconds
    while True:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Could not verify GPU admission state: {exc}") from exc
        rows = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            free_mib, utilization = (int(value.strip()) for value in line.split(",", 1))
            rows.append({"free_bytes": free_mib * 1024**2, "utilization_percent": utilization})
        if not rows:
            raise RuntimeError("nvidia-smi reported no visible GPUs.")
        selected = rows[torch.cuda.current_device() if torch.cuda.is_available() else 0]
        if selected["utilization_percent"] <= MAX_GPU_UTILIZATION:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU utilization remains {selected['utilization_percent']}%; refusing to compete "
                f"above the {MAX_GPU_UTILIZATION}% admission limit."
            )
        time.sleep(min(5, max(0.0, deadline - time.monotonic())))
    if selected["free_bytes"] < MINIMUM_FREE_GPU_BYTES:
        raise RuntimeError(
            f"GPU has {selected['free_bytes']} free bytes; "
            f"the registered minimum is {MINIMUM_FREE_GPU_BYTES}."
        )
    free_disk = shutil.disk_usage(OUT).free
    if free_disk < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(f"Only {free_disk} disk bytes remain in the output filesystem.")
    return {
        "gpu": selected,
        "visible_gpus": len(rows),
        "free_disk_bytes": free_disk,
        "minimum_free_gpu_bytes": MINIMUM_FREE_GPU_BYTES,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "maximum_gpu_utilization_percent": MAX_GPU_UTILIZATION,
    }


def _job_id(seed: int, model_type: str, condition: str) -> str:
    return f"seed{seed}_{model_type}_{condition}"


def _attempt_directory(seed: int, model_type: str, condition: str) -> Path:
    job_dir = OUT / "attempts" / _job_id(seed, model_type, condition)
    incomplete = [
        path
        for path in sorted(job_dir.glob("attempt_*"))
        if (path / "interruption.pt").is_file() and not (path / "result.json").is_file()
    ]
    if len(incomplete) > 1:
        raise RuntimeError(
            f"Multiple incomplete attempts exist for {_job_id(seed, model_type, condition)}; "
            "select and resolve one manually."
        )
    if incomplete:
        return incomplete[0]
    return job_dir / f"attempt_{time.time_ns()}"


def _completed_result(seed: int, model_type: str, condition: str) -> dict | None:
    status_path = OUT / "status" / f"{_job_id(seed, model_type, condition)}.json"
    if not status_path.is_file():
        return None
    status = json_read(status_path)
    if status.get("status") != "COMPLETE":
        return None
    result_path = Path(status["result"])
    result = json_read(result_path)
    checkpoint = Path(result["final_checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != result["final_checkpoint_sha256"]:
        raise RuntimeError(f"Completed result hash verification failed: {result_path}")
    return result


def _endpoint_ce(attempt: str | Path) -> float:
    with (Path(attempt) / "trajectory_metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["global_step"]) == 1500:
                return float(row["ce"])
    raise RuntimeError(f"Attempt has no global-step-1500 validation readout: {attempt}")


def _run_job(seed: int, model_type: str, condition: str, device: torch.device) -> dict:
    completed = _completed_result(seed, model_type, condition)
    if completed is not None:
        return completed
    job_id = _job_id(seed, model_type, condition)
    claim_path = OUT / "status" / "claims" / f"{job_id}.json"
    with exclusive_claim(claim_path):
        completed = _completed_result(seed, model_type, condition)
        if completed is not None:
            return completed
        admission = gpu_admission_check(wait_for_idle_seconds=GPU_IDLE_WAIT_SECONDS)
        attempt_dir = _attempt_directory(seed, model_type, condition)
        json_save(
            OUT / "status" / f"{job_id}.json",
            {
                "status": "RUNNING",
                "seed": seed,
                "model_type": model_type,
                "condition": condition,
                "attempt": str(attempt_dir),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "heartbeat_unix": time.time(),
                "resource_admission": admission,
            },
        )
        try:
            return run_condition(
                seed=seed,
                model_type=model_type,
                condition=condition,
                device=device,
                attempt_dir=attempt_dir,
            )
        except Exception as exc:
            json_save(
                OUT / "status" / f"{job_id}.json",
                {
                    "status": "FAILED",
                    "seed": seed,
                    "model_type": model_type,
                    "condition": condition,
                    "attempt": str(attempt_dir),
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "heartbeat_unix": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise


def _ensure_prefix(seed: int, device: torch.device) -> dict:
    prefix_dir = OUT / "prefixes" / f"seed{seed}"
    verification = prefix_dir / "prefix_verification.json"
    status_path = OUT / "status" / f"seed{seed}_prefix.json"
    with exclusive_claim(OUT / "status" / "claims" / f"prefix-seed{seed}.json"):
        if verification.is_file():
            record = json_read(verification)
            if record.get("status") == "verified":
                return record
        json_save(
            status_path,
            {
                "status": "RUNNING",
                "seed": seed,
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "heartbeat_unix": time.time(),
            },
        )
        try:
            record = reconstruct_prefix(seed, device)
        except Exception as exc:
            json_save(
                status_path,
                {
                    "status": "FAILED",
                    "seed": seed,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "heartbeat_unix": time.time(),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        json_save(
            status_path,
            {
                "status": "COMPLETE",
                "seed": seed,
                "prefix_verification": str(verification),
                "prefix_sha256": sha256(prefix_dir / "reconstruction_interruption.pt"),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "heartbeat_unix": time.time(),
            },
        )
        return record


def run_pilot(device: torch.device, admission: dict) -> dict:
    _ensure_prefix(17, device)
    results = []
    try:
        for model_type in MODEL_TYPES:
            for condition in CONDITIONS:
                results.append(_run_job(17, model_type, condition, device))
        by_type: dict[str, list[dict]] = {}
        for result in results:
            by_type.setdefault(result["model_type"], []).append(result)
        comparisons = {}
        for model_type, rows in by_type.items():
            starts = []
            for row in rows:
                branch_path = Path(row["attempt"]) / "branch_start.json"
                branch = json_read(branch_path)
                starts.append(branch)
            state_hashes = {row["state_sha256"] for row in starts}
            losses = [float(row["baseline_ce"]) for row in starts]
            if len(state_hashes) != 1:
                raise RuntimeError(f"{model_type} pilot conditions did not start from identical model states.")
            if max(losses) - min(losses) > 1e-6:
                raise RuntimeError(f"{model_type} pilot baseline CE differs across conditions.")
            comparisons[model_type] = {
                "branch_start_state_hashes": sorted(state_hashes),
                "baseline_ce_by_condition": {
                    row["condition"]: float(row["baseline_ce"]) for row in starts
                },
                "baseline_ce_range": max(losses) - min(losses),
                "branch_count": len(rows),
            }
        record = {
            "status": "PASS",
            "seed": 17,
            "branches": [
                {
                    "model_type": row["model_type"],
                    "condition": row["condition"],
                    "attempt": row["attempt"],
                    "endpoint_ce": _endpoint_ce(row["attempt"]),
                    "final_checkpoint_sha256": row["final_checkpoint_sha256"],
                    "attempted_updates": row["attempted_updates"],
                    "applied_updates": row["applied_updates"],
                    "amp_skips": row["amp_skips"],
                }
                for row in results
            ],
            "baseline_equivalence": comparisons,
            "resource_admission": admission,
            "effect_direction_used_as_gate": False,
        }
    except Exception as exc:
        record = {
            "status": "FAIL",
            "seed": 17,
            "completed_branches": len(results),
            "error": f"{type(exc).__name__}: {exc}",
            "effect_direction_used_as_gate": False,
        }
        json_save(OUT / "pilot_validation.json", record)
        raise
    json_save(OUT / "pilot_validation.json", record)
    return record


def run_cohort(device: torch.device) -> dict:
    pilot_path = OUT / "pilot_validation.json"
    if not pilot_path.is_file() or json_read(pilot_path).get("status") != "PASS":
        raise RuntimeError("Cohort dispatch is blocked until the seed-17 pilot passes.")
    seed_results = []
    for seed in SEEDS:
        if seed == 17:
            continue
        _ensure_prefix(seed, device)
        for model_type in MODEL_TYPES:
            for condition in CONDITIONS:
                seed_results.append(_run_job(seed, model_type, condition, device))
    result = {
        "status": "COMPLETE",
        "seeds": [seed for seed in SEEDS if seed != 17],
        "branches": len(seed_results),
        "completed_unix": time.time(),
    }
    json_save(OUT / "cohort_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--pilot", action="store_true")
    choices.add_argument("--cohort", action="store_true")
    choices.add_argument("--job", nargs=3, metavar=("SEED", "MODEL_TYPE", "CONDITION"))
    args = parser.parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("The registered worker requires CUDA.")
    torch.cuda.set_device(0)
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", socket.gethostname())
    with exclusive_claim(OUT / "status" / "claims" / f"host-{host}.json"):
        admission = gpu_admission_check()
        json_save(OUT / "status" / "last_resource_admission.json", admission)
        if args.pilot:
            run_pilot(device, admission)
        elif args.cohort:
            run_cohort(device)
        else:
            seed = int(args.job[0])
            model_type, condition = args.job[1:]
            if seed not in SEEDS or model_type not in MODEL_TYPES or condition not in CONDITIONS:
                parser.error("Unknown seed, model type, or intervention condition.")
            if seed != 17 and (
                not (OUT / "pilot_validation.json").is_file()
                or json_read(OUT / "pilot_validation.json").get("status") != "PASS"
            ):
                raise RuntimeError("Non-pilot jobs are blocked until seed-17 validation passes.")
            _ensure_prefix(seed, device)
            _run_job(seed, model_type, condition, device)


if __name__ == "__main__":
    main()
