from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts/imp_recovery_order_stability_v1"
SPEC = importlib.util.spec_from_file_location("shared_gpu_worker_under_test", SCRIPT_DIR / "shared_gpu_worker.py")
sgw = importlib.util.module_from_spec(SPEC)
sys.modules["shared_gpu_worker_under_test"] = sgw
SPEC.loader.exec_module(sgw)

POLICY = sgw.GpuPolicy(job_memory_mib=40000, safety_margin_mib=12000, utilization_samples=4, sample_interval_s=0)


def sample(used=13225, util=0, total=95830, pids=(1371616, 1372332)):
    return {
        "memory_total_mib": total,
        "memory_used_mib": used,
        "own_memory_mib": 0,
        "utilization_percent": util,
        "external_compute_pids": list(pids),
    }


def decide(sampler):
    return sgw.decide_admission(POLICY, 0, sampler=sampler, sleep=lambda _s: None)


def test_admits_with_external_processes_low_util_and_enough_memory():
    admit, record = decide(lambda _d: sample())
    assert admit and record["reason"] == "shared_gpu_admitted"
    assert record["external_compute_pids"] == [1371616, 1372332]


def test_denies_insufficient_free_memory():
    admit, record = decide(lambda _d: sample(used=50000))
    assert not admit and record["reason"] == "insufficient_free_memory"


def test_waits_on_sustained_high_utilization():
    admit, record = decide(lambda _d: sample(util=95))
    assert not admit and record["reason"] == "sustained_high_utilization"
    strict = sgw.GpuPolicy(40000, 12000, 4, 0, max_mean_utilization=30, max_sample_utilization=90)
    readings = iter([5, 5, 80, 80])
    admit, _ = sgw.decide_admission(strict, 0, sampler=lambda _d: sample(util=next(readings)), sleep=lambda _s: None)
    assert not admit


def test_monitor_failure_is_conservative():
    def failing(_d):
        raise sgw.GpuMonitorError("nvidia-smi unavailable")

    admit, record = decide(failing)
    assert not admit and record["reason"].startswith("monitor_error")


def test_query_gpu_rejects_garbage_and_missing_binary():
    def garbage(*_a, **_k):
        class R:
            stdout = "not,a,number\n"

        return R()

    with pytest.raises(sgw.GpuMonitorError):
        sgw.query_gpu(0, run=garbage)

    def missing(*_a, **_k):
        raise FileNotFoundError("nvidia-smi")

    with pytest.raises(sgw.GpuMonitorError):
        sgw.query_gpu(0, run=missing)


def make_out(tmp_path, statuses, checkpoints=(), attempts=()):
    rows = []
    for seed in (7, 43, 61):
        for order in ("A", "B"):
            rows.append({"job_id": f"e32_seed{seed}_order_{order}_imp80", "seed": str(seed),
                         "order": order, "model_type": "imp80", "pilot": "False"})
    (tmp_path / "status").mkdir()
    for job_id, status in statuses.items():
        (tmp_path / "status" / f"{job_id}.json").write_text(json.dumps({"status": status}))
    for job_id in attempts:
        (tmp_path / "jobs" / job_id / "attempts" / "attempt_1").mkdir(parents=True)
    for job_id in checkpoints:
        path = tmp_path / "jobs" / job_id / "attempts" / "attempt_1" / "segment_0"
        path.mkdir(parents=True, exist_ok=True)
        (path / "interruption.pt").write_bytes(b"x")
    return rows


def plan(tmp_path, rows, retries=1, assigned=(43, 61)):
    return sgw.plan_jobs(tmp_path, rows, set(assigned), sgw.RetryPolicy(retries))


def test_completed_never_scheduled_pending_scheduled_host_respected(tmp_path):
    done = "e32_seed43_order_A_imp80"
    rows = make_out(tmp_path, {done: "COMPLETE"})
    runnable, skipped = plan(tmp_path, rows)
    ids = {row["job_id"]: action for row, action in runnable}
    assert done not in ids
    assert ids == {
        "e32_seed43_order_B_imp80": "fresh",
        "e32_seed61_order_A_imp80": "fresh",
        "e32_seed61_order_B_imp80": "fresh",
    }
    reasons = {s["job_id"]: s["reason"] for s in skipped}
    assert reasons["e32_seed7_order_A_imp80"] == "assigned_to_other_host"
    assert reasons[done] == "complete"


def test_interrupted_job_resumes_only_with_checkpoint(tmp_path):
    with_ckpt, without = "e32_seed43_order_A_imp80", "e32_seed43_order_B_imp80"
    rows = make_out(
        tmp_path,
        {with_ckpt: "RUNNING", without: "RUNNING"},
        checkpoints=[with_ckpt],
        attempts=[without],
    )
    runnable, _ = plan(tmp_path, rows)
    actions = {row["job_id"]: action for row, action in runnable}
    assert actions[with_ckpt] == "resume"
    assert actions[without] == "fresh"


def test_failed_retry_policy(tmp_path):
    failed = "e32_seed43_order_A_imp80"
    rows = make_out(tmp_path, {failed: "FAILED"}, attempts=[failed])
    assert failed in {r["job_id"]: a for r, a in plan(tmp_path, rows, retries=1)[0]}
    assert failed not in {r["job_id"]: a for r, a in plan(tmp_path, rows, retries=0)[0]}


def test_job_lock_prevents_duplicate_scheduling(tmp_path):
    job = "e32_seed43_order_A_imp80"
    rows = make_out(tmp_path, {})
    (tmp_path / "locks").mkdir()
    with (tmp_path / "locks" / f"{job}.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        runnable, skipped = plan(tmp_path, rows)
    assert job not in {r["job_id"] for r, _ in runnable}
    assert {"job_id": job, "reason": "locked_by_active_worker"} in skipped


def test_runner_provenance_still_verifies_and_runner_is_unmodified():
    run = sgw.load_runner()
    registration = run._verify_registration()
    assert registration["production_ready"]
    assert run._sha256_file(SCRIPT_DIR / "run.py") == (
        "d9decbc86ac65718a261acfc45d78b6992be091888f840b2e44ba31d3dd320b4"
    )
