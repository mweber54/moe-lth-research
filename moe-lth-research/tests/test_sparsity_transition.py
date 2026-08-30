from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from moe_lth.experiments.run_sparsity_transition import (
    HISTORICAL_SPARSITIES,
    REFERENCE_SEEDS,
    ROUTER_AGES,
    ROUTER_STEPS,
    TransitionPreflightError,
    _load_historical_rows,
    _next_attempt_dir,
    _record_matches,
)


FIELDS = [
    "reference_seed",
    "sparsity",
    "router_age",
    "router_step",
    "sparse_final_loss",
    "dense_final_loss",
    "ticket_gap",
    "mask_hash",
    "router_hash",
    "shared_state_hash",
    "training_sequence_hash",
    "validation_sequence_hash",
    "audit_passed",
]


def _historical(path: Path, *, duplicate: bool = False, nonfinite: bool = False) -> None:
    rows = []
    for seed in REFERENCE_SEEDS:
        for sparsity in HISTORICAL_SPARSITIES:
            for age in ROUTER_AGES:
                dense = 1.0 + seed / 1000
                gap = sparsity / 10 + age / 10000
                rows.append({
                    "reference_seed": seed,
                    "sparsity": sparsity,
                    "router_age": age,
                    "router_step": ROUTER_STEPS[age],
                    "sparse_final_loss": dense + gap,
                    "dense_final_loss": dense,
                    "ticket_gap": "nan" if nonfinite and not rows else gap,
                    "mask_hash": f"mask-{seed}-{sparsity}",
                    "router_hash": f"router-{seed}-{age}",
                    "shared_state_hash": f"shared-{seed}",
                    "training_sequence_hash": f"train-{seed}",
                    "validation_sequence_hash": "validation",
                    "audit_passed": True,
                })
    if duplicate:
        rows.append(dict(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_historical_loader_requires_exact_unique_finite_grid(tmp_path: Path):
    complete = tmp_path / "complete.csv"
    _historical(complete)
    rows = _load_historical_rows(complete)
    assert len(rows) == 45
    assert {row["router_training_step"] for row in rows} == {0, 500, 2500}

    duplicate = tmp_path / "duplicate.csv"
    _historical(duplicate, duplicate=True)
    with pytest.raises(TransitionPreflightError, match="duplicate"):
        _load_historical_rows(duplicate)

    nonfinite = tmp_path / "nonfinite.csv"
    _historical(nonfinite, nonfinite=True)
    with pytest.raises(TransitionPreflightError, match="nonfinite"):
        _load_historical_rows(nonfinite)


def test_dense_compatibility_requires_router_file_and_protocol_identity():
    expected = {
        "condition_type": "dense_control",
        "reference_seed": 7,
        "sparsity": 0.0,
        "router_age": 20,
        "router_training_step": 500,
        "router_checkpoint_hash": "router",
        "router_checkpoint_file_sha256": "checkpoint",
        "mask_hash": "dense_no_mask",
        "expert_state_hash": "expert",
        "shared_state_hash": "shared",
        "training_sequence_hash": "train",
        "validation_sequence_hash": "validation",
        "recovery_steps": 2500,
        "training_evaluation_config_id": "protocol",
    }
    record = {
        **expected,
        "completion_status": "complete",
        "protocol_version": "standard_moe_sparsity_transition_v1",
        "audit_passed": True,
        "final_validation_loss": 1.25,
    }
    assert _record_matches(record, expected)
    for field in ("router_checkpoint_hash", "router_checkpoint_file_sha256", "training_evaluation_config_id"):
        invalid = dict(record)
        invalid[field] = "wrong"
        assert not _record_matches(invalid, expected)


def test_retry_directory_never_overwrites_partial_artifacts(tmp_path: Path):
    base = tmp_path / "R20"
    base.mkdir()
    (base / "partial.txt").write_text("partial", encoding="utf-8")
    retry_one = _next_attempt_dir(base)
    assert retry_one.name == "R20_retry_001"
    retry_one.mkdir()
    (retry_one / "partial.txt").write_text("partial", encoding="utf-8")
    assert _next_attempt_dir(base).name == "R20_retry_002"


def test_historical_loader_rejects_wrong_router_step(tmp_path: Path):
    path = tmp_path / "wrong-step.csv"
    _historical(path)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    rows[0]["router_step"] = "250"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(TransitionPreflightError, match="step mismatch"):
        _load_historical_rows(path)
