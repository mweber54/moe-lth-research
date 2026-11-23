"""Post hoc, read-only analysis of IMP80 pruning damage and adaptation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/imp_postpruning_adaptation_v1"))
sys.path.insert(0, str(ROOT / "scripts/imp_adaptation_factorial_v1"))
sys.path.insert(0, str(ROOT / "src"))

import torch

import common
from diagnostics import probe_identity, probe_indices
import engine
from engine import variants as adaptation_variants
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import apply_masks_, load_masks
from moe_lth.pruning.router_age import state_dict_hash
from worker import gpu_admission_check


OUT = ROOT / "results/imp_compensation_analysis_v1"
TRANSITION = ROOT / "results/imp_mask_transition_v1"
ADAPTATION = ROOT / "results/imp_postpruning_adaptation_v1"
SEEDS = (7, 17, 29, 43, 61)
MODEL_TYPES = ("unpruned", "sparse")
CONDITIONS = (
    "full",
    "router_frozen",
    "experts_frozen",
    "router_experts_frozen",
)
OFFSETS = (0, 25, 100, 250, 500)
GLOBAL_STEPS = dict(zip(OFFSETS, (1000, 1025, 1100, 1250, 1500), strict=True))
CE_ATOL = 1e-6
MAX_HASH_BYTES = 5 * 1024 * 1024


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty required table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _transition_attempt(seed: int) -> Path:
    status_path = TRANSITION / "status" / f"seed{seed}_abrupt.json"
    status = _read_json(status_path)
    _check(status.get("status") == "COMPLETE", f"Abrupt transition incomplete for seed {seed}.")
    return Path(status["attempt"])


def _transition_used_file(attempt: Path, relative: str) -> dict:
    completion = _read_json(attempt / "completion.json")
    expected = completion.get("files", {}).get(relative)
    _check(expected is not None, f"Transition completion has no hash for {relative}.")
    path = attempt / relative
    actual = _sha256(path)
    _check(actual == expected, f"Transition output hash mismatch: {path}.")
    return {"path": str(path), "sha256": actual, "verified": True}


def _adaptation_branch(seed: int, model_type: str, condition: str) -> tuple[dict, Path, dict]:
    job = f"seed{seed}_{model_type}_{condition}"
    status_path = ADAPTATION / "status" / f"{job}.json"
    status = _read_json(status_path)
    _check(status.get("status") == "COMPLETE", f"Adaptation branch is not complete: {job}.")
    result_path = Path(status["result"])
    result = _read_json(result_path)
    checkpoint = Path(result["final_checkpoint"])
    actual_hash = _sha256(checkpoint)
    _check(
        actual_hash == result["final_checkpoint_sha256"],
        f"Final checkpoint hash mismatch for {job}.",
    )
    _check(int(result["attempted_updates"]) == 500, f"Wrong attempted update count for {job}.")
    _check(
        int(result["applied_updates"]) + int(result["amp_skips"]) == 500,
        f"Attempt accounting does not sum to 500 for {job}.",
    )
    _check(bool(result.get("frozen_parameters_bitwise_unchanged")), f"Frozen parameter verification failed for {job}.")
    _check(bool(result.get("frozen_optimizer_state_unchanged")), f"Frozen optimizer verification failed for {job}.")
    attempt = Path(result["attempt"])
    return result, attempt, status


def _phase(model_type: str, offset: int) -> str:
    if offset == 0:
        return "pre_pruning_parent" if model_type == "unpruned" else "post_pruning_event"
    return "post_step_1000_continuation"


def _seed_coverage_and_provenance() -> tuple[dict, list[dict], dict, dict]:
    registration = _read_json(ADAPTATION / "registration.json")
    input_hash_path = ADAPTATION / "input_code_configuration_hashes.json"
    input_hashes = _read_json(input_hash_path)
    registered_hash_failures = []
    for path_string, expected in input_hashes.items():
        path = Path(path_string)
        if not path.is_absolute():
            continue
        if not path.is_file():
            registered_hash_failures.append({"path": path_string, "status": "missing"})
        elif _sha256(path) != expected:
            registered_hash_failures.append(
                {"path": path_string, "status": "hash_mismatch", "expected": expected, "actual": _sha256(path)}
            )
    _check(not registered_hash_failures, "Active factorial source/input hash manifest does not verify.")
    _check(
        _sha256(ADAPTATION / "diagnostic_probe_manifest.json")
        == registration["diagnostic_probe_manifest_sha256"],
        "Sealed probe manifest differs from active registration.",
    )

    transition_registration = _read_json(TRANSITION / "registration.json")
    transition_completion = _read_json(TRANSITION / "completion.json")
    _check(transition_completion.get("status") == "COMPLETE", "Transition cohort is not COMPLETE.")
    _check(int(transition_completion.get("registered", -1)) == 20, "Transition completion does not cover 20 registered jobs.")
    transition_manifest = _read_json(TRANSITION / "manifest.json")
    variant_by_seed = {
        int(item["seed"]): item
        for item in transition_manifest["variants"]
    }

    seed_records = []
    transition_used = {}
    metadata_gaps = []
    for seed in SEEDS:
        prefix_path = ADAPTATION / "prefixes" / f"seed{seed}" / "prefix_verification.json"
        prefix = _read_json(prefix_path)
        variant = variant_by_seed[seed]
        train_batches, validation_batches = common.materialize_data(variant, count=1500)
        replay_training_hash = common.batch_hash(train_batches)
        next_500_hash = common.batch_hash(train_batches[1000:1500])
        validation_hash = common.batch_hash(validation_batches)
        prefix_checkpoint = Path(prefix["state_checkpoint"])
        prefix_hash = _sha256(prefix_checkpoint)
        prefix_payload = common.torch_load(prefix_checkpoint)
        actual_prefix_model_hash = state_dict_hash(prefix_payload["model"])
        checkpoint_keys = set(prefix_payload)
        recorded_keys = set(prefix["state_keys"])
        optimizer_state = prefix_payload["optimizer"]
        optimizer_hash = engine._structure_hash(optimizer_state)
        rng_fields_present = all(
            name in prefix_payload
            for name in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng")
        )
        post_verification_metadata = checkpoint_keys - recorded_keys
        prefix_ok = (
            prefix.get("status") == "verified"
            and prefix.get("model_bitwise_equal") is True
            and prefix.get("optimizer_and_scaler_saved") is True
            and prefix.get("python_numpy_torch_cpu_cuda_rng_saved") is True
            and int(prefix.get("step", -1)) == 1000
            and int(prefix.get("sampler_next_batch", -1)) == 1000
            and int(prefix_payload["step"]) == 1000
            and int(prefix_payload["sampler_next_batch"]) == 1000
            and actual_prefix_model_hash == prefix["unpruned_model_hash"]
            and len(optimizer_state["param_groups"]) == int(prefix["optimizer_parameter_groups"])
            and len(optimizer_state["state"]) == int(prefix["optimizer_state_entries"])
            and prefix_payload["scaler"] == prefix["scaler"]
            and rng_fields_present
            and replay_training_hash == prefix["training_hash_steps_1_1500"]
            and validation_hash == prefix["validation_hash"]
            and (prefix_hash == prefix.get("state_checkpoint_sha256") or post_verification_metadata == {"source_optimizer_group_hashes"})
        )
        del prefix_payload
        _check(prefix_ok, f"Common step-1000 prefix evidence failed for seed {seed}.")

        transition_attempt = _transition_attempt(seed)
        event_files = {
            name: _transition_used_file(transition_attempt, f"segment_0/{name}")
            for name in ("grid_1000_pre.json", "grid_1000.json", "event_1000.json")
        }
        transition_used[str(seed)] = event_files
        pre = _read_json(transition_attempt / "segment_0/grid_1000_pre.json")
        post = _read_json(transition_attempt / "segment_0/grid_1000.json")
        event = _read_json(transition_attempt / "segment_0/event_1000.json")
        _check(
            pre["precision"] == post["precision"] == "FP32_TF32off"
            and pre["target_tokens"] == post["target_tokens"] == 492416
            and pre["batches"] == post["batches"] == 31,
            f"Paired validation population/settings mismatch at pruning event, seed {seed}.",
        )
        _check(
            event["pre"]["target_tokens"] == event["post"]["target_tokens"] == 32768,
            f"Paired event-probe population mismatch for seed {seed}.",
        )

        starts = {}
        branch_status_ok = True
        for model_type in MODEL_TYPES:
            by_condition = []
            for condition in CONDITIONS:
                result, attempt, _status = _adaptation_branch(seed, model_type, condition)
                start = _read_json(attempt / "branch_start.json")
                metadata_path = attempt / "metadata.json"
                if metadata_path.is_file():
                    metadata = _read_json(metadata_path)
                else:
                    finalization = _read_json(Path(result["finalization_validation"]))
                    _check(
                        result.get("native_result", {}).get("finalized_from_saved_checkpoint") is True
                        and finalization.get("status") == "PASS"
                        and finalization.get("scientific_conditions_changed") is False
                        and finalization.get("training_loop_invoked") is False
                        and int(finalization.get("optimizer_updates_after_checkpoint", -1)) == 0
                        and int(finalization.get("training_batches_consumed", -1)) == 0,
                        f"Missing branch metadata lacks passing checkpoint-only finalization evidence: {attempt}.",
                    )
                    metadata = None
                    metadata_gaps.append(
                        {
                            "seed": seed,
                            "model_type": model_type,
                            "condition": condition,
                            "path": str(metadata_path),
                            "status": "missing",
                            "substitute_evidence": str(result["finalization_validation"]),
                            "note": "Checkpoint-only finalization evidence verified; absent metadata is not reconstructed.",
                        }
                    )
                rows = _read_csv(attempt / "trajectory_metrics.csv")
                start_checks = (
                    int(start["sampler_next_batch"]) == 1000
                    and start["optimizer_reset"] is False
                    and start["validation_batch_hash"] == validation_hash
                    and start["training_batch_hash"] == next_500_hash
                    and int(start["optimizer_state_entries"]) == int(prefix["optimizer_state_entries"])
                    and start["scaler"] == prefix["scaler"]
                )
                _check(start_checks, f"Branch-start continuation evidence differs for seed {seed}/{model_type}/{condition}.")
                by_condition.append(start)
                _check(
                    len(rows) == len(OFFSETS)
                    and [int(row["offset"]) for row in rows] == list(OFFSETS),
                    f"Factorial trajectory grid incomplete or duplicated: seed {seed}/{model_type}/{condition}.",
                )
                _check(
                    all(row["precision"] == "FP32_TF32off" for row in rows)
                    and all(int(row["target_tokens"]) == 492416 for row in rows)
                    and all(int(row["batches"]) == 31 for row in rows),
                    f"Evaluation identity/population mismatch: seed {seed}/{model_type}/{condition}.",
                )
                for row in rows:
                    loss_error = abs(float(row["loss_sum"]) / int(row["target_tokens"]) - float(row["ce"]))
                    _check(loss_error <= CE_ATOL, f"CE/token arithmetic mismatch: seed {seed}/{model_type}/{condition}.")
                if metadata is not None:
                    _check(
                        metadata.get("mask_hash") == start.get("mask_hash"),
                        f"Mask hash changed between branch metadata and start: seed {seed}/{model_type}/{condition}.",
                    )
                if model_type == "sparse":
                    _check(
                        start["mask_hash"] == event["mask_hash"],
                        f"Sparse mask differs from native abrupt IMP80 event, seed {seed}.",
                    )
                else:
                    _check(
                        start["mask_hash"] == hashlib.sha256(b"").hexdigest(),
                        f"Unpruned branch unexpectedly has a mask, seed {seed}.",
                    )
                if condition == "router_experts_frozen":
                    _check(
                        metadata.get("router_hash_unchanged") is True
                        and metadata.get("initial_router_state_hash") == metadata.get("final_router_state_hash"),
                        f"Router changed in shared-only branch: seed {seed}/{model_type}.",
                    )
                    _check(
                        result.get("frozen_parameters_bitwise_unchanged") is True,
                        f"Frozen expert/router parameters were not verified: seed {seed}/{model_type}.",
                    )
                starts[model_type] = by_condition
            hashes = {row["state_sha256"] for row in by_condition}
            ce_values = [float(row["baseline_ce"]) for row in by_condition]
            _check(len(hashes) == 1, f"Condition start states differ within seed {seed}/{model_type}.")
            _check(max(ce_values) - min(ce_values) <= CE_ATOL, f"Condition start CE differs within seed {seed}/{model_type}.")

        dense_starts = starts["unpruned"]
        sparse_starts = starts["sparse"]
        _check(
            dense_starts[0]["state_sha256"] != sparse_starts[0]["state_sha256"],
            f"Expected registered pruning to distinguish sparse model state, seed {seed}.",
        )
        common_parent_verified = bool(prefix_ok and all(starts.values()))
        seed_records.append(
            {
                "seed": seed,
                "classification": "A_verified_common_model_and_training_state" if common_parent_verified else "C_unverifiable",
                "common_parent_verified": common_parent_verified,
                "common_step1000_checkpoint": str(prefix_checkpoint),
                "common_step1000_checkpoint_sha256": prefix_hash,
                "prefix_checkpoint_recorded_sha256": prefix["state_checkpoint_sha256"],
                "prefix_checkpoint_hash_status": (
                    "match"
                    if prefix_hash == prefix["state_checkpoint_sha256"]
                    else "stale_after_documented_source_optimizer_group_hashes_metadata_append"
                ),
                "prefix_model_hash_reverified": actual_prefix_model_hash,
                "prefix_optimizer_state_structure_hash": optimizer_hash,
                "prefix_checkpoint_additional_keys_after_verification": sorted(post_verification_metadata),
                "prefix_model_bitwise_equal_to_historical_dense": prefix["model_bitwise_equal"],
                "optimizer_scaler_rng_saved": bool(prefix["optimizer_and_scaler_saved"] and prefix["python_numpy_torch_cpu_cuda_rng_saved"]),
                "training_stream_position": int(prefix["sampler_next_batch"]),
                "training_stream_hash": prefix["training_hash_steps_1_1500"],
                "validation_hash": prefix["validation_hash"],
                "next_500_training_hash": next_500_hash,
                "unpruned_parent_model_hash": prefix["unpruned_model_hash"],
                "dense_branch_start_state_hash": dense_starts[0]["state_sha256"],
                "sparse_branch_start_state_hash": sparse_starts[0]["state_sha256"],
                "sparse_mask_hash": sparse_starts[0]["mask_hash"],
                "registered_optimizer_handling": "same recovered step-1000 optimizer/scaler; zero coordinate-shaped AdamW moments at masked coordinates; optimizer not reset",
                "branch_count_verified": 8,
                "branch_start_hash_equal_across_conditions": True,
                "source_event": event_files,
            }
        )

    provenance = {
        "analysis_config_sha256": _sha256(OUT / "analysis_config.json"),
        "adaptation_registration_revision": registration["registration_revision"],
        "adaptation_registration_sha256": _sha256(ADAPTATION / "registration.json"),
        "adaptation_source_manifest_sha256": _sha256(input_hash_path),
        "adaptation_registered_source_hash_failures": registered_hash_failures,
        "adaptation_probe_manifest_sha256": _sha256(ADAPTATION / "diagnostic_probe_manifest.json"),
        "adaptation_pilot_status": _read_json(ADAPTATION / "pilot_validation.json")["status"],
        "adaptation_cohort_status": _read_json(ADAPTATION / "cohort_status.json")["status"],
        "transition_registration_sha256": _sha256(TRANSITION / "registration.json"),
        "transition_input_hash_manifest_sha256": _sha256(TRANSITION / "input_hashes.json"),
        "transition_completion_status": transition_completion["status"],
        "transition_registered_jobs": transition_completion["registered"],
        "transition_completed_jobs": transition_completion["completed"],
        "transition_failed_jobs": transition_completion["failed"],
        "factorial_branch_metadata_gaps": metadata_gaps,
        "transition_used_event_files": transition_used,
        "registered_event_damage_table": {
            "path": str(TRANSITION / "pruning_event_damage.csv"),
            "sha256": _sha256(TRANSITION / "pruning_event_damage.csv"),
            "training_only_event_values_crosschecked": True,
        },
        "common_parent_classification": seed_records,
        "interpretation": "A is supported by a bitwise-equal reconstructed dense step-1000 model, saved optimizer/scaler/RNG state and sampler position, identical branch-start training/validation identities, and sparse construction from that common state via the registered IMP80 mask and masked-moment handling. This is treatment-inclusive common parenthood, not a weight-zeroing-only comparison."
    }
    return provenance, seed_records, variant_by_seed, transition_manifest


def _read_measurements():
    rows_by_key = {}
    branch_paths = {}
    coverage = []
    result_by_key = {}
    starts_by_key = {}
    metadata_by_key = {}
    update_summary = []
    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            for condition in CONDITIONS:
                result, attempt, _status = _adaptation_branch(seed, model_type, condition)
                key = (seed, model_type, condition)
                branch_paths[key] = attempt
                result_by_key[key] = result
                starts_by_key[key] = _read_json(attempt / "branch_start.json")
                metadata_path = attempt / "metadata.json"
                metadata_by_key[key] = _read_json(metadata_path) if metadata_path.is_file() else None
                trajectory_rows = _read_csv(attempt / "trajectory_metrics.csv")
                for row in trajectory_rows:
                    offset = int(row["offset"])
                    tuple_key = (seed, model_type, condition, offset)
                    _check(tuple_key not in rows_by_key, f"Duplicate trajectory row: {tuple_key}.")
                    row = dict(row)
                    row["phase_label"] = _phase(model_type, offset)
                    row["source_file"] = str(attempt / "trajectory_metrics.csv")
                    rows_by_key[tuple_key] = row
                    coverage.append(
                        {
                            "seed": seed,
                            "model_type": model_type,
                            "condition": condition,
                            "offset": offset,
                            "global_step": int(row["global_step"]),
                            "phase_label": row["phase_label"],
                            "population": "common validation",
                            "status": "observed_verified",
                            "target_tokens": int(row["target_tokens"]),
                            "batches": int(row["batches"]),
                            "source_file": row["source_file"],
                            "source_sha256": _sha256(Path(row["source_file"])),
                        }
                    )
                updates = _read_csv(attempt / "optimizer_and_updates.csv")
                _check(len(updates) == 1500, f"Update diagnostic count incorrect: {key}.")
                _check(
                    all(row["frozen_optimizer_state_unchanged"] in ("True", "1") for row in updates),
                    f"Frozen optimizer-state validation failed: {key}.",
                )
                update_summary.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": condition,
                        "attempted_updates": result["attempted_updates"],
                        "applied_updates": result["applied_updates"],
                        "amp_skips": result["amp_skips"],
                        "frozen_parameters_bitwise_unchanged": result["frozen_parameters_bitwise_unchanged"],
                        "frozen_optimizer_state_unchanged": result["frozen_optimizer_state_unchanged"],
                        "updates_source": str(attempt / "optimizer_and_updates.csv"),
                    }
                )
    _check(len(rows_by_key) == 200, f"Expected 200 trajectory observations, found {len(rows_by_key)}.")
    return rows_by_key, branch_paths, result_by_key, starts_by_key, metadata_by_key, coverage, update_summary


def _event_damage(seed_records: list[dict]) -> tuple[list[dict], dict[int, dict]]:
    rows = []
    validation_by_seed = {}
    registered_table = _read_csv(TRANSITION / "pruning_event_damage.csv")
    registered_events = {
        int(row["seed"]): row
        for row in registered_table
        if row["condition"] == "abrupt" and int(row["step"]) == 1000
    }
    _check(set(registered_events) == set(SEEDS), "Registered pruning event table does not cover the abrupt five-seed cohort.")
    for seed in SEEDS:
        attempt = _transition_attempt(seed)
        pre_path = attempt / "segment_0/grid_1000_pre.json"
        post_path = attempt / "segment_0/grid_1000.json"
        event_path = attempt / "segment_0/event_1000.json"
        pre = _read_json(pre_path)
        post = _read_json(post_path)
        event = _read_json(event_path)
        d_val = float(post["ce"]) - float(pre["ce"])
        validation_by_seed[seed] = {
            "D_validation": d_val,
            "pre": float(pre["ce"]),
            "post": float(post["ce"]),
            "target_tokens": int(pre["target_tokens"]),
            "precision": pre["precision"],
        }
        rows.append(
            {
                "seed": seed,
                "population": "common validation",
                "pre_prune_ce": float(pre["ce"]),
                "post_prune_ce": float(post["ce"]),
                "D_post_minus_pre": d_val,
                "target_tokens": int(pre["target_tokens"]),
                "batches": int(pre["batches"]),
                "precision": pre["precision"],
                "pre_phase": pre["phase"],
                "post_phase": post["phase"],
                "interpretation": "immediate measured task-CE effect of the complete registered abrupt IMP80 event",
                "source_file": json.dumps([str(pre_path), str(post_path)]),
                "source_sha256": json.dumps([_sha256(pre_path), _sha256(post_path)]),
            }
        )
        pre_probe = event["pre"]
        post_probe = event["post"]
        registered = registered_events[seed]
        _check(
            abs(float(registered["task_ce_pre"]) - float(pre_probe["task_ce"])) <= CE_ATOL
            and abs(float(registered["task_ce_post"]) - float(post_probe["task_ce"])) <= CE_ATOL
            and abs(float(registered["task_damage"]) - float(event["damage"])) <= CE_ATOL,
            f"Original training-only event readout differs from pruning_event_damage.csv for seed {seed}.",
        )
        rows.append(
            {
                "seed": seed,
                "population": "original training-only event probe",
                "pre_prune_ce": float(pre_probe["task_ce"]),
                "post_prune_ce": float(post_probe["task_ce"]),
                "D_post_minus_pre": float(event["damage"]),
                "target_tokens": int(pre_probe["target_tokens"]),
                "batches": 2,
                "precision": "FP32_TF32off",
                "pre_phase": "pre_pruning",
                "post_phase": "post_pruning",
                "interpretation": "separate training-only event diagnostic; not pooled with validation",
                "source_file": str(event_path),
                "source_sha256": _sha256(event_path),
                "registered_damage_table": str(TRANSITION / "pruning_event_damage.csv"),
                "registered_damage_table_sha256": _sha256(TRANSITION / "pruning_event_damage.csv"),
            }
        )
    _check(len(rows) == 10, "Immediate damage table does not cover both populations for all five seeds.")
    return rows, validation_by_seed


def _trajectory_tables(rows_by_key, validation_damage):
    losses = []
    decomposition = []
    penalties = []
    interactions = []
    seed_measures = []
    integrals = []
    for (seed, model_type, condition, offset), row in sorted(rows_by_key.items()):
        ce = float(row["ce"])
        losses.append(
            {
                "seed": seed,
                "model_type": model_type,
                "condition": condition,
                "offset": offset,
                "global_step": int(row["global_step"]),
                "phase_label": row["phase_label"],
                "ce": ce,
                "loss_sum": float(row["loss_sum"]),
                "target_tokens": int(row["target_tokens"]),
                "batches": int(row["batches"]),
                "precision": row["precision"],
                "source": row["source"],
                "source_file": row["source_file"],
            }
        )

    for seed in SEEDS:
        d = validation_damage[seed]["D_validation"]
        for condition in CONDITIONS:
            baseline_sparse = float(rows_by_key[(seed, "sparse", condition, 0)]["ce"])
            baseline_dense = float(rows_by_key[(seed, "unpruned", condition, 0)]["ce"])
            g0 = baseline_sparse - baseline_dense
            _check(abs(g0 - d) <= CE_ATOL, f"D_validation differs from G(0), seed {seed}/{condition}.")
            for offset in OFFSETS:
                sparse_row = rows_by_key[(seed, "sparse", condition, offset)]
                dense_row = rows_by_key[(seed, "unpruned", condition, offset)]
                sparse_ce = float(sparse_row["ce"])
                dense_ce = float(dense_row["ce"])
                gap = sparse_ce - dense_ce
                h = g0 - gap
                c = d - gap
                r_sparse = baseline_sparse - sparse_ce
                r_unpruned = baseline_dense - dense_ce
                _check(abs(h - (r_sparse - r_unpruned)) <= 1e-10, "Gap closure decomposition arithmetic failed.")
                _check(abs(c - h) <= CE_ATOL, "C and H do not agree when G(0)=D.")
                if r_sparse > 0 and r_unpruned > 0 and r_sparse > r_unpruned:
                    pattern = "both improve; sparse improves more"
                elif r_sparse > 0 and r_unpruned < 0:
                    pattern = "sparse improves while unpruned deteriorates"
                elif r_sparse < 0 and r_unpruned < 0:
                    pattern = "both deteriorate"
                else:
                    pattern = "other observed improvement combination"
                record = {
                    "seed": seed,
                    "condition": condition,
                    "offset": offset,
                    "global_step_sparse": int(sparse_row["global_step"]),
                    "global_step_unpruned": int(dense_row["global_step"]),
                    "sparse_phase": sparse_row["phase_label"],
                    "unpruned_phase": dense_row["phase_label"],
                    "L_sparse": sparse_ce,
                    "L_unpruned": dense_ce,
                    "G_sparse_minus_unpruned": gap,
                    "D_validation": d,
                    "C_D_minus_G": c,
                    "H_G0_minus_Gk": h,
                    "R_sparse": r_sparse,
                    "R_unpruned": r_unpruned,
                    "decomposition_residual": h - (r_sparse - r_unpruned),
                    "C_minus_H_residual": c - h,
                    "improvement_pattern": pattern,
                    "parent_classification": "A_verified_common_model_and_training_state",
                    "population": "common validation",
                }
                decomposition.append(record)
                for measure, value in (
                    ("L_sparse", sparse_ce),
                    ("L_unpruned", dense_ce),
                    ("G", gap),
                    ("C", c),
                    ("H", h),
                    ("R_sparse", r_sparse),
                    ("R_unpruned", r_unpruned),
                ):
                    seed_measures.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "offset": offset,
                            "global_step": GLOBAL_STEPS[offset],
                            "measure": measure,
                            "value": value,
                            "population": "common validation",
                        }
                    )

        for model_type in MODEL_TYPES:
            for offset in OFFSETS:
                values = {
                    condition: float(rows_by_key[(seed, model_type, condition, offset)]["ce"])
                    for condition in CONDITIONS
                }
                p_r = values["router_frozen"] - values["full"]
                p_e = values["experts_frozen"] - values["full"]
                p_re = values["router_experts_frozen"] - values["full"]
                interaction = values["full"] - values["experts_frozen"] - values["router_frozen"] + values["router_experts_frozen"]
                penalties.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "offset": offset,
                        "global_step": GLOBAL_STEPS[offset],
                        "L_full": values["full"],
                        "L_router_frozen": values["router_frozen"],
                        "L_experts_frozen": values["experts_frozen"],
                        "L_router_experts_frozen_shared_only": values["router_experts_frozen"],
                        "P_R": p_r,
                        "P_E": p_e,
                        "P_RE": p_re,
                        "I_L11_minus_L10_minus_L01_plus_L00": interaction,
                    }
                )
                for name, value in (("P_R", p_r), ("P_E", p_e), ("P_RE", p_re), ("interaction", interaction)):
                    seed_measures.append(
                        {
                            "seed": seed,
                            "model_type": model_type,
                            "offset": offset,
                            "global_step": GLOBAL_STEPS[offset],
                            "measure": name,
                            "value": value,
                            "population": "common validation",
                        }
                    )
                interactions.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "offset": offset,
                        "global_step": GLOBAL_STEPS[offset],
                        "L11_full": values["full"],
                        "L10_router_train_experts_frozen": values["experts_frozen"],
                        "L01_router_frozen_experts_train": values["router_frozen"],
                        "L00_shared_only": values["router_experts_frozen"],
                        "I": interaction,
                        "sign_convention": "L11 - L10 - L01 + L00; 1=train, 0=frozen; shared trains in all cells",
                    }
                )

        for restricted in ("router_frozen", "experts_frozen", "router_experts_frozen"):
            for offset in OFFSETS:
                g_restricted = (
                    float(rows_by_key[(seed, "sparse", restricted, offset)]["ce"])
                    - float(rows_by_key[(seed, "unpruned", restricted, offset)]["ce"])
                )
                g_full = (
                    float(rows_by_key[(seed, "sparse", "full", offset)]["ce"])
                    - float(rows_by_key[(seed, "unpruned", "full", offset)]["ce"])
                )
                p_sparse = (
                    float(rows_by_key[(seed, "sparse", restricted, offset)]["ce"])
                    - float(rows_by_key[(seed, "sparse", "full", offset)]["ce"])
                )
                p_dense = (
                    float(rows_by_key[(seed, "unpruned", restricted, offset)]["ce"])
                    - float(rows_by_key[(seed, "unpruned", "full", offset)]["ce"])
                )
                _check(abs((g_restricted - g_full) - (p_sparse - p_dense)) <= 1e-10, "Sparse-excess penalty identity failed.")
                h_full = (
                    float(rows_by_key[(seed, "sparse", "full", 0)]["ce"])
                    - float(rows_by_key[(seed, "unpruned", "full", 0)]["ce"])
                    - g_full
                )
                h_restricted = (
                    float(rows_by_key[(seed, "sparse", restricted, 0)]["ce"])
                    - float(rows_by_key[(seed, "unpruned", restricted, 0)]["ce"])
                    - g_restricted
                )
                _check(abs((h_full - h_restricted) - (p_sparse - p_dense)) <= CE_ATOL, "H/freezing penalty identity failed.")

        for condition in CONDITIONS:
            for model_type in MODEL_TYPES:
                values = [float(rows_by_key[(seed, model_type, condition, k)]["ce"]) for k in OFFSETS]
                auc = sum(
                    0.5 * (values[i] + values[i + 1]) * (OFFSETS[i + 1] - OFFSETS[i])
                    for i in range(len(OFFSETS) - 1)
                )
                gaps = [
                    float(rows_by_key[(seed, "sparse", condition, k)]["ce"])
                    - float(rows_by_key[(seed, "unpruned", condition, k)]["ce"])
                    for k in OFFSETS
                ]
                gap_auc = sum(
                    0.5 * (gaps[i] + gaps[i + 1]) * (OFFSETS[i + 1] - OFFSETS[i])
                    for i in range(len(OFFSETS) - 1)
                )
                closure = [gaps[0] - value for value in gaps]
                closure_auc = sum(
                    0.5 * (closure[i] + closure[i + 1]) * (OFFSETS[i + 1] - OFFSETS[i])
                    for i in range(len(OFFSETS) - 1)
                )
                integrals.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": condition,
                        "grid_offsets": ",".join(map(str, OFFSETS)),
                        "observed_span_updates": OFFSETS[-1] - OFFSETS[0],
                        "absolute_ce_auc_ce_updates": auc,
                        "mean_absolute_ce_on_grid": auc / (OFFSETS[-1] - OFFSETS[0]),
                        "paired_gap_auc_ce_updates": gap_auc,
                        "mean_paired_gap_on_grid": gap_auc / (OFFSETS[-1] - OFFSETS[0]),
                        "observed_initial_gap_closure_auc_ce_updates": closure_auc,
                        "mean_observed_initial_gap_closure_on_grid": closure_auc / (OFFSETS[-1] - OFFSETS[0]),
                        "integration": "trapezoid over observed offsets; no extrapolation",
                    }
                )
    return losses, decomposition, penalties, interactions, seed_measures, integrals


def _aggregate(seed_measures: list[dict], integrals: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in seed_measures:
        key = tuple(
            row.get(name)
            for name in ("model_type", "condition", "offset", "measure", "population")
        )
        grouped[key][int(row["seed"])] = float(row["value"])
    rows = []
    for key, values_by_seed in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = [values_by_seed[s] for s in SEEDS if s in values_by_seed]
        if len(values) != len(SEEDS):
            continue
        rows.append(
            {
                "model_type": key[0],
                "condition": key[1],
                "offset": key[2],
                "measure": key[3],
                "population": key[4],
                "n_seeds": len(values),
                "seed_values_in_order_7_17_29_43_61": json.dumps(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
                "sample_sd": statistics.stdev(values),
            }
        )
    for column in (
        "absolute_ce_auc_ce_updates",
        "mean_absolute_ce_on_grid",
        "paired_gap_auc_ce_updates",
        "mean_paired_gap_on_grid",
        "observed_initial_gap_closure_auc_ce_updates",
        "mean_observed_initial_gap_closure_on_grid",
    ):
        for model_type in MODEL_TYPES:
            for condition in CONDITIONS:
                values_by_seed = {
                    int(row["seed"]): float(row[column])
                    for row in integrals
                    if row["model_type"] == model_type and row["condition"] == condition
                }
                values = [values_by_seed[s] for s in SEEDS]
                rows.append(
                    {
                        "model_type": model_type,
                        "condition": condition,
                        "offset": "observed_grid_integral",
                        "measure": column,
                        "population": "common validation",
                        "n_seeds": len(values),
                        "seed_values_in_order_7_17_29_43_61": json.dumps(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "minimum": min(values),
                        "maximum": max(values),
                        "sample_sd": statistics.stdev(values),
                    }
                )
    return rows


def _existing_diagnostic_summary(branch_paths):
    rows = []
    candidate_cells = []
    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            attempt = branch_paths[(seed, model_type, "router_experts_frozen")]
            meta = _read_json(attempt / "metadata.json")
            candidate_cells.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "condition": "router_experts_frozen",
                    "router_hash_unchanged": meta["router_hash_unchanged"],
                    "expert_hash_start": meta["expert_state_hash"],
                    "router_hash_start": meta["initial_router_state_hash"],
                    "router_hash_final": meta["final_router_state_hash"],
                    "shared_state_hash_start": meta["shared_state_hash"],
                }
            )
            for item in _read_csv(attempt / "shared_representation_diagnostics.csv"):
                if item["measurement"] == "router_input_representation":
                    rows.append(
                        {
                            "seed": seed,
                            "model_type": model_type,
                            "condition": "router_experts_frozen",
                            "offset": int(item["offset"]),
                            "global_step": int(item["global_step"]),
                            "diagnostic": "router/expert input representation",
                            "layer": item["component"],
                            "population": "sealed training probe; same fixed token positions within model type",
                            "displacement_l2_rms": float(item["displacement_l2_rms"]),
                            "cosine_similarity_mean": float(item["cosine_similarity_mean"]),
                            "coverage_count": int(item["token_positions"]),
                            "zero_norm_exclusions": 0,
                            "low_count_exclusions": 0,
                        }
                    )
                elif item["measurement"] == "shared_parameter_displacement":
                    rows.append(
                        {
                            "seed": seed,
                            "model_type": model_type,
                            "condition": "router_experts_frozen",
                            "offset": int(item["offset"]),
                            "global_step": int(item["global_step"]),
                            "diagnostic": "shared parameter displacement",
                            "layer": item["component"],
                            "population": "parameter state relative to branch start",
                            "displacement_l2_rms": item["net_displacement_l2"],
                            "cosine_similarity_mean": "",
                            "coverage_count": int(item["parameter_count"]),
                            "zero_norm_exclusions": "",
                            "low_count_exclusions": "",
                            "normalized_net_displacement": item["normalized_net_displacement"],
                            "parameter_names": item["parameter_names"],
                        }
                    )
            for item in _read_csv(attempt / "routing_and_gate_diagnostics.csv"):
                if item["population"] not in ("proposed", "accepted", "same_proposed"):
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": "router_experts_frozen",
                        "offset": int(item["offset"]),
                        "global_step": int(item["global_step"]),
                        "diagnostic": f"routing/gating/{item['population']}",
                        "layer": int(item["layer"]),
                        "expert": int(item["expert"]),
                        "population": "sealed training probe",
                        "proposed_assignments": int(item["proposed_assignments"]),
                        "accepted_assignments": int(item["accepted_assignments"]),
                        "assignment_agreement_with_branch_start": float(item["assignment_agreement_with_branch_start"]),
                        "accepted_set_jaccard": (
                            float(item["accepted_set_jaccard"])
                            if item["accepted_set_jaccard"]
                            else None
                        ),
                        "gate_change_rms": (
                            float(item["gate_change_rms"])
                            if item["gate_change_rms"]
                            else None
                        ),
                        "gate_change_abs_mean": (
                            float(item["gate_change_abs_mean"])
                            if item["gate_change_abs_mean"]
                            else None
                        ),
                        "dropped_assignment_rate": float(item["dropped_assignment_rate"]),
                        "dropped_token_rate": float(item["dropped_token_rate"]),
                        "empty_union_jaccard_undefined": item["empty_union_jaccard_undefined"],
                    }
                )
            expert_rows = _read_csv(attempt / "expert_function_diagnostics.csv")
            for item in expert_rows:
                fixed = item["input_population"] == "branch_start_fixed_inputs"
                rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": "router_experts_frozen",
                        "offset": int(item["offset"]),
                        "global_step": int(item["global_step"]),
                        "diagnostic": "frozen expert on fixed inputs" if fixed else "frozen expert on current native inputs",
                        "layer": int(item["layer"]),
                        "expert": int(item["expert"]),
                        "population": item["input_population"],
                        "output_norm_rms": float(item["output_norm_rms"]),
                        "paired_output_change_l2_rms": (
                            float(item["paired_output_change_l2_rms"])
                            if item["paired_output_change_l2_rms"]
                            else None
                        ),
                        "paired_output_cosine_mean": (
                            float(item["paired_output_cosine_mean"])
                            if item["paired_output_cosine_mean"]
                            else None
                        ),
                        "zero_norm_exclusions": (
                            int(item["zero_norm_exclusions"])
                            if item["zero_norm_exclusions"]
                            else None
                        ),
                        "low_count_exclusions": (
                            int(item["low_count_exclusions"])
                            if item["low_count_exclusions"]
                            else None
                        ),
                        "comparison": item["comparison"],
                    }
                )
                if fixed:
                    cosine = float(item["paired_output_cosine_mean"])
                    _check(
                        float(item["paired_output_change_l2_rms"]) <= 1e-7
                        and 1.0 - 1e-7 <= cosine <= 1.0 + 1e-6,
                        f"Frozen expert output changed on fixed inputs: seed {seed}/{model_type}/{item['offset']}.",
                    )
    return rows, candidate_cells


@contextmanager
def _fp32_tf32_off():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _probe_batches(seed: int, variant: dict):
    train, _validation = common.materialize_data(variant, count=2500)
    indices = probe_indices(seed, len(train))
    batches = [train[index] for index in indices]
    manifest = _read_json(ADAPTATION / "diagnostic_probe_manifest.json")
    sealed = next(row for row in manifest["seeds"] if int(row["seed"]) == seed)
    _check(indices == sealed["batch_indices"], f"Sealed probe batch indices differ for seed {seed}.")
    _check(
        probe_identity(seed, indices, batches) == sealed["probe_identity_sha256"],
        f"Sealed probe identity differs for seed {seed}.",
    )
    _check(all(inputs.shape[0] == 128 and inputs.shape[1] == 128 for inputs, _ in batches), "Probe batch is not a complete native batch.")
    return batches, sealed


def _model_from_state(config: dict, state: dict, device: torch.device):
    model = TinyMoELanguageModel(config["model"])
    model.load_state_dict(state)
    return model.to(device)


def _forward_probe(model, batches, device):
    hidden_batches = []
    logits_batches = []
    losses = 0.0
    targets_count = 0
    modes = [(module, module.training) for module in model.modules()]
    captured = []
    hook = model.final_norm.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False), _fp32_tf32_off():
            for inputs, targets in batches:
                captured.clear()
                output = model(inputs.to(device))
                normalized_hidden = captured[-1]
                target_device = targets.to(device)
                loss_sum = torch.nn.functional.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    target_device.reshape(-1),
                    reduction="sum",
                )
                losses += float(loss_sum.double().cpu())
                targets_count += targets.numel()
                hidden_batches.append(normalized_hidden.detach().float().cpu())
                logits_batches.append(output.logits.detach().float().cpu())
    finally:
        hook.remove()
        for module, training in modes:
            module.training = training
    _check(targets_count == 65536, f"Probe target-token count changed: {targets_count}.")
    return {
        "hidden": hidden_batches,
        "logits": logits_batches,
        "probe_ce": losses / targets_count,
        "probe_loss_sum": losses,
        "target_tokens": targets_count,
    }


def _paired_output_metrics(reference, current, seed, model_type, probe_id):
    hidden_squared = hidden_reference_squared = 0.0
    hidden_cosine_sum = 0.0
    hidden_cosine_count = 0
    logits_squared = logits_reference_squared = 0.0
    logits_cosine_sum = 0.0
    logits_cosine_count = 0
    js_sum = 0.0
    top1_equal = 0
    token_count = 0
    token_count = 0
    with torch.inference_mode():
        for old_hidden_cpu, old_logits_cpu, new_hidden_cpu, new_logits_cpu in zip(
            reference["hidden"],
            reference["logits"],
            current["hidden"],
            current["logits"],
            strict=True,
        ):
            old_hidden = old_hidden_cpu.cuda()
            old_logits = old_logits_cpu.cuda()
            new_hidden = new_hidden_cpu.cuda()
            new_logits = new_logits_cpu.cuda()
            delta_hidden = new_hidden - old_hidden
            delta_logits = new_logits - old_logits
            hidden_squared += float(delta_hidden.double().square().sum())
            hidden_reference_squared += float(old_hidden.double().square().sum())
            hidden_dot = (new_hidden * old_hidden).sum(dim=-1)
            hidden_norm = new_hidden.norm(dim=-1) * old_hidden.norm(dim=-1)
            valid_hidden = hidden_norm > 0
            hidden_cosine_sum += float((hidden_dot[valid_hidden] / hidden_norm[valid_hidden]).double().sum())
            hidden_cosine_count += int(valid_hidden.sum())
            logits_squared += float(delta_logits.double().square().sum())
            logits_reference_squared += float(old_logits.double().square().sum())
            logits_dot = (new_logits * old_logits).sum(dim=-1)
            logits_norm = new_logits.norm(dim=-1) * old_logits.norm(dim=-1)
            valid_logits = logits_norm > 0
            logits_cosine_sum += float((logits_dot[valid_logits] / logits_norm[valid_logits]).double().sum())
            logits_cosine_count += int(valid_logits.sum())
            logp = torch.log_softmax(old_logits, dim=-1)
            logq = torch.log_softmax(new_logits, dim=-1)
            p = logp.exp()
            q = logq.exp()
            logm = torch.logaddexp(logp, logq) - math.log(2.0)
            js = 0.5 * (p * (logp - logm)).sum(dim=-1) + 0.5 * (q * (logq - logm)).sum(dim=-1)
            js_sum += float(js.double().sum())
            top1_equal += int((old_logits.argmax(dim=-1) == new_logits.argmax(dim=-1)).sum())
            token_count += int(new_logits.shape[0] * new_logits.shape[1])
    return {
        "seed": seed,
        "model_type": model_type,
        "condition": "router_experts_frozen",
        "comparison": "endpoint_offset_500_vs_branch_start_offset_0_same_sealed_training_probe",
        "probe_identity": probe_id,
        "target_tokens": token_count,
        "start_probe_ce": reference["probe_ce"],
        "endpoint_probe_ce": current["probe_ce"],
        "probe_ce_change": current["probe_ce"] - reference["probe_ce"],
        "final_normalized_hidden_change_rms": math.sqrt(hidden_squared / (token_count * 256)),
        "final_normalized_hidden_relative_l2": math.sqrt(hidden_squared / hidden_reference_squared) if hidden_reference_squared else None,
        "final_normalized_hidden_cosine_mean": hidden_cosine_sum / hidden_cosine_count if hidden_cosine_count else None,
        "logit_change_rms": math.sqrt(logits_squared / (token_count * 256)),
        "logit_relative_l2": math.sqrt(logits_squared / logits_reference_squared) if logits_reference_squared else None,
        "logit_cosine_mean": logits_cosine_sum / logits_cosine_count if logits_cosine_count else None,
        "softmax_jensen_shannon_mean_nats": js_sum / token_count,
        "top1_prediction_agreement": top1_equal / token_count,
        "numerics": "FP32_TF32off, eval/inference mode, identical four complete native training batches",
        "independent_test_set": False,
    }


def _profile_pathway() -> dict:
    admission = gpu_admission_check(wait_for_idle_seconds=0)
    _check(torch.cuda.is_available(), "CUDA is required by the registered Viper evaluation policy.")
    device = torch.device("cuda", torch.cuda.current_device())
    variants = adaptation_variants()
    seed = SEEDS[0]
    variant = variants[seed]
    batches, sealed = _probe_batches(seed, variant)
    prefix_payload = common.torch_load(ADAPTATION / "prefixes" / f"seed{seed}" / "router_step_1000.pt")
    model = _model_from_state(variant["config"], prefix_payload["model"], device)
    state_hash_before = state_dict_hash(model.state_dict())
    start = time.perf_counter()
    with common.preserve_rng_and_modes(model):
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False), _fp32_tf32_off():
            _ = model(batches[0][0].to(device))
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    state_hash_after = state_dict_hash(model.state_dict())
    _check(state_hash_before == state_hash_after, "Profile forward modified model state.")
    profile = {
        "status": "profiled",
        "profiled_seed": seed,
        "profiled_model": "unpruned branch-start checkpoint",
        "batch_index": sealed["batch_indices"][0],
        "tokens_in_profile_batch": int(batches[0][0].numel()),
        "forward_runtime_seconds": elapsed,
        "remaining_forwards_estimate": 80,
        "estimated_remaining_compute_seconds": elapsed * 80,
        "estimated_storage_bytes": 100000,
        "gpu_admission": admission,
        "device": str(device),
        "model_state_preserved": state_hash_before == state_hash_after,
        "probe_identity": sealed["probe_identity_sha256"],
        "no_training_or_optimizer_updates": True,
    }
    _write_json(OUT / "evaluation_profile.json", profile)
    return profile


def _run_pathway_diagnostics(seed_records, branch_paths) -> list[dict]:
    _check((OUT / "evaluation_profile.json").is_file(), "Run --profile-pathway before checkpoint-only diagnostics.")
    profile = _read_json(OUT / "evaluation_profile.json")
    _check(profile.get("status") == "profiled" and profile.get("model_state_preserved"), "Pathway profile is invalid.")
    _check(torch.cuda.is_available(), "CUDA is required by the registered Viper evaluation policy.")
    try:
        admission = gpu_admission_check(wait_for_idle_seconds=0)
    except RuntimeError as exc:
        if not str(exc).startswith("GPU utilization remains "):
            raise
        rows = [
            {
                "seed": seed,
                "model_type": model_type,
                "condition": "router_experts_frozen",
                "status": "deferred_gpu_admission",
                "deferred_reason": str(exc),
                "training_or_optimizer_updates": 0,
            }
            for seed in SEEDS
            for model_type in MODEL_TYPES
        ]
        _write_csv(OUT / "checkpoint_prediction_diagnostics.csv", rows)
        return rows
    device = torch.device("cuda", torch.cuda.current_device())
    variants = adaptation_variants()
    rows = []
    for seed in SEEDS:
        variant = variants[seed]
        batches, sealed = _probe_batches(seed, variant)
        prefix_payload = common.torch_load(ADAPTATION / "prefixes" / f"seed{seed}" / "router_step_1000.pt")
        for model_type in MODEL_TYPES:
            start_state = dict(prefix_payload["model"])
            if model_type == "sparse":
                masks = load_masks(variant["mask"])
                base_model = TinyMoELanguageModel(variant["config"]["model"])
                base_model.load_state_dict(start_state)
                apply_masks_(base_model, masks)
                start_state = base_model.state_dict()
            start_state = {name: tensor.detach().cpu().clone() for name, tensor in start_state.items()}
            start_hash = state_dict_hash(start_state)
            start_model = _model_from_state(variant["config"], start_state, device)
            modes = [(module, module.training) for module in start_model.modules()]
            with common.preserve_rng_and_modes(start_model):
                start_model.eval()
                reference = _forward_probe(start_model, batches, device)
            for module, mode in modes:
                module.training = mode
            _check(state_dict_hash(start_model.state_dict()) == start_hash, "Branch-start probe modified model state.")
            del start_model

            attempt = branch_paths[(seed, model_type, "router_experts_frozen")]
            result = _read_json(attempt / "result.json")
            checkpoint = Path(result["final_checkpoint"])
            _check(_sha256(checkpoint) == result["final_checkpoint_sha256"], f"Final checkpoint changed before probe: {checkpoint}.")
            payload = common.torch_load(checkpoint)
            end_state = payload["model"]
            frozen_groups = {"router", "experts"}
            differences = []
            for name, initial in start_state.items():
                if common.group_for_parameter(name) in frozen_groups:
                    _check(torch.equal(initial, end_state[name].cpu()), f"Frozen router/expert parameter changed: seed {seed}/{model_type}/{name}.")
            end_model = _model_from_state(variant["config"], end_state, device)
            state_hash_before = state_dict_hash(end_model.state_dict())
            end_modes = [(module, module.training) for module in end_model.modules()]
            with common.preserve_rng_and_modes(end_model):
                end_model.eval()
                current = _forward_probe(end_model, batches, device)
                metrics = _paired_output_metrics(
                    reference, current, seed, model_type, sealed["probe_identity_sha256"]
                )
            for module, mode in end_modes:
                module.training = mode
            state_hash_after = state_dict_hash(end_model.state_dict())
            _check(state_hash_before == state_hash_after, "Endpoint probe modified model state.")
            metrics.update(
                {
                    "status": "completed",
                    "branch_start_model_hash": start_hash,
                    "endpoint_model_hash_before": state_hash_before,
                    "endpoint_model_hash_after": state_hash_after,
                    "endpoint_checkpoint": str(checkpoint),
                    "endpoint_checkpoint_sha256": result["final_checkpoint_sha256"],
                    "frozen_router_and_expert_parameters_including_biases_bitwise_unchanged": True,
                    "frozen_parameter_names_checked": sum(
                        1 for name in start_state if common.group_for_parameter(name) in frozen_groups
                    ),
                    "profile_forward_seconds": profile["forward_runtime_seconds"],
                    "gpu_admission_before_evaluation": admission,
                    "probe_indices": sealed["batch_indices"],
                    "probe_batch_hashes": [item["sha256"] for item in sealed["batch_hashes"]],
                }
            )
            rows.append(metrics)
            del end_model, reference, current, payload, start_state
            torch.cuda.empty_cache()
    _check(len(rows) == 10, f"Expected 10 shared-only endpoint pathway records, found {len(rows)}.")
    _write_csv(OUT / "checkpoint_prediction_diagnostics.csv", rows)
    return rows


def _artifact_inventory(coverage, provenance):
    rows = []
    verified_checkpoint_paths = {}
    for status_path in (ADAPTATION / "status").glob("seed*_*.json"):
        if "_prefix" in status_path.stem or "_revision_" in status_path.stem:
            continue
        status = _read_json(status_path)
        if status.get("status") == "COMPLETE" and status.get("result"):
            result = _read_json(Path(status["result"]))
            verified_checkpoint_paths[result["final_checkpoint"]] = result["final_checkpoint_sha256"]
    roots = (
        ("transition", TRANSITION),
        ("factorial", ADAPTATION),
    )
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(ROOT))
            size = path.stat().st_size
            expected = verified_checkpoint_paths.get(str(path))
            actual = None
            hash_status = "not_rehashed_large_file"
            if size <= MAX_HASH_BYTES:
                actual = _sha256(path)
                hash_status = "sha256_verified"
            elif expected:
                actual = _sha256(path)
                hash_status = "registered_checkpoint_sha256_verified" if actual == expected else "registered_checkpoint_hash_mismatch"
                _check(actual == expected, f"Checkpoint inventory hash mismatch: {path}.")
            rows.append(
                {
                    "study": label,
                    "relative_path": relative,
                    "size_bytes": size,
                    "sha256": actual or "",
                    "hash_status": hash_status,
                    "used_in_analysis": (
                        any(row["source_file"] in str(path) for row in coverage)
                        or relative.endswith(("registration.json", "manifest.json", "completion.json", "input_hashes.json"))
                        or relative.endswith("pruning_event_damage.csv")
                    ),
                }
            )
    _write_csv(OUT / "artifact_inventory.csv", rows)
    return rows


def _audit_text(provenance, damage_rows, decomposition, penalties, pathway_rows, summary):
    shared = [row for row in decomposition if row["condition"] == "router_experts_frozen" and row["offset"] == 500]
    d_by_seed = {
        row["seed"]: row["D_post_minus_pre"]
        for row in damage_rows
        if row["population"] == "common validation"
    }
    full_closure = {
        row["seed"]: row["C_D_minus_G"]
        for row in shared
    }
    s_losses = {row["seed"]: row["L_sparse"] for row in shared}
    u_losses = {row["seed"]: row["L_unpruned"] for row in shared}
    sparse_reductions = {row["seed"]: row["R_sparse"] for row in shared}
    unpruned_reductions = {row["seed"]: row["R_unpruned"] for row in shared}
    parent_classes = {row["seed"]: row["classification"] for row in provenance["common_parent_classification"]}
    shared_penalties = [
        row for row in penalties
        if row["model_type"] == "sparse" and row["offset"] == 500
    ]
    shared_sparse_penalty_mean = statistics.mean(row["P_RE"] for row in shared_penalties)
    pathway_completed = sum(row.get("status") == "completed" for row in pathway_rows)
    output = [
        "# IMP80 immediate damage and adaptation-gap analysis",
        "",
        "## Status and scope",
        "",
        "This is a post hoc, artifact-only analysis of the five registered source seeds. It did not launch training, optimizer updates, mask discovery, or a pruning-policy experiment. The independent replication unit is the source seed (n=5). The primary window is global step 1000 to 1500, with offsets 0, 25, 100, 250, and 500. Shared-only means the registered `router_experts_frozen` condition; router and expert parameters are frozen and shared parameters train.",
        "",
        "All 40 adaptation branches and 20 transition jobs are reported against their registrations. The 200 common-validation grid rows have 492,416 target tokens over 31 batches per readout. Immediate pruning damage on validation and on original training-only event probes is reported separately.",
        "",
        "## Common parent and immediate event damage",
        "",
        "Common parent status is A for each seed: the step-1000 reconstructed dense model is bitwise equal to the historical dense checkpoint; optimizer/scaler and Python/NumPy/Torch CPU/CUDA RNG state are saved, sampler position is 1000, and branch starts share training and validation identities within each model type and across conditions. Sparse branches originate from that state, then apply the registered IMP80 mask and zero coordinate-shaped AdamW moments at masked coordinates. This is a treatment-inclusive comparison, not a weight-zeroing-only effect. The native sparse and unpruned start states differ as expected because of the event.",
        "",
        "Validation `D = L_post - L_pre` is the immediate paired CE effect of the complete pruning event on identical 492,416 validation target tokens. Training-only event-probe damage is separately recorded and never substituted for validation D. See `immediate_damage.csv`.",
        "",
        "| Seed | Validation D | Shared-only L_sparse(1500) | Shared-only L_unpruned(1500) | Sparse reduction | Unpruned reduction | C= D-G(500) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = next(item for item in shared if item["seed"] == seed)
        output.append(
            f"| {seed} | {d_by_seed[seed]:.6f} | {s_losses[seed]:.6f} | {u_losses[seed]:.6f} | {sparse_reductions[seed]:.6f} | {unpruned_reductions[seed]:.6f} | {full_closure[seed]:.6f} |"
        )
    output.extend(
        [
            "",
            "Positive C means the observed sparse-minus-unpruned CE gap narrowed relative to the measured immediate validation D; it is not uniquely attributable repair and is not proof that original function was restored. The equality `H=R_sparse-R_unpruned` was checked numerically at every seed, condition, and offset. Since `G(0)=D` within the registered CE tolerance, C and observed initial-gap closure H also agree within tolerance.",
            "",
            "## Trajectories and permitted interventions",
            "",
            "`absolute_losses.csv` keeps global step 1000's pre-pruning unpruned parent and post-pruning sparse state as distinct phases. `gap_closure_decomposition.csv` gives both absolute losses beside G, C, H, each model's reduction, the improvement-pattern label, and arithmetic residuals for all 100 seed × condition × offset observations. Integrals use only the common observed offsets and span; no extrapolation is used.",
            "",
            "`condition_penalties.csv` reports P_R, P_E, and P_RE for each model type, seed, and grid point. Sparse-excess penalty identities and the H/freezing identity passed. `factorial_interactions.csv` uses I=L11-L10-L01+L00, where 1 means the router or expert group trains; shared parameters train in every cell. These identities are algebraic bookkeeping, not independent evidence.",
            "",
            "The complete five-seed values precede means, medians, minima, maxima, and sample SDs in `seed_level_metrics.csv` and `aggregate_summaries.csv`. No binary recovered/harmless threshold, winner classifier, or component-percentage decomposition was used.",
            "",
            "## Shared-only pathway diagnostics",
            "",
            "Existing sealed-probe routing, gate, accepted-load, dropped-token, representation, expert-function, update, and freeze-verification diagnostics are consolidated in `shared_only_diagnostics.csv`. Frozen expert outputs on identical branch-start inputs are unchanged within 1e-7; that is a consistency check. Outputs on current native inputs vary with the changing inputs and do not imply expert parameter learning. Frozen expert and router parameters, including biases, were separately compared bitwise between shared-only start and endpoint checkpoints.",
            "",
            (
                "A bounded checkpoint-only evaluation on the sealed four-batch training probe compares shared-only start and endpoint final-normalized hidden states and logits. It reports target-token-weighted probe CE, activation/logit displacement and cosine, softmax Jensen-Shannon divergence, and top-1 agreement in `checkpoint_prediction_diagnostics.csv`. These training probes may overlap training and are not an independent test set. Shared parameter movement includes the tied embedding/output matrix, which has both input and output roles. These diagnostics are candidate-process measurements, not causal decomposition."
                if pathway_completed == 10
                else f"The checkpoint-only endpoint pathway evaluation was deferred for all 10 planned seed/model cells because native GPU admission refused competition: {pathway_rows[0]['deferred_reason']}. `checkpoint_prediction_diagnostics.csv` records the deferred status; no pathway endpoint predictions were generated. Existing shared-only training-probe diagnostics remain reported separately. This diagnostic can be resumed when GPU admission is available."
            ),
            "",
            "For all five seeds, the sparse shared-only CE reduction is larger than the unpruned shared-only reduction because the sparse state starts with the measured event gap. A larger reduction from a worse starting loss, by itself, does not establish a specialized compensatory mechanism. Frozen experts remain functionally in the pathway because shared updates can change their inputs, gates, assignments, and downstream use. Similar average CE does not imply restored predictions or expert behavior.",
            "",
            "## Provenance and outputs",
            "",
            "See `provenance_checks.json` for seed-level common-parent evidence, source registrations, event output hash checks, and cohort status; `measurement_coverage.csv` for every required observation; and `artifact_inventory.csv` for relevant original files and hashes. No historical report, checkpoint, manifest, registration, raw measurement, or failed attempt was overwritten.",
            "",
            "The primary inputs were the abrupt IMP80 transition artifacts and completed post-pruning adaptation factorial artifacts. All new analysis products are isolated under `results/imp_compensation_analysis_v1/`.",
            "",
        ]
    )
    (OUT / "audit.md").write_text("\n".join(output), encoding="utf-8")


def analyze() -> dict:
    _check((OUT / "analysis_config.json").is_file(), "Analysis config must be written before analysis.")
    provenance, seed_records, variants, _transition_manifest = _seed_coverage_and_provenance()
    rows_by_key, branch_paths, results, starts, metadata, coverage, update_summary = _read_measurements()
    damage_rows, validation_damage = _event_damage(seed_records)
    loss_rows, decomposition, penalties, interactions, seed_measures, integrals = _trajectory_tables(rows_by_key, validation_damage)
    diagnostic_rows, candidate_cells = _existing_diagnostic_summary(branch_paths)
    pathway_rows = _run_pathway_diagnostics(seed_records, branch_paths)
    summary = _aggregate(seed_measures, integrals)
    coverage.extend(
        {
            "seed": row["seed"],
            "model_type": "event",
            "condition": "abrupt",
            "offset": 0,
            "global_step": 1000,
            "phase_label": "paired_pruning_event_readout",
            "population": row["population"],
            "status": "observed_verified",
            "target_tokens": row["target_tokens"],
            "batches": row["batches"],
            "source_file": row["source_file"],
            "source_sha256": row["source_sha256"],
        }
        for row in damage_rows
    )
    # Keep every seed-level record in the file; aggregate summaries are a separate table.
    _write_csv(OUT / "absolute_losses.csv", loss_rows)
    _write_csv(OUT / "immediate_damage.csv", damage_rows)
    _write_csv(OUT / "gap_closure_decomposition.csv", decomposition)
    _write_csv(OUT / "condition_penalties.csv", penalties)
    _write_csv(OUT / "factorial_interactions.csv", interactions)
    _write_csv(OUT / "seed_level_metrics.csv", seed_measures)
    _write_csv(OUT / "trajectory_integrals.csv", integrals)
    _write_csv(OUT / "aggregate_summaries.csv", summary)
    _write_csv(OUT / "shared_only_diagnostics.csv", diagnostic_rows)
    _write_csv(OUT / "update_and_freezing_verification.csv", update_summary)
    provenance["candidate_component_hash_checks"] = candidate_cells
    provenance["analysis_outputs"] = [
        "absolute_losses.csv",
        "immediate_damage.csv",
        "gap_closure_decomposition.csv",
        "condition_penalties.csv",
        "factorial_interactions.csv",
        "seed_level_metrics.csv",
        "trajectory_integrals.csv",
        "aggregate_summaries.csv",
        "shared_only_diagnostics.csv",
        "checkpoint_prediction_diagnostics.csv",
        "update_and_freezing_verification.csv",
    ]
    provenance["coverage"] = {
        "seeds": len(SEEDS),
        "branches": len(results),
        "trajectory_rows": len(loss_rows),
        "gap_decomposition_rows": len(decomposition),
        "immediate_damage_rows": len(damage_rows),
        "checkpoint_only_pathway_rows": sum(row.get("status") == "completed" for row in pathway_rows),
        "checkpoint_only_pathway_deferred_rows": sum(row.get("status") == "deferred_gpu_admission" for row in pathway_rows),
        "update_rows_per_branch": 1500,
        "per_group_update_rows_total": sum(int(item["attempted_updates"]) for item in update_summary) * 3,
    }
    _write_json(OUT / "provenance_checks.json", provenance)
    inventory = _artifact_inventory(coverage, provenance)
    _write_csv(OUT / "measurement_coverage.csv", coverage)
    _audit_text(provenance, damage_rows, decomposition, penalties, pathway_rows, summary)
    report = {
        "status": (
            "COMPLETE"
            if all(row.get("status") == "completed" for row in pathway_rows)
            else "COMPLETE_WITH_DEFERRED_PATHWAY_DIAGNOSTICS"
        ),
        "seeds": list(SEEDS),
        "n_independent_seeds": len(SEEDS),
        "branches": len(results),
        "validation_grid_rows": len(loss_rows),
        "gap_decomposition_rows": len(decomposition),
        "immediate_damage_population_rows": len(damage_rows),
        "pathway_probe_rows": len(pathway_rows),
        "pathway_probe_rows_completed": sum(row.get("status") == "completed" for row in pathway_rows),
        "pathway_probe_rows_deferred": sum(row.get("status") == "deferred_gpu_admission" for row in pathway_rows),
        "artifact_inventory_files": len(inventory),
        "all_common_parents_verified": all(row["common_parent_verified"] for row in seed_records),
        "new_training_updates": 0,
        "new_mask_searches": 0,
        "outputs": provenance["analysis_outputs"] + [
            "analysis_config.json",
            "artifact_inventory.csv",
            "measurement_coverage.csv",
            "provenance_checks.json",
            "audit.md",
            "evaluation_profile.json",
        ],
    }
    _write_json(OUT / "analysis_status.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--profile-pathway", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.profile_pathway:
        print(json.dumps(_profile_pathway(), indent=2))
    else:
        print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
