from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/imp_capacity_diagnostic_v1"
TRANSITION = ROOT / "results/imp_mask_transition_v1"
ADAPTATION = ROOT / "results/imp_postpruning_adaptation_v1"
COMPENSATION = ROOT / "results/imp_compensation_analysis_v1"
SEEDS = (7, 17, 29, 43, 61)
MODEL_TYPES = ("sparse", "unpruned")
OFFSETS = (0, 25, 100, 250, 500)
PRIMARY_OFFSETS = (0, 500)
POLICY_VERSION = "capacity-rejection-only-v1"
EXPECTED_TOKENS = 492_416
EXPECTED_BATCHES = 31
CE_ATOL = 1e-6
MODEL_ATOL = 1e-6
MODEL_RTOL = 1e-5

sys.path.insert(0, str(ROOT / "scripts/imp_postpruning_adaptation_v1"))
sys.path.insert(0, str(ROOT / "scripts/imp_adaptation_factorial_v1"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import apply_masks_, load_masks
from moe_lth.pruning.router_age import state_dict_hash

import af_native
import common
from engine import variants as adaptation_variants
from worker import GPU_IDLE_WAIT_SECONDS, exclusive_claim, gpu_admission_check


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty results table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = af_native.decode_numpy(payload)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a checkpoint mapping in {path}.")
    return payload


def _model_state(payload: dict, path: Path) -> dict[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no model state mapping: {path}")
    if not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise RuntimeError(f"Checkpoint model state is not a tensor mapping: {path}")
    return state


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _group_hash(state: dict[str, torch.Tensor], group: str) -> str:
    selected = {
        name: value
        for name, value in state.items()
        if common.group_for_parameter(name) == group
    }
    return state_dict_hash(selected)


def _mask_hash(masks: dict[str, torch.Tensor]) -> str:
    return state_dict_hash({name: mask.to(torch.uint8) for name, mask in masks.items()})


def _assert_same_model_values(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    context: str,
) -> int:
    _need(set(left) == set(right), f"Model-state keys differ: {context}.")
    signed_zero_differences = 0
    for name in left:
        first = left[name]
        second = right[name]
        _need(first.dtype == second.dtype, f"Model-state dtypes differ: {context}/{name}.")
        _need(torch.equal(first, second), f"Model-state values differ: {context}/{name}.")
        if first.is_floating_point():
            signed_zero_differences += int(
                ((first == 0) & (second == 0) & (torch.signbit(first) != torch.signbit(second)))
                .sum()
                .item()
            )
    return signed_zero_differences


def _source_row(
    path: Path,
    role: str,
    expected_hash: str | None = None,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Registered {role} is missing: {path}")
    actual = _sha256(path)
    if expected_hash is not None and actual != expected_hash:
        raise RuntimeError(f"Registered {role} hash mismatch: {path}")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected_hash or "",
        "hash_status": "verified" if expected_hash is None or actual == expected_hash else "mismatch",
    }


def _transition_attempt(seed: int) -> tuple[Path, dict]:
    status_path = TRANSITION / "status" / f"seed{seed}_abrupt.json"
    status = _json_read(status_path)
    _need(status.get("status") == "COMPLETE", f"Transition source is incomplete for seed {seed}.")
    attempt = Path(status["attempt"])
    completion = _json_read(attempt / "completion.json")
    _need(completion.get("verified") is True, f"Transition completion is unverified for seed {seed}.")
    _need(
        _sha256(attempt / "completion.json") == status["completion_sha256"],
        f"Transition completion hash mismatch for seed {seed}.",
    )
    return attempt, completion


def _branch(seed: int, model_type: str) -> tuple[dict, Path, dict]:
    job = f"seed{seed}_{model_type}_router_experts_frozen"
    status_path = ADAPTATION / "status" / f"{job}.json"
    status = _json_read(status_path)
    _need(status.get("status") == "COMPLETE", f"Shared-only source branch is incomplete: {job}.")
    result_path = Path(status["result"])
    result = _json_read(result_path)
    checkpoint = Path(result["final_checkpoint"])
    _need(
        _sha256(checkpoint) == result["final_checkpoint_sha256"],
        f"Endpoint checkpoint hash mismatch for {job}.",
    )
    _need(int(result.get("attempted_updates", -1)) == 500, f"Unexpected update count for {job}.")
    _need(
        int(result.get("applied_updates", -1)) + int(result.get("amp_skips", -1)) == 500,
        f"Update accounting mismatch for {job}.",
    )
    _need(
        result.get("frozen_parameters_bitwise_unchanged") is True,
        f"Registered frozen-parameter check failed for {job}.",
    )
    return result, Path(result["attempt"]), status


def _start_state(seed: int, model_type: str, variant: dict) -> tuple[dict, dict, Path]:
    prefix_path = ADAPTATION / "prefixes" / f"seed{seed}" / "reconstruction_interruption.pt"
    prefix_meta_path = ADAPTATION / "prefixes" / f"seed{seed}" / "prefix_verification.json"
    prefix_meta = _json_read(prefix_meta_path)
    _need(prefix_meta.get("status") == "verified", f"Step-1000 parent is not verified for seed {seed}.")
    payload = _load_payload(prefix_path)
    _need(int(payload.get("step", -1)) == 1000, f"Wrong parent checkpoint step for seed {seed}.")
    _need(int(payload.get("sampler_next_batch", -1)) == 1000, f"Wrong parent sampler position for seed {seed}.")
    state = _clone_state(_model_state(payload, prefix_path))
    _need(
        state_dict_hash(state) == prefix_meta["unpruned_model_hash"],
        f"Step-1000 parent model hash mismatch for seed {seed}.",
    )
    if model_type == "sparse":
        masks = load_masks(variant["mask"])
        model = TinyMoELanguageModel(variant["config"]["model"])
        model.load_state_dict(state)
        apply_masks_(model, masks)
        state = _clone_state(model.state_dict())
        expected_mask_hash = variant["mask_hash"]
        _need(_mask_hash(masks) == expected_mask_hash, f"IMP80 mask hash mismatch for seed {seed}.")
        del model
    else:
        masks = {}
    return state, masks, prefix_path


def _endpoint_state(seed: int, model_type: str) -> tuple[dict, dict, Path, dict]:
    result, attempt, _status = _branch(seed, model_type)
    checkpoint = Path(result["final_checkpoint"])
    payload = _load_payload(checkpoint)
    state = _clone_state(_model_state(payload, checkpoint))
    return state, result, checkpoint, _json_read(attempt / "branch_start.json")


def _validation_batches(seed: int, variant: dict) -> tuple[list, str]:
    rng_state = _rng_snapshot()
    try:
        recovery.seed_everything(seed)
        previous_cwd = Path.cwd()
        try:
            import os

            os.chdir(ROOT)
            _train, validation_loader = build_dataloaders(
                variant["config"]["data"], 128, seed
            )
            batches = recovery._materialize_validation_batches(validation_loader, 32)
        finally:
            os.chdir(previous_cwd)
    finally:
        _restore_rng(rng_state)
    digest = common.batch_hash(batches)
    expected = variant["canonical_pair"]["validation_hash"]
    _need(digest == expected, f"Canonical validation hash mismatch for seed {seed}.")
    count = sum(int(targets.numel()) for _inputs, targets in batches)
    batch_target_counts = [int(targets.numel()) for _, targets in batches]
    _need(len(batches) == EXPECTED_BATCHES, f"Expected 31 canonical validation batches for seed {seed}.")
    _need(count == EXPECTED_TOKENS, f"Expected 492416 validation targets for seed {seed}.")
    _need(
        batch_target_counts[-1] < max(batch_target_counts)
        and all(value == max(batch_target_counts) for value in batch_target_counts[:-1]),
        f"Canonical validation partial batch/order mismatch for seed {seed}.",
    )
    return batches, digest


def _training_suffix_hash(seed: int, variant: dict) -> tuple[str, str]:
    rng_state = _rng_snapshot()
    try:
        recovery.seed_everything(seed)
        previous_cwd = Path.cwd()
        try:
            import os

            os.chdir(ROOT)
            train_loader, _validation_loader = build_dataloaders(
                variant["config"]["data"], 128, seed
            )
            train = recovery._materialize_batches(train_loader, 1500)
        finally:
            os.chdir(previous_cwd)
    finally:
        _restore_rng(rng_state)
    return common.batch_hash(train), common.batch_hash(train[1000:1500])


def audit_sources() -> tuple[dict, list[dict], dict[int, dict]]:
    transition_manifest = _json_read(TRANSITION / "manifest.json")
    transition_by_seed = {int(row["seed"]): row for row in transition_manifest["variants"]}
    source_inventory: list[dict] = []
    transition_input_hashes = _json_read(TRANSITION / "input_hashes.json")
    transition_supplemental_hashes = _json_read(TRANSITION / "supplemental_input_hashes.json")
    reuse_candidates_path = ADAPTATION / "historical_full_endpoint_reuse_candidates.csv"
    reuse_candidates = _read_csv(reuse_candidates_path)
    _need(
        len(transition_input_hashes) > 0 and len(transition_supplemental_hashes) > 0,
        "Transition source hash manifests are empty.",
    )
    _need(
        len(reuse_candidates) == 2 * len(SEEDS)
        and all(row.get("full_trajectory_reusable") == "False" for row in reuse_candidates),
        "Historical endpoint candidates were incorrectly marked as full-trajectory reusable.",
    )
    _need(
        {
            (int(row["seed"]), row["model_type"])
            for row in reuse_candidates
        } == {(seed, model_type) for seed in SEEDS for model_type in MODEL_TYPES},
        "Historical endpoint candidate inventory does not cover the registered cohort.",
    )
    registered_hashes = _json_read(ADAPTATION / "input_code_configuration_hashes.json")
    probe_manifest = _json_read(ADAPTATION / "diagnostic_probe_manifest.json")
    probe_identities = {
        int(row["seed"]): row["probe_identity_sha256"]
        for row in probe_manifest["seeds"]
    }
    known_identities = {}
    source_hash_audit = {}
    additive_model_extensions = {
        (ROOT / "src/moe_lth/models/moe_layer.py").resolve(),
        (ROOT / "src/moe_lth/models/transformer.py").resolve(),
    }
    pseudo_hashes = {}
    for path_string, expected_hash in registered_hashes.items():
        candidate = Path(path_string)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if candidate.is_file():
            actual_hash = _sha256(candidate)
            if actual_hash != expected_hash:
                _need(
                    candidate.resolve() in additive_model_extensions,
                    f"Unexpected adaptation registered-input hash mismatch: {candidate}",
                )
                status = "intentional_evaluation_api_extension"
            else:
                status = "verified"
            source_hash_audit[path_string] = {
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": status,
            }
            source_inventory.append(
                {
                    "role": f"registered_adaptation_input:{path_string}",
                    "path": str(candidate),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "hash_status": status,
                }
            )
        else:
            pseudo_hashes[path_string] = expected_hash
    expected_pseudo_keys = {
        f"diagnostic-probe-seed-{seed}" for seed in SEEDS
    } | {
        f"training-seed-{seed}-next500" for seed in SEEDS
    } | {
        f"validation-seed-{seed}" for seed in SEEDS
    }
    _need(
        set(pseudo_hashes) == expected_pseudo_keys,
        "Adaptation registration has unknown or missing non-file identity entries.",
    )
    provenance = {
        "status": "PASS",
        "seeds": {},
        "source_registrations": {
            "transition": str(TRANSITION / "registration.json"),
            "adaptation": str(ADAPTATION / "registration.json"),
            "compensation": str(COMPENSATION / "analysis_config.json"),
        },
        "validation_contract": {
            "expected_batches": EXPECTED_BATCHES,
            "expected_target_tokens": EXPECTED_TOKENS,
            "precision": "FP32, TF32 disabled, eval/inference mode",
            "intermediate_offsets": {},
        },
    }
    _need(_json_read(TRANSITION / "completion.json").get("scientifically_complete") is True,
          "The IMP transition source cohort is not scientifically complete.")
    _need(_json_read(COMPENSATION / "analysis_status.json").get("status") == "COMPLETE",
          "The compensation analysis is not marked complete.")
    _need(_json_read(ADAPTATION / "cohort_status.json").get("status") == "COMPLETE",
          "The adaptation cohort is not marked complete.")

    for path, role in (
        (TRANSITION / "registration.json", "transition_registration"),
        (TRANSITION / "completion.json", "transition_completion"),
        (TRANSITION / "manifest.json", "transition_manifest"),
        (TRANSITION / "input_hashes.json", "transition_input_hash_manifest"),
        (TRANSITION / "supplemental_input_hashes.json", "transition_supplemental_input_hash_manifest"),
        (ADAPTATION / "registration.json", "adaptation_registration"),
        (ADAPTATION / "input_code_configuration_hashes.json", "adaptation_hash_manifest"),
        (ADAPTATION / "diagnostic_probe_manifest.json", "adaptation_probe_manifest"),
        (reuse_candidates_path, "historical_full_endpoint_reuse_candidates"),
        (COMPENSATION / "analysis_config.json", "compensation_analysis_config"),
        (COMPENSATION / "analysis_status.json", "compensation_analysis_status"),
        (COMPENSATION / "provenance_checks.json", "compensation_provenance"),
        (COMPENSATION / "shared_only_diagnostics.csv", "shared_only_diagnostic_definitions_and_source_metrics"),
        (ROOT / "scripts/imp_capacity_diagnostic_v1/run.py", "capacity_diagnostic_implementation"),
        (ROOT / "scripts/imp_capacity_diagnostic_v1/README.md", "capacity_diagnostic_documentation"),
        (ROOT / "src/moe_lth/models/moe_layer.py", "capacity_dispatch_model_extension"),
        (ROOT / "src/moe_lth/models/transformer.py", "capacity_policy_model_extension"),
        (ROOT / "tests/test_model.py", "capacity_policy_unit_tests"),
        (ROOT / "tests/test_imp_capacity_diagnostic_v1.py", "capacity_readout_unit_tests"),
    ):
        source_inventory.append(_source_row(path, role))

    for role, registered_files in (
        ("transition_registered_input", transition_input_hashes),
        ("transition_supplemental_registered_input", transition_supplemental_hashes),
    ):
        for path_string, expected_hash in registered_files.items():
            path = Path(path_string)
            if not path.is_file():
                raise FileNotFoundError(f"Registered {role} is missing: {path}")
            actual_hash = _sha256(path)
            if actual_hash == expected_hash:
                hash_status = "verified"
            else:
                _need(
                    path.resolve() in additive_model_extensions,
                    f"Unexpected transition registered-input hash mismatch: {path}",
                )
                hash_status = "intentional_evaluation_api_extension"
            source_inventory.append(
                {
                    "role": role,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": actual_hash,
                    "expected_sha256": expected_hash,
                    "hash_status": hash_status,
                }
            )

    source_abs = {
        (int(row["seed"]), row["model_type"], int(row["offset"])): row
        for row in _read_csv(COMPENSATION / "absolute_losses.csv")
        if row["condition"] == "router_experts_frozen"
    }
    damage_rows = {
        int(row["seed"]): row
        for row in _read_csv(COMPENSATION / "immediate_damage.csv")
        if row.get("population") == "common validation"
    }
    variants = {int(seed): row for seed, row in adaptation_variants().items()}

    for seed in SEEDS:
        variant = variants[seed]
        transition_variant = transition_by_seed[seed]
        _need(transition_variant["mask"] == variant["mask"], f"Mask registration mismatch for seed {seed}.")
        attempt, completion = _transition_attempt(seed)
        for relative in ("segment_0/grid_1000_pre.json", "segment_0/grid_1000.json", "segment_0/event_1000.json"):
            expected_hash = completion.get("files", {}).get(relative)
            _need(expected_hash is not None, f"Transition output hash is absent: seed {seed}/{relative}.")
            source_inventory.append(_source_row(attempt / relative, f"transition_{seed}_{relative}", expected_hash))
        transition_snapshot = attempt / "snapshots/step_1000.pt"
        transition_snapshot_hash = completion.get("files", {}).get("snapshots/step_1000.pt")
        _need(transition_snapshot_hash is not None, f"Transition snapshot hash is absent for seed {seed}.")
        source_inventory.append(
            _source_row(transition_snapshot, f"transition_step1000_snapshot_seed{seed}", transition_snapshot_hash)
        )
        event = _json_read(attempt / "segment_0/event_1000.json")
        _need(event.get("mask_hash") == transition_variant["mask_hash"], f"Transition event mask mismatch for seed {seed}.")

        mask_path = Path(variant["mask"])
        registered_mask_hash = transition_input_hashes.get(str(mask_path))
        _need(registered_mask_hash is not None, f"Transition input hash manifest omits the IMP80 mask for seed {seed}.")
        masks = load_masks(mask_path)
        mask_digest = _mask_hash(masks)
        _need(mask_digest == transition_variant["mask_hash"], f"Registered mask contents mismatch for seed {seed}.")
        source_inventory.append(_source_row(mask_path, f"imp80_mask_seed{seed}", registered_mask_hash))

        prefix_path = ADAPTATION / "prefixes" / f"seed{seed}" / "reconstruction_interruption.pt"
        prefix_meta_path = ADAPTATION / "prefixes" / f"seed{seed}" / "prefix_verification.json"
        prefix_meta = _json_read(prefix_meta_path)
        _need(prefix_meta.get("status") == "verified", f"Prefix verification failed for seed {seed}.")
        prefix_hash = _sha256(prefix_path)
        prefix_payload = _load_payload(prefix_path)
        post_verification_metadata = set(prefix_payload) - set(prefix_meta.get("state_keys", []))
        _need(
            prefix_hash == prefix_meta.get("state_checkpoint_sha256")
            or post_verification_metadata == {"source_optimizer_group_hashes"},
            f"Step-1000 full-state checkpoint hash mismatch for seed {seed}.",
        )
        source_inventory.append(_source_row(prefix_path, f"step1000_parent_seed{seed}"))
        source_inventory.append(_source_row(prefix_meta_path, f"prefix_verification_seed{seed}"))
        payload = prefix_payload
        dense_state = _clone_state(_model_state(payload, prefix_path))
        _need(state_dict_hash(dense_state) == prefix_meta["unpruned_model_hash"],
              f"Common dense parent hash mismatch for seed {seed}.")
        training_hash, training_suffix_hash = _training_suffix_hash(seed, variant)
        _need(
            training_hash == prefix_meta["training_hash_steps_1_1500"],
            f"Step-1000 parent training batch identity mismatch for seed {seed}.",
        )
        sparse_model = TinyMoELanguageModel(variant["config"]["model"])
        sparse_model.load_state_dict(dense_state)
        apply_masks_(sparse_model, masks)
        sparse_state = _clone_state(sparse_model.state_dict())
        del sparse_model
        transition_snapshot_state = _clone_state(
            _model_state(_load_payload(transition_snapshot), transition_snapshot)
        )
        signed_zero_differences = _assert_same_model_values(
            sparse_state,
            transition_snapshot_state,
            f"reconstructed sparse start vs transition snapshot for seed {seed}",
        )

        checkpoint_entries = {}
        pair_starts = {}
        for model_type, state in (("unpruned", dense_state), ("sparse", sparse_state)):
            result, branch_attempt, _branch_status = _branch(seed, model_type)
            start = _json_read(branch_attempt / "branch_start.json")
            _need(start.get("condition") == "router_experts_frozen", f"Wrong condition for {seed}/{model_type}.")
            _need(
                start.get("validation_batch_hash") == prefix_meta.get("validation_hash"),
                f"Shared-only validation identity mismatch for {seed}/{model_type}.",
            )
            _need(
                start.get("training_batch_hash") == training_suffix_hash,
                f"Shared-only training identity mismatch for {seed}/{model_type}.",
            )
            expected_start_hash = start.get("state_sha256")
            _need(expected_start_hash is not None, f"Shared-only start state hash missing for {seed}/{model_type}.")
            _need(state_dict_hash(state) == expected_start_hash,
                  f"Reconstructed shared-only start state mismatch for {seed}/{model_type}.")
            expected_mask = transition_variant["mask_hash"] if model_type == "sparse" else hashlib.sha256(b"").hexdigest()
            _need(start.get("mask_hash") == expected_mask, f"Shared-only mask identity mismatch for {seed}/{model_type}.")
            endpoint_state, endpoint_result, endpoint_path, _endpoint_start = _endpoint_state(seed, model_type)
            _need(
                _group_hash(endpoint_state, "router") == _group_hash(state, "router"),
                f"Router changed in frozen shared-only branch: {seed}/{model_type}.",
            )
            _need(
                _group_hash(endpoint_state, "experts") == _group_hash(state, "experts"),
                f"Experts changed in frozen shared-only branch: {seed}/{model_type}.",
            )
            if model_type == "sparse":
                for name, mask in masks.items():
                    if name not in endpoint_state:
                        raise RuntimeError(f"IMP80 mask has no endpoint parameter: {name}")
                    _need(
                        torch.count_nonzero(endpoint_state[name][~mask.bool()]).item() == 0,
                        f"Masked coordinate is nonzero at endpoint: {seed}/{name}.",
                    )
            source_inventory.append(
                _source_row(endpoint_path, f"step1500_{model_type}_seed{seed}", endpoint_result["final_checkpoint_sha256"])
            )
            source_inventory.append(_source_row(branch_attempt / "branch_start.json", f"branch_start_{model_type}_seed{seed}"))
            source_inventory.append(_source_row(branch_attempt / "result.json", f"branch_result_{model_type}_seed{seed}"))

            source_entry = source_abs.get((seed, model_type, 0))
            _need(source_entry is not None, f"Compensation readout missing for {seed}/{model_type}/0.")
            if model_type == "sparse":
                _need(source_entry["phase_label"] == "post_pruning_event", f"Sparse start phase mismatch for seed {seed}.")
            else:
                _need(source_entry["phase_label"] == "pre_pruning_parent", f"Unpruned start phase mismatch for seed {seed}.")
            checkpoint_entries[model_type] = {
                "step1000": {
                    "phase": source_entry["phase_label"],
                    "model_state_sha256": state_dict_hash(state),
                    "parent_checkpoint": str(prefix_path),
                    "parent_checkpoint_sha256": prefix_hash,
                    "mask_sha256": mask_digest if model_type == "sparse" else None,
                    "source_common_readout_ce": float(source_entry["ce"]),
                    **(
                        {
                            "transition_snapshot": str(transition_snapshot),
                            "transition_snapshot_sha256": transition_snapshot_hash,
                            "transition_snapshot_model_state_sha256": state_dict_hash(transition_snapshot_state),
                            "transition_snapshot_numeric_values_equal": True,
                            "different_signed_zero_encodings": signed_zero_differences,
                            "raw_state_hash_equal": state_dict_hash(state) == state_dict_hash(transition_snapshot_state),
                        }
                        if model_type == "sparse"
                        else {}
                    ),
                },
                "step1500": {
                    "checkpoint": str(endpoint_path),
                    "checkpoint_sha256": endpoint_result["final_checkpoint_sha256"],
                    "model_state_sha256": state_dict_hash(endpoint_state),
                    "source_common_readout_ce": float(source_abs[(seed, model_type, 500)]["ce"]),
                },
            }
            pair_starts[model_type] = start

        _need(
            pair_starts["sparse"]["validation_batch_hash"] == pair_starts["unpruned"]["validation_batch_hash"],
            f"Sparse/unpruned validation batches differ for seed {seed}.",
        )
        _need(
            pair_starts["sparse"]["training_batch_hash"] == pair_starts["unpruned"]["training_batch_hash"],
            f"Sparse/unpruned training batches differ for seed {seed}.",
        )
        pre = _json_read(attempt / "segment_0/grid_1000_pre.json")
        post = _json_read(attempt / "segment_0/grid_1000.json")
        _need(
            pre["precision"] == post["precision"] == "FP32_TF32off"
            and int(pre["target_tokens"]) == int(post["target_tokens"]) == EXPECTED_TOKENS
            and int(pre["batches"]) == int(post["batches"]) == EXPECTED_BATCHES,
            f"Immediate event validation identity mismatch for seed {seed}.",
        )
        _need(seed in damage_rows, f"Validation event damage is missing for seed {seed}.")
        source_gap = (
            float(source_abs[(seed, "sparse", 0)]["ce"])
            - float(source_abs[(seed, "unpruned", 0)]["ce"])
        )
        _need(
            abs(source_gap - float(damage_rows[seed]["D_post_minus_pre"])) <= CE_ATOL,
            f"Compensation gap does not reproduce registered immediate pruning damage for seed {seed}.",
        )
        _need(
            abs(
                float(post["ce"])
                - float(pre["ce"])
                - float(damage_rows[seed]["D_post_minus_pre"])
            )
            <= CE_ATOL,
            f"Transition event readout does not reproduce registered validation damage for seed {seed}.",
        )
        batches, validation_hash = _validation_batches(seed, variant)
        _need(validation_hash == prefix_meta["validation_hash"], f"Source validation hash differs for seed {seed}.")
        known_identities[f"diagnostic-probe-seed-{seed}"] = probe_identities[seed]
        known_identities[f"training-seed-{seed}-next500"] = training_suffix_hash
        known_identities[f"validation-seed-{seed}"] = validation_hash

        intermediate = []
        for model_type in MODEL_TYPES:
            result, branch_attempt, _ = _branch(seed, model_type)
            checkpoint_dir = branch_attempt / "checkpoints"
            saved_offsets = []
            if checkpoint_dir.is_dir():
                for candidate in checkpoint_dir.glob("*.pt"):
                    stem = candidate.stem.lower()
                    for offset, step in zip(OFFSETS, (1000, 1025, 1100, 1250, 1500), strict=True):
                        if str(step) in stem or f"offset_{offset}" in stem:
                            saved_offsets.append(offset)
            intermediate.extend(saved_offsets)
        available = sorted(set((0, 500, *intermediate)))
        provenance["validation_contract"]["intermediate_offsets"][str(seed)] = [
            offset for offset in available if offset not in PRIMARY_OFFSETS
        ]
        provenance["seeds"][str(seed)] = {
            "validation_sha256": validation_hash,
            "validation_batches": len(batches),
            "validation_target_tokens": sum(targets.numel() for _, targets in batches),
            "validation_batch_target_counts": [int(targets.numel()) for _, targets in batches],
            "training_steps_1_1500_sha256": training_hash,
            "shared_only_next_500_training_sha256": training_suffix_hash,
            "transition_mask_sha256": mask_digest,
            "immediate_event_damage_validation": float(damage_rows[seed]["D_post_minus_pre"]),
            "checkpoints": checkpoint_entries,
            "available_offsets": available,
            "source_training_condition": "router_experts_frozen; shared parameters train, router and experts fixed",
            "model_config": variant["config"]["model"],
            "common_parent_verified": True,
            "transition_sparse_snapshot_values_verified": True,
            "transition_sparse_snapshot_signed_zero_representation_differences": signed_zero_differences,
            "mask_zeros_verified_at_step1000_and1500": True,
            "frozen_router_and_experts_verified": True,
        }

    for identity_key, actual_hash in known_identities.items():
        _need(
            pseudo_hashes[identity_key] == actual_hash,
            f"Registered adaptation identity hash mismatch: {identity_key}.",
        )
        source_hash_audit[identity_key] = {
            "expected_sha256": pseudo_hashes[identity_key],
            "actual_sha256": actual_hash,
            "status": "verified",
        }
    provenance["registered_adaptation_input_hashes"] = source_hash_audit
    provenance["registered_adaptation_hash_status"] = (
        "PASS_WITH_DOCUMENTED_EVALUATION_API_EXTENSION"
        if any(value["status"] == "intentional_evaluation_api_extension" for value in source_hash_audit.values())
        else "PASS"
    )
    provenance["status"] = provenance["registered_adaptation_hash_status"]
    provenance["engineering_extension"] = {
        "files": [
            "src/moe_lth/models/moe_layer.py",
            "src/moe_lth/models/transformer.py",
        ],
        "change": "optional capacity_enabled flag and expert computation chunking; default capacity path retains native acceptance and gate behavior",
        "registered_hash_effect": "current file hashes differ from the factorial's registered implementation hashes because of this additive diagnostic API; all other registered file and data identity hashes were checked",
        "native_behavior_test": "default and explicitly enabled capacity paths are exactly equal in test_model.py",
    }
    provenance["historical_endpoint_reuse_candidates"] = {
        "inventory_path": str(reuse_candidates_path),
        "candidate_count": len(reuse_candidates),
        "reusable_as_full_trajectory": False,
        "included_as_additional_readout_states": False,
        "reason": "candidate inventory requires offset-25/100 checkpoints and diagnostic measurements that are not available",
    }
    _csv_write(OUT / "artifact_inventory.csv", source_inventory)
    _json_write(OUT / "checkpoint_and_validation_provenance.json", provenance)
    return provenance, source_inventory, variants


def _analysis_config(provenance: dict) -> dict:
    config = {
        "analysis_id": "imp_capacity_diagnostic_v1",
        "purpose": "checkpoint-only controlled evaluation of native versus capacity rejection disabled",
        "seeds": list(SEEDS),
        "independent_replication_unit": "source seed",
        "model_types": list(MODEL_TYPES),
        "condition": "router_experts_frozen",
        "offsets_requested": list(OFFSETS),
        "primary_offsets": list(PRIMARY_OFFSETS),
        "available_offsets_by_seed": {
            seed: provenance["seeds"][str(seed)]["available_offsets"] for seed in SEEDS
        },
        "primary_policies": ["NATIVE", "NO_DROP_ALL"],
        "layer_local_policies": [f"NO_DROP_L{layer}" for layer in range(4)],
        "validation": {
            "expected_batches": EXPECTED_BATCHES,
            "expected_target_tokens": EXPECTED_TOKENS,
            "batching": "identical original canonical 31-batch sequence per seed, including partial last batch",
            "precision": "FP32",
            "tf32": False,
            "evaluation_mode": "eval/inference",
            "loss": "task cross-entropy, summed NLL divided by total target tokens",
            "independent_test_set": False,
        },
        "capacity": {
            "unit": "all token positions in the complete original input batch at each MoE layer",
            "capacity_formula": "max(1, ceil(capacity_factor * batch_size * sequence_length * top_k / num_experts))",
            "overflow_order": "descending selected gate confidence per expert using native topk(capacity), unchanged",
            "no_drop": "remove capacity rejection only; preserve top-k, full softmax, selected gate, expert, residual and downstream rerouting",
        },
        "tolerances": {"ce_absolute": CE_ATOL, "model_atol": MODEL_ATOL, "model_rtol": MODEL_RTOL},
        "source_paths": {
            "transition": str(TRANSITION),
            "adaptation": str(ADAPTATION),
            "compensation": str(COMPENSATION),
            "uploaded_shared_only_diagnostics": "definitions audited from the attached CSV header and scripts/imp_postpruning_adaptation_v1/diagnostics.py",
        },
        "interpretation_limits": [
            "K and J are controlled checkpoint-evaluation contrasts, not uniquely attributable mediation effects.",
            "No outcome is a training run without capacity limits.",
            "The repeated validation set is not an independent test set.",
            "Seeds are the only independent replication units; tokens, batches, experts, and layers are not independent replications.",
            "No seed selection, threshold, clamping, or winner classifier is used.",
        ],
        "provenance_status": provenance["status"],
    }
    _json_write(OUT / "analysis_config.json", config)
    return config


def _metric_definitions() -> str:
    return """# Metric definition audit

## What the source diagnostic calls a dropped token

`RouteTrace.accepted_mask` has shape `(flattened batch positions, top_k)` and
marks accepted token–expert assignments. The source `MoEFeedForward` computes
capacity over the complete flattened input batch (`batch_size * sequence_length`)
at each layer. It does not compute capacity independently per sequence. For each
expert, assignments above capacity are ordered by the selected gate confidence
(unless an explicit `RouteOverride.capacity_scores` is supplied) and only the
highest-confidence `capacity` assignments are accepted. For the registered
top-1 model, a dropped assignment is exactly a token position whose single
selected assignment is rejected. In the shared-only probe implementation,
`dropped_assignment_rate = 1 - accepted_assignments / proposed_assignments`;
`dropped_token_rate = mean(~accepted_mask.all(dim=-1))`, with all flattened
positions as the token denominator. With top-1 these rates coincide. These
CSV fields are counts/rates on the sealed training probe, not validation-set
measurements.

The attached shared-only CSV records those counts and rates, but the columns do
not themselves encode the capacity formula or dispatch boundary; those are
audited against `src/moe_lth/models/moe_layer.py`. A batch is one complete
`model(inputs)` call. Splitting a batch for capacity accounting would change
the acceptance context and is prohibited.

## Dropped token computation

The expert-loop output starts at zero. A rejected assignment makes no selected
expert contribution. The transformer block still adds the pre-MoE `hidden`
residual to the attention output (`hidden + expert_output`); the residual is
not dropped or scaled. No-drop preserves each selected expert and its original
softmax gate, does not redistribute or renormalize, and recomputes subsequent
layers from the changed hidden states.

## Validation and loss

For each policy, record batch summed task NLL and target count. Full-validation
CE is total NLL divided by total targets (492,416), preserving all 31 original
batches, including the final partial batch. Auxiliary loss is reported
separately and is never added to task CE. Prediction JS, top-1 agreement, and
rejected-position contrasts compare NO_DROP_ALL with the same-checkpoint native
baseline. Native rejection masks are fixed before policy comparison. Rejected
subsets can overlap across layers and are not additive loss decompositions.

## Estimands

`G_native = L_sparse,native - L_unpruned,native`;
`G_nodrop = L_sparse,nodrop - L_unpruned,nodrop`;
`B_sparse = L_sparse,native - L_sparse,nodrop`;
`B_unpruned = L_unpruned,native - L_unpruned,nodrop`;
`K = G_native - G_nodrop = B_sparse - B_unpruned`.
`H_native = G_native(0) - G_native(500)`;
`H_nodrop = G_nodrop(0) - G_nodrop(500)`;
`J = H_native - H_nodrop = K(0) - K(500)`.
Layer-local `B`, `K`, and start-to-end change are secondary and must not be
summed across layers. All signs are retained.

## Routing denominators and scopes

Assignment rates use proposed assignments as the denominator; token rates use
flattened token positions. Loads are per expert within one full original batch.
Capacity and maximum proposed load, load CVs, assignment agreement, accepted-set
Jaccard, accepted/rejected gate summaries, and coverage are layer-level values
repeated on expert rows for convenience; the row has an explicit scope field.
Expert proposal/acceptance/rejection counts are expert-level. Undefined
zero-denominator metrics remain blank and have a corresponding flag.
"""


def _write_metric_definitions() -> None:
    path = OUT / "metric_definitions.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_metric_definitions(), encoding="utf-8")


@contextmanager
def _precision_guard():
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _rng_snapshot() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _numpy_rng_equal(left, right) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _rng_equal(left: dict, right: dict) -> bool:
    cuda_equal = left["torch_cuda"] is None or (
        len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))
    )
    return (
        left["python"] == right["python"]
        and _numpy_rng_equal(left["numpy"], right["numpy"])
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and cuda_equal
    )


def _gradient_signature(model: torch.nn.Module) -> dict:
    result = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        result[name] = None if grad is None else (
            grad.data_ptr(),
            grad._version,
            tuple(grad.shape),
            hashlib.sha256(grad.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        )
    return result


def _routes_cpu(output) -> list[dict]:
    routes = []
    for trace in output.routes:
        routes.append(
            {
                "ids": trace.selected_expert_indices.detach().cpu().clone(),
                "gates": trace.selected_probabilities.detach().float().cpu().clone(),
                "accepted": trace.accepted_mask.detach().reshape(
                    *trace.selected_expert_indices.shape
                ).cpu().clone(),
            }
        )
    return routes


def _cv(values: list[int]) -> float | None:
    average = sum(values) / len(values) if values else 0.0
    return statistics.pstdev(values) / average if average else None


def _summary(values: torch.Tensor) -> dict:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "q10": None, "q50": None, "q90": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "q10": float(torch.quantile(values, 0.1)),
        "q50": float(torch.quantile(values, 0.5)),
        "q90": float(torch.quantile(values, 0.9)),
    }


def _accepted_set(route: dict, num_experts: int) -> torch.Tensor:
    ids = route["ids"].reshape(-1, route["ids"].shape[-1])
    accepted = route["accepted"].reshape_as(ids)
    result = torch.zeros((ids.shape[0], num_experts), dtype=torch.bool)
    for slot in range(ids.shape[1]):
        valid = accepted[:, slot]
        if valid.any():
            result[valid, ids[valid, slot]] = True
    return result


def _routing_rows(
    *,
    model: TinyMoELanguageModel,
    checkpoint_id: str,
    seed: int,
    model_type: str,
    offset: int,
    policy: str,
    routes: list[dict],
    native_routes: list[dict],
    batch_index: int | None = None,
) -> list[dict]:
    rows = []
    for layer, current in enumerate(routes):
        native = native_routes[layer]
        ids = current["ids"].reshape(-1, current["ids"].shape[-1])
        gates = current["gates"].reshape_as(ids).float()
        accepted = current["accepted"].reshape_as(ids)
        native_ids = native["ids"].reshape_as(ids)
        native_accepted = native["accepted"].reshape_as(ids)
        proposed_load = torch.bincount(ids.reshape(-1), minlength=model.blocks[layer].moe.num_experts).tolist()
        accepted_ids = ids[accepted]
        accepted_load = torch.bincount(accepted_ids, minlength=model.blocks[layer].moe.num_experts).tolist()
        rejected_gates = native["gates"].reshape_as(ids).float()
        accepted_gate_summary = _summary(rejected_gates[native_accepted])
        rejected_gate_summary = _summary(rejected_gates[~native_accepted])
        current_native_accepted_gates = gates[native_accepted]
        current_native_rejected_gates = gates[~native_accepted]
        native_set = _accepted_set(native, model.blocks[layer].moe.num_experts)
        current_set = _accepted_set(current, model.blocks[layer].moe.num_experts)
        union = int((native_set | current_set).sum())
        overlap = (
            int((native_set & current_set).sum()) / union if union else None
        )
        assignment_agreement = float((ids == native_ids).float().mean())
        total_proposed = int(ids.numel())
        total_rejected = int((~accepted).sum())
        token_denominator = int(ids.shape[0])
        capacity = max(
            1,
            math.ceil(
                model.blocks[layer].moe.capacity_factor
                * token_denominator
                * model.blocks[layer].moe.top_k
                / model.blocks[layer].moe.num_experts
            ),
        )
        for expert in range(model.blocks[layer].moe.num_experts):
            proposed = int((ids == expert).sum())
            accepted_count = int(((ids == expert) & accepted).sum())
            rows.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "offset": offset,
                    "global_step": 1000 + offset,
                    "checkpoint_id": checkpoint_id,
                    "policy": policy,
                    "layer": layer,
                    **({"batch_index": batch_index} if batch_index is not None else {}),
                    "expert": expert,
                    "row_scope": "expert_counts; layer_metrics_repeated",
                    "proposed_assignments": proposed,
                    "accepted_assignments": accepted_count,
                    "rejected_assignments": proposed - accepted_count,
                    "proposed_assignment_denominator": total_proposed,
                    "rejected_assignment_rate": total_rejected / total_proposed if total_proposed else None,
                    "rejected_token_count_any_assignment": int((~accepted.all(dim=-1)).sum()),
                    "token_position_denominator": token_denominator,
                    "rejected_token_rate": float((~accepted.all(dim=-1)).float().mean()) if token_denominator else None,
                    "expert_capacity_per_batch": capacity,
                    "maximum_proposed_expert_load": max(proposed_load, default=0),
                    "proposed_load_cv_population": _cv(proposed_load),
                    "accepted_load_cv_population": _cv(accepted_load),
                    "accepted_load_cv_undefined": _cv(accepted_load) is None,
                    "assignment_agreement_with_native": assignment_agreement,
                    "accepted_set_jaccard_with_native": overlap,
                    "accepted_set_jaccard_undefined": union == 0,
                    "layer_proposed_assignments": total_proposed,
                    "layer_accepted_assignments": int(accepted.sum()),
                    "layer_rejected_assignments": total_rejected,
                    "layer_rejected_assignment_rate": total_rejected / total_proposed if total_proposed else None,
                    "layer_accepted_token_coverage": float(accepted.any(dim=-1).float().mean()) if token_denominator else None,
                    "layer_native_accepted_gate_count": accepted_gate_summary["count"],
                    "layer_native_accepted_gate_mean": accepted_gate_summary["mean"],
                    "layer_native_accepted_gate_std": accepted_gate_summary["std"],
                    "layer_native_accepted_gate_q10": accepted_gate_summary["q10"],
                    "layer_native_accepted_gate_q50": accepted_gate_summary["q50"],
                    "layer_native_accepted_gate_q90": accepted_gate_summary["q90"],
                    "layer_native_rejected_gate_count": rejected_gate_summary["count"],
                    "layer_native_rejected_gate_mean": rejected_gate_summary["mean"],
                    "layer_native_rejected_gate_std": rejected_gate_summary["std"],
                    "layer_native_rejected_gate_q10": rejected_gate_summary["q10"],
                    "layer_native_rejected_gate_q50": rejected_gate_summary["q50"],
                    "layer_native_rejected_gate_q90": rejected_gate_summary["q90"],
                    "readout_gate_on_native_accepted_mean": float(current_native_accepted_gates.mean()) if current_native_accepted_gates.numel() else None,
                    "readout_gate_on_native_rejected_mean": float(current_native_rejected_gates.mean()) if current_native_rejected_gates.numel() else None,
                    "readout_gate_on_native_accepted_undefined": current_native_accepted_gates.numel() == 0,
                    "readout_gate_on_native_rejected_undefined": current_native_rejected_gates.numel() == 0,
                }
            )
    return rows


def _capacity_policy_mapping(layer_count: int, offset: int) -> dict[str, tuple[bool, ...]]:
    if layer_count <= 0:
        raise ValueError("A capacity policy requires at least one transformer layer.")
    policies = {
        "NATIVE": (True,) * layer_count,
        "NO_DROP_ALL": (False,) * layer_count,
    }
    if offset in PRIMARY_OFFSETS:
        for layer in range(layer_count):
            flags = [True] * layer_count
            flags[layer] = False
            policies[f"NO_DROP_L{layer}"] = tuple(flags)
    return policies


def _make_policies(layer_count: int, offset: int) -> dict[str, tuple[bool, ...]]:
    return _capacity_policy_mapping(layer_count, offset)


def _validate_routing_rows(
    routing_rows: list[dict],
    *,
    layer_count: int,
    offset: int,
    expected_batch_count: int | None = None,
) -> None:
    policy_mapping = _capacity_policy_mapping(layer_count, offset)
    expected_policies = set(policy_mapping)
    rows_by_policy_batch_layer: dict[tuple[str, int, int], list[dict]] = {}
    seen_experts: dict[tuple[str, int, int], set[int]] = {}
    rows_by_policy: dict[str, list[dict]] = {}
    policies_seen = set()

    def integer_count(row: dict, field: str, context: str) -> int:
        value = row.get(field)
        _need(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"Invalid nonnegative integer count {field} in {context}.",
        )
        return value

    for row in routing_rows:
        _need(isinstance(row, dict), "Routing rows must be mappings.")
        policy = row.get("policy")
        _need(
            isinstance(policy, str) and policy in policy_mapping,
            f"Unknown capacity policy in routing rows: {policy!r}.",
        )
        layer = row.get("layer")
        _need(
            isinstance(layer, int)
            and not isinstance(layer, bool)
            and 0 <= layer < layer_count,
            f"Invalid routing layer index for {policy!r}: {layer!r}.",
        )
        expert = row.get("expert")
        _need(
            isinstance(expert, int) and not isinstance(expert, bool) and expert >= 0,
            f"Invalid expert index for {policy!r}/layer{layer}: {expert!r}.",
        )
        context = f"{policy}/layer{layer}/expert{expert}"
        proposed = integer_count(row, "proposed_assignments", context)
        accepted = integer_count(row, "accepted_assignments", context)
        rejected = integer_count(row, "rejected_assignments", context)
        _need(accepted <= proposed, f"Accepted assignments exceed proposed assignments in {context}.")
        _need(
            accepted + rejected == proposed,
            f"Assignment counts are inconsistent in {context}.",
        )
        rows_by_policy.setdefault(policy, []).append(row)
        policies_seen.add(policy)

    _need(
        policies_seen == expected_policies,
        f"Routing rows do not contain the expected capacity policies: "
        f"expected {sorted(expected_policies)}, got {sorted(policies_seen)}.",
    )

    batch_indices_by_policy = {}
    for policy, policy_rows in rows_by_policy.items():
        has_batch_index = ["batch_index" in row for row in policy_rows]
        _need(
            all(has_batch_index) or not any(has_batch_index),
            f"Routing rows mix batch-indexed and legacy rows for {policy}.",
        )
        if all(has_batch_index):
            batch_indices = set()
            for row in policy_rows:
                batch_index = row["batch_index"]
                _need(
                    isinstance(batch_index, int)
                    and not isinstance(batch_index, bool)
                    and batch_index >= 0,
                    f"Invalid batch index for {policy}: {batch_index!r}.",
                )
                batch_indices.add(batch_index)
                key = (policy, batch_index, int(row["layer"]))
                expert = int(row["expert"])
                _need(
                    expert not in seen_experts.setdefault(key, set()),
                    f"Duplicate expert routing row in {policy}/batch{batch_index}/layer{row['layer']}/expert{expert}.",
                )
                seen_experts[key].add(expert)
                rows_by_policy_batch_layer.setdefault(key, []).append(row)
            batch_indices_by_policy[policy] = batch_indices
            continue

        first_layer_rows = 0
        for row in policy_rows:
            if row["layer"] != 0:
                break
            first_layer_rows += 1
        _need(first_layer_rows > 0, f"Cannot infer legacy routing batch shape for {policy}.")
        rows_per_batch = layer_count * first_layer_rows
        _need(
            len(policy_rows) % rows_per_batch == 0,
            f"Legacy routing rows do not form complete layer batches for {policy}.",
        )
        batch_indices = set()
        for batch_index, start in enumerate(range(0, len(policy_rows), rows_per_batch)):
            batch_indices.add(batch_index)
            batch_rows = policy_rows[start : start + rows_per_batch]
            for row_index, row in enumerate(batch_rows):
                expected_layer = row_index // first_layer_rows
                _need(
                    row["layer"] == expected_layer,
                    f"Legacy routing layer order is invalid for {policy}/batch{batch_index}.",
                )
                layer = int(row["layer"])
                expert = int(row["expert"])
                key = (policy, batch_index, layer)
                _need(
                    expert not in seen_experts.setdefault(key, set()),
                    f"Duplicate expert routing row in {policy}/batch{batch_index}/layer{layer}/expert{expert}.",
                )
                seen_experts[key].add(expert)
                rows_by_policy_batch_layer.setdefault(key, []).append(row)
        batch_indices_by_policy[policy] = batch_indices

    reference_batches = batch_indices_by_policy[next(iter(policy_mapping))]
    for policy, batch_indices in batch_indices_by_policy.items():
        _need(batch_indices == reference_batches, f"Routing batch coverage differs for {policy}.")
    if expected_batch_count is not None:
        _need(
            reference_batches == set(range(expected_batch_count)),
            f"Routing rows do not cover the expected {expected_batch_count} batches.",
        )

    expected_experts_by_layer: dict[int, set[int]] = {}
    for policy, enabled_layers in policy_mapping.items():
        for batch_index in reference_batches:
            for layer, capacity_enabled in enumerate(enabled_layers):
                key = (policy, batch_index, layer)
                layer_rows = rows_by_policy_batch_layer.get(key, [])
                _need(layer_rows, f"Routing rows are missing {policy}/batch{batch_index}/layer{layer}.")
                expert_indices = seen_experts[key]
                if layer not in expected_experts_by_layer:
                    expected_experts_by_layer[layer] = expert_indices
                _need(
                    expert_indices == expected_experts_by_layer[layer],
                    f"Expert routing rows differ across policies for layer {layer}.",
                )
                proposed_total = sum(int(row["proposed_assignments"]) for row in layer_rows)
                accepted_total = sum(int(row["accepted_assignments"]) for row in layer_rows)
                rejected_total = sum(int(row["rejected_assignments"]) for row in layer_rows)
                _need(
                    accepted_total + rejected_total == proposed_total
                    and accepted_total <= proposed_total,
                    f"Layer assignment counts are inconsistent in {policy}/batch{batch_index}/layer{layer}.",
                )
                layer_fields = (
                    ("layer_proposed_assignments", proposed_total),
                    ("layer_accepted_assignments", accepted_total),
                    ("layer_rejected_assignments", rejected_total),
                )
                for row in layer_rows:
                    context = f"{policy}/batch{batch_index}/layer{layer}/expert{row['expert']}"
                    for field, expected in layer_fields:
                        value = integer_count(row, field, context)
                        _need(value == expected, f"Repeated layer count {field} is inconsistent in {context}.")
                if not capacity_enabled:
                    _need(
                        accepted_total == proposed_total and rejected_total == 0,
                        f"Capacity-disabled layer rejected assignments: {policy}/layer{layer}.",
                    )


def _ce_vector(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).detach().float().cpu()


def _js_sum_and_agreement(
    native_logits: torch.Tensor,
    policy_logits: torch.Tensor,
    chunk_tokens: int = 2048,
) -> tuple[float, int, int]:
    native_flat = native_logits.reshape(-1, native_logits.shape[-1])
    policy_flat = policy_logits.reshape_as(native_flat)
    js_sum = 0.0
    agreement = int((native_flat.argmax(dim=-1) == policy_flat.argmax(dim=-1)).sum())
    for start in range(0, native_flat.shape[0], chunk_tokens):
        left = native_flat[start : start + chunk_tokens].float()
        right = policy_flat[start : start + chunk_tokens].float()
        p = torch.softmax(left, dim=-1)
        q = torch.softmax(right, dim=-1)
        midpoint = (p + q) * 0.5
        log_midpoint = midpoint.clamp_min(1e-30).log()
        left_kl = (p * (p.clamp_min(1e-30).log() - log_midpoint)).sum(dim=-1)
        right_kl = (q * (q.clamp_min(1e-30).log() - log_midpoint)).sum(dim=-1)
        js_sum += float(((left_kl + right_kl) * 0.5).double().sum())
    return js_sum, agreement, int(native_flat.shape[0])


def _policy_specs(
    model: TinyMoELanguageModel,
    policy: str,
    routes: list[dict],
) -> list[dict]:
    return _routing_rows(
        model=model,
        checkpoint_id="",
        seed=0,
        model_type="",
        offset=0,
        policy=policy,
        routes=routes,
        native_routes=routes,
    )


def _evaluate_checkpoint(
    *,
    state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    config: dict,
    batches: list,
    seed: int,
    model_type: str,
    offset: int,
    checkpoint_id: str,
    checkpoint_hash: str,
    validation_hash: str,
    device: torch.device,
    include_profile: bool = False,
) -> tuple[dict, dict]:
    entry_rng = _rng_snapshot()
    model = TinyMoELanguageModel(config).to(device)
    model.load_state_dict(state)
    initial_modes = [(module, module.training) for module in model.modules()]
    model.train(initial_modes[0][1])
    _restore_rng(entry_rng)
    for name, mask in masks.items():
        _need(
            torch.count_nonzero(dict(model.named_parameters())[name][~mask.to(device).bool()]).item() == 0,
            f"Masked zero is not exact before evaluation: {seed}/{model_type}/{name}.",
        )
    initial_model_hash = state_dict_hash(model.state_dict())
    initial_rng = _rng_snapshot()
    initial_gradients = _gradient_signature(model)
    policies = _capacity_policy_mapping(len(model.blocks), offset)
    expected_batch_count = len(batches)
    expected_target_count = sum(int(targets.numel()) for _inputs, targets in batches)
    accum = {
        name: {
            "loss_sum": 0.0,
            "target_tokens": 0,
            "batches": 0,
            "auxiliary_target_weighted_sum": 0.0,
            "batch_rows": [],
            "routing_rows": [],
        }
        for name in policies
    }
    prediction = {
        "js_sum": 0.0,
        "agreement": 0,
        "targets": 0,
        "rejected": {},
    }
    numerical = {
        "disabled_intervention_identical_to_native": True,
        "first_intervened_layer_raw_scores_identical": True,
        "first_intervened_layer_probabilities_ids_gates_identical": True,
        "layer_local_upstream_unchanged": True,
        "native_vs_policy_accepted_contribution_equal": True,
    }
    profile = {"batch_seconds": [], "peak_memory_bytes": 0}
    previous_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode(), _precision_guard(), torch.autocast(device_type=device.type, enabled=False):
            for batch_index, (inputs_cpu, targets_cpu) in enumerate(batches):
                inputs = inputs_cpu.to(device)
                targets = targets_cpu.to(device)
                native_start = time.perf_counter()
                native_output = model(inputs)
                native_seconds = time.perf_counter() - native_start
                native_logits = native_output.logits.detach()
                native_routes = _routes_cpu(native_output)
                native_hidden = [
                    value.detach().float().cpu().clone()
                    for value in native_output.pre_router_hidden_states
                ]
                native_ce = _ce_vector(native_logits, targets)
                native_aux = float(native_output.auxiliary_loss.detach().float().cpu())
                native_positions = int(targets.numel())
                native_logits_cpu = native_logits.float().cpu()
                if batch_index == 0:
                    explicit_native = model(
                        inputs,
                        capacity_enabled_layers=[True] * len(model.blocks),
                    )
                    _need(
                        torch.equal(native_logits, explicit_native.logits),
                        "Explicitly disabled intervention does not reproduce native output.",
                    )
                    numerical["disabled_intervention_identical_to_native"] &= True
                    del explicit_native
                native_output = None
                native_logits = None
                accum["NATIVE"]["loss_sum"] += float(native_ce.double().sum())
                accum["NATIVE"]["target_tokens"] += native_positions
                accum["NATIVE"]["batches"] += 1
                accum["NATIVE"]["auxiliary_target_weighted_sum"] += native_aux * native_positions
                accum["NATIVE"]["batch_rows"].append(
                    {
                        "batch_index": batch_index,
                        "target_tokens": native_positions,
                        "nll_sum": float(native_ce.double().sum()),
                        "task_ce": float(native_ce.double().mean()),
                        "auxiliary_loss": native_aux,
                    }
                )
                accum["NATIVE"]["routing_rows"].extend(
                    _routing_rows(
                        model=model,
                        checkpoint_id=checkpoint_id,
                        seed=seed,
                        model_type=model_type,
                        offset=offset,
                        policy="NATIVE",
                        routes=native_routes,
                        native_routes=native_routes,
                        batch_index=batch_index,
                    )
                )

                batch_policy_rows = {"NATIVE": accum["NATIVE"]["batch_rows"][-1]}
                native_rejected_by_layer = []
                native_ids_by_layer = []
                for route in native_routes:
                    accepted = route["accepted"].reshape(-1, route["accepted"].shape[-1])
                    native_rejected_by_layer.append(~accepted.all(dim=-1))
                    native_ids_by_layer.append(route["ids"])
                native_rejected_any = torch.stack(native_rejected_by_layer).any(dim=0)

                for policy, capacity_enabled_layers in policies.items():
                    if policy == "NATIVE":
                        continue
                    policy_start = time.perf_counter()
                    output = model(
                        inputs,
                        capacity_enabled_layers=list(capacity_enabled_layers),
                    )
                    elapsed = time.perf_counter() - policy_start
                    policy_logits = output.logits.detach()
                    routes = _routes_cpu(output)
                    disabled_layers = [
                        i for i, enabled in enumerate(capacity_enabled_layers) if not enabled
                    ]
                    for layer in disabled_layers:
                        if not routes[layer]["accepted"].all():
                            raise RuntimeError(f"{policy} rejected assignments in disabled layer {layer}.")
                        if routes[layer]["ids"].shape[-1] != 1:
                            raise RuntimeError("The registered top-1 cohort did not retain exactly one expert per token.")
                    if disabled_layers:
                        layer = disabled_layers[0]
                        base_hidden = native_hidden[layer].to(device)
                        policy_hidden = output.pre_router_hidden_states[layer].detach()
                        _need(
                            torch.allclose(base_hidden, policy_hidden, atol=MODEL_ATOL, rtol=MODEL_RTOL),
                            f"Layer-local intervention changed its own upstream input: {seed}/{policy}/layer{layer}.",
                        )
                        router = model.blocks[layer].moe.router
                        base_scores = router.projection(base_hidden) / router.temperature
                        policy_scores = router.projection(policy_hidden) / router.temperature
                        _need(
                            torch.allclose(base_scores, policy_scores, atol=MODEL_ATOL, rtol=MODEL_RTOL),
                            f"Raw first-intervention router scores differ: {seed}/{policy}/layer{layer}.",
                        )
                        _need(
                            torch.allclose(
                                torch.softmax(base_scores, dim=-1),
                                torch.softmax(policy_scores, dim=-1),
                                atol=MODEL_ATOL,
                                rtol=MODEL_RTOL,
                            )
                            and torch.equal(
                                native_routes[layer]["ids"],
                                routes[layer]["ids"],
                            )
                            and torch.equal(
                                native_routes[layer]["gates"],
                                routes[layer]["gates"],
                            ),
                            f"First-intervention routing probabilities, IDs, or gates changed: {seed}/{policy}/layer{layer}.",
                        )
                        for upstream in range(layer):
                            _need(
                                torch.allclose(
                                    native_hidden[upstream],
                                    output.pre_router_hidden_states[upstream].detach().float().cpu(),
                                    atol=MODEL_ATOL,
                                    rtol=MODEL_RTOL,
                                ),
                                f"Layer-local intervention changed upstream hidden states: {seed}/{policy}/layer{upstream}.",
                            )
                            _need(
                                torch.equal(
                                    native_routes[upstream]["ids"],
                                    routes[upstream]["ids"],
                                ),
                                f"Layer-local intervention changed upstream assignments: {seed}/{policy}/layer{upstream}.",
                            )
                    if disabled_layers:
                        numerical["first_intervened_layer_raw_scores_identical"] &= True
                        numerical["first_intervened_layer_probabilities_ids_gates_identical"] &= True
                        numerical["layer_local_upstream_unchanged"] &= True
                    policy_ce = _ce_vector(policy_logits, targets)
                    auxiliary = float(output.auxiliary_loss.detach().float().cpu())
                    accum[policy]["loss_sum"] += float(policy_ce.double().sum())
                    accum[policy]["target_tokens"] += native_positions
                    accum[policy]["batches"] += 1
                    accum[policy]["auxiliary_target_weighted_sum"] += auxiliary * native_positions
                    policy_row = {
                        "batch_index": batch_index,
                        "target_tokens": native_positions,
                        "nll_sum": float(policy_ce.double().sum()),
                        "task_ce": float(policy_ce.double().mean()),
                        "auxiliary_loss": auxiliary,
                    }
                    if policy == "NO_DROP_ALL":
                        js_sum, agreement, compared = _js_sum_and_agreement(
                            native_logits_cpu.to(device), policy_logits, chunk_tokens=2048
                        )
                        prediction["js_sum"] += js_sum
                        prediction["agreement"] += agreement
                        prediction["targets"] += compared
                        delta = policy_ce - native_ce
                        rejected_stats = []
                        for layer, rejected in enumerate(native_rejected_by_layer):
                            count = int(rejected.sum())
                            rejected_stats.append(
                                {
                                    "subset": f"native_rejected_layer_{layer}",
                                    "comparison": "NO_DROP_ALL minus NATIVE",
                                    "positions": count,
                                    "native_nll_sum": float(native_ce[rejected].double().sum()),
                                    "policy_nll_sum": float(policy_ce[rejected].double().sum()),
                                    "paired_nll_difference_sum": float(delta[rejected].double().sum()),
                                    "native_ce": float(native_ce[rejected].double().mean()) if count else None,
                                    "policy_ce": float(policy_ce[rejected].double().mean()) if count else None,
                                    "paired_ce_difference": float(delta[rejected].double().mean()) if count else None,
                                }
                            )
                        union_count = int(native_rejected_any.sum())
                        rejected_stats.append(
                            {
                                "subset": "native_rejected_any_layer",
                                "comparison": "NO_DROP_ALL minus NATIVE",
                                "positions": union_count,
                                "native_nll_sum": float(native_ce[native_rejected_any].double().sum()),
                                "policy_nll_sum": float(policy_ce[native_rejected_any].double().sum()),
                                "paired_nll_difference_sum": float(delta[native_rejected_any].double().sum()),
                                "native_ce": float(native_ce[native_rejected_any].double().mean()) if union_count else None,
                                "policy_ce": float(policy_ce[native_rejected_any].double().mean()) if union_count else None,
                                "paired_ce_difference": float(delta[native_rejected_any].double().mean()) if union_count else None,
                            }
                        )
                        policy_row["paired_nll_difference_sum"] = float(delta.double().sum())
                        policy_row["paired_ce_difference"] = float(delta.double().mean())
                        policy_row["top1_agreement"] = agreement / compared
                        policy_row["js_divergence_mean"] = js_sum / compared
                        policy_row["native_rejected_subsets"] = rejected_stats
                        prediction["rejected"][str(batch_index)] = rejected_stats
                    accum[policy]["batch_rows"].append(policy_row)
                    accum[policy]["routing_rows"].extend(
                        _routing_rows(
                            model=model,
                            checkpoint_id=checkpoint_id,
                            seed=seed,
                            model_type=model_type,
                            offset=offset,
                            policy=policy,
                            routes=routes,
                            native_routes=native_routes,
                            batch_index=batch_index,
                        )
                    )

                    if include_profile and batch_index == 0:
                        profile["batch_seconds"].append(elapsed)
                        if device.type == "cuda":
                            profile["peak_memory_bytes"] = max(
                                profile["peak_memory_bytes"],
                                int(torch.cuda.max_memory_allocated(device)),
                            )
                    del output, policy_logits

                if include_profile and batch_index == 0:
                    profile["batch_seconds"].append(native_seconds)
                    if device.type == "cuda":
                        profile["peak_memory_bytes"] = max(
                            profile["peak_memory_bytes"],
                            int(torch.cuda.max_memory_allocated(device)),
                        )
                del native_logits_cpu, native_ce
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
    finally:
        for module, training in previous_modes:
            module.training = training

    final_model_hash = state_dict_hash(model.state_dict())
    final_rng = _rng_snapshot()
    final_gradients = _gradient_signature(model)
    final_modes = [(module, module.training) for module in model.modules()]
    _need(initial_model_hash == final_model_hash, "Evaluation mutated model weights or buffers.")
    _need(_rng_equal(initial_rng, final_rng), "Evaluation changed an RNG state.")
    _need(initial_gradients == final_gradients, "Evaluation changed gradient state.")
    _need(
        [value for _module, value in initial_modes] == [value for _module, value in final_modes],
        "Evaluation failed to restore module modes.",
    )
    for name, mask in masks.items():
        _need(
            torch.count_nonzero(dict(model.named_parameters())[name][~mask.to(device).bool()]).item() == 0,
            f"Evaluation changed exact masked zeros: {name}.",
        )

    readouts = {}
    per_batch_rows = []
    routing_rows = []
    for policy, record in accum.items():
        _need(record["target_tokens"] == expected_target_count, f"Wrong token denominator for {seed}/{model_type}/{offset}/{policy}.")
        _need(record["batches"] == expected_batch_count, f"Wrong batch count for {seed}/{model_type}/{offset}/{policy}.")
        ce = record["loss_sum"] / record["target_tokens"]
        _need(math.isfinite(ce), "Nonfinite task CE.")
        readouts[policy] = {
            "loss_sum": record["loss_sum"],
            "target_tokens": record["target_tokens"],
            "batches": record["batches"],
            "task_ce": ce,
            "auxiliary_loss_target_weighted_mean": record["auxiliary_target_weighted_sum"] / record["target_tokens"],
            "model_state_sha256_before_after": initial_model_hash,
            "checkpoint_sha256": checkpoint_hash,
            "validation_sha256": validation_hash,
            "precision": "FP32_TF32off",
        }
        for row in record["batch_rows"]:
            per_batch_rows.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "offset": offset,
                    "global_step": 1000 + offset,
                    "checkpoint_id": checkpoint_id,
                    "policy": policy,
                    "scope": "full_validation_batch",
                    **{key: value for key, value in row.items() if key != "native_rejected_subsets"},
                }
            )
            for subset in row.get("native_rejected_subsets", []):
                per_batch_rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "offset": offset,
                        "global_step": 1000 + offset,
                        "checkpoint_id": checkpoint_id,
                        "policy": policy,
                        "scope": "native_rejected_subset",
                        "batch_index": row["batch_index"],
                        "target_tokens": row["target_tokens"],
                        "subset_positions": subset["positions"],
                        "nll_sum": subset["policy_nll_sum"],
                        "task_ce": subset["policy_ce"],
                        "auxiliary_loss": None,
                        "subset": subset["subset"],
                        "comparison": subset["comparison"],
                        "native_nll_sum": subset["native_nll_sum"],
                        "paired_nll_difference_sum": subset["paired_nll_difference_sum"],
                        "paired_ce_difference": subset["paired_ce_difference"],
                    }
                )
        routing_rows.extend(record["routing_rows"])

    prediction_summary = {
        "mean_probability_distribution_js_divergence": (
            prediction["js_sum"] / prediction["targets"] if prediction["targets"] else None
        ),
        "prediction_top1_agreement": (
            prediction["agreement"] / prediction["targets"] if prediction["targets"] else None
        ),
        "target_count": prediction["targets"],
        "native_rejected_subsets_by_batch": prediction["rejected"],
    }
    numeric_rows = {
        "disabled_intervention_identical_to_native": numerical["disabled_intervention_identical_to_native"],
        "first_intervened_layer_raw_scores_identical": numerical["first_intervened_layer_raw_scores_identical"],
        "first_intervened_layer_probabilities_ids_gates_identical": numerical["first_intervened_layer_probabilities_ids_gates_identical"],
        "layer_local_upstream_unchanged": numerical["layer_local_upstream_unchanged"],
        "native_vs_policy_accepted_contribution_equal": numerical["native_vs_policy_accepted_contribution_equal"],
        "model_state_hash_preserved": initial_model_hash == final_model_hash,
        "rng_preserved": _rng_equal(initial_rng, final_rng),
        "gradient_state_preserved": initial_gradients == final_gradients,
        "module_modes_preserved": [value for _, value in initial_modes] == [value for _, value in final_modes],
        "masked_zeros_preserved": True,
    }
    result = {
        "seed": seed,
        "model_type": model_type,
        "offset": offset,
        "global_step": 1000 + offset,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_hash,
        "validation_sha256": validation_hash,
        "policies": readouts,
        "prediction_diagnostics": prediction_summary,
        "per_batch_rows": per_batch_rows,
        "routing_rows": routing_rows,
        "numerical_checks": numeric_rows,
        "profile": profile,
    }
    del model
    return result, numeric_rows


def _state_for(
    seed: int,
    model_type: str,
    offset: int,
    variant: dict,
) -> tuple[dict, dict, str, str]:
    if offset == 0:
        state, masks, path = _start_state(seed, model_type, variant)
        source_hash = _sha256(path)
        return state, masks, f"step1000_{model_type}_seed{seed}", source_hash
    if offset == 500:
        state, result, path, _start = _endpoint_state(seed, model_type)
        masks = load_masks(variant["mask"]) if model_type == "sparse" else {}
        return state, masks, f"step1500_{model_type}_seed{seed}", result["final_checkpoint_sha256"]
    raise RuntimeError(f"No verified model snapshot for offset {offset}; refusing reconstruction.")


def _profile_one_batch(
    seed: int,
    variant: dict,
    device: torch.device,
    provenance: dict,
) -> dict:
    batches, validation_hash = _validation_batches(seed, variant)
    state, masks, checkpoint_id, checkpoint_hash = _state_for(seed, "sparse", 0, variant)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result, _checks = _evaluate_checkpoint(
        state=state,
        masks=masks,
        config=variant["config"]["model"],
        batches=batches[:1],
        seed=seed,
        model_type="sparse",
        offset=0,
        checkpoint_id=checkpoint_id,
        checkpoint_hash=checkpoint_hash,
        validation_hash=validation_hash,
        device=device,
        include_profile=True,
    )
    _need(result["policies"]["NATIVE"]["batches"] == 1, "Profile did not execute one full native batch.")
    return {
        "status": "PASS",
        "seed": seed,
        "batch_index": 0,
        "batch_target_tokens": int(batches[0][1].numel()),
        "batch_count_in_validation": len(batches),
        "memory_peak_bytes": result["profile"]["peak_memory_bytes"],
        "per_policy_batch_seconds": result["profile"]["batch_seconds"],
        "device": str(device),
        "precision": "FP32_TF32off",
        "checkpoint": checkpoint_id,
        "checkpoint_sha256": checkpoint_hash,
    }


def _read_cache(path: Path, identity: dict) -> dict | None:
    if not path.is_file():
        return None
    cached = _json_read(path)
    if cached.get("cache_identity") != identity or cached.get("complete") is not True:
        return None
    return cached["result"]


def _preserve_failed_attempt() -> Path:
    attempt_dir = OUT / "attempts" / "capacity_assertion_scope_fix_20261027"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    record_path = attempt_dir / "correction_record.json"
    if record_path.is_file():
        existing = _json_read(record_path)
        current_hash = _sha256(Path(__file__))
        if existing.get("new_source_sha256") != current_hash:
            history = existing.setdefault("source_revision_history", [])
            history.append(existing.get("new_source_sha256"))
            existing["new_source_sha256"] = current_hash
            _json_write(record_path, existing)
        return attempt_dir

    inventory_path = OUT / "artifact_inventory.csv"
    _need(inventory_path.is_file(), "Cannot preserve the failed attempt without the prior source inventory.")
    prior_inventory = _read_csv(inventory_path)
    prior_hashes = {row["role"]: row["sha256"] for row in prior_inventory}
    old_runner_hash = prior_hashes.get("capacity_diagnostic_implementation")
    _need(old_runner_hash is not None, "Prior inventory does not contain the diagnostic source hash.")

    unchanged_sources = {
        "capacity_dispatch_model_extension": ROOT / "src/moe_lth/models/moe_layer.py",
        "capacity_policy_model_extension": ROOT / "src/moe_lth/models/transformer.py",
    }
    unchanged_source_hashes = {}
    for role, path in unchanged_sources.items():
        _need(role in prior_hashes, f"Prior inventory does not contain {role}.")
        current_hash = _sha256(path)
        _need(
            current_hash == prior_hashes[role],
            f"Forward implementation changed since the cached evaluation: {path}",
        )
        unchanged_source_hashes[str(path)] = {
            "prior_sha256": prior_hashes[role],
            "current_sha256": current_hash,
            "unchanged": True,
        }

    preserved_files = {}
    for name in (
        "artifact_inventory.csv",
        "analysis_config.json",
        "completion.json",
        "compute_accounting.json",
        "evaluation_profile.json",
        "findings.md",
        "metric_definitions.md",
        "numerical_checks.json",
        "gpu_admission.json",
        "checkpoint_and_validation_provenance.json",
    ):
        source = OUT / name
        if source.is_file():
            destination = attempt_dir / f"prior_{name}"
            shutil.copy2(source, destination)
            preserved_files[name] = {
                "path": str(destination),
                "sha256": _sha256(destination),
            }

    cache_files = []
    for cache_path in sorted((OUT / "cache").glob("*.json")):
        cache = _json_read(cache_path)
        if cache.get("complete") is True:
            cache_files.append(
                {
                    "path": str(cache_path),
                    "sha256": _sha256(cache_path),
                    "cache_identity": cache.get("cache_identity"),
                    "result_seed": cache.get("result", {}).get("seed"),
                    "result_model_type": cache.get("result", {}).get("model_type"),
                    "result_offset": cache.get("result", {}).get("offset"),
                }
            )

    _json_write(
        record_path,
        {
            "status": "PRESERVED_BEFORE_CORRECTION",
            "correction": "Capacity-disabled runtime assertion incorrectly applied to all NO_DROP-prefixed layers.",
            "impact": "Validation-only defect; capacity policy execution and cached numerical readouts are unchanged.",
            "old_source_sha256": old_runner_hash,
            "new_source_sha256": _sha256(Path(__file__)),
            "prior_attempt_files": preserved_files,
            "prior_cache_files": cache_files,
            "unchanged_forward_implementation_hashes": unchanged_source_hashes,
            "historical_registration_hashes_modified": False,
            "cache_compatibility": {
                "policy_version_unchanged": POLICY_VERSION,
                "forward_implementation_unchanged": True,
                "checkpoint_and_validation_identities": "verified against current audited state before each cache reuse",
                "numerical_settings": "FP32_TF32off; cached readout precision and 31-batch/492416-target shape checked before reuse",
                "policy_mapping": {
                    "legacy": "NATIVE all enabled; NO_DROP_ALL all disabled; NO_DROP_Lk disables only layer k",
                    "current": {
                        policy: list(enabled)
                        for policy, enabled in _capacity_policy_mapping(4, 0).items()
                    },
                    "policy_version_unchanged": True,
                },
                "cached_routing_rows": "revalidated with count consistency and per-disabled-layer acceptance assertions before output use",
            },
        },
    )
    return attempt_dir


def _validate_cached_readout(
    result: dict,
    *,
    identity: dict,
    checkpoint_id: str,
    checkpoint_hash: str,
    validation_hash: str,
    layer_count: int,
    offset: int,
) -> None:
    _need(result.get("checkpoint_id") == checkpoint_id, "Cached checkpoint ID does not match the current source checkpoint.")
    _need(result.get("checkpoint_sha256") == checkpoint_hash, "Cached checkpoint hash does not match the current source checkpoint.")
    _need(result.get("validation_sha256") == validation_hash == identity["validation_sha256"],
          "Cached validation identity does not match the current canonical batch sequence.")
    policy_mapping = _capacity_policy_mapping(layer_count, offset)
    _need(set(result.get("policies", {})) == set(policy_mapping),
          "Cached policy records do not match the verified capacity policy mapping.")
    for policy, record in result["policies"].items():
        _need(record.get("checkpoint_sha256") == checkpoint_hash,
              f"Cached {policy} checkpoint hash does not match the current source checkpoint.")
        _need(record.get("validation_sha256") == validation_hash,
              f"Cached {policy} validation identity does not match the current batches.")
        _need(record.get("precision") == "FP32_TF32off",
              f"Cached {policy} readout uses unexpected numerical settings.")
        _need(record.get("batches") == EXPECTED_BATCHES and record.get("target_tokens") == EXPECTED_TOKENS,
              f"Cached {policy} readout has an unexpected validation shape.")
    _need(result.get("prediction_diagnostics", {}).get("target_count") == EXPECTED_TOKENS,
          "Cached prediction diagnostics have an unexpected validation denominator.")
    _need(result.get("numerical_checks", {}).get("model_state_hash_preserved") is True,
          "Cached model-state preservation check did not pass.")
    _need(result.get("numerical_checks", {}).get("rng_preserved") is True,
          "Cached RNG preservation check did not pass.")
    for key in (
        "disabled_intervention_identical_to_native",
        "first_intervened_layer_raw_scores_identical",
        "first_intervened_layer_probabilities_ids_gates_identical",
        "layer_local_upstream_unchanged",
        "native_vs_policy_accepted_contribution_equal",
        "model_state_hash_preserved",
        "rng_preserved",
        "gradient_state_preserved",
        "module_modes_preserved",
        "masked_zeros_preserved",
    ):
        _need(result["numerical_checks"].get(key) is True,
              f"Cached numerical check {key} did not pass.")
    _validate_routing_rows(
        result.get("routing_rows", []),
        layer_count=layer_count,
        offset=offset,
        expected_batch_count=EXPECTED_BATCHES,
    )


def _record_cache_revalidation(attempt_dir: Path, entry: dict) -> None:
    path = attempt_dir / "cache_revalidation.json"
    if path.is_file():
        record = _json_read(path)
    else:
        record = {
            "status": "PASS",
            "policy_version": POLICY_VERSION,
            "cache_entries_revalidated": [],
        }
    for index, existing in enumerate(record["cache_entries_revalidated"]):
        if existing["path"] == entry["path"]:
            record["cache_entries_revalidated"][index] = {**existing, **entry}
            break
    else:
        record["cache_entries_revalidated"].append(entry)
    _json_write(path, record)


def _revalidate_existing_caches(
    variants: dict[int, dict],
    attempt_dir: Path,
) -> None:
    for cache_path in sorted((OUT / "cache").glob("*.json")):
        envelope = _json_read(cache_path)
        if envelope.get("complete") is not True:
            continue
        result = envelope.get("result")
        _need(isinstance(result, dict), f"Complete cache has no result mapping: {cache_path}.")
        seed = result.get("seed")
        model_type = result.get("model_type")
        offset = result.get("offset")
        _need(seed in variants and model_type in MODEL_TYPES, f"Unknown cached readout cohort identity: {cache_path}.")
        _need(offset in PRIMARY_OFFSETS, f"Unknown cached readout offset: {cache_path}.")
        variant = variants[seed]
        state, masks, checkpoint_id, checkpoint_hash = _state_for(
            seed, model_type, offset, variant
        )
        batches, validation_hash = _validation_batches(seed, variant)
        identity = {
            "version": POLICY_VERSION,
            "model_type": model_type,
            "offset": offset,
            "model_state_sha256": state_dict_hash(state),
            "mask_sha256": _mask_hash(masks) if masks else hashlib.sha256(b"").hexdigest(),
            "validation_sha256": validation_hash,
        }
        if envelope.get("cache_identity") != identity:
            continue
        _validate_cached_readout(
            result,
            identity=identity,
            checkpoint_id=checkpoint_id,
            checkpoint_hash=checkpoint_hash,
            validation_hash=validation_hash,
            layer_count=int(variant["config"]["model"]["num_layers"]),
            offset=offset,
        )
        _record_cache_revalidation(
            attempt_dir,
            {
                "path": str(cache_path),
                "sha256": _sha256(cache_path),
                "cache_identity": identity,
                "checkpoint_id": checkpoint_id,
                "checkpoint_sha256": checkpoint_hash,
                "validation_sha256": validation_hash,
                "validation_batches": len(batches),
                "validation_target_tokens": sum(int(targets.numel()) for _, targets in batches),
                "precision": "FP32_TF32off",
                "policy_mapping": {
                    policy: list(flags)
                    for policy, flags in _capacity_policy_mapping(
                        int(variant["config"]["model"]["num_layers"]),
                        offset,
                    ).items()
                },
                "routing_validation": "PASS",
                "numerical_checks": result["numerical_checks"],
                "cache_consumed_by_final_report": False,
            },
        )


def _cache_write(path: Path, identity: dict, result: dict) -> None:
    _json_write(path, {"complete": True, "cache_identity": identity, "result": result})


def _contrast_tables(
    readouts: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    absolute = []
    per_batch = []
    routing = []
    predictions = []
    lookup = {}
    for checkpoint in readouts:
        seed = int(checkpoint["seed"])
        model_type = checkpoint["model_type"]
        offset = int(checkpoint["offset"])
        lookup[(seed, model_type, offset)] = checkpoint
        for policy, value in checkpoint["policies"].items():
            absolute.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "offset": offset,
                    "global_step": 1000 + offset,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "policy": policy,
                    "task_ce": value["task_ce"],
                    "total_nll_sum": value["loss_sum"],
                    "target_tokens": value["target_tokens"],
                    "batches": value["batches"],
                    "auxiliary_loss_target_weighted_mean": value["auxiliary_loss_target_weighted_mean"],
                    "precision": value["precision"],
                }
            )
        per_batch.extend(checkpoint["per_batch_rows"])
        routing.extend(checkpoint["routing_rows"])
        diagnostics = checkpoint["prediction_diagnostics"]
        predictions.append(
            {
                "seed": seed,
                "model_type": model_type,
                "offset": offset,
                "global_step": 1000 + offset,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "comparison": "NO_DROP_ALL_minus_NATIVE",
                "prediction_top1_agreement": diagnostics["prediction_top1_agreement"],
                "mean_probability_distribution_js_divergence": diagnostics["mean_probability_distribution_js_divergence"],
                "target_count": diagnostics["target_count"],
                "native_rejected_subset_diagnostics_json": json.dumps(
                    diagnostics["native_rejected_subsets_by_batch"], separators=(",", ":"), allow_nan=False
                ),
            }
        )

    contrasts = []
    for seed in SEEDS:
        for offset in PRIMARY_OFFSETS:
            sparse = lookup[(seed, "sparse", offset)]["policies"]
            unpruned = lookup[(seed, "unpruned", offset)]["policies"]
            g_native = sparse["NATIVE"]["task_ce"] - unpruned["NATIVE"]["task_ce"]
            g_nodrop = sparse["NO_DROP_ALL"]["task_ce"] - unpruned["NO_DROP_ALL"]["task_ce"]
            b_sparse = sparse["NATIVE"]["task_ce"] - sparse["NO_DROP_ALL"]["task_ce"]
            b_unpruned = unpruned["NATIVE"]["task_ce"] - unpruned["NO_DROP_ALL"]["task_ce"]
            k = g_native - g_nodrop
            _need(abs(k - (b_sparse - b_unpruned)) <= 1e-12, "Primary K identity failed.")
            contrasts.append(
                {
                    "seed": seed,
                    "offset": offset,
                    "global_step": 1000 + offset,
                    "G_native": g_native,
                    "G_no_drop_all": g_nodrop,
                    "B_sparse": b_sparse,
                    "B_unpruned": b_unpruned,
                    "K": k,
                    "identity_residual_K": k - (b_sparse - b_unpruned),
                    "sparse_native_ce": sparse["NATIVE"]["task_ce"],
                    "sparse_no_drop_all_ce": sparse["NO_DROP_ALL"]["task_ce"],
                    "unpruned_native_ce": unpruned["NATIVE"]["task_ce"],
                    "unpruned_no_drop_all_ce": unpruned["NO_DROP_ALL"]["task_ce"],
                }
            )
        start = next(row for row in contrasts if row["seed"] == seed and row["offset"] == 0)
        end = next(row for row in contrasts if row["seed"] == seed and row["offset"] == 500)
        h_native = start["G_native"] - end["G_native"]
        h_nodrop = start["G_no_drop_all"] - end["G_no_drop_all"]
        j = h_native - h_nodrop
        _need(abs(j - (start["K"] - end["K"])) <= 1e-12, "Primary J identity failed.")
        start.update(
            {
                "H_native": h_native,
                "H_no_drop_all": h_nodrop,
                "J": j,
                "identity_residual_J": j - (start["K"] - end["K"]),
            }
        )

    layer_rows = []
    layer_lookup = {}
    for seed in SEEDS:
        for offset in PRIMARY_OFFSETS:
            for layer in range(4):
                values = {}
                for model_type in MODEL_TYPES:
                    policies = lookup[(seed, model_type, offset)]["policies"]
                    native_ce = policies["NATIVE"]["task_ce"]
                    local_ce = policies[f"NO_DROP_L{layer}"]["task_ce"]
                    values[model_type] = (native_ce, local_ce, native_ce - local_ce)
                sparse_b = values["sparse"][2]
                unpruned_b = values["unpruned"][2]
                k_layer = sparse_b - unpruned_b
                layer_lookup[(seed, layer, offset)] = k_layer
                layer_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "offset": offset,
                        "global_step": 1000 + offset,
                        "B_sparse": sparse_b,
                        "B_unpruned": unpruned_b,
                        "K_layer": k_layer,
                        "sparse_native_ce": values["sparse"][0],
                        "sparse_layer_local_ce": values["sparse"][1],
                        "unpruned_native_ce": values["unpruned"][0],
                        "unpruned_layer_local_ce": values["unpruned"][1],
                    }
                )
    for row in layer_rows:
        if row["offset"] == 0:
            end_k = layer_lookup[(row["seed"], row["layer"], 500)]
            row["J_layer_start_to_end"] = row["K_layer"] - end_k
            row["K_layer_endpoint"] = end_k
        else:
            row["J_layer_start_to_end"] = None
            row["K_layer_endpoint"] = None

    aggregate = []
    metric_names = (
        "G_native",
        "G_no_drop_all",
        "B_sparse",
        "B_unpruned",
        "K",
        "H_native",
        "H_no_drop_all",
        "J",
    )
    for offset in PRIMARY_OFFSETS:
        for metric in metric_names:
            values = [
                row[metric]
                for row in contrasts
                if row["offset"] == offset and row.get(metric) is not None
            ]
            if not values:
                continue
            aggregate.append(
                {
                    "scope": "seed-level primary",
                    "offset": offset,
                    "metric": metric,
                    "seed_values_json": json.dumps(
                        [
                            {"seed": row["seed"], "value": row[metric]}
                            for row in contrasts
                            if row["offset"] == offset and row.get(metric) is not None
                        ],
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "n_seeds": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                }
            )
    for model_type in MODEL_TYPES:
        for policy in ("NATIVE", "NO_DROP_ALL"):
            for offset in PRIMARY_OFFSETS:
                values = [
                    row
                    for row in absolute
                    if row["model_type"] == model_type
                    and row["policy"] == policy
                    and row["offset"] == offset
                ]
                seed_values = [
                    {"seed": row["seed"], "value": row["task_ce"]}
                    for row in sorted(values, key=lambda row: row["seed"])
                ]
                scores = [row["task_ce"] for row in values]
                aggregate.append(
                    {
                        "scope": f"absolute {model_type} {policy}",
                        "offset": offset,
                        "metric": "task_ce",
                        "seed_values_json": json.dumps(seed_values, separators=(",", ":"), allow_nan=False),
                        "n_seeds": len(scores),
                        "mean": statistics.mean(scores),
                        "median": statistics.median(scores),
                        "minimum": min(scores),
                        "maximum": max(scores),
                        "sample_sd": statistics.stdev(scores),
                    }
                )
    for layer in range(4):
        for offset in PRIMARY_OFFSETS:
            values = [row["K_layer"] for row in layer_rows if row["layer"] == layer and row["offset"] == offset]
            aggregate.append(
                {
                    "scope": f"secondary layer {layer}",
                    "offset": offset,
                    "metric": "K_layer",
                    "seed_values_json": json.dumps(
                        [
                            {"seed": row["seed"], "value": row["K_layer"]}
                            for row in layer_rows
                            if row["layer"] == layer and row["offset"] == offset
                        ],
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "n_seeds": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "sample_sd": statistics.stdev(values),
                }
            )
    return absolute, contrasts, layer_rows, aggregate, per_batch, routing, predictions


def _findings(
    contrasts: list[dict],
    layer_rows: list[dict],
    absolute_rows: list[dict],
    prediction_rows: list[dict],
    routing_rows: list[dict],
) -> str:
    by_seed = {}
    for seed in SEEDS:
        start = next(row for row in contrasts if row["seed"] == seed and row["offset"] == 0)
        end = next(row for row in contrasts if row["seed"] == seed and row["offset"] == 500)
        by_seed[seed] = (start, end)

    def values(field: str, offset: int) -> list[tuple[int, float]]:
        return [(seed, by_seed[seed][0 if offset == 0 else 1][field]) for seed in SEEDS]

    lines = [
        "# IMP capacity diagnostic findings",
        "",
        "## Scope and status",
        "",
        "Checkpoint-only FP32 evaluation of the existing abrupt IMP80 shared-only trajectory. No training, optimizer updates, mask discovery, or new recovery branch was run. The same canonical validation set was examined repeatedly; it is not an independent test set. Source seed is the replication unit (n=5).",
        "",
        "Positive `B` means disabling capacity rejection reduced that model's CE. Positive `K` means it reduced the sparse–unpruned gap. `J` is the difference in observed gap closure between native and no-drop evaluation, not a uniquely attributable mediation effect.",
        "",
        "## Seed-level primary contrasts",
        "",
        "| Seed | G native (0) | G no-drop (0) | B sparse (0) | B unpruned (0) | K (0) | G native (500) | G no-drop (500) | B sparse (500) | B unpruned (500) | K (500) | H native | H no-drop | J |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        start, end = by_seed[seed]
        lines.append(
            f"| {seed} | {start['G_native']:.8f} | {start['G_no_drop_all']:.8f} | "
            f"{start['B_sparse']:.8f} | {start['B_unpruned']:.8f} | {start['K']:.8f} | "
            f"{end['G_native']:.8f} | {end['G_no_drop_all']:.8f} | {end['B_sparse']:.8f} | "
            f"{end['B_unpruned']:.8f} | {end['K']:.8f} | {start['H_native']:.8f} | "
            f"{start['H_no_drop_all']:.8f} | {start['J']:.8f} |"
        )

    def summarize(field: str, offset: int) -> str:
        vals = [row[field] for row in contrasts if row["offset"] == offset and row.get(field) is not None]
        return (
            f"mean {statistics.mean(vals):.8f}, median {statistics.median(vals):.8f}, "
            f"min {min(vals):.8f}, max {max(vals):.8f}, sample SD {statistics.stdev(vals):.8f}"
        )

    start_ks = [by_seed[seed][0]["K"] for seed in SEEDS]
    end_ks = [by_seed[seed][1]["K"] for seed in SEEDS]
    b_sp0 = [by_seed[seed][0]["B_sparse"] for seed in SEEDS]
    b_un0 = [by_seed[seed][0]["B_unpruned"] for seed in SEEDS]
    b_sp5 = [by_seed[seed][1]["B_sparse"] for seed in SEEDS]
    b_un5 = [by_seed[seed][1]["B_unpruned"] for seed in SEEDS]
    j_values = [by_seed[seed][0]["J"] for seed in SEEDS]
    layer_summary = []
    for layer in range(4):
        start_values = {
            seed: next(
                row["K_layer"]
                for row in layer_rows
                if row["seed"] == seed and row["layer"] == layer and row["offset"] == 0
            )
            for seed in SEEDS
        }
        end_values = {
            seed: next(
                row["K_layer"]
                for row in layer_rows
                if row["seed"] == seed and row["layer"] == layer and row["offset"] == 500
            )
            for seed in SEEDS
        }
        layer_summary.append(
            {
                "layer": layer,
                "mean_absolute_start": statistics.mean(abs(value) for value in start_values.values()),
                "mean_absolute_end": statistics.mean(abs(value) for value in end_values.values()),
                "start_positive": sum(value > 0 for value in start_values.values()),
                "start_negative": sum(value < 0 for value in start_values.values()),
                "end_positive": sum(value > 0 for value in end_values.values()),
                "end_negative": sum(value < 0 for value in end_values.values()),
                "start": start_values,
                "end": end_values,
            }
        )
    largest_layer = max(layer_summary, key=lambda row: row["mean_absolute_start"])
    positives = lambda data: sum(value > 0 for value in data)
    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"A. Immediate gap: K(0) is positive in {positives(start_ks)}/5 seeds; K(0) values by seed are {json.dumps(dict(zip(SEEDS, start_ks)), sort_keys=True)}. `G_native(0)` versus `G_no_drop_all(0)` is shown seed by seed above.",
            f"B. Immediate model benefits: `B_sparse(0)` is positive in {positives(b_sp0)}/5 seeds and `B_unpruned(0)` is positive in {positives(b_un0)}/5. Values are retained signed; no-drop may worsen CE.",
            f"C. Endpoint effect: K(500) is positive in {positives(end_ks)}/5 seeds; signed values are {json.dumps(dict(zip(SEEDS, end_ks)), sort_keys=True)}. `B_sparse(500)` positive in {positives(b_sp5)}/5 and `B_unpruned(500)` positive in {positives(b_un5)}/5.",
            f"D. Layer-local K values are listed for every seed in `layer_local_contrasts.csv`. The largest by mean absolute start effect is L{largest_layer['layer']} (mean |K|={largest_layer['mean_absolute_start']:.8f}); this is descriptive, not a significance ranking. Per-layer signed values and sign counts (positive/negative out of five) are: " + "; ".join(
                f"L{row['layer']} start={json.dumps(row['start'], sort_keys=True)} ({row['start_positive']}/{row['start_negative']}), end={json.dumps(row['end'], sort_keys=True)} ({row['end_positive']}/{row['end_negative']})"
                for row in layer_summary
            ) + ".",
            f"E. The no-drop gap is positive in {sum(row['G_no_drop_all'] > 0 for row in contrasts if row['offset'] == 0)}/5 seeds at offset 0 and {sum(row['G_no_drop_all'] > 0 for row in contrasts if row['offset'] == 500)}/5 at offset 500. `G_no_drop_all(0)` is {summarize('G_no_drop_all', 0)}; at 500 it is {summarize('G_no_drop_all', 500)}. Signed results are not clamped.",
            f"F. NO_DROP_ALL worsens sparse CE at offset 0 in {sum(v < 0 for v in b_sp0)}/5 seeds and unpruned CE in {sum(v < 0 for v in b_un0)}/5; at offset 500 the counts are {sum(v < 0 for v in b_sp5)}/5 and {sum(v < 0 for v in b_un5)}/5, respectively.",
            f"G. Capacity-policy sensitivity of observed gap closure J is {json.dumps(dict(zip(SEEDS, j_values)), sort_keys=True)}; {summarize('J', 0)}. Positive J means observed native gap closure is smaller when both checkpoints are evaluated without dropping.",
            "",
            "## Routing and prediction diagnostics",
            "",
            "`routing_capacity_metrics.csv` records batch/layer/expert counts. Layer-level load CV, capacity, maximum load, gate summaries, agreement, overlap, and coverage are repeated across expert rows with an explicit row-scope label; they are not independent expert replications. `prediction_diagnostics.csv` reports paired top-1 agreement, mean probability JS divergence, and native-rejected-position summaries. Conditional rejected-position subsets overlap across layers and are not additive.",
            "",
            "## Interpretation boundary",
            "",
            "A favorable result supports a capacity-policy contribution to checkpoint performance under this evaluation intervention. It does not establish why IMP discovers tickets, why sparse beats dense, why the observed behavior occurs, or that capacity-aware pruning improves training. Drop rates alone do not quantify performance impact. K and J are not mediation fractions and do not describe training without capacity limits.",
            "",
            "## Display the share bundle",
            "",
            "```sh",
            "cat results/imp_capacity_diagnostic_v1/summary_bundle/absolute_losses.csv",
            "cat results/imp_capacity_diagnostic_v1/summary_bundle/seed_level_capacity_contrasts.csv",
            "cat results/imp_capacity_diagnostic_v1/summary_bundle/findings.md",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _plots(out: Path, absolute: list[dict], contrasts: list[dict], layer_rows: list[dict], routing: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {seed: color for seed, color in zip(SEEDS, plt.cm.tab10.colors, strict=False)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, offset in zip(axes, PRIMARY_OFFSETS, strict=True):
        for seed in SEEDS:
            row = next(item for item in contrasts if item["seed"] == seed and item["offset"] == offset)
            axis.plot(
                ["Native", "No drop"],
                [row["G_native"], row["G_no_drop_all"]],
                marker="o",
                color=colors[seed],
                label=f"seed {seed}",
            )
        axis.set_title(f"Offset {offset}")
        axis.set_ylabel("Sparse − unpruned task CE")
        axis.grid(alpha=0.25)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(figures / "native_vs_nodrop_gap.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for seed in SEEDS:
        rows = [row for row in contrasts if row["seed"] == seed]
        axis.plot([row["offset"] for row in rows], [row["K"] for row in rows], marker="o", color=colors[seed], label=f"seed {seed}")
    axis.set_xlabel("Shared-only updates from step 1000")
    axis.set_ylabel("K = G native − G no-drop")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "K_over_time.png", dpi=180)
    plt.close(fig)

    native = [row for row in routing if row["policy"] == "NATIVE"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, model_type in zip(axes, MODEL_TYPES, strict=True):
        for layer in range(4):
            means = []
            for offset in PRIMARY_OFFSETS:
                rates = [
                    row["layer_rejected_assignment_rate"]
                    for row in native
                    if row["model_type"] == model_type and row["layer"] == layer and row["offset"] == offset and row["expert"] == 0
                ]
                means.append(float(np.mean(rates)) if rates else float("nan"))
            axis.plot(PRIMARY_OFFSETS, means, marker="o", label=f"layer {layer}")
        axis.set_title(model_type)
        axis.set_xlabel("Offset")
        axis.set_ylabel("Native rejected assignment rate")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "native_rejected_rates_by_layer.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, offset in zip(axes, PRIMARY_OFFSETS, strict=True):
        for layer in range(4):
            values = [
                row["K_layer"]
                for row in layer_rows
                if row["offset"] == offset and row["layer"] == layer
            ]
            x = np.full(len(values), layer, dtype=float) + np.linspace(-0.08, 0.08, len(values))
            axis.scatter(x, values, label=f"layer {layer}" if offset == 0 else None)
            axis.plot([layer - 0.15, layer + 0.15], [float(np.mean(values))] * 2)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Offset {offset}")
        axis.set_xlabel("Layer")
        axis.set_ylabel("Layer-local K")
        axis.set_xticks(range(4))
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "layer_local_K.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for model_type in MODEL_TYPES:
        for policy in ("NATIVE", "NO_DROP_ALL"):
            rows = [
                row
                for row in absolute
                if row["model_type"] == model_type and row["policy"] == policy
            ]
            grouped = {}
            for row in rows:
                grouped.setdefault(row["offset"], []).append(row["task_ce"])
            offsets = sorted(grouped)
            means = [float(np.mean(grouped[offset])) for offset in offsets]
            axis.plot(offsets, means, marker="o", label=f"{model_type} · {policy}")
    axis.set_xlabel("Shared-only updates from step 1000")
    axis.set_ylabel("Task cross-entropy")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "absolute_losses_primary.png", dpi=180)
    plt.close(fig)


def _write_outputs(readouts: list[dict], numerical_checks: dict, compute: dict) -> None:
    absolute, contrasts, layer_rows, aggregates, per_batch, routing, predictions = _contrast_tables(readouts)
    identity_residuals = {
        "K_max_abs": max(abs(row["identity_residual_K"]) for row in contrasts),
        "J_max_abs": max(abs(row["identity_residual_J"]) for row in contrasts if row.get("identity_residual_J") is not None),
    }
    _need(
        identity_residuals["K_max_abs"] <= 1e-12
        and identity_residuals["J_max_abs"] <= 1e-12,
        "Algebraic capacity contrast identities failed numerical verification.",
    )
    numerical_checks["algebraic_identity_max_abs_residuals"] = identity_residuals
    numerical_checks["algebraic_identities"] = "PASS"
    _csv_write(OUT / "absolute_losses.csv", absolute)
    _csv_write(OUT / "seed_level_capacity_contrasts.csv", contrasts)
    _csv_write(OUT / "layer_local_contrasts.csv", layer_rows)
    _csv_write(OUT / "aggregate_summaries.csv", aggregates)
    _csv_write(OUT / "per_batch_losses.csv", per_batch)
    _csv_write(OUT / "routing_capacity_metrics.csv", routing)
    _csv_write(OUT / "prediction_diagnostics.csv", predictions)
    _json_write(OUT / "numerical_checks.json", numerical_checks)
    _json_write(OUT / "compute_accounting.json", compute)
    _plots(OUT, absolute, contrasts, layer_rows, routing)
    findings = _findings(contrasts, layer_rows, absolute, predictions, routing)
    (OUT / "findings.md").write_text(findings, encoding="utf-8")
    bundle = OUT / "summary_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("absolute_losses.csv", "seed_level_capacity_contrasts.csv", "findings.md"):
        source = OUT / name
        temporary = bundle / f".{name}.{time.time_ns()}.tmp"
        shutil.copyfile(source, temporary)
        temporary.replace(bundle / name)
    _json_write(
        OUT / "completion.json",
        {
            "status": "COMPLETE",
            "scientifically_complete": True,
            "seeds": list(SEEDS),
            "primary_offsets": list(PRIMARY_OFFSETS),
            "validated_readouts": len(readouts),
            "outputs": [
                "analysis_config.json",
                "artifact_inventory.csv",
                "metric_definitions.md",
                "checkpoint_and_validation_provenance.json",
                "numerical_checks.json",
                "absolute_losses.csv",
                "seed_level_capacity_contrasts.csv",
                "layer_local_contrasts.csv",
                "routing_capacity_metrics.csv",
                "prediction_diagnostics.csv",
                "per_batch_losses.csv",
                "aggregate_summaries.csv",
                "compute_accounting.json",
                "findings.md",
                "figures/",
                "summary_bundle/",
            ],
        },
    )


def _record_blocked(reason: str, provenance: dict, synthetic_tests: dict) -> None:
    correction_attempt = OUT / "attempts" / "capacity_assertion_scope_fix_20261027"
    cache_revalidation_path = correction_attempt / "cache_revalidation.json"
    revalidated_cache_count = (
        len(_json_read(cache_revalidation_path).get("cache_entries_revalidated", []))
        if cache_revalidation_path.is_file()
        else 0
    )
    _json_write(
        OUT / "compute_accounting.json",
        {
            "status": "BLOCKED_BEFORE_PROFILE",
            "reason": reason,
            "profile_completed": False,
            "evaluation_forwards": 0,
            "training_updates": 0,
            "optimizer_updates": 0,
            "mask_discovery_runs": 0,
            "source_checkpoint_audit_completed": True,
            "existing_cache_entries_revalidated": revalidated_cache_count,
            "seeds": list(SEEDS),
            "gpu_policy": "The registered adaptation worker requires utilization <=90%, >=24 GiB free GPU memory, and >=200 GiB free output filesystem space.",
            "source_provenance_status": provenance["status"],
        },
    )
    _json_write(
        OUT / "numerical_checks.json",
        {
            "status": "NOT_RUN_GPU_ADMISSION_BLOCKED",
            "source_hashes_masks_parent_checks": "PASS",
            "synthetic_capacity_and_chunking_tests": synthetic_tests,
            "native_common_readout_reproduction": "NOT_RUN",
            "disabled_vs_native_runtime_identity": "NOT_RUN",
            "no_drop_overflow_runtime_checks": "NOT_RUN",
            "model_rng_mode_gradient_preservation": "NOT_RUN",
            "checkpoint_evaluation": "NOT_RUN",
        },
    )
    _json_write(
        OUT / "completion.json",
        {
            "status": "BLOCKED",
            "scientifically_complete": False,
            "reason": reason,
            "source_audit_complete": True,
            "evaluation_started": False,
            "no_training_or_optimizer_updates": True,
            "existing_cache_entries_revalidated": revalidated_cache_count,
            "required_follow_up": "Re-run scripts/imp_capacity_diagnostic_v1/run.py --evaluate when source GPU admission passes.",
        },
    )
    (OUT / "findings.md").write_text(
        "# IMP capacity diagnostic status\n\n"
        "The source audit completed, but the full checkpoint cohort could not be aggregated because GPU admission failed. "
        "This is not a completed scientific result, and no capacity-effect contrast is inferred from historical drop rates.\n\n"
        f"Corrected routing assertions revalidated {revalidated_cache_count} pre-existing complete cache entr"
        f"{'y' if revalidated_cache_count == 1 else 'ies'} against current checkpoint, validation, precision, and policy identities. "
        "These cached readouts were not used to produce a cohort report.\n\n"
        f"Admission reason: `{reason}`.\n\n"
        "No training, optimizer update, mask discovery, or recovery branch was launched. "
        "Run the exact checkpoint-only analysis after the registered GPU admission limit is met.\n",
        encoding="utf-8",
    )


def run_evaluation(provenance: dict, variants: dict[int, dict]) -> None:
    correction_attempt = OUT / "attempts" / "capacity_assertion_scope_fix_20261027"
    test_python = ROOT / ".venv/bin/python"
    python_executable = str(test_python) if test_python.is_file() else sys.executable
    test_command = [
        python_executable,
        "-m",
        "pytest",
        "tests/test_model.py",
        "tests/test_imp_capacity_diagnostic_v1.py",
        "-q",
    ]
    test_run = subprocess.run(
        test_command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    synthetic_tests = {
        "command": " ".join(test_command),
        "return_code": test_run.returncode,
        "output": test_run.stdout.strip(),
        "status": "PASS" if test_run.returncode == 0 else "FAIL",
    }
    if test_run.returncode != 0:
        raise RuntimeError(
            "Synthetic capacity/numerical tests failed before GPU admission:\n"
            + test_run.stdout
            + test_run.stderr
        )
    _revalidate_existing_caches(variants, correction_attempt)
    try:
        admission = gpu_admission_check(wait_for_idle_seconds=GPU_IDLE_WAIT_SECONDS)
    except RuntimeError as exc:
        _record_blocked(str(exc), provenance, synthetic_tests)
        print(f"GPU admission blocked evaluation: {exc}")
        return
    _json_write(OUT / "gpu_admission.json", admission)
    with exclusive_claim(OUT / "evaluation.lock"):
        admission = gpu_admission_check()
        _json_write(OUT / "gpu_admission.json", admission)
        device = torch.device("cuda", torch.cuda.current_device())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        profile_path = OUT / "evaluation_profile.json"
        if not profile_path.is_file():
            profile = _profile_one_batch(7, variants[7], device, provenance)
            _json_write(profile_path, profile)

        all_readouts = []
        numerical_checks = {
            "status": "PASS",
            "source_ce_reproduction": {},
            "disabled_intervention_identical_to_native": True,
            "no_drop_zero_rejections_and_all_assignments_accepted": True,
            "top1_one_expert_per_token": True,
            "first_intervened_layer_raw_scores_identical": True,
            "first_intervened_layer_probabilities_ids_gates_identical": True,
            "layer_local_upstream_unchanged": True,
            "native_accepted_contributions_equal": "PASS in synthetic overflow unit test",
            "all_model_states_rng_gradients_modes_masks_preserved": True,
            "synthetic_checks": synthetic_tests,
            "TF32_disabled": True,
            "precision": "FP32",
        }
        forwards = 0
        peak_eval_memory_bytes = 0
        started = time.perf_counter()
        for seed in SEEDS:
            variant = variants[seed]
            batches, validation_hash = _validation_batches(seed, variant)
            layer_count = int(variant["config"]["model"]["num_layers"])
            for model_type in MODEL_TYPES:
                for offset in (0, 500):
                    state, masks, checkpoint_id, checkpoint_hash = _state_for(
                        seed, model_type, offset, variant
                    )
                    checkpoint_state_hash = state_dict_hash(state)
                    mask_digest = _mask_hash(masks) if masks else hashlib.sha256(b"").hexdigest()
                    identity = {
                        "version": POLICY_VERSION,
                        "model_type": model_type,
                        "offset": offset,
                        "model_state_sha256": checkpoint_state_hash,
                        "mask_sha256": mask_digest,
                        "validation_sha256": validation_hash,
                    }
                    cache_key = hashlib.sha256(
                        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    cache_path = OUT / "cache" / f"readout_{cache_key}.json"
                    cached = _read_cache(cache_path, identity)
                    did_evaluate = cached is None
                    if cached is None:
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        result, checks = _evaluate_checkpoint(
                            state=state,
                            masks=masks,
                            config=variant["config"]["model"],
                            batches=batches,
                            seed=seed,
                            model_type=model_type,
                            offset=offset,
                            checkpoint_id=checkpoint_id,
                            checkpoint_hash=checkpoint_hash,
                            validation_hash=validation_hash,
                            device=device,
                        )
                        result["peak_cuda_memory_bytes"] = (
                            int(torch.cuda.max_memory_allocated(device))
                            if device.type == "cuda"
                            else 0
                        )
                    else:
                        result = json.loads(json.dumps(cached))
                        checks = result["numerical_checks"]
                    _validate_cached_readout(
                        result,
                        identity=identity,
                        checkpoint_id=checkpoint_id,
                        checkpoint_hash=checkpoint_hash,
                        validation_hash=validation_hash,
                        layer_count=layer_count,
                        offset=offset,
                    )
                    if did_evaluate:
                        _cache_write(cache_path, identity, result)
                    else:
                        _record_cache_revalidation(
                            correction_attempt,
                            {
                                "path": str(cache_path),
                                "sha256": _sha256(cache_path),
                                "cache_identity": identity,
                                "checkpoint_id": checkpoint_id,
                                "checkpoint_sha256": checkpoint_hash,
                                "validation_sha256": validation_hash,
                                "precision": "FP32_TF32off",
                                "policy_mapping": {
                                    policy: list(enabled)
                                    for policy, enabled in _capacity_policy_mapping(
                                        layer_count,
                                        offset,
                                    ).items()
                                },
                                "routing_validation": "PASS",
                                "cache_consumed_by_final_report": True,
                            },
                        )
                    all_readouts.append(result)
                    peak_eval_memory_bytes = max(
                        peak_eval_memory_bytes,
                        int(result.get("peak_cuda_memory_bytes", 0)),
                    )
                    if did_evaluate:
                        forwards += EXPECTED_BATCHES * len(result["policies"])
                    for key in (
                        "model_state_hash_preserved",
                        "rng_preserved",
                        "gradient_state_preserved",
                        "module_modes_preserved",
                        "masked_zeros_preserved",
                    ):
                        numerical_checks["all_model_states_rng_gradients_modes_masks_preserved"] &= bool(
                            checks[key]
                        )
                        numerical_checks["disabled_intervention_identical_to_native"] &= bool(
                            checks["disabled_intervention_identical_to_native"]
                        )
                        numerical_checks["first_intervened_layer_raw_scores_identical"] &= bool(
                            checks["first_intervened_layer_raw_scores_identical"]
                        )
                        numerical_checks["first_intervened_layer_probabilities_ids_gates_identical"] &= bool(
                            checks["first_intervened_layer_probabilities_ids_gates_identical"]
                        )
                        numerical_checks["layer_local_upstream_unchanged"] &= bool(
                            checks["layer_local_upstream_unchanged"]
                        )
                    numerical_checks["no_drop_zero_rejections_and_all_assignments_accepted"] &= True
                    source_ce = _read_csv(COMPENSATION / "absolute_losses.csv")
                    source_by = {
                        (int(row["seed"]), row["model_type"], int(row["offset"])): float(row["ce"])
                        for row in source_ce
                        if row["condition"] == "router_experts_frozen"
                    }
                    for policy in ("NATIVE",):
                        actual = result["policies"][policy]["task_ce"]
                        expected = source_by[(seed, model_type, offset)]
                        error = abs(actual - expected)
                        numerical_checks["source_ce_reproduction"][
                            f"{seed}:{model_type}:{offset}"
                        ] = {
                            "actual": actual,
                            "registered": expected,
                            "absolute_error": error,
                            "tolerance": CE_ATOL,
                            "pass": error <= CE_ATOL,
                        }
                        _need(error <= CE_ATOL, f"Native registered CE reproduction failed: {seed}/{model_type}/{offset}.")
                    if device.type == "cuda":
                        del state
                        torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        compute = {
            "status": "COMPLETE",
            "profile": _json_read(profile_path),
            "readout_checkpoint_count": len(all_readouts),
            "readout_policy_forward_count": forwards,
            "profile_forward_count": len(_json_read(profile_path)["per_policy_batch_seconds"]),
            "total_forward_count_including_profile": forwards + len(_json_read(profile_path)["per_policy_batch_seconds"]),
            "evaluation_seconds_excluding_profile_and_audit": elapsed,
            "evaluation_forwards": forwards,
            "training_updates": 0,
            "optimizer_updates": 0,
            "mask_discovery_runs": 0,
            "peak_cuda_memory_bytes": max(
                peak_eval_memory_bytes,
                int(_json_read(profile_path)["memory_peak_bytes"]),
            ),
            "precision": "FP32_TF32off",
            "all_readouts_cached_atomically": True,
            "cache_directory": str(OUT / "cache"),
        }
        _write_outputs(all_readouts, numerical_checks, compute)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint-only IMP capacity diagnostic.")
    parser.add_argument("--audit-only", action="store_true", help="Verify and inventory source artifacts without evaluation.")
    parser.add_argument("--evaluate", action="store_true", help="Run the authorized checkpoint-only evaluation.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        _preserve_failed_attempt()
    provenance, _inventory, variants = audit_sources()
    _analysis_config(provenance)
    _write_metric_definitions()
    if args.audit_only:
        _json_write(
            OUT / "completion.json",
            {
                "status": "SOURCE_AUDIT_COMPLETE",
                "scientifically_complete": False,
                "evaluation_started": False,
                "seeds": list(SEEDS),
                "available_offsets": {seed: provenance["seeds"][str(seed)]["available_offsets"] for seed in SEEDS},
            },
        )
        return
    if not args.evaluate:
        parser.error("Choose --audit-only or --evaluate.")
    run_evaluation(provenance, variants)


if __name__ == "__main__":
    main()
