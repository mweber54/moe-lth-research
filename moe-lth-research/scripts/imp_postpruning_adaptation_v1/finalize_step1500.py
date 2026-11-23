"""Finalize a saved step-1500 branch without re-entering the training loop."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import af_native
import engine
from common import (
    GLOBAL_STEPS,
    OUT,
    TRANSITION,
    csv_save,
    group_for_parameter,
    json_read,
    json_save,
    sha256,
    torch_load,
    torch_save_atomic,
)
from diagnostics import collect_diagnostics
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.router_age import parameter_group, state_dict_hash


def _historical_dense_row(seed: int, step: int) -> dict:
    identifier = f"seed{seed}_dense"
    with (TRANSITION / "trajectory_metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["id"] == identifier and int(row["step"]) == step:
                return row
    raise RuntimeError(f"Historical dense readout is missing for {identifier} at step {step}.")


def _require_unique(rows: list[dict], key, label: str) -> None:
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate {label} rows are present.")


def _component_hashes(state: dict[str, object]) -> dict[str, str]:
    return {
        group: state_dict_hash(
            {name: value for name, value in state.items() if parameter_group(name) == group}
        )
        for group in ("expert", "shared", "router")
    }


def finalize_saved_attempt(attempt_dir: Path, device: torch.device) -> dict:
    started = time.monotonic()
    attempt_dir = attempt_dir.resolve()
    if not attempt_dir.is_relative_to((OUT / "attempts").resolve()):
        raise ValueError(f"Attempt path is outside the registered output namespace: {attempt_dir}")
    parts = attempt_dir.parts
    if len(parts) < 3 or parts[-2] != "seed17_unpruned_full":
        raise ValueError("Finalization-only repair is limited to the failed seed-17 unpruned/full pilot.")

    status_path = OUT / "status/seed17_unpruned_full.json"
    status = json_read(status_path)
    if status.get("status") != "FAILED" or "component_checkpoints does not exist" not in str(
        status.get("error", "")
    ):
        raise RuntimeError("The saved status does not match the registered finalization failure.")
    if Path(status["attempt"]).resolve() != attempt_dir:
        raise RuntimeError("Failed status points to a different attempt.")
    if (attempt_dir / "result.json").exists():
        raise FileExistsError("Attempt already has a result; refusing to overwrite it.")

    required_absent = (
        "checkpoints/final_recovered.pt",
        "component_checkpoints/final_router.pt",
        "optimizer_and_updates.csv",
        "trajectory_metrics.csv",
        "routing_and_gate_diagnostics.csv",
        "expert_function_diagnostics.csv",
        "shared_representation_diagnostics.csv",
        "diagnostic_manifest.json",
        "diagnostic_recomputation.json",
        "finalization_validation.json",
        "result.json",
    )
    existing = [attempt_dir / relative for relative in required_absent if (attempt_dir / relative).exists()]
    if existing:
        raise FileExistsError(
            "Finalization outputs already exist; inspect before retrying: "
            + ", ".join(map(str, existing))
        )

    registration = json_read(OUT / "registration.json")
    if int(registration.get("registration_revision", 0)) < 4:
        raise RuntimeError("An engineering registration revision 4 or later is required.")
    source_hash_path = Path(registration["source_hash_manifest"])
    if sha256(source_hash_path) != registration.get("source_hash_manifest_sha256"):
        raise RuntimeError("Engineering revision-3 source-hash manifest has changed.")
    for source_path, expected_hash in json_read(source_hash_path).items():
        path = Path(source_path)
        if path.is_absolute() and (not path.is_file() or sha256(path) != expected_hash):
            raise RuntimeError(f"Registered source/input hash mismatch: {path}")
    probe_path = OUT / "diagnostic_probe_manifest.json"
    if sha256(probe_path) != registration.get("diagnostic_probe_manifest_sha256"):
        raise RuntimeError("Current probe manifest differs from registration revision 3.")
    probe_row = next(
        row
        for row in json_read(probe_path)["seeds"]
        if int(row["seed"]) == 17
    )
    if registration["probe_identities"]["17"] != probe_row["probe_identity_sha256"]:
        raise RuntimeError("Seed-17 probe identity differs from the registered manifest.")

    prefix_dir = OUT / "prefixes/seed17"
    prefix_verification = json_read(prefix_dir / "prefix_verification.json")
    if prefix_verification.get("status") != "verified":
        raise RuntimeError("Seed-17 exact prefix is not verified.")
    prefix = af_native.decode_numpy(torch_load(prefix_dir / "reconstruction_interruption.pt"))
    if int(prefix["step"]) != 1000:
        raise RuntimeError("The saved source prefix is not at global step 1000.")
    checkpoint_path = attempt_dir / "interruption.pt"
    checkpoint_hash_before = sha256(checkpoint_path)
    checkpoint = af_native.decode_numpy(torch_load(checkpoint_path))
    if (
        int(checkpoint["step"]) != 1500
        or int(checkpoint["global_step"]) != 1500
        or int(checkpoint["sampler_next_batch"]) != 1500
    ):
        raise RuntimeError("Saved interruption checkpoint is not at the step-1500 boundary.")
    if checkpoint.get("study_condition") != "full":
        raise RuntimeError("Saved interruption checkpoint is not the full condition.")
    if checkpoint.get("trainability") != {
        "router": True,
        "experts": True,
        "shared": True,
    }:
        raise RuntimeError("Saved checkpoint trainability differs from the pilot branch.")
    if (
        checkpoint["identity"].get("seed") != 17
        or checkpoint["identity"].get("router_mode") != "trainable"
        or checkpoint["identity"].get("steps") != 1500
    ):
        raise RuntimeError("Saved native checkpoint identity differs from the registered branch.")

    branch_start = json_read(attempt_dir / "branch_start.json")
    prefix_state_hash = state_dict_hash(prefix["model"])
    if prefix_state_hash != branch_start["state_sha256"]:
        raise RuntimeError("Verified prefix no longer matches the saved branch-start state.")
    final_state = checkpoint["model"]
    model_hash_before = state_dict_hash(final_state)
    components_before = _component_hashes(final_state)
    prefix_router_state = {
        name: value
        for name, value in prefix["model"].items()
        if parameter_group(name) == "router"
    }
    if state_dict_hash(prefix_router_state) != checkpoint["identity"]["router_hash"]:
        raise RuntimeError("Saved native router identity differs from the verified prefix.")
    if checkpoint["optimizer_steps_applied"] + checkpoint["amp_overflow_steps"] != 500:
        raise RuntimeError("Checkpoint update accounting does not sum to 500 attempts.")
    optimizer_hash_before = engine._structure_hash(checkpoint["optimizer"])
    scaler_hash_before = engine._structure_hash(checkpoint["scaler"])
    saved_rng_hash_before = engine._structure_hash(
        {
            key: checkpoint[key]
            for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng")
        }
    )

    update_rows = engine._read_jsonl(attempt_dir / "study_optimizer_and_updates.jsonl")
    update_keys = [(int(row["global_step"]), row["group"]) for row in update_rows]
    _require_unique(update_rows, lambda row: (int(row["global_step"]), row["group"]), "update")
    expected_update_keys = {
        (step, group)
        for step in range(1001, 1501)
        for group in ("router", "experts", "shared")
    }
    if len(update_rows) != 1500 or set(update_keys) != expected_update_keys:
        raise RuntimeError("Saved update rows are incomplete or duplicated.")
    stats = engine._read_jsonl(attempt_dir / "gradient_stats/gradient_stats.jsonl")
    stat_steps = [int(row["step"]) for row in stats]
    if len(stats) != 500 or stat_steps != list(range(1001, 1501)):
        raise RuntimeError("Native gradient rows do not cover each attempted step exactly once.")
    applied = sum(bool(row["optimizer_step_applied"]) for row in stats)
    skipped = len(stats) - applied
    shared_updates = [row for row in update_rows if row["group"] == "shared"]
    if (
        applied != int(checkpoint["optimizer_steps_applied"])
        or skipped != int(checkpoint["amp_overflow_steps"])
        or sum(bool(row["applied_update"]) for row in shared_updates) != applied
        or sum(bool(row["amp_skip"]) for row in shared_updates) != skipped
    ):
        raise RuntimeError("Checkpoint, native gradient, and study update accounting disagree.")

    route_rows = engine._read_jsonl(attempt_dir / "study_routing_and_gate_diagnostics.jsonl")
    expert_rows = engine._read_jsonl(attempt_dir / "study_expert_function_diagnostics.jsonl")
    shared_rows = engine._read_jsonl(attempt_dir / "study_shared_representation_diagnostics.jsonl")
    trajectory_rows = engine._read_jsonl(attempt_dir / "study_trajectory_metrics.jsonl")
    registered_offsets = (25, 100, 250, 500)
    for rows, expected, key, label in (
        (
            route_rows,
            384,
            lambda row: (int(row["global_step"]), row["layer"], row["expert"], row["population"]),
            "routing diagnostic",
        ),
        (
            expert_rows,
            256,
            lambda row: (int(row["global_step"]), row["layer"], row["expert"], row["input_population"]),
            "expert diagnostic",
        ),
        (
            shared_rows,
            9,
            lambda row: (int(row["global_step"]), row["measurement"], row["component"]),
            "shared diagnostic",
        ),
    ):
        _require_unique(rows, key, label)
        counts = {}
        for row in rows:
            step = int(row["global_step"])
            if step not in (1025, 1100, 1250, 1500):
                raise RuntimeError(f"Unexpected global step in {label} rows: {step}.")
            counts[step] = counts.get(step, 0) + 1
        if counts != {1000 + offset: expected for offset in registered_offsets}:
            raise RuntimeError(f"Saved {label} grid is incomplete: {counts}.")
    if sorted(int(row["global_step"]) for row in trajectory_rows) != [1025, 1100, 1250, 1500]:
        raise RuntimeError("Saved validation trajectory is incomplete or duplicated.")

    expected_probe_rows = {
        "routing": 384,
        "expert": 256,
        "shared": 9,
    }
    cache_path = prefix_dir / "diagnostics/unpruned_start.pt"
    if not cache_path.is_file():
        raise FileNotFoundError("Verified branch-start diagnostic cache is missing.")
    cache = torch_load(cache_path)
    if (
        cache["probe_identity"] != probe_row["probe_identity_sha256"]
        or cache["probe_indices"] != probe_row["batch_indices"]
        or cache["positions"] != probe_row["fixed_token_position_indices"]
        or cache["start_state_hash"] != branch_start["state_sha256"]
    ):
        raise RuntimeError("Existing diagnostic cache differs from sealed branch-start identity.")

    variant = engine.variants()[17]
    with torch.random.fork_rng(devices=[]):
        baseline_model = TinyMoELanguageModel(variant["config"]["model"])
    baseline_model.load_state_dict(prefix["model"])
    baseline_model = baseline_model.to(device)
    diagnostic_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=float(variant["config"]["training"]["learning_rate"]),
        weight_decay=float(variant["config"]["training"]["weight_decay"]),
    )
    diagnostic_optimizer.load_state_dict(prefix["optimizer"])
    diagnostic_scaler = engine.create_grad_scaler(
        variant["config"]["training"]["precision"] == "fp16"
    )
    diagnostic_scaler.load_state_dict(prefix["scaler"])
    diagnostic_optimizer_hash = engine._structure_hash(diagnostic_optimizer.state_dict())
    diagnostic_scaler_hash = engine._structure_hash(diagnostic_scaler.state_dict())
    if (
        diagnostic_optimizer_hash != engine._structure_hash(prefix["optimizer"])
        or diagnostic_scaler_hash != engine._structure_hash(prefix["scaler"])
    ):
        raise RuntimeError("Diagnostic state clone differs from the saved step-1000 optimizer/scaler.")
    recomputation_started = time.monotonic()
    baseline_diagnostic = collect_diagnostics(
        model=baseline_model,
        optimizer=diagnostic_optimizer,
        scaler=diagnostic_scaler,
        cache=cache,
        probe_batches=[],
        seed=17,
        model_type="unpruned",
        condition="full",
        offset=0,
        device=device,
        data_position=1000,
        current_routes_override=cache["routes"],
    )
    if not baseline_diagnostic[
        "diagnostic_rng_modes_gradients_buffers_parameters_optimizer_scaler_preserved"
    ]:
        raise RuntimeError("Branch-start diagnostic recomputation did not preserve model state.")
    recomputation_seconds = time.monotonic() - recomputation_started
    if state_dict_hash(baseline_model.state_dict()) != branch_start["state_sha256"]:
        raise RuntimeError("Branch-start model changed during diagnostic recomputation.")
    baseline_ce = float(branch_start["baseline_ce"])
    if abs(baseline_ce - engine.historical_ce(17, "unpruned", 1000)) > engine.CE_ATOL:
        raise RuntimeError("Branch-start validation CE no longer matches historical dense step 1000.")

    baseline_row = _historical_dense_row(17, 1000)
    baseline_trajectory = {
        "seed": 17,
        "model_type": "unpruned",
        "condition": "full",
        "offset": 0,
        "global_step": 1000,
        "ce": baseline_ce,
        "loss_sum": baseline_ce * int(baseline_row["target_tokens"]),
        "target_tokens": int(baseline_row["target_tokens"]),
        "batches": int(baseline_row["batches"]),
        "precision": baseline_row["precision"],
        "applied_updates": 0,
        "amp_skips": 0,
        "source": "reconstructed complete branch start",
    }
    all_trajectory_rows = [baseline_trajectory] + trajectory_rows
    if sorted(int(row["global_step"]) for row in all_trajectory_rows) != list(GLOBAL_STEPS):
        raise RuntimeError("Final validation grid differs from the registered endpoints.")
    endpoint_ce = float(
        next(row["ce"] for row in all_trajectory_rows if int(row["global_step"]) == 1500)
    )
    if abs(endpoint_ce - engine.historical_ce(17, "unpruned", 1500)) > engine.CE_ATOL:
        raise RuntimeError("Full-control pilot endpoint differs from the historical dense endpoint.")

    all_route_rows = baseline_diagnostic["routing_rows"] + route_rows
    all_expert_rows = baseline_diagnostic["expert_rows"] + expert_rows
    all_shared_rows = baseline_diagnostic["shared_rows"] + shared_rows
    for rows, kind, expected_count in (
        (all_route_rows, "routing", expected_probe_rows["routing"]),
        (all_expert_rows, "expert", expected_probe_rows["expert"]),
        (all_shared_rows, "shared", expected_probe_rows["shared"]),
    ):
        _require_unique(
            rows,
            lambda row: (
                int(row["global_step"]),
                row.get("layer", row.get("measurement")),
                row.get("expert", row.get("component")),
                row.get("population", row.get("input_population")),
            ),
            f"combined {kind} diagnostic",
        )
        counts = {}
        for row in rows:
            step = int(row["global_step"])
            counts[step] = counts.get(step, 0) + 1
        if counts != {step: expected_count for step in GLOBAL_STEPS}:
            raise RuntimeError(f"Combined {kind} diagnostics differ from registered grid: {counts}.")

    engine.ensure_native_output_directories(attempt_dir)
    final_router_state = {
        name: value.detach().cpu()
        for name, value in final_state.items()
        if parameter_group(name) == "router"
    }
    final_router_hash = state_dict_hash(final_router_state)
    if final_router_hash == checkpoint["identity"]["router_hash"]:
        raise RuntimeError("Trainable router did not change during the completed branch.")
    checkpoint_payload = {
        "step": 1500,
        "model": final_state,
        "validation_loss": endpoint_ce,
        "config": variant["config"],
    }
    torch_save_atomic(attempt_dir / "checkpoints/final_recovered.pt", checkpoint_payload)
    torch_save_atomic(attempt_dir / "component_checkpoints/final_router.pt", final_router_state)

    saved_final = torch_load(attempt_dir / "checkpoints/final_recovered.pt")
    saved_router = torch_load(attempt_dir / "component_checkpoints/final_router.pt")
    model_hash_after = state_dict_hash(saved_final["model"])
    components_after = _component_hashes(saved_final["model"])
    if saved_final["step"] != 1500 or model_hash_after != model_hash_before:
        raise RuntimeError("Final model checkpoint differs from the saved interruption state.")
    if components_after != components_before or state_dict_hash(saved_router) != final_router_hash:
        raise RuntimeError("Model-component hashes changed during finalization.")
    if sha256(checkpoint_path) != checkpoint_hash_before:
        raise RuntimeError("Original step-1500 interruption checkpoint changed during finalization.")
    if (
        engine._structure_hash(checkpoint["optimizer"]) != optimizer_hash_before
        or engine._structure_hash(checkpoint["scaler"]) != scaler_hash_before
        or engine._structure_hash(
            {key: checkpoint[key] for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng")}
        )
        != saved_rng_hash_before
    ):
        raise RuntimeError("Saved optimizer, scaler, or RNG state changed during finalization.")

    csv_save(attempt_dir / "optimizer_and_updates.csv", update_rows)
    csv_save(attempt_dir / "trajectory_metrics.csv", all_trajectory_rows)
    csv_save(attempt_dir / "routing_and_gate_diagnostics.csv", all_route_rows)
    csv_save(attempt_dir / "expert_function_diagnostics.csv", all_expert_rows)
    csv_save(attempt_dir / "shared_representation_diagnostics.csv", all_shared_rows)
    json_save(
        attempt_dir / "diagnostic_recomputation.json",
        {
            "reason": "Offset-zero diagnostic rows were held in memory and not streamed before native finalization failed.",
            "checkpoint_global_step": 1500,
            "source_cache": str(cache_path),
            "source_cache_sha256": sha256(cache_path),
            "probe_identity": cache["probe_identity"],
            "probe_indices": cache["probe_indices"],
            "fixed_token_positions": cache["positions"],
            "training_batches_consumed": 0,
            "model_forwards_on_training_batches": 0,
            "optimizer_updates": 0,
            "diagnostic_optimizer_source": "read-only clone restored from the exact step-1000 optimizer state",
            "diagnostic_scaler_source": "restored from the exact step-1000 scaler state",
            "diagnostic_optimizer_state_hash_before": diagnostic_optimizer_hash,
            "diagnostic_optimizer_state_hash_after": engine._structure_hash(
                diagnostic_optimizer.state_dict()
            ),
            "diagnostic_scaler_state_hash_before": diagnostic_scaler_hash,
            "diagnostic_scaler_state_hash_after": engine._structure_hash(
                diagnostic_scaler.state_dict()
            ),
            "recomputed_rows": {
                "routing": len(baseline_diagnostic["routing_rows"]),
                "expert": len(baseline_diagnostic["expert_rows"]),
                "shared": len(baseline_diagnostic["shared_rows"]),
            },
            "diagnostic_runtime_seconds": recomputation_seconds,
            "state_preservation_verified": True,
        },
    )
    json_save(
        attempt_dir / "diagnostic_manifest.json",
        {
            "probe_indices": cache["probe_indices"],
            "probe_identity": cache["probe_identity"],
            "probe_cache": str(cache_path),
            "branch_start_state_hash": cache["start_state_hash"],
            "fixed_token_positions": cache["positions"],
            "fixed_token_position_count": len(cache["positions"]),
            "cache_generation_seconds_this_invocation": 0.0,
            "probe_batches_may_overlap_training": True,
            "probe_is_independent_test_set": False,
            "diagnostic_preservation_verified_at_all_offsets": True,
            "diagnostic_recomputation": "offset-zero rows rebuilt from sealed start cache; other offsets reused from persisted JSONL",
        },
    )

    finalization_record = {
        "status": "PASS",
        "mode": "checkpoint-only finalization",
        "training_loop_invoked": False,
        "training_batches_consumed": 0,
        "optimizer_updates_after_checkpoint": 0,
        "saved_checkpoint_step": 1500,
        "interruption_checkpoint_sha256_before": checkpoint_hash_before,
        "interruption_checkpoint_sha256_after": sha256(checkpoint_path),
        "full_model_hash_before": model_hash_before,
        "full_model_hash_after": model_hash_after,
        "component_hashes_before": components_before,
        "component_hashes_after": components_after,
        "optimizer_state_hash_before": optimizer_hash_before,
        "optimizer_state_hash_after": engine._structure_hash(checkpoint["optimizer"]),
        "scaler_state_hash_before": scaler_hash_before,
        "scaler_state_hash_after": engine._structure_hash(checkpoint["scaler"]),
        "saved_rng_hash_before": saved_rng_hash_before,
        "saved_rng_hash_after": engine._structure_hash(
            {key: checkpoint[key] for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng")}
        ),
        "attempted_update_rows": len(update_rows),
        "unique_attempted_steps": len(stats),
        "applied_updates": applied,
        "amp_skips": skipped,
        "baseline_diagnostic_recomputed": True,
        "existing_nonbaseline_diagnostics_reused": True,
        "scientific_conditions_changed": False,
        "outputs_created": [
            "checkpoints/final_recovered.pt",
            "component_checkpoints/final_router.pt",
            "optimizer_and_updates.csv",
            "trajectory_metrics.csv",
            "routing_and_gate_diagnostics.csv",
            "expert_function_diagnostics.csv",
            "shared_representation_diagnostics.csv",
        ],
    }
    json_save(attempt_dir / "finalization_validation.json", finalization_record)

    result = {
        "seed": 17,
        "model_type": "unpruned",
        "condition": "full",
        "attempt": str(attempt_dir),
        "attempted_updates": 500,
        "applied_updates": applied,
        "amp_skips": skipped,
        "native_result": {
            "finalized_from_saved_checkpoint": True,
            "native_training_loop_invoked": False,
            "training_batches_consumed": 0,
            "optimizer_updates_after_checkpoint": 0,
        },
        "final_checkpoint": str(attempt_dir / "checkpoints/final_recovered.pt"),
        "final_checkpoint_sha256": sha256(attempt_dir / "checkpoints/final_recovered.pt"),
        "wall_seconds": time.monotonic() - started,
        "branch_start_validation": {
            "ce": baseline_ce,
            "target_tokens": branch_start["baseline_target_tokens"],
            "historical_ce": branch_start["historical_baseline_ce"],
        },
        "frozen_parameters_bitwise_unchanged": True,
        "frozen_optimizer_state_unchanged": True,
        "mask_persisted": None,
        "mask_applicability": "not applicable to unpruned branch",
        "optimizer_reset": False,
        "global_step": 1500,
        "finalization_validation": str(attempt_dir / "finalization_validation.json"),
    }
    json_save(attempt_dir / "result.json", result)
    json_save(
        status_path,
        {
            "status": "COMPLETE",
            "attempt": str(attempt_dir),
            "result": str(attempt_dir / "result.json"),
            "finalization_only": True,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Finalization requires the registered CUDA device for diagnostics.")
    torch.cuda.set_device(0)
    result = finalize_saved_attempt(args.attempt, torch.device("cuda"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
