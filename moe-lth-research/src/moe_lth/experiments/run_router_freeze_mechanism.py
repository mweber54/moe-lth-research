"""Frozen-router mechanism intervention for the standard MoE ticket protocol.

This runner never changes routing behavior: it disables only router parameter
updates, while reusing the shared recovery loop and its E0-under-ET-mask ticket.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import torch

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.router_age import (
    assemble_router_age_model,
    build_fixed_pruned_base,
    component_state_dict,
    load_model_from_checkpoint,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything

PROTOCOL_VERSION = "router_freeze_mechanism_v1"
SPARSITIES = (0.75, 0.85, 0.95)
ROUTER_AGES = (0, 20, 100)
ROUTER_STEPS = {0: 0, 20: 500, 100: 2500}
RECOVERY_STEPS = 2500


def _expert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "expert"}


def _shared_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "shared"}


def _assert_rewound(initial: dict[str, torch.Tensor], ticket: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> None:
    for name, mask in masks.items():
        keep = mask.detach().cpu().bool()
        actual, expected = ticket[name].detach().cpu(), initial[name].detach().cpu()
        if not torch.equal(actual[keep], expected[keep]) or not torch.equal(actual[~keep], torch.zeros_like(actual[~keep])):
            raise RuntimeError(f"Rewind assertion failed for {name}.")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_aggregate(root: Path, records: list[dict]) -> None:
    sparse = [row for row in records if row["condition_type"] == "sparse_ticket"]
    dense = {(row["reference_seed"], row["router_age"]): row for row in records if row["condition_type"] == "dense_control"}
    seed_rows = []
    for row in sparse:
        baseline = dense[(row["reference_seed"], row["router_age"])]
        seed_rows.append({
            "reference_seed": row["reference_seed"], "sparsity": row["sparsity"], "router_age": row["router_age"],
            "router_mode": "frozen", "sparse_final_loss": row["final_validation_loss"],
            "dense_final_loss": baseline["final_validation_loss"],
            "ticket_gap": row["final_validation_loss"] - baseline["final_validation_loss"],
        })
    aggregate_dir = root / "aggregate"; aggregate_dir.mkdir(exist_ok=True)
    if seed_rows:
        with (aggregate_dir / "seed_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0])); writer.writeheader(); writer.writerows(seed_rows)
        groups = {}
        for row in seed_rows: groups.setdefault((row["sparsity"], row["router_age"], row["router_mode"]), []).append(row)
        rows = []
        for (sparsity, age, mode), group in sorted(groups.items()):
            def values(name): return [float(row[name]) for row in group]
            gaps = values("ticket_gap")
            rows.append({"sparsity": sparsity, "router_age": age, "router_mode": mode, "mean_sparse_final_loss": mean(values("sparse_final_loss")), "std_sparse_final_loss": stdev(values("sparse_final_loss")) if len(group)>1 else 0.0, "mean_dense_final_loss": mean(values("dense_final_loss")), "std_dense_final_loss": stdev(values("dense_final_loss")) if len(group)>1 else 0.0, "mean_ticket_gap": mean(gaps), "std_ticket_gap": stdev(gaps) if len(group)>1 else 0.0, "standard_error": stdev(gaps)/math.sqrt(len(group)) if len(group)>1 else 0.0, "num_seeds": len(group)})
        with (aggregate_dir / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_router_freeze_mechanism(
    config_paths: list[str],
    output_dir: str,
    *,
    recovery_steps: int = RECOVERY_STEPS,
    smoke: bool = False,
    smoke_device: str | None = None,
) -> dict:
    """Run frozen dense/sparse pairs. Smoke mode executes only seed 7, s=.85, R20."""
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing mechanism output: {root}")
    if smoke_device is not None and not smoke:
        raise ValueError("--smoke-device is permitted only with --smoke.")
    root.mkdir(parents=True, exist_ok=True)
    selected_sparsities = (0.85,) if smoke else SPARSITIES
    selected_ages = (20,) if smoke else ROUTER_AGES
    selected_configs = [path for path in config_paths if int(load_config(path)["seed"]) == 7] if smoke else config_paths
    if not selected_configs:
        raise RuntimeError("Smoke mode requires the seed 7 reference config.")
    records: list[dict] = []
    audit = {"all_pass": True, "protocol_version": PROTOCOL_VERSION, "router_mode": "frozen", "smoke": smoke, "seeds": {}}

    for config_path in selected_configs:
        config = load_config(config_path); seed = int(config["seed"]); run_dir = recovery._ensure_reference(config_path)
        total = int(config["training"]["steps"])
        if not smoke and recovery_steps != RECOVERY_STEPS: raise ValueError("Production recovery_steps must be 2500.")
        device = resolve_device(smoke_device or config["device"]); configure_device(device); seed_everything(seed)
        initial_path, _ = recovery._checkpoint_for_percent(run_dir, total, 0); final_path, _ = recovery._checkpoint_for_percent(run_dir, total, 100)
        trained = load_model_from_checkpoint(config["model"], str(final_path), device); initial_model = load_model_from_checkpoint(config["model"], str(initial_path), device)
        initial = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}
        dense_base = initial; shared_hash = state_dict_hash(_shared_state(initial)); dense_expert_hash = state_dict_hash(_expert_state(initial))
        loader, validation_loader = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config))
        train_batches = recovery._materialize_batches(loader, recovery_steps); validation_batches = recovery._materialize_validation_batches(validation_loader, int(config["data"]["validation_blocks"]))
        train_hash, validation_hash = recovery._batch_sequence_hash(train_batches), recovery._batch_sequence_hash(validation_batches)
        calibration = recovery._calibration_batches(validation_batches)
        dense_reference_loss = recovery.evaluate_language_model(trained, validation_batches, device, max_batches=len(validation_batches))["loss"]
        audit["seeds"][str(seed)] = {"shared_state_hash": shared_hash, "training_sequence_hash": train_hash, "validation_sequence_hash": validation_hash, "routers": {}}
        dense_by_age = {}
        for age in selected_ages:
            checkpoint, step = recovery._checkpoint_for_percent(run_dir, total, age)
            if step != ROUTER_STEPS[age]: raise RuntimeError(f"Router checkpoint mismatch: R{age} loaded step {step}.")
            router_hash = state_dict_hash(component_state_dict(load_model_from_checkpoint(config["model"], str(checkpoint), torch.device("cpu")), "router"))
            audit["seeds"][str(seed)]["routers"][f"R{age}"] = {"requested_step": ROUTER_STEPS[age], "loaded_step": step, "router_hash": router_hash}
            reference = assemble_router_age_model(config["model"], dense_base, str(final_path), {}, device)
            dense = recovery._run_recovery_condition(config=config, condition_name=f"freeze_seed{seed}_R{age}_dense", pruned_base_state=dense_base, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=step, masks={}, expert_hash=dense_expert_hash, shared_hash=shared_hash, mask_hash="dense_no_mask", reference_selected=selected_experts_per_batch(reference, calibration, device), calibration_batches=calibration, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=recovery_steps, dense_loss=dense_reference_loss, output_dir=root / "dense" / f"seed_{seed}" / f"R{age}", confidence_control=False, target_confidence=None, seed=seed, sparsity=0.0, router_mode="frozen")
            dense.update({"reference_seed": seed, "router_age": age, "router_mode": "frozen", "condition_type": "dense_control"}); records.append(dense); dense_by_age[age] = dense
        for sparsity in selected_sparsities:
            masks = expert_local_magnitude_masks(trained, sparsity); ticket = build_fixed_pruned_base(config["model"], str(initial_path), masks, device); _assert_rewound(initial, ticket, masks)
            mask_hash, expert_hash = recovery._mask_hash(masks), state_dict_hash(_expert_state(ticket)); mask_dir = root / "masks" / f"s{sparsity:.2f}" / f"seed_{seed}"; mask_dir.mkdir(parents=True, exist_ok=True); save_masks(masks, mask_dir / "pruning_mask.pt")
            for age in selected_ages:
                checkpoint, step = recovery._checkpoint_for_percent(run_dir, total, age); dense = dense_by_age[age]
                reference = assemble_router_age_model(config["model"], ticket, str(final_path), masks, device)
                sparse = recovery._run_recovery_condition(config=config, condition_name=f"freeze_seed{seed}_s{sparsity:.2f}_R{age}_sparse", pruned_base_state=ticket, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=step, masks=masks, expert_hash=expert_hash, shared_hash=shared_hash, mask_hash=mask_hash, reference_selected=selected_experts_per_batch(reference, calibration, device), calibration_batches=calibration, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=recovery_steps, dense_loss=dense["final_validation_loss"], output_dir=root / "sparse" / f"s{sparsity:.2f}" / f"seed_{seed}" / f"R{age}", confidence_control=False, target_confidence=None, seed=seed, sparsity=sparsity, router_mode="frozen")
                if not sparse["router_hash_unchanged"] or sparse["router_parameter_drift_final"] != 0.0: raise RuntimeError("Frozen router hash/drift assertion failed.")
                sparse.update({"reference_seed": seed, "router_age": age, "router_mode": "frozen", "condition_type": "sparse_ticket", "matched_dense_final_validation_loss": dense["final_validation_loss"], "ticket_gap": sparse["final_validation_loss"] - dense["final_validation_loss"]}); records.append(sparse)
    audit["smoke_device_override"] = smoke_device
    _write_json(root / "preflight" / "preflight_audit.json", audit); _write_json(root / "mechanism_records.json", records); _write_aggregate(root, records)
    return {"output_dir": str(root), "audit": audit, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recovery-steps", type=int, default=RECOVERY_STEPS); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-device", choices=("cpu", "cuda"))
    args = parser.parse_args(); print(json.dumps(run_router_freeze_mechanism(args.configs, args.output_dir, recovery_steps=args.recovery_steps, smoke=args.smoke, smoke_device=args.smoke_device)["audit"], indent=2))


if __name__ == "__main__":
    main()
