"""Architecture replication of trainable-router sparse recovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch

from moe_lth.config import load_config, save_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.router_age import assemble_router_age_model, component_state_dict, routing_statistics, selected_experts_per_batch, state_dict_hash
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything

EXPERT_COUNTS = (8, 12)
SEEDS = (7, 17, 29)
SPARSITIES = (0.75, 0.85, 0.95)
AGES = (0, 20, 100)
RECOVERY_STEPS = 2500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_state(state, group):
    return {name: tensor for name, tensor in state.items() if recovery.parameter_group(name) == group}


def assert_rewound(initial, ticket, masks):
    for name, mask in masks.items():
        keep = mask.detach().cpu().bool()
        if not torch.equal(ticket[name].detach().cpu()[keep], initial[name].detach().cpu()[keep]):
            raise RuntimeError(f"Surviving weights for {name} do not come from E0.")
        if ticket[name].detach().cpu()[~keep].abs().any():
            raise RuntimeError(f"Pruned weights for {name} are nonzero.")


def config_for(root, experts, seed):
    config = load_config(Path("configs/expert_count_replication") / f"e{experts}_seed7.yaml")
    config["seed"] = seed
    config["model"]["num_experts"] = experts
    config["output_dir"] = str(root / f"e{experts}" / "reference" / f"seed_{seed}")
    config_path = root / "configs" / f"e{experts}_seed{seed}.yaml"
    save_config(config, config_path)
    return config_path, config


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def run(output_dir: str, *, smoke: bool = False) -> None:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite replication output: {root}")
    root.mkdir(parents=True)
    selected_experts = EXPERT_COUNTS
    selected_seeds = (7,) if smoke else SEEDS
    selected_sparsities = (0.85,) if smoke else SPARSITIES
    selected_ages = (20,) if smoke else AGES
    recovery_steps = 2 if smoke else RECOVERY_STEPS
    config_rows, manifest, dense_rows, sparse_rows, sanity = [], [], [], [], []
    for experts in selected_experts:
        for seed in selected_seeds:
            config_path, config = config_for(root, experts, seed)
            if smoke:
                config = deepcopy(config)
                config["training"]["steps"] = 10
                config["training"]["checkpoint_steps"] = [0, 2, 10]
                config["output_dir"] = str(root / f"e{experts}" / "smoke_reference" / f"seed_{seed}")
                save_config(config, config_path)
            reference_dir = recovery._ensure_reference(str(config_path))
            device = resolve_device(config["device"]); configure_device(device); seed_everything(seed)
            total = int(config["training"]["steps"])
            expected_steps = {age: round(total * age / 100) for age in selected_ages}
            initial_path, _ = recovery._checkpoint_for_percent(reference_dir, total, 0)
            final_path, _ = recovery._checkpoint_for_percent(reference_dir, total, 100)
            dense_model = recovery.load_model_from_checkpoint(config["model"], str(final_path), device)
            initial_model = recovery.load_model_from_checkpoint(config["model"], str(initial_path), device)
            initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
            loader, validation_loader = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config))
            train_batches = recovery._materialize_batches(loader, recovery_steps)
            validation_batches = recovery._materialize_validation_batches(validation_loader, int(config["data"]["validation_blocks"]))
            train_hash = recovery._batch_sequence_hash(train_batches); validation_hash = recovery._batch_sequence_hash(validation_batches)
            calibration = recovery._calibration_batches(validation_batches)
            dense_reference_loss = recovery.evaluate_language_model(dense_model, validation_batches, device, max_batches=len(validation_batches))["loss"]
            config_rows.append({"expert_count": experts, "seed": seed, "config_path": str(config_path.relative_to(root)), "architecture_config_hash": sha256_file(config_path), "num_layers": config["model"]["num_layers"], "d_model": config["model"]["d_model"], "expert_hidden_size": config["model"]["expert_hidden_size"], "top_k": config["model"]["top_k"], "capacity_factor": config["model"]["capacity_factor"], "aux_loss_weight": config["routing"]["aux_loss_weight"], "reference_steps": total, "recovery_steps": recovery_steps, "precision": config["training"]["precision"]})
            router_models = {}
            for age in selected_ages:
                checkpoint, loaded_step = recovery._checkpoint_for_percent(reference_dir, total, age)
                if loaded_step != expected_steps[age]:
                    raise RuntimeError(f"R{age} requested step {expected_steps[age]}, loaded {loaded_step}.")
                router = component_state_dict(recovery.load_model_from_checkpoint(config["model"], str(checkpoint), torch.device("cpu")), "router")
                manifest.append({"expert_count": experts, "seed": seed, "router_age": age, "requested_router_checkpoint_step": expected_steps[age], "loaded_router_checkpoint_step": loaded_step, "checkpoint_path": str(checkpoint), "checkpoint_file_sha256": sha256_file(checkpoint), "initial_router_hash": state_dict_hash(router)})
                router_models[age] = (checkpoint, loaded_step)
                probe = assemble_router_age_model(config["model"], initial_state, str(checkpoint), {}, device)
                stats = routing_statistics(probe, calibration, device)
                utilization = stats["top1_assignment_distribution"]
                sanity.append({"expert_count": experts, "seed": seed, "router_age": age, "routing_entropy": stats["routing_entropy"], "mean_top1_confidence": stats["mean_selected_probability"], "top1_top2_margin": stats["top1_top2_margin"], "utilization": json.dumps(utilization), "utilization_cv": float(torch.tensor(utilization).float().std(unbiased=False) / torch.tensor(utilization).float().mean()), "inactive_experts": sum(item == 0 for item in utilization), "capacity_overflow": sum(stats["dropped_fraction_by_layer"]) / len(stats["dropped_fraction_by_layer"]), "token_drop_rate": sum(stats["dropped_fraction_by_layer"]) / len(stats["dropped_fraction_by_layer"])})
            dense_by_age = {}
            for age, (checkpoint, router_step) in router_models.items():
                output = root / f"e{experts}" / "dense" / f"seed_{seed}" / f"R{age}"
                record = recovery._run_recovery_condition(config=config, condition_name=f"e{experts}_seed{seed}_R{age}_dense", pruned_base_state=initial_state, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=router_step, masks={}, expert_hash=state_dict_hash(group_state(initial_state, "expert")), shared_hash=state_dict_hash(group_state(initial_state, "shared")), mask_hash="dense_no_mask", reference_selected=selected_experts_per_batch(dense_model, calibration, device), calibration_batches=calibration, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=recovery_steps, dense_loss=dense_reference_loss, output_dir=output, confidence_control=False, target_confidence=None, seed=seed, sparsity=0.0, router_mode="trainable")
                record["expert_count"] = experts; record["condition_dir"] = str(output); dense_rows.append(record); dense_by_age[age] = record
            for sparsity in selected_sparsities:
                masks = expert_local_magnitude_masks(dense_model, sparsity)
                ticket = recovery.build_fixed_pruned_base(config["model"], str(initial_path), masks, device)
                assert_rewound(initial_state, ticket, masks)
                mask_hash = recovery._mask_hash(masks)
                save_masks(masks, root / f"e{experts}" / "masks" / f"s{sparsity:.2f}" / f"seed_{seed}" / "pruning_mask.pt")
                for age, (checkpoint, router_step) in router_models.items():
                    output = root / f"e{experts}" / "sparse" / f"s{sparsity:.2f}" / f"seed_{seed}" / f"R{age}"
                    record = recovery._run_recovery_condition(config=config, condition_name=f"e{experts}_seed{seed}_s{sparsity:.2f}_R{age}_sparse", pruned_base_state=ticket, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=router_step, masks=masks, expert_hash=state_dict_hash(group_state(ticket, "expert")), shared_hash=state_dict_hash(group_state(ticket, "shared")), mask_hash=mask_hash, reference_selected=selected_experts_per_batch(dense_model, calibration, device), calibration_batches=calibration, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=recovery_steps, dense_loss=dense_by_age[age]["final_validation_loss"], output_dir=output, confidence_control=False, target_confidence=None, seed=seed, sparsity=sparsity, router_mode="trainable")
                    record["expert_count"] = experts; record["matched_dense_final_loss"] = dense_by_age[age]["final_validation_loss"]; record["ticket_gap"] = record["final_validation_loss"] - dense_by_age[age]["final_validation_loss"]; record["condition_dir"] = str(output); sparse_rows.append(record)
    write_csv(root / "architecture_configs.csv", config_rows); write_csv(root / "reference_checkpoint_manifest.csv", manifest); write_csv(root / "dense_baselines.csv", dense_rows); write_csv(root / "sparse_results.csv", sparse_rows); write_csv(root / "routing_sanity_metrics.csv", sanity)
    (root / "preflight_architecture_audit.md").write_text("# Architecture Audit\n\nThe current experiment architecture is 8 experts, top-k=1, 4 MoE layers, d_model=256, expert hidden size=1024, capacity factor=1.25, learned linear routers, and auxiliary routing loss weight=0.1. Dense and recovery budgets are 2500 steps, fp16, AdamW, fresh optimizer, and deterministic materialized train/validation schedules. The replication changes only `num_experts` (8 or 12); all other model/data/training settings are inherited unchanged. Expert-local magnitude sparsity remains the fraction pruned within each expert, never whole-expert removal. Router ages are derived as fractions of each architecture's configured dense-reference budget.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/expert_count_replication")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, smoke=args.smoke)  