"""Map the 12-expert router-development state against sparse ticket gaps."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

import torch

from moe_lth.config import load_config, save_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.router_age import assemble_router_age_model, component_state_dict, routing_statistics, selected_experts_per_batch, state_dict_hash
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = RESULTS / "router_development_12expert"
OLD = RESULTS / "expert_count_replication"
SEEDS = (7, 17, 29)
AGES = (0, 10, 20, 40, 60, 80, 100)
STEPS = {age: round(2500 * age / 100) for age in AGES}
SPARSITIES = (0.60, 0.70, 0.75, 0.85, 0.90, 0.95)
MISSING_AGES = {10, 40, 60, 80}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group(state, name):
    return {key: value for key, value in state.items() if recovery.parameter_group(key) == name}


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def make_config(seed, reference_root):
    config = load_config(ROOT / "configs" / "expert_count_replication" / "e12_seed7.yaml")
    config["seed"] = seed; config["model"]["num_experts"] = 12
    config["training"]["checkpoint_steps"] = [0, 250, 500, 1000, 1500, 2000, 2500]
    config["output_dir"] = str(reference_root / f"reference/seed_{seed}")
    path = reference_root / "configs" / f"e12_seed{seed}.yaml"; save_config(config, path)
    return path, config


def fixed_probe_metrics(model, batches, device):
    model.eval(); layers = len(model.blocks); experts = model.blocks[0].moe.num_experts
    token_counts = [[Counter() for _ in range(experts)] for _ in range(layers)]
    assignments = [[] for _ in range(layers)]
    with torch.no_grad():
        for token_ids in batches:
            output = model(token_ids.to(device))
            for layer, trace in enumerate(output.routes):
                ids = token_ids.detach().cpu().reshape(-1).tolist(); routes = trace.selected_experts.detach().cpu().reshape(-1).tolist()
                assignments[layer].extend(routes)
                for token, expert in zip(ids, routes): token_counts[layer][expert][token] += 1
    stats = routing_statistics(model, batches, device)
    rows = []
    for layer in range(layers):
        total = len(assignments[layer]); counts = Counter(assignments[layer]); util = [counts[e] / max(1, total) for e in range(experts)]
        global_tokens = Counter(token for expert in token_counts[layer] for token in expert.elements())
        mi = 0.0
        for expert in range(experts):
            for token, count in token_counts[layer][expert].items():
                p_joint = count / total; p_e = util[expert]; p_x = global_tokens[token] / total
                mi += p_joint * math.log(p_joint / max(1e-12, p_e * p_x))
        rows.append({"layer": layer, "routing_entropy": sum(trace["routing_entropy"] for trace in [stats]) / 1, "mean_top1_confidence": stats["mean_selected_probability"], "top1_top2_margin": stats["top1_top2_margin"], "utilization_entropy": -sum(item * math.log(item) for item in util if item > 0), "utilization_cv": float(torch.tensor(util).std(unbiased=False) / torch.tensor(util).mean()), "max_utilization": max(util), "min_utilization": min(util), "inactive_experts": sum(item == 0 for item in util), "specialization_mi_token_id": mi, "normalized_specialization_mi": mi / max(1e-12, math.log(experts)), "utilization": json.dumps(util)})
    model.train()
    return rows


def main(output_dir=str(OUT)):
    global OUT
    OUT = Path(output_dir)
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    reference_root = OUT / "references"; reference_root.mkdir()
    configs, manifests, dense_rows, sparse_rows, sanity_rows, development_rows = [], [], [], [], [], []
    old_dense = { (int(row["expert_count"]), int(row["seed"]), int(row["router_age_percent"])): row for row in csv.DictReader(open(OLD / "dense_baselines.csv")) if int(row["expert_count"]) == 12 }
    old_sparse = { (int(row["expert_count"]), int(row["seed"]), float(row["sparsity"]), int(row["router_age_percent"])): row for row in csv.DictReader(open(OLD / "sparse_results.csv")) if int(row["expert_count"]) == 12 }
    for seed in SEEDS:
        config_path, config = make_config(seed, reference_root)
        reference_dir = recovery._ensure_reference(str(config_path)); device = resolve_device(config["device"]); configure_device(device); seed_everything(seed)
        total = int(config["training"]["steps"]); initial_path, _ = recovery._checkpoint_for_percent(reference_dir, total, 0); final_path, _ = recovery._checkpoint_for_percent(reference_dir, total, 100)
        dense_model = recovery.load_model_from_checkpoint(config["model"], str(final_path), device); initial_model = recovery.load_model_from_checkpoint(config["model"], str(initial_path), device)
        initial_state = {name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()}
        loader, validation_loader = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config))
        train_batches = recovery._materialize_batches(loader, 2500); validation_batches = recovery._materialize_validation_batches(validation_loader, int(config["data"]["validation_blocks"])); probe = recovery._calibration_batches(validation_batches)
        train_hash = recovery._batch_sequence_hash(train_batches); validation_hash = recovery._batch_sequence_hash(validation_batches); dense_loss = recovery.evaluate_language_model(dense_model, validation_batches, device, max_batches=len(validation_batches))["loss"]
        router_info = {}
        for age in AGES:
            checkpoint, loaded = recovery._checkpoint_for_percent(reference_dir, total, age)
            if loaded != STEPS[age]: raise RuntimeError(f"R{age}: expected {STEPS[age]}, loaded {loaded}")
            router = component_state_dict(recovery.load_model_from_checkpoint(config["model"], str(checkpoint), torch.device("cpu")), "router"); router_hash = state_dict_hash(router)
            manifest = {"expert_count": 12, "seed": seed, "router_age": age, "requested_step": STEPS[age], "loaded_step": loaded, "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "router_hash": router_hash, "training_sequence_hash": train_hash, "validation_sequence_hash": validation_hash}
            manifests.append(manifest); router_info[age] = (checkpoint, loaded)
            for metric in fixed_probe_metrics(assemble_router_age_model(config["model"], initial_state, str(checkpoint), {}, device), probe, device): development_rows.append({"seed": seed, "router_age": age, **metric})
        masks_by_sparsity = {}
        for sparsity in SPARSITIES:
            masks = expert_local_magnitude_masks(dense_model, sparsity); ticket = recovery.build_fixed_pruned_base(config["model"], str(initial_path), masks, device); mask_hash = recovery._mask_hash(masks); masks_by_sparsity[sparsity] = (masks, ticket, mask_hash)
            save_masks(masks, OUT / "masks" / f"s{sparsity:.2f}" / f"seed_{seed}" / "pruning_mask.pt")
        for age in AGES:
            checkpoint, router_step = router_info[age]
            dense = old_dense.get((12, seed, age)) if age not in MISSING_AGES else None
            if dense is None:
                output = OUT / "dense" / f"seed_{seed}" / f"R{age}"
                record = recovery._run_recovery_condition(config=config, condition_name=f"dev12_seed{seed}_R{age}_dense", pruned_base_state=initial_state, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=router_step, masks={}, expert_hash=state_dict_hash(group(initial_state, "expert")), shared_hash=state_dict_hash(group(initial_state, "shared")), mask_hash="dense_no_mask", reference_selected=selected_experts_per_batch(dense_model, probe, device), calibration_batches=probe, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=2500, dense_loss=dense_loss, output_dir=output, confidence_control=False, target_confidence=None, seed=seed, sparsity=0.0, router_mode="trainable"); record["condition_dir"] = str(output); dense_rows.append(record); dense = record
            else:
                dense = {**dense, "final_validation_loss": float(dense["final_validation_loss"]), "initial_validation_loss": float(dense["initial_validation_loss"]), "reused_from": "expert_count_replication/e12"}
                dense_rows.append(dense)
            for sparsity in SPARSITIES:
                old = old_sparse.get((12, seed, sparsity, age)) if age not in MISSING_AGES else None
                if old is not None:
                    old = {**old, "final_validation_loss": float(old["final_validation_loss"]), "matched_dense_final_loss": float(old["matched_dense_final_loss"]), "ticket_gap": float(old["ticket_gap"]), "reused_from": "expert_count_replication/e12"}
                    sparse_rows.append(old); continue
                masks, ticket, mask_hash = masks_by_sparsity[sparsity]; output = OUT / "sparse" / f"s{sparsity:.2f}" / f"seed_{seed}" / f"R{age}"
                record = recovery._run_recovery_condition(config=config, condition_name=f"dev12_seed{seed}_s{sparsity:.2f}_R{age}_sparse", pruned_base_state=ticket, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=router_step, masks=masks, expert_hash=state_dict_hash(group(ticket, "expert")), shared_hash=state_dict_hash(group(ticket, "shared")), mask_hash=mask_hash, reference_selected=selected_experts_per_batch(dense_model, probe, device), calibration_batches=probe, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=2500, dense_loss=dense["final_validation_loss"], output_dir=output, confidence_control=False, target_confidence=None, seed=seed, sparsity=sparsity, router_mode="trainable"); record.update({"expert_count": 12, "matched_dense_final_loss": dense["final_validation_loss"], "ticket_gap": record["final_validation_loss"] - dense["final_validation_loss"], "condition_dir": str(output)}); sparse_rows.append(record)
    write_csv(OUT / "architecture_configs.csv", configs); write_csv(OUT / "checkpoint_manifest.csv", manifests); write_csv(OUT / "dense_baselines.csv", dense_rows); write_csv(OUT / "matched_runs.csv", sparse_rows + dense_rows); write_csv(OUT / "ticket_gap_seed_level.csv", sparse_rows); write_csv(OUT / "routing_specialization.csv", development_rows); write_csv(OUT / "routing_certainty.csv", development_rows); write_csv(OUT / "routing_load_structure.csv", development_rows); write_csv(OUT / "router_development_vectors.csv", development_rows)
    (OUT / "preflight_audit.md").write_text("# 12-Expert Router Development Preflight\n\nArchitecture: 12 experts, top-k=1, four MoE layers, d_model=256, expert hidden size=1024, capacity factor 1.25, learned linear router, auxiliary loss 0.1. Only router age and expert count study scope differ from the established protocol; all ages are fractions of each 2500-step dense reference. Sparse tickets use masks discovered from E_T and surviving values rewound from E_0. Existing compatible R0/R20/R100 e12 cells are reused; missing ages are new runs.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default=str(OUT)); args = parser.parse_args(); main(args.output_dir)