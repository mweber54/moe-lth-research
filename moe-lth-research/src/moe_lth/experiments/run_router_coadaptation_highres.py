"""Run the minimal high-resolution, trainable-router R0/R20/R100 diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.router_age import assemble_router_age_model, component_state_dict, parameter_group, selected_experts_per_batch, state_dict_hash
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything

SPARSITY = 0.85
RECOVERY_STEPS = 2500
AGES = (0, 20, 100)
DIAGNOSTIC_STEPS = (0, 100, 250, 400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1500, 2000, 2500)


def _component(state, group):
    return {name: tensor for name, tensor in state.items() if parameter_group(name) == group}


def _probe_metadata(batches):
    digest = hashlib.sha256()
    records = []
    for sample_id, batch in enumerate(batches):
        values = batch.detach().cpu().tolist()
        for position in range(batch.shape[1]):
            for row in range(batch.shape[0]):
                token_id = values[row][position]
                stable_id = f"{sample_id}:{row}:{position}"
                digest.update(f"{stable_id}:{token_id}".encode())
                records.append({"stable_token_id": stable_id, "sample_id": sample_id, "batch_row": row, "position": position, "token_id": token_id})
    return records, digest.hexdigest()


def _make_probe_callback(probe_batches, run_label, sink):
    def callback(model, step):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for sample_id, token_ids in enumerate(probe_batches):
                output = model(token_ids.to(next(model.parameters()).device))
                for layer, trace in enumerate(output.routes):
                    probabilities = trace.probabilities.detach().cpu()
                    top2 = probabilities.topk(k=min(2, probabilities.shape[-1]), dim=-1)
                    accepted = trace.accepted_mask[:, 0].reshape_as(trace.selected_experts).detach().cpu()
                    top1 = trace.selected_experts.detach().cpu()
                    for row in range(token_ids.shape[0]):
                        for position in range(token_ids.shape[1]):
                            expert = int(top1[row, position])
                            sink.append({"run": run_label, "step": step, "stable_token_id": f"{sample_id}:{row}:{position}", "sample_id": sample_id, "batch_row": row, "position": position, "token_id": int(token_ids[row, position]), "layer": layer, "top1_expert": expert, "top2_expert": int(top2.indices[row, position, 1]), "top1_probability": float(top2.values[row, position, 0]), "top2_probability": float(top2.values[row, position, 1]), "margin": float(trace.margin[row, position].detach().cpu()), "entropy": float(trace.entropy[row, position].detach().cpu()), "probabilities": probabilities[row, position].tolist(), "accepted_by_capacity": bool(accepted[row, position]), "dropped_or_overflowed": not bool(accepted[row, position]), "executed_expert": expert if bool(accepted[row, position]) else -1})
        model.train(was_training)
    return callback


def run(config_path: str, output_dir: str) -> None:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing high-resolution output: {root}")
    config = load_config(config_path)
    if int(config["seed"]) != 7:
        raise ValueError("The minimal diagnostic is locked to reference seed 7.")
    root.mkdir(parents=True)
    reference_dir = recovery._ensure_reference(config_path)
    device = resolve_device(config["device"]); configure_device(device); seed_everything(7)
    total_steps = int(config["training"]["steps"])
    initial_path, _ = recovery._checkpoint_for_percent(reference_dir, total_steps, 0)
    final_path, _ = recovery._checkpoint_for_percent(reference_dir, total_steps, 100)
    trained = recovery.load_model_from_checkpoint(config["model"], str(final_path), device)
    loader, validation_loader = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config))
    train_batches = recovery._materialize_batches(loader, RECOVERY_STEPS)
    validation_batches = recovery._materialize_validation_batches(validation_loader, int(config["data"]["validation_blocks"]))
    probe_batches = recovery._calibration_batches(validation_batches)
    probe_records, probe_hash = _probe_metadata(probe_batches)
    (root / "probe_metadata.json").write_text(json.dumps({"probe_hash": probe_hash, "logical_token_count": len(probe_records), "batches": len(probe_batches), "stable_identifier": "sample_id:batch_row:position", "tokens": probe_records}, indent=2) + "\n")
    masks = expert_local_magnitude_masks(trained, SPARSITY)
    ticket = recovery.build_fixed_pruned_base(config["model"], str(initial_path), masks, device)
    mask_hash = recovery._mask_hash(masks); save_masks(masks, root / "pruning_mask.pt")
    expert_hash = state_dict_hash(_component(ticket, "expert")); shared_hash = state_dict_hash(_component(ticket, "shared"))
    train_hash = recovery._batch_sequence_hash(train_batches); validation_hash = recovery._batch_sequence_hash(validation_batches)
    dense_loss = recovery.evaluate_language_model(trained, validation_batches, device, max_batches=len(validation_batches))["loss"]
    token_rows = []
    records = []
    for age in AGES:
        checkpoint, router_step = recovery._checkpoint_for_percent(reference_dir, total_steps, age)
        label = f"R{age}"
        reference = assemble_router_age_model(config["model"], ticket, str(final_path), masks, device)
        condition_dir = root / "runs" / label
        record = recovery._run_recovery_condition(config=config, condition_name=f"highres_seed7_{label}", pruned_base_state=ticket, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=router_step, masks=masks, expert_hash=expert_hash, shared_hash=shared_hash, mask_hash=mask_hash, reference_selected=selected_experts_per_batch(reference, probe_batches, device), calibration_batches=probe_batches, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=RECOVERY_STEPS, dense_loss=dense_loss, output_dir=condition_dir, confidence_control=False, target_confidence=None, seed=7, sparsity=SPARSITY, router_mode="trainable", diagnostic_steps=DIAGNOSTIC_STEPS, highres_probe_callback=_make_probe_callback(probe_batches, label, token_rows), save_router_checkpoints=True)
        records.append(record)
    pq.write_table(pa.Table.from_pylist(token_rows), root / "token_routing_trajectory.parquet")
    manifest = []
    for label in ("R0", "R20", "R100"):
        for checkpoint in sorted((root / "runs" / label / "router_checkpoints").glob("*.pt")):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            manifest.append({"run": label, "step": int(checkpoint.stem.split("_")[-1]), "path": str(checkpoint.relative_to(root)), "router_hash": state_dict_hash(state)})
    with (root / "router_checkpoint_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    (root / "preflight_audit.md").write_text(f"# Preflight Audit\n\n- Protocol: existing seed-7 85% trainable recovery.\n- Conditions: R0, R20, R100; 2500 steps; no dense/control reruns.\n- Train hash: `{train_hash}`\n- Validation hash: `{validation_hash}`\n- Probe hash: `{probe_hash}`\n- Mask hash: `{mask_hash}`\n- Expert/shared hashes: `{expert_hash}` / `{shared_hash}`\n- Diagnostic steps: {list(DIAGNOSTIC_STEPS)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/router_age_reference_seed7.yaml")
    parser.add_argument("--output-dir", default="results/router_coadaptation_highres")
    arguments = parser.parse_args()
    run(arguments.config, arguments.output_dir)