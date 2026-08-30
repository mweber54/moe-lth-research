"""Endpoint sparsity sweep for the corrected router-conditioned LTH protocol."""

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
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import save_masks
from moe_lth.pruning.router_age import (
    build_fixed_pruned_base,
    component_state_dict,
    load_model_from_checkpoint,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything
from moe_lth.experiments import run_router_age_recovery as recovery

SPARSITIES_TO_SWEEP = (0.60, 0.70, 0.90, 0.95)
ROUTER_AGES_PERCENT_ENDPOINTS = (0, 100)
DEFAULT_EXISTING_80_DIR = "results/router_age_lth_80pct_dense_v2"


def _expert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "expert"}


def _shared_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "shared"}


def _record_id(record: dict) -> str:
    return f"{record['reference_seed']}:{record['router_age_percent']}:{record['condition']}"


def _finite(value: object) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _assert_rewound_ticket(
    initial_state: dict[str, torch.Tensor], ticket_state: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
) -> None:
    """Prove that retained ticket coordinates are E_0 and pruned coordinates are zero."""
    for name, mask in masks.items():
        initial = initial_state[name].detach().cpu()
        ticket = ticket_state[name].detach().cpu()
        keep = mask.detach().cpu().bool()
        if not torch.equal(ticket[keep], initial[keep]):
            raise RuntimeError(f"Rewind assertion failed for retained coordinates in {name}.")
        if not torch.equal(ticket[~keep], torch.zeros_like(ticket[~keep])):
            raise RuntimeError(f"Rewind assertion failed for pruned coordinates in {name}.")


def _load_records(path: Path) -> list[dict]:
    records_path = path / "router_age_recovery_all_records.json"
    if not records_path.exists():
        return []
    return json.loads(records_path.read_text(encoding="utf-8"))


def _compatible_external_dense(
    record: dict, *, seed: int, age: int, step: int, dense_expert_hash: str,
    shared_hash: str, train_hash: str, validation_hash: str, recovery_steps: int,
) -> bool:
    return (
        record.get("condition_type") == "dense_control"
        and not record.get("confidence_control", False)
        and record.get("reference_seed") == seed
        and record.get("router_age_percent") == age
        and record.get("router_step") == step
        and record.get("expert_state_hash") == dense_expert_hash
        and record.get("shared_state_hash") == shared_hash
        and record.get("training_batch_sequence_hash") == train_hash
        and record.get("validation_batch_sequence_hash") == validation_hash
        and record.get("recovery_steps") == recovery_steps
        and record.get("optimizer") == "fresh_AdamW"
        and record.get("scheduler") == "none"
        and _finite(record.get("final_validation_loss"))
    )


def _annotate_gradient_diagnostics(record: dict, condition_dir: Path) -> None:
    """Summarize existing gradient logs without allowing nonfinite metrics to abort a run."""
    path = condition_dir / "gradient_stats" / "gradient_stats.jsonl"
    values: list[float] = []
    nonfinite = False
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            value = row.get("expert_grad_norm")
            if _finite(value):
                values.append(float(value))
            elif value is not None:
                nonfinite = True
    record["mean_expert_gradient_norm"] = mean(values) if values else None
    record["nonfinite_gradient_metrics"] = nonfinite


def _run_dense(
    *, config: dict, base_state: dict[str, torch.Tensor], checkpoint: Path, age: int, step: int,
    expert_hash: str, shared_hash: str, reference_selected: list[torch.Tensor], calibration_batches: list[torch.Tensor],
    train_batches: list, validation_batches: list, train_hash: str, validation_hash: str, device: torch.device,
    recovery_steps: int, dense_reference_loss: float, output_dir: Path, seed: int,
) -> dict:
    record = recovery._run_recovery_condition(
        config=config, condition_name=f"seed{seed}_age{age}_dense", pruned_base_state=base_state,
        router_checkpoint=str(checkpoint), router_age_percent=age, router_step=step, masks={},
        expert_hash=expert_hash, shared_hash=shared_hash, mask_hash="dense_no_mask",
        reference_selected=reference_selected, calibration_batches=calibration_batches,
        train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash,
        validation_batch_hash=validation_hash, device=device, recovery_steps=recovery_steps,
        dense_loss=dense_reference_loss, output_dir=output_dir, confidence_control=False,
        target_confidence=None, seed=seed, sparsity=0.0,
    )
    record.update({
        "reference_seed": seed, "condition_type": "dense_control", "dense_baseline_reused": False,
        "dense_baseline_record_id": _record_id({"reference_seed": seed, "router_age_percent": age, "condition": f"seed{seed}_age{age}_dense"}),
        "expert_surviving_weight_source": "E0", "mask_source": "none", "shared_state_source": "E0",
        "requested_router_step": step, "loaded_router_step": step,
    })
    _annotate_gradient_diagnostics(record, output_dir)
    return record


def _run_sparse(
    *, config: dict, ticket_state: dict[str, torch.Tensor], checkpoint: Path, age: int, step: int,
    masks: dict[str, torch.Tensor], expert_hash: str, shared_hash: str, mask_hash: str,
    reference_selected: list[torch.Tensor], calibration_batches: list[torch.Tensor], train_batches: list,
    validation_batches: list, train_hash: str, validation_hash: str, device: torch.device, recovery_steps: int,
    dense_loss: float, output_dir: Path, seed: int, sparsity: float, dense_record: dict,
) -> dict:
    record = recovery._run_recovery_condition(
        config=config, condition_name=f"seed{seed}_s{sparsity:.2f}_age{age}_sparse", pruned_base_state=ticket_state,
        router_checkpoint=str(checkpoint), router_age_percent=age, router_step=step, masks=masks,
        expert_hash=expert_hash, shared_hash=shared_hash, mask_hash=mask_hash,
        reference_selected=reference_selected, calibration_batches=calibration_batches,
        train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash,
        validation_batch_hash=validation_hash, device=device, recovery_steps=recovery_steps,
        dense_loss=dense_loss, output_dir=output_dir, confidence_control=False,
        target_confidence=None, seed=seed, sparsity=sparsity,
    )
    record.update({
        "reference_seed": seed, "condition_type": "sparse_ticket", "expert_surviving_weight_source": "E0",
        "mask_source": "ET", "shared_state_source": "E0", "requested_router_step": step,
        "loaded_router_step": step, "dense_baseline_reused": bool(dense_record.get("dense_baseline_reused", False)),
        "dense_baseline_record_id": dense_record["dense_baseline_record_id"],
        "dense_baseline_final_loss": dense_record["final_validation_loss"],
        "ticket_gap": record["final_validation_loss"] - dense_record["final_validation_loss"],
    })
    _annotate_gradient_diagnostics(record, output_dir)
    return record


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(records: list[dict]) -> list[dict]:
    sparse = [row for row in records if row.get("condition_type") == "sparse_ticket"]
    by_key = {(row["reference_seed"], row["sparsity"], row["router_age_percent"]): row for row in sparse}
    rows = []
    for seed, sparsity in sorted({(row["reference_seed"], row["sparsity"]) for row in sparse}):
        r0, r100 = by_key.get((seed, sparsity, 0)), by_key.get((seed, sparsity, 100))
        if r0 is None or r100 is None:
            continue
        gap0, gap100 = r0["ticket_gap"], r100["ticket_gap"]
        reduction = gap0 - gap100
        proportional = reduction / gap0 if abs(gap0) > 1e-8 else None
        rows.append({
            "reference_seed": seed, "sparsity": sparsity,
            "sparse_R0_final": r0["final_validation_loss"], "dense_R0_final": r0["dense_baseline_final_loss"],
            "ticket_gap_R0": gap0, "sparse_R100_final": r100["final_validation_loss"],
            "dense_R100_final": r100["dense_baseline_final_loss"], "ticket_gap_R100": gap100,
            "gap_reduction": reduction, "proportional_gap_reduction": proportional if proportional is not None else "undefined",
        })
    return rows


def _aggregate_rows(records: list[dict]) -> list[dict]:
    sparse = [row for row in records if row.get("condition_type") == "sparse_ticket"]
    grouped: dict[tuple[float, int], list[dict]] = {}
    for row in sparse:
        grouped.setdefault((float(row["sparsity"]), int(row["router_age_percent"])), []).append(row)
    rows = []
    for (sparsity, age), group in sorted(grouped.items()):
        sparse_losses = [row["final_validation_loss"] for row in group]
        dense_losses = [row["dense_baseline_final_loss"] for row in group]
        gaps = [row["ticket_gap"] for row in group]
        auc_gaps = [row["early_auc"] - row.get("dense_baseline_early_auc", 0.0) for row in group]
        gradients = [row["mean_expert_gradient_norm"] for row in group if _finite(row.get("mean_expert_gradient_norm"))]
        rows.append({
            "sparsity": sparsity, "router_age": age, "mean_sparse_final_loss": mean(sparse_losses),
            "std_sparse_final_loss": stdev(sparse_losses) if len(sparse_losses) > 1 else 0.0,
            "mean_dense_final_loss": mean(dense_losses),
            "std_dense_final_loss": stdev(dense_losses) if len(dense_losses) > 1 else 0.0,
            "mean_ticket_gap": mean(gaps), "std_ticket_gap": stdev(gaps) if len(gaps) > 1 else 0.0,
            "mean_early_auc_gap": mean(auc_gaps), "mean_expert_gradient_norm": mean(gradients) if gradients else None,
            "num_seeds": len(group),
        })
    return rows


def _import_compatible_80(existing_root: Path, existing_records: list[dict], audit: dict) -> list[dict]:
    """Import only native 80% endpoint tickets whose prior audit proves the rewind protocol."""
    imported = []
    dense_by_key = {
        (row.get("reference_seed"), row.get("router_age_percent")): row
        for row in existing_records
        if row.get("condition_type") == "dense_control" and not row.get("confidence_control", False)
    }
    for row in existing_records:
        if (
            row.get("condition_type") != "sparse_ticket"
            or row.get("confidence_control", False)
            or row.get("router_age_percent") not in ROUTER_AGES_PERCENT_ENDPOINTS
            or not math.isclose(float(row.get("sparsity", -1)), 0.8)
        ):
            continue
        seed = row.get("reference_seed")
        prior_audit_path = existing_root / f"seed_{seed}" / "lth_isolation_audit.json"
        dense = dense_by_key.get((seed, row.get("router_age_percent")))
        if not prior_audit_path.exists() or dense is None:
            audit["warnings"].append(f"80% seed {seed}, age {row.get('router_age_percent')} was not imported: missing audit or dense control.")
            continue
        prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
        compatible = (
            prior_audit.get("ticket_matches_initial_expert_values_under_final_mask") is True
            and prior_audit.get("all_router_age_conditions_share_same_shared_state") is True
            and row.get("shared_state_hash") == dense.get("shared_state_hash")
            and row.get("training_batch_sequence_hash") == dense.get("training_batch_sequence_hash")
            and row.get("validation_batch_sequence_hash") == dense.get("validation_batch_sequence_hash")
            and row.get("recovery_steps") == dense.get("recovery_steps")
        )
        if not compatible:
            audit["warnings"].append(f"80% seed {seed}, age {row.get('router_age_percent')} was not imported: protocol metadata mismatch.")
            continue
        imported_row = dict(row)
        imported_row.update({
            "expert_surviving_weight_source": "E0",
            "mask_source": "ET",
            "shared_state_source": "E0",
            "requested_router_step": row["router_step"],
            "loaded_router_step": row["router_step"],
            "dense_baseline_reused": True,
            "dense_baseline_record_id": f"existing_80:{_record_id(dense)}",
            "dense_baseline_final_loss": dense["final_validation_loss"],
            "ticket_gap": row["final_validation_loss"] - dense["final_validation_loss"],
            "imported_from_existing_80": True,
            "mean_expert_gradient_norm": None,
            "nonfinite_gradient_metrics": False,
        })
        imported.append(imported_row)
    return imported


def _write_figures(root: Path, paired: list[dict]) -> None:
    """Write portable SVG figures with individual seed lines and mean curves."""
    if not paired:
        return
    def write_plot(filename: str, series: list[tuple[str, str, list[tuple[float, float]]]], ylabel: str) -> None:
        width, height, left, top, right, bottom = 760, 460, 82, 36, 24, 72
        values = [value for _label, _color, points in series for _x, value in points]
        minimum, maximum = min(values + [0.0]), max(values + [0.0])
        span = max(maximum - minimum, 1e-9)
        minimum -= span * 0.08
        maximum += span * 0.08
        x_values = sorted({x for _label, _color, points in series for x, _y in points})
        x_min, x_max = min(x_values), max(x_values)
        def x_pos(value: float) -> float:
            return left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - left - right)
        def y_pos(value: float) -> float:
            return top + (maximum - value) / (maximum - minimum) * (height - top - bottom)
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
                 f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
                 f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="sans-serif">Expert sparsity</text>',
                 f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif">{ylabel}</text>']
        for x in x_values:
            lines.append(f'<text x="{x_pos(x):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">{x:.2f}</text>')
        for index, (label, color, points) in enumerate(series):
            coordinates = " ".join(f"{x_pos(x):.1f},{y_pos(y):.1f}" for x, y in points)
            stroke_width = "2.5" if label.endswith("mean") else "1"
            opacity = "1" if label.endswith("mean") else "0.35"
            lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}"/>')
            for x, y in points:
                lines.append(f'<circle cx="{x_pos(x):.1f}" cy="{y_pos(y):.1f}" r="3" fill="{color}" opacity="{opacity}"/>')
            lines.append(f'<text x="{left + 8 + index * 150}" y="{top + 16}" font-family="sans-serif" font-size="12" fill="{color}">{label}</text>')
        lines.append("</svg>")
        (root / filename).write_text("\n".join(lines), encoding="utf-8")

    seeds = sorted({int(row["reference_seed"]) for row in paired})
    def points(field: str, seed: int | None = None) -> list[tuple[float, float]]:
        rows = [row for row in paired if seed is None or int(row["reference_seed"]) == seed]
        xs = sorted({float(row["sparsity"]) for row in rows})
        return [(x, mean(float(row[field]) for row in rows if float(row["sparsity"]) == x)) for x in xs]

    write_plot("sparsity_ticket_gap.svg", [(f"seed {seed} R0", "#c65d28", points("ticket_gap_R0", seed)) for seed in seeds] + [("R0 mean", "#b04316", points("ticket_gap_R0")), ("R100 mean", "#096a64", points("ticket_gap_R100"))], "L_sparse - L_dense")
    write_plot("sparsity_final_loss.svg", [("Sparse R0 mean", "#b04316", points("sparse_R0_final")), ("Sparse R100 mean", "#096a64", points("sparse_R100_final")), ("Dense R0 mean", "#777777", points("dense_R0_final")), ("Dense R100 mean", "#222222", points("dense_R100_final"))], "Final validation loss")
    write_plot("sparsity_router_benefit.svg", [(f"seed {seed}", "#7760a8", points("gap_reduction", seed)) for seed in seeds] + [("Mean", "#4a397d", points("gap_reduction"))], "Gap reduction: R0 - R100")


def _write_summary(root: Path, paired: list[dict], audit: dict, epsilon: float | None) -> None:
    lines = ["# Router-Conditioned Sparsity Sweep", "", f"All required audits passed: `{audit['all_pass']}`.", ""]
    lines.append("## Paired Results")
    lines.append("")
    lines.append("| Seed | Sparsity | Gap R0 | Gap R100 | Reduction | Proportional reduction |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in paired:
        lines.append("| {reference_seed} | {sparsity:.2f} | {ticket_gap_R0:.6f} | {ticket_gap_R100:.6f} | {gap_reduction:.6f} | {proportional_gap_reduction} |".format(**row))
    if epsilon is not None:
        lines.extend(["", f"Ticket-like tolerance was pre-specified as `epsilon={epsilon}`; raw gaps above remain the primary result."])
    warnings = audit.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    (root / "sparsity_sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sparsity_sweep(
    config_paths: list[str], output_dir: str, recovery_steps: int | None = None,
    existing_80_dir: str | None = DEFAULT_EXISTING_80_DIR, ticket_tolerance: float | None = None,
) -> dict:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite experiment artifacts in {root}")
    root.mkdir(parents=True)
    existing_records = _load_records(Path(existing_80_dir)) if existing_80_dir else []
    new_records: list[dict] = []
    audit = {"all_pass": True, "seeds": {}, "external_dense_reuse": [], "warnings": []}
    dense_cache: dict[tuple[int, int], dict] = {}

    for config_path in config_paths:
        config = load_config(config_path)
        seed = int(config["seed"])
        run_dir = recovery._ensure_reference(config_path)
        device = resolve_device(config["device"])
        configure_device(device)
        total_steps = int(config["training"]["steps"])
        condition_steps = total_steps if recovery_steps is None else int(recovery_steps)
        initial_checkpoint, _ = recovery._checkpoint_for_percent(run_dir, total_steps, 0)
        final_checkpoint, _ = recovery._checkpoint_for_percent(run_dir, total_steps, 100)
        seed_everything(seed)
        trained = load_model_from_checkpoint(config["model"], str(final_checkpoint), device)
        initial_model = load_model_from_checkpoint(config["model"], str(initial_checkpoint), device)
        initial_state = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}
        dense_base = initial_state
        dense_expert_hash = state_dict_hash(_expert_state(dense_base))
        shared_hash = state_dict_hash(_shared_state(dense_base))
        loader, validation_loader = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config))
        train_batches = recovery._materialize_batches(loader, condition_steps)
        validation_batches = recovery._materialize_validation_batches(validation_loader, int(config["data"]["validation_blocks"]))
        train_hash = recovery._batch_sequence_hash(train_batches)
        validation_hash = recovery._batch_sequence_hash(validation_batches)
        dense_reference_loss = recovery.evaluate_language_model(trained, validation_batches, device, max_batches=len(validation_batches))["loss"]
        calibration_batches = recovery._calibration_batches(validation_batches)
        audit["seeds"][str(seed)] = {"shared_state_hash": shared_hash, "train_hash": train_hash, "validation_hash": validation_hash, "routers": {}}

        for age in ROUTER_AGES_PERCENT_ENDPOINTS:
            checkpoint, step = recovery._checkpoint_for_percent(run_dir, total_steps, age)
            if step != recovery._expected_router_step_for_percent(total_steps, age):
                raise RuntimeError(f"Router checkpoint mismatch: requested age {age}, loaded step {step}.")
            router_model = load_model_from_checkpoint(config["model"], str(checkpoint), torch.device("cpu"))
            router_hash = state_dict_hash(component_state_dict(router_model, "router"))
            audit["seeds"][str(seed)]["routers"][str(age)] = {"requested_step": step, "loaded_step": step, "router_hash": router_hash}
            reference_model = recovery.assemble_router_age_model(config["model"], dense_base, str(final_checkpoint), {}, device)
            reference_selected = selected_experts_per_batch(reference_model, calibration_batches, device)
            cache_key = (seed, age)
            if cache_key not in dense_cache:
                external = next((row for row in existing_records if _compatible_external_dense(
                    row, seed=seed, age=age, step=step, dense_expert_hash=dense_expert_hash, shared_hash=shared_hash,
                    train_hash=train_hash, validation_hash=validation_hash, recovery_steps=condition_steps)), None)
                if external:
                    dense = dict(external)
                    dense.update({"dense_baseline_reused": True, "dense_baseline_record_id": f"external:{_record_id(external)}"})
                    audit["external_dense_reuse"].append(dense["dense_baseline_record_id"])
                else:
                    dense = _run_dense(config=config, base_state=dense_base, checkpoint=checkpoint, age=age, step=step,
                        expert_hash=dense_expert_hash, shared_hash=shared_hash, reference_selected=reference_selected,
                        calibration_batches=calibration_batches, train_batches=train_batches, validation_batches=validation_batches,
                        train_hash=train_hash, validation_hash=validation_hash, device=device, recovery_steps=condition_steps,
                        dense_reference_loss=dense_reference_loss, output_dir=root / f"seed_{seed}" / f"age_{age:03d}_dense", seed=seed)
                    new_records.append(dense)
                dense_cache[cache_key] = dense

        if audit["seeds"][str(seed)]["routers"]["0"]["router_hash"] == audit["seeds"][str(seed)]["routers"]["100"]["router_hash"]:
            raise RuntimeError(f"Router hash collision between R0 and R100 for seed {seed}.")

        for sparsity in SPARSITIES_TO_SWEEP:
            masks = expert_local_magnitude_masks(trained, sparsity)
            mask_hash = recovery._mask_hash(masks)
            ticket = build_fixed_pruned_base(config["model"], str(initial_checkpoint), masks, device)
            _assert_rewound_ticket(initial_state, ticket, masks)
            expert_hash = state_dict_hash(_expert_state(ticket))
            if expert_hash == state_dict_hash(_expert_state(trained.state_dict())):
                raise RuntimeError("Ticket expert state unexpectedly equals E_T.")
            level_dir = root / f"sparsity_{sparsity:.2f}" / f"seed_{seed}"
            level_dir.mkdir(parents=True, exist_ok=True)
            save_masks(masks, level_dir / "pruning_mask.pt")
            prunable = sum(mask.numel() for mask in masks.values())
            retained = sum(int(mask.sum()) for mask in masks.values())
            (level_dir / "pruning_metadata.json").write_text(json.dumps({
                "reference_seed": seed, "sparsity": sparsity, "total_expert_parameters": sum(p.numel() for n, p in trained.named_parameters() if parameter_group(n) == "expert"),
                "number_prunable_expert_weights": prunable, "number_pruned": prunable - retained,
                "number_retained": retained, "realized_sparsity": (prunable - retained) / prunable,
                "mask_sha256": mask_hash, "pruning_method": "expert_local_magnitude", "mask_source": "ET",
            }, indent=2), encoding="utf-8")
            for age in ROUTER_AGES_PERCENT_ENDPOINTS:
                checkpoint, step = recovery._checkpoint_for_percent(run_dir, total_steps, age)
                dense = dense_cache[(seed, age)]
                reference_model = recovery.assemble_router_age_model(config["model"], ticket, str(final_checkpoint), masks, device)
                reference_selected = selected_experts_per_batch(reference_model, calibration_batches, device)
                dense_for_ticket = dict(dense)
                dense_for_ticket["dense_baseline_reused"] = True
                sparse = _run_sparse(config=config, ticket_state=ticket, checkpoint=checkpoint, age=age, step=step,
                    masks=masks, expert_hash=expert_hash, shared_hash=shared_hash, mask_hash=mask_hash,
                    reference_selected=reference_selected, calibration_batches=calibration_batches, train_batches=train_batches,
                    validation_batches=validation_batches, train_hash=train_hash, validation_hash=validation_hash, device=device,
                    recovery_steps=condition_steps, dense_loss=dense["final_validation_loss"],
                    output_dir=level_dir / f"age_{age:03d}_sparse", seed=seed, sparsity=sparsity, dense_record=dense_for_ticket)
                new_records.append(sparse)

    imported_80 = _import_compatible_80(Path(existing_80_dir), existing_records, audit) if existing_80_dir else []
    all_records = new_records + imported_80
    paired = _paired_rows(all_records)
    aggregate = _aggregate_rows(all_records)
    (root / "sparsity_sweep_all_records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    _write_csv(root / "sparsity_sweep_aggregate.csv", aggregate)
    _write_csv(root / "sparsity_sweep_paired.csv", paired)
    (root / "sparsity_sweep_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_summary(root, paired, audit, ticket_tolerance)
    _write_figures(root, paired)
    return {"records": all_records, "output_dir": str(root), "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recovery-steps", type=int)
    parser.add_argument("--existing-80-dir", default=DEFAULT_EXISTING_80_DIR)
    parser.add_argument("--ticket-tolerance", type=float)
    args = parser.parse_args()
    run_sparsity_sweep(args.configs, args.output_dir, args.recovery_steps, args.existing_80_dir, args.ticket_tolerance)


if __name__ == "__main__":
    main()
