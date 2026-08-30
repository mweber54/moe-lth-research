"""Complete the router-age by sparsity rewound-ticket matrix without rerunning existing cells.

The runner imports audited endpoint records for 60/70/90/95%, imports all validated
80% records and dense E0 controls from the previous router-age run, and executes
only the 60 missing sparse intermediate-age recovery conditions.
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
    build_fixed_pruned_base,
    component_state_dict,
    load_model_from_checkpoint,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)
from moe_lth.utils import configure_device, resolve_data_seed, resolve_device, seed_everything

SPARSITIES = (0.60, 0.70, 0.80, 0.90, 0.95)
ROUTER_AGES = (0, 10, 20, 40, 60, 80, 100)
MISSING_AGES = (10, 20, 40, 60, 80)
MISSING_SPARSITIES = (0.60, 0.70, 0.90, 0.95)
LONG_FIELDS = (
    "reference_seed", "sparsity", "router_age", "router_step", "sparse_final_loss",
    "dense_final_loss", "ticket_gap", "sparse_initial_validation_loss",
    "dense_initial_validation_loss", "early_auc_sparse", "early_auc_dense", "early_auc_gap",
    "expert_gradient_norm_sparse", "mask_hash", "router_hash", "shared_state_hash",
    "training_sequence_hash", "validation_sequence_hash", "dense_baseline_reused", "audit_passed",
)


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _expert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "expert"}


def _shared_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "shared"}


def _mask_hash(masks: dict[str, torch.Tensor]) -> str:
    return recovery._mask_hash(masks)


def _assert_rewound(initial: dict[str, torch.Tensor], ticket: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> None:
    for name, mask in masks.items():
        keep = mask.detach().cpu().bool()
        expected = initial[name].detach().cpu()
        actual = ticket[name].detach().cpu()
        if not torch.equal(actual[keep], expected[keep]) or not torch.equal(actual[~keep], torch.zeros_like(actual[~keep])):
            raise RuntimeError(f"Ticket rewind assertion failed for {name}.")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean_gradient_norm(condition_dir: Path) -> tuple[float | None, bool]:
    path = condition_dir / "gradient_stats" / "gradient_stats.jsonl"
    if not path.exists():
        return None, False
    values, nonfinite = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line).get("expert_grad_norm")
        if _finite(value):
            values.append(float(value))
        elif value is not None:
            nonfinite = True
    return (mean(values) if values else None), nonfinite


def _dense_compatible(row: dict, *, seed: int, age: int, step: int, expert_hash: str, shared_hash: str, train_hash: str, validation_hash: str, recovery_steps: int) -> bool:
    return (
        row.get("condition_type") == "dense_control"
        and not row.get("confidence_control", False)
        and row.get("reference_seed") == seed
        and row.get("router_age_percent") == age
        and row.get("router_step") == step
        and row.get("expert_state_hash") == expert_hash
        and row.get("shared_state_hash") == shared_hash
        and row.get("training_batch_sequence_hash") == train_hash
        and row.get("validation_batch_sequence_hash") == validation_hash
        and row.get("recovery_steps") == recovery_steps
        and row.get("optimizer") == "fresh_AdamW"
        and row.get("scheduler") == "none"
        and _finite(row.get("final_validation_loss"))
    )


def _as_long(sparse: dict, dense: dict, dense_reused: bool) -> dict:
    sparse_gradient, nonfinite = _mean_gradient_norm(Path(sparse.get("condition_dir", "")))
    return {
        "reference_seed": sparse["reference_seed"], "sparsity": float(sparse["sparsity"]),
        "router_age": sparse["router_age_percent"], "router_step": sparse["router_step"],
        "sparse_final_loss": sparse["final_validation_loss"], "dense_final_loss": dense["final_validation_loss"],
        "ticket_gap": sparse["final_validation_loss"] - dense["final_validation_loss"],
        "sparse_initial_validation_loss": sparse["initial_validation_loss"],
        "dense_initial_validation_loss": dense["initial_validation_loss"],
        "early_auc_sparse": sparse["early_auc"], "early_auc_dense": dense["early_auc"],
        "early_auc_gap": sparse["early_auc"] - dense["early_auc"],
        "expert_gradient_norm_sparse": sparse_gradient,
        "mask_hash": sparse["mask_hash"], "router_hash": sparse["initial_router_state_hash"],
        "shared_state_hash": sparse["shared_state_hash"],
        "training_sequence_hash": sparse["training_batch_sequence_hash"],
        "validation_sequence_hash": sparse["validation_batch_sequence_hash"],
        "dense_baseline_reused": dense_reused, "audit_passed": sparse.get("integrity_checks_passed", False) and not nonfinite,
    }


def _write_csv(path: Path, fields: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _matrix(long_rows: list[dict], statistic) -> list[dict]:
    output = []
    for sparsity in SPARSITIES:
        row = {"sparsity": f"{sparsity:.2f}"}
        for age in ROUTER_AGES:
            values = [float(item["ticket_gap"]) for item in long_rows if item["sparsity"] == sparsity and item["router_age"] == age]
            row[f"R{age}"] = statistic(values) if len(values) == 3 else "incompatible"
        output.append(row)
    return output


def _aggregate(long_rows: list[dict]) -> list[dict]:
    rows = []
    for sparsity in SPARSITIES:
        for age in ROUTER_AGES:
            group = [row for row in long_rows if row["sparsity"] == sparsity and row["router_age"] == age]
            if len(group) != 3:
                continue
            def values(key: str) -> list[float]:
                return [float(row[key]) for row in group]
            gradients = [float(row["expert_gradient_norm_sparse"]) for row in group if _finite(row["expert_gradient_norm_sparse"])]
            rows.append({
                "sparsity": sparsity, "router_age": age, "router_step": group[0]["router_step"],
                "mean_sparse_final_loss": mean(values("sparse_final_loss")), "std_sparse_final_loss": stdev(values("sparse_final_loss")),
                "mean_dense_final_loss": mean(values("dense_final_loss")), "std_dense_final_loss": stdev(values("dense_final_loss")),
                "mean_ticket_gap": mean(values("ticket_gap")), "std_ticket_gap": stdev(values("ticket_gap")),
                "mean_early_auc_gap": mean(values("early_auc_gap")),
                "mean_expert_gradient_norm": mean(gradients) if gradients else None, "num_seeds": len(group),
            })
    return rows


def _write_svg_heatmap(path: Path, matrix: list[dict]) -> None:
    values = [float(row[f"R{age}"]) for row in matrix for age in ROUTER_AGES if isinstance(row[f"R{age}"], float)]
    low, high = min(values), max(values)
    width, height, left, top, cell_w, cell_h = 940, 410, 120, 60, 110, 55
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>']
    lines.append('<text x="470" y="28" text-anchor="middle" font-family="sans-serif" font-size="17">Mean sparse-dense ticket gap</text>')
    for column, age in enumerate(ROUTER_AGES):
        lines.append(f'<text x="{left + column * cell_w + cell_w / 2}" y="52" text-anchor="middle" font-family="sans-serif">R{age}</text>')
    for row_index, row in enumerate(matrix):
        y = top + row_index * cell_h
        lines.append(f'<text x="75" y="{y + 33}" text-anchor="middle" font-family="sans-serif">{float(row["sparsity"])*100:.0f}%</text>')
        for column, age in enumerate(ROUTER_AGES):
            value = row[f"R{age}"]
            x = left + column * cell_w
            if isinstance(value, float):
                t = (value - low) / max(high - low, 1e-9)
                color = f'rgb({int(250 - 150*t)},{int(242 - 150*t)},{int(225 - 80*t)})'
                text = f"{value:.4f}"
            else:
                color, text = "#dddddd", "N/A"
            lines.extend([f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-2}" fill="{color}" stroke="#ffffff"/>', f'<text x="{x + cell_w/2}" y="{y + 33}" text-anchor="middle" font-family="sans-serif" font-size="12">{text}</text>'])
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_svg_curves(path: Path, long_rows: list[dict]) -> None:
    width, height, left, top, right, bottom = 850, 460, 75, 42, 24, 70
    means = {(s, age): mean(float(row["ticket_gap"]) for row in long_rows if row["sparsity"] == s and row["router_age"] == age) for s in SPARSITIES for age in ROUTER_AGES}
    values = list(means.values())
    lower, upper = min(values), max(values)
    pad = max((upper-lower)*.1, .01); lower -= pad; upper += pad
    colors = ["#b04316", "#096a64", "#4a397d", "#b56d00", "#a12d55"]
    def x(age): return left + ROUTER_AGES.index(age) * (width-left-right) / (len(ROUTER_AGES)-1)
    def y(value): return top + (upper-value) / (upper-lower) * (height-top-bottom)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>', f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>', f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="sans-serif">Router age</text>', f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif">Mean ticket gap</text>']
    for age in ROUTER_AGES: lines.append(f'<text x="{x(age):.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="sans-serif" font-size="12">R{age}</text>')
    for sparsity, color in zip(SPARSITIES, colors):
        points = " ".join(f"{x(age):.1f},{y(means[(sparsity, age)]):.1f}" for age in ROUTER_AGES)
        lines.extend([f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>', f'<text x="{left + 10 + SPARSITIES.index(sparsity)*115}" y="{top+16}" font-family="sans-serif" font-size="12" fill="{color}">{sparsity:.2f}</text>'])
    lines.append('</svg>'); path.write_text("\n".join(lines), encoding="utf-8")


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    """Invert a full-rank square matrix with Gauss-Jordan elimination."""
    size = len(matrix)
    augmented = [row[:] + [float(index == row_index) for index in range(size)] for row_index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise RuntimeError("Repeated-measures design matrix is rank deficient.")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column])]
    return [row[size:] for row in augmented]


def _statistics(long_rows: list[dict]) -> dict:
    """Seed-fixed-effect factorial OLS; seeds are repeated measures rather than independent samples."""
    y = [float(row["ticket_gap"]) for row in long_rows]
    seeds = sorted({row["reference_seed"] for row in long_rows})
    base_s, base_a = SPARSITIES[0], ROUTER_AGES[0]
    columns, names = [[1.0] * len(long_rows)], ["intercept"]
    for seed in seeds[1:]: columns.append([float(row["reference_seed"] == seed) for row in long_rows]); names.append(f"seed[{seed}]")
    for sparsity in SPARSITIES[1:]: columns.append([float(row["sparsity"] == sparsity) for row in long_rows]); names.append(f"sparsity[{sparsity}]")
    for age in ROUTER_AGES[1:]: columns.append([float(row["router_age"] == age) for row in long_rows]); names.append(f"router_age[{age}]")
    for sparsity in SPARSITIES[1:]:
        for age in ROUTER_AGES[1:]: columns.append([float(row["sparsity"] == sparsity and row["router_age"] == age) for row in long_rows]); names.append(f"interaction[{sparsity},{age}]")
    design = [list(row) for row in zip(*columns)]
    parameter_count = len(columns)
    cross_product = [[sum(row[left] * row[right] for row in design) for right in range(parameter_count)] for left in range(parameter_count)]
    inverse = _inverse(cross_product)
    cross_response = [sum(row[column] * value for row, value in zip(design, y)) for column in range(parameter_count)]
    coefficients = [sum(inverse[row][column] * cross_response[column] for column in range(parameter_count)) for row in range(parameter_count)]
    residual = [value - sum(weight * coefficient for weight, coefficient in zip(row, coefficients)) for row, value in zip(design, y)]
    dof = len(y) - parameter_count
    mse = sum(value * value for value in residual) / dof
    coefficient_rows = [{"term": name, "estimate": value, "ci95": [value - 1.96 * math.sqrt(max(mse * inverse[index][index], 0)), value + 1.96 * math.sqrt(max(mse * inverse[index][index], 0))]} for index, (name, value) in enumerate(zip(names, coefficients))]
    interaction_rows = [row for row in coefficient_rows if row["term"].startswith("interaction[")]
    sparsity_means = {str(sparsity): mean(float(row["ticket_gap"]) for row in long_rows if row["sparsity"] == sparsity) for sparsity in SPARSITIES}
    router_means = {f"R{age}": mean(float(row["ticket_gap"]) for row in long_rows if row["router_age"] == age) for age in ROUTER_AGES}
    largest_interaction = max(interaction_rows, key=lambda row: abs(row["estimate"]))
    return {"model": "ticket_gap ~ categorical_sparsity + categorical_router_age + interaction + seed_fixed_effect", "n_observations": len(y), "residual_df": dof, "residual_mse": mse, "coefficients": coefficient_rows, "effects": {"sparsity_means": sparsity_means, "sparsity_range": max(sparsity_means.values()) - min(sparsity_means.values()), "router_age_means": router_means, "router_age_range": max(router_means.values()) - min(router_means.values()), "largest_interaction": {**largest_interaction, "ci95_excludes_zero": largest_interaction["ci95"][0] > 0 or largest_interaction["ci95"][1] < 0}}, "interpretation": "Seed fixed effects account for repeated measurements; report observed ranges and interaction confidence intervals rather than significance alone."}


def run_matrix_completion(config_paths: list[str], output_dir: str, endpoint_records: str, router_age_records: str, recovery_steps: int | None = None) -> dict:
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()): raise FileExistsError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    endpoints = _load_json(Path(endpoint_records))
    prior = _load_json(Path(router_age_records))
    prior_dense = {(row.get("reference_seed"), row.get("router_age_percent")): row for row in prior if row.get("condition_type") == "dense_control" and not row.get("confidence_control", False)}
    audit = {"all_pass": True, "new_sparse_runs": 0, "existing_sparse_reused": 0, "dense_baselines_reused": 0, "warnings": [], "seeds": {}}
    final_rows: list[dict] = []

    for config_path in config_paths:
        config = load_config(config_path); seed = int(config["seed"]); run_dir = recovery._ensure_reference(config_path)
        device = resolve_device(config["device"]); configure_device(device); total = int(config["training"]["steps"]); steps = total if recovery_steps is None else int(recovery_steps)
        initial_checkpoint, _ = recovery._checkpoint_for_percent(run_dir, total, 0); final_checkpoint, _ = recovery._checkpoint_for_percent(run_dir, total, 100)
        seed_everything(seed); trained = load_model_from_checkpoint(config["model"], str(final_checkpoint), device); initial_model = load_model_from_checkpoint(config["model"], str(initial_checkpoint), device)
        initial = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}; dense_expert_hash = state_dict_hash(_expert_state(initial)); shared_hash = state_dict_hash(_shared_state(initial))
        loader, val_loader = build_dataloaders(config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)); train_batches = recovery._materialize_batches(loader, steps); validation_batches = recovery._materialize_validation_batches(val_loader, int(config["data"]["validation_blocks"]))
        train_hash, validation_hash = recovery._batch_sequence_hash(train_batches), recovery._batch_sequence_hash(validation_batches); calibration = recovery._calibration_batches(validation_batches)
        audit["seeds"][str(seed)] = {"shared_state_hash": shared_hash, "train_hash": train_hash, "validation_hash": validation_hash, "routers": {}}
        dense_by_age = {}
        for age in ROUTER_AGES:
            checkpoint, step = recovery._checkpoint_for_percent(run_dir, total, age)
            requested = recovery._expected_router_step_for_percent(total, age)
            if requested != step: raise RuntimeError(f"Router checkpoint mismatch for seed {seed}, R{age}: requested {requested}, loaded {step}.")
            router_hash = state_dict_hash(component_state_dict(load_model_from_checkpoint(config["model"], str(checkpoint), torch.device("cpu")), "router"))
            audit["seeds"][str(seed)]["routers"][str(age)] = {"requested_step": requested, "loaded_step": step, "router_hash": router_hash}
            dense = prior_dense.get((seed, age))
            if not dense or not _dense_compatible(dense, seed=seed, age=age, step=step, expert_hash=dense_expert_hash, shared_hash=shared_hash, train_hash=train_hash, validation_hash=validation_hash, recovery_steps=steps):
                raise RuntimeError(f"No compatible dense baseline for seed {seed}, R{age}; refusing to run sparse condition.")
            dense_by_age[age] = dense; audit["dense_baselines_reused"] += 1
        for sparsity in SPARSITIES:
            masks = expert_local_magnitude_masks(trained, sparsity); ticket = build_fixed_pruned_base(config["model"], str(initial_checkpoint), masks, device); _assert_rewound(initial, ticket, masks)
            mask_hash, expert_hash = _mask_hash(masks), state_dict_hash(_expert_state(ticket)); level_dir = root / f"sparsity_{sparsity:.2f}" / f"seed_{seed}"; level_dir.mkdir(parents=True, exist_ok=True); save_masks(masks, level_dir / "pruning_mask.pt")
            for age in ROUTER_AGES:
                dense = dense_by_age[age]
                old = next((row for row in endpoints + prior if row.get("condition_type") == "sparse_ticket" and row.get("reference_seed") == seed and row.get("router_age_percent") == age and math.isclose(float(row.get("sparsity", -1)), sparsity) and not row.get("confidence_control", False)), None)
                if old:
                    checks = old.get("expert_state_hash") == expert_hash and old.get("mask_hash") == mask_hash and old.get("shared_state_hash") == shared_hash and old.get("training_batch_sequence_hash") == train_hash and old.get("validation_batch_sequence_hash") == validation_hash and old.get("router_step") == recovery._expected_router_step_for_percent(total, age)
                    if not checks: raise RuntimeError(f"Existing result incompatible for seed {seed}, s={sparsity}, R{age}.")
                    final_rows.append(_as_long(old, dense, True)); audit["existing_sparse_reused"] += 1; continue
                if sparsity not in MISSING_SPARSITIES or age not in MISSING_AGES: raise RuntimeError(f"Unexpected missing cell seed={seed} sparsity={sparsity} age={age}.")
                checkpoint, step = recovery._checkpoint_for_percent(run_dir, total, age)
                reference = recovery.assemble_router_age_model(config["model"], ticket, str(final_checkpoint), masks, device)
                record = recovery._run_recovery_condition(config=config, condition_name=f"seed{seed}_s{sparsity:.2f}_age{age}_sparse", pruned_base_state=ticket, router_checkpoint=str(checkpoint), router_age_percent=age, router_step=step, masks=masks, expert_hash=expert_hash, shared_hash=shared_hash, mask_hash=mask_hash, reference_selected=selected_experts_per_batch(reference, calibration, device), calibration_batches=calibration, train_batches=train_batches, validation_batches=validation_batches, train_batch_hash=train_hash, validation_batch_hash=validation_hash, device=device, recovery_steps=steps, dense_loss=dense["final_validation_loss"], output_dir=level_dir / f"age_{age:03d}_sparse", confidence_control=False, target_confidence=None, seed=seed, sparsity=sparsity)
                record.update({"reference_seed": seed, "condition_type": "sparse_ticket", "condition_dir": str(level_dir / f"age_{age:03d}_sparse")})
                final_rows.append(_as_long(record, dense, True)); audit["new_sparse_runs"] += 1
    if len(final_rows) != 105: raise RuntimeError(f"Expected 105 compatible sparse cells, got {len(final_rows)}.")
    matrix = _matrix(final_rows, mean); std_matrix = _matrix(final_rows, stdev); aggregate = _aggregate(final_rows); stats = _statistics(final_rows)
    _write_csv(root / "routing_age_sparsity_seed_level.csv", LONG_FIELDS, final_rows); _write_csv(root / "routing_age_sparsity_ticket_gap_matrix.csv", ["sparsity", *[f"R{age}" for age in ROUTER_AGES]], matrix); _write_csv(root / "routing_age_sparsity_ticket_gap_std_matrix.csv", ["sparsity", *[f"R{age}" for age in ROUTER_AGES]], std_matrix); _write_csv(root / "routing_age_sparsity_aggregate_long.csv", list(aggregate[0]), aggregate)
    (root / "routing_age_sparsity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8"); (root / "routing_age_sparsity_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8"); _write_svg_heatmap(root / "routing_age_sparsity_ticket_gap_heatmap.svg", matrix); _write_svg_curves(root / "routing_age_sparsity_ticket_gap_curves.svg", final_rows)
    header = "sparsity " + " ".join(f"R{age:>7}" for age in ROUTER_AGES); lines = ["# Router-Age x Sparsity Ticket-Gap Matrix", "", f"New sparse runs: {audit['new_sparse_runs']}", f"Existing sparse rows reused: {audit['existing_sparse_reused']}", f"Dense baseline reuses validated: {audit['dense_baselines_reused']}", "", "```", header]
    lines.extend(f"{float(row['sparsity'])*100:>5.0f}% " + " ".join(f"{float(row[f'R{age}']):8.4f}" for age in ROUTER_AGES) for row in matrix); lines.extend(["```", "", "## Standard Deviations", "", "```", header]); lines.extend(f"{float(row['sparsity'])*100:>5.0f}% " + " ".join(f"{float(row[f'R{age}']):8.4f}" for age in ROUTER_AGES) for row in std_matrix); lines.append("```")
    (root / "routing_age_sparsity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[7:]))
    return {"output_dir": str(root), "audit": audit, "matrix": matrix, "statistics": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint-records", required=True); parser.add_argument("--router-age-records", required=True)
    parser.add_argument("--recovery-steps", type=int)
    args = parser.parse_args(); run_matrix_completion(args.configs, args.output_dir, args.endpoint_records, args.router_age_records, args.recovery_steps)


if __name__ == "__main__":
    main()
