"""Router-age recovery experiment.

Central question: if the sparse expert weights and pruning mask are held
fixed, does changing only the router checkpoint change how quickly and how
completely those experts recover during retraining?

For each reference seed, this script:
  1. Loads the final (100%) reference checkpoint and prunes its experts once
     (magnitude, expert-local, matching `moe_lth.pruning.magnitude_prune`).
  2. Builds M_t = (S_T, R_t, E_T^{80%}) for each router-age checkpoint t,
     swapping in only the router parameters from checkpoint t.
  3. Retrains each M_t from a fresh optimizer for a fixed recovery budget,
     logging loss, gradient norms (expert/router/shared), and routing
     statistics (entropy, mean selected probability, margin, utilization,
     assignment agreement with the final router).
  4. Optionally repeats a subset of ages with router-logit temperature
     calibrated to match mean selected probability (confidence-matched
     control), to separate "routing structure" from "gating amplitude".

Existing reference/rewind artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import MaskDict, apply_masks_, save_masks
from moe_lth.pruning.rewind import register_mask_gradient_hooks
from moe_lth.pruning.router_age import (
    assemble_router_age_model,
    assignment_agreement,
    build_fixed_pruned_base,
    calibrate_temperature,
    component_state_dict,
    grad_norms_by_group,
    load_model_from_checkpoint,
    mean_selected_probability,
    parameter_group,
    selected_experts_per_batch,
    set_router_temperature,
    state_dict_hash,
)
from moe_lth.training.checkpoint import save_checkpoint
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.utils import (
    append_jsonl,
    configure_device,
    resolve_data_seed,
    resolve_device,
    seed_everything,
)

ROUTER_AGES_PERCENT = (0, 10, 20, 40, 60, 80, 100)
CONFIDENCE_CONTROL_AGES_PERCENT = (0, 40, 80, 100)
DEFAULT_SPARSITY = 0.8
RECOVERY_EVAL_INTERVAL = 50
EARLY_AUC_WINDOW_FRACTION = 0.5
THRESHOLDS = {"within_5pct": 1.05, "within_10pct": 1.10}


def _checkpoint_for_percent(run_dir: Path, total_steps: int, percent: int) -> tuple[Path, int]:
    target_step = round(total_steps * percent / 100)
    available = {
        int(path.stem.split("_")[-1]): path for path in (run_dir / "checkpoints").glob("step_*.pt")
    }
    if not available:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}/checkpoints")
    closest_step = min(available, key=lambda step: abs(step - target_step))
    return available[closest_step], closest_step


def _calibration_batches(validation_loader, device: torch.device, max_batches: int = 8) -> list[torch.Tensor]:
    batches = []
    for batch_id, (token_ids, _targets) in enumerate(validation_loader):
        if batch_id >= max_batches:
            break
        batches.append(token_ids.to(device))
    return batches


def _mask_hash(masks: MaskDict) -> str:
    return state_dict_hash({name: mask.to(torch.uint8) for name, mask in masks.items()})


def _run_recovery_condition(
    *,
    config: dict,
    condition_name: str,
    pruned_base_state: dict[str, torch.Tensor],
    router_checkpoint: str,
    router_age_percent: int,
    router_step: int,
    masks: MaskDict,
    expert_hash: str,
    shared_hash: str,
    mask_hash: str,
    reference_selected: list[torch.Tensor],
    calibration_batches: list[torch.Tensor],
    device: torch.device,
    recovery_steps: int,
    dense_loss: float,
    output_dir: Path,
    confidence_control: bool,
    target_confidence: float | None,
    seed: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_stats").mkdir(exist_ok=True)
    (output_dir / "gradient_stats").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    seed_everything(seed)
    model = assemble_router_age_model(config["model"], pruned_base_state, router_checkpoint, masks, device)

    # --- Integrity checks (fail loudly) ---
    observed_expert_hash = state_dict_hash(component_state_dict(model, "expert"))
    observed_shared_hash = state_dict_hash(component_state_dict(model, "shared"))
    if observed_expert_hash != expert_hash:
        raise RuntimeError(
            f"Integrity violation: expert weights differ from the fixed pruned state in {condition_name}."
        )
    if observed_shared_hash != shared_hash:
        raise RuntimeError(
            f"Integrity violation: shared weights differ from the fixed reference state in {condition_name}."
        )
    for name in masks:
        if parameter_group(name) != "expert":
            raise RuntimeError(f"Integrity violation: non-expert parameter {name} present in mask dict.")

    temperature = 1.0
    achieved_confidence = None
    agreement_before_after = None
    if confidence_control:
        assert target_confidence is not None
        temperature, achieved_confidence, agreement_before_after = calibrate_temperature(
            model, calibration_batches, device, target_confidence, reference_selected
        )
        if agreement_before_after < 0.999:
            raise RuntimeError(
                f"Integrity violation: confidence calibration changed top-1 assignment in {condition_name} "
                f"(agreement={agreement_before_after:.4f})."
            )
    set_router_temperature(model, temperature)

    handles = register_mask_gradient_hooks(model, masks)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    train_loader, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    iterator = cycle(train_loader)

    def evaluate() -> dict:
        model.eval()
        metrics = evaluate_language_model(
            model, validation_loader, device, max_batches=int(config["data"]["validation_blocks"])
        )
        model.train()
        return metrics

    def routing_snapshot() -> dict:
        stats = mean_selected_probability(model, calibration_batches, device)
        candidate_selected = selected_experts_per_batch(model, calibration_batches, device)
        stats["assignment_agreement_with_final_router"] = assignment_agreement(reference_selected, candidate_selected)
        stats["router_logit_norm"] = float(
            sum(block.moe.router.projection.weight.detach().float().pow(2).sum() for block in model.blocks).cpu()
            ** 0.5
        )
        return stats

    recovery_curve: list[dict] = []
    initial_metrics = evaluate()
    initial_routing = routing_snapshot()
    recovery_curve.append({"step": 0, "loss": initial_metrics["loss"]})
    append_jsonl(output_dir / "routing_stats" / "routing_stats.jsonl", {"step": 0, **initial_routing})

    model.train()
    for step in range(1, recovery_steps + 1):
        token_ids, targets = next(iterator)
        token_ids, targets = token_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(token_ids)
        language_loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1)
        )
        loss = language_loss + float(config["routing"]["aux_loss_weight"]) * output.auxiliary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip"]))
        grad_norms = grad_norms_by_group(model)
        optimizer.step()
        apply_masks_(model, masks)

        append_jsonl(
            output_dir / "gradient_stats" / "gradient_stats.jsonl",
            {
                "step": step,
                "expert_grad_norm": grad_norms["expert"],
                "router_grad_norm": grad_norms["router"],
                "shared_grad_norm": grad_norms["shared"],
                "train_loss": float(loss.detach().cpu()),
            },
        )

        if step % RECOVERY_EVAL_INTERVAL == 0 or step == recovery_steps:
            metrics = evaluate()
            recovery_curve.append({"step": step, "loss": metrics["loss"]})
            append_jsonl(output_dir / "routing_stats" / "routing_stats.jsonl", {"step": step, **routing_snapshot()})

    for handle in handles:
        handle.remove()

    # --- Post-hoc integrity check: pruned weights must remain exactly zero. ---
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        pruned_positions = ~mask.to(parameters[name].device)
        if pruned_positions.any():
            residual = parameters[name].detach()[pruned_positions].abs().max().item()
            if residual > 0.0:
                raise RuntimeError(
                    f"Integrity violation: pruned weight {name} became non-zero during recovery "
                    f"in {condition_name} (max residual {residual})."
                )

    for record in recovery_curve:
        append_jsonl(output_dir / "metrics.jsonl", record)

    final_loss = recovery_curve[-1]["loss"]
    initial_loss = recovery_curve[0]["loss"]

    early_window_step = recovery_steps * EARLY_AUC_WINDOW_FRACTION
    early_points = [row for row in recovery_curve if row["step"] <= early_window_step]
    early_auc = 0.0
    for previous, current in zip(early_points, early_points[1:]):
        early_auc += 0.5 * (previous["loss"] + current["loss"]) * (current["step"] - previous["step"])

    recovery_fraction = None
    denominator = initial_loss - dense_loss
    if denominator > 0:
        recovery_fraction = (initial_loss - final_loss) / denominator

    time_to_threshold = {}
    for name, factor in THRESHOLDS.items():
        threshold_value = dense_loss * factor
        reached = next((row["step"] for row in recovery_curve if row["loss"] <= threshold_value), None)
        time_to_threshold[name] = reached if reached is not None else "unreached"

    final_router_state = component_state_dict(model, "router")
    final_confidence = routing_snapshot()

    save_checkpoint(
        output_dir / "checkpoints" / "final_recovered.pt",
        model,
        None,
        recovery_steps,
        final_loss,
        config,
    )

    metadata = {
        "condition": condition_name,
        "seed": seed,
        "router_age_percent": router_age_percent,
        "router_checkpoint": str(router_checkpoint),
        "router_step": router_step,
        "sparsity": DEFAULT_SPARSITY,
        "pruning_method": "expert_local_magnitude",
        "confidence_control": confidence_control,
        "temperature": temperature,
        "recovery_steps": recovery_steps,
        "expert_state_hash": observed_expert_hash,
        "shared_state_hash": observed_shared_hash,
        "mask_hash": mask_hash,
        "router_state_hash": state_dict_hash(final_router_state),
        "optimizer": "fresh_AdamW",
        "integrity_checks_passed": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "early_auc": early_auc,
        "recovery_fraction": recovery_fraction,
        "time_to_threshold": time_to_threshold,
        "dense_reference_loss": dense_loss,
        "mean_selected_probability_initial": initial_routing["mean_selected_probability"],
        "mean_selected_probability_final": final_confidence["mean_selected_probability"],
        "routing_entropy_initial": initial_routing["routing_entropy"],
        "routing_entropy_final": final_confidence["routing_entropy"],
        "assignment_agreement_with_final_router_initial": initial_routing["assignment_agreement_with_final_router"],
        "assignment_agreement_with_final_router_final": final_confidence["assignment_agreement_with_final_router"],
        "achieved_confidence": achieved_confidence,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {**metadata, **summary}


def _ensure_reference(config_path: str) -> Path:
    from moe_lth.training.train import train_from_config

    config = load_config(config_path)
    run_dir = Path(config["output_dir"])
    checkpoint_steps = sorted(int(step) for step in config["training"]["checkpoint_steps"])
    existing = {
        int(path.stem.split("_")[-1]) for path in (run_dir / "checkpoints").glob("step_*.pt")
    } if (run_dir / "checkpoints").exists() else set()
    if set(checkpoint_steps).issubset(existing):
        return run_dir
    train_from_config(config)
    return run_dir


def run_router_age_recovery(
    config_paths: list[str],
    output_dir: str,
    sparsity: float = DEFAULT_SPARSITY,
    recovery_steps: int | None = None,
    router_ages_percent: tuple[int, ...] = ROUTER_AGES_PERCENT,
    confidence_control_ages: tuple[int, ...] = CONFIDENCE_CONTROL_AGES_PERCENT,
    confidence_control_seed_indices: tuple[int, ...] | None = None,
) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    # Default: run the confidence-matched control only for the first (primary) reference
    # seed to bound compute, per "if compute allows" guidance; the native sweep still
    # covers every seed x every router age.
    if confidence_control_seed_indices is None:
        confidence_control_seed_indices = (0,)

    for config_index, config_path in enumerate(config_paths):
        config = load_config(config_path)
        seed = int(config["seed"])
        run_dir = _ensure_reference(config_path)
        device = resolve_device(config["device"])
        configure_device(device)
        total_steps = int(config["training"]["steps"])
        if recovery_steps is None:
            recovery_steps = round(total_steps * max(config["pruning"]["rewind_fractions"]))

        final_checkpoint, final_step = _checkpoint_for_percent(run_dir, total_steps, 100)
        seed_everything(seed)
        dense_model = load_model_from_checkpoint(config["model"], str(final_checkpoint), device)
        _, validation_loader_for_dense = build_dataloaders(
            config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
        )
        dense_metrics = evaluate_language_model(
            dense_model, validation_loader_for_dense, device, max_batches=int(config["data"]["validation_blocks"])
        )
        dense_loss = dense_metrics["loss"]

        masks = expert_local_magnitude_masks(dense_model, sparsity)
        seed_dir = root / f"seed_{seed}"
        mask_path = seed_dir / "pruning_mask.pt"
        save_masks(masks, mask_path)

        expert_params = sum(mask.numel() for mask in masks.values())
        surviving = sum(int(mask.sum().item()) for mask in masks.values())
        pruned = expert_params - surviving
        pruning_stats = {
            "total_expert_parameters": expert_params,
            "pruned_parameters": pruned,
            "surviving_parameters": surviving,
            "realized_sparsity": pruned / expert_params,
            "pruning_method": "expert_local_magnitude (top-k retained by magnitude per expert)",
            "pruning_threshold": "none (rank-based top-k selection, not an absolute threshold)",
            "mask_hash": _mask_hash(masks),
        }
        (seed_dir / "pruning_metadata.json").write_text(json.dumps(pruning_stats, indent=2), encoding="utf-8")

        pruned_base_state = build_fixed_pruned_base(config["model"], str(final_checkpoint), masks, device)
        expert_hash = state_dict_hash({n: t for n, t in pruned_base_state.items() if parameter_group(n) == "expert"})
        shared_hash = state_dict_hash({n: t for n, t in pruned_base_state.items() if parameter_group(n) == "shared"})

        # Reference router (final, R_T) selections on a fixed calibration set, used for
        # assignment-agreement comparisons and confidence-target calibration.
        reference_model = assemble_router_age_model(
            config["model"], pruned_base_state, str(final_checkpoint), masks, device
        )
        _, calibration_loader = build_dataloaders(
            config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
        )
        calibration_batches = _calibration_batches(calibration_loader, device)
        reference_selected = selected_experts_per_batch(reference_model, calibration_batches, device)
        reference_confidence = mean_selected_probability(reference_model, calibration_batches, device)
        target_confidence = min(
            reference_confidence["mean_selected_probability"],
            mean_selected_probability(
                assemble_router_age_model(
                    config["model"],
                    pruned_base_state,
                    _checkpoint_for_percent(run_dir, total_steps, 0)[0],
                    masks,
                    device,
                ),
                calibration_batches,
                device,
            )["mean_selected_probability"],
        )

        for percent in router_ages_percent:
            router_checkpoint, router_step = _checkpoint_for_percent(run_dir, total_steps, percent)
            condition_dir = seed_dir / f"age_{percent:03d}pct_native"
            record = _run_recovery_condition(
                config=config,
                condition_name=f"seed{seed}_age{percent}_native",
                pruned_base_state=pruned_base_state,
                router_checkpoint=str(router_checkpoint),
                router_age_percent=percent,
                router_step=router_step,
                masks=masks,
                expert_hash=expert_hash,
                shared_hash=shared_hash,
                mask_hash=pruning_stats["mask_hash"],
                reference_selected=reference_selected,
                calibration_batches=calibration_batches,
                device=device,
                recovery_steps=recovery_steps,
                dense_loss=dense_loss,
                output_dir=condition_dir,
                confidence_control=False,
                target_confidence=None,
                seed=seed,
            )
            record.update({"reference_seed": seed, "final_step": final_step})
            all_records.append(record)
            _write_partial_csv(all_records, root)

            if percent in confidence_control_ages and config_index in confidence_control_seed_indices:
                condition_dir = seed_dir / f"age_{percent:03d}pct_confmatched"
                record = _run_recovery_condition(
                    config=config,
                    condition_name=f"seed{seed}_age{percent}_confmatched",
                    pruned_base_state=pruned_base_state,
                    router_checkpoint=str(router_checkpoint),
                    router_age_percent=percent,
                    router_step=router_step,
                    masks=masks,
                    expert_hash=expert_hash,
                    shared_hash=shared_hash,
                    mask_hash=pruning_stats["mask_hash"],
                    reference_selected=reference_selected,
                    calibration_batches=calibration_batches,
                    device=device,
                    recovery_steps=recovery_steps,
                    dense_loss=dense_loss,
                    output_dir=condition_dir,
                    confidence_control=True,
                    target_confidence=target_confidence,
                    seed=seed,
                )
                record.update({"reference_seed": seed, "final_step": final_step})
                all_records.append(record)
                _write_partial_csv(all_records, root)

    _write_partial_csv(all_records, root)
    (root / "router_age_recovery_all_records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    _write_report(all_records, root)
    return {"records": all_records, "output_dir": str(root)}


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_report(records: list[dict], root: Path) -> Path:
    native = sorted(
        (row for row in records if not row["confidence_control"]),
        key=lambda row: (row["reference_seed"], row["router_age_percent"]),
    )
    conf_matched = sorted(
        (row for row in records if row["confidence_control"]),
        key=lambda row: (row["reference_seed"], row["router_age_percent"]),
    )

    def rows_table(rows: list[dict]) -> str:
        lines = []
        for row in rows:
            final_router_row = next(
                r
                for r in native
                if r["reference_seed"] == row["reference_seed"] and r["router_age_percent"] == 100
            )
            initial_router_row = next(
                r
                for r in native
                if r["reference_seed"] == row["reference_seed"] and r["router_age_percent"] == 0
            )
            delta_final = row["final_validation_loss"] - final_router_row["final_validation_loss"]
            delta_initial = row["final_validation_loss"] - initial_router_row["final_validation_loss"]
            lines.append(
                f"| {row['reference_seed']} | {row['router_age_percent']} | {row['router_step']} | "
                f"{_fmt(row['initial_validation_loss'])} | {_fmt(row['early_auc'])} | "
                f"{_fmt(row['final_validation_loss'])} | {delta_final:+.4f} | {delta_initial:+.4f} | "
                f"{row['time_to_threshold']['within_5pct']} | {row['time_to_threshold']['within_10pct']} | "
                f"{_fmt(row['mean_selected_probability_final'])} | {_fmt(row['routing_entropy_final'])} | "
                f"{_fmt(row['assignment_agreement_with_final_router_final'])} |"
            )
        return "\n".join(lines)

    seeds = sorted({row["reference_seed"] for row in records})
    pruning_rows = []
    for seed in seeds:
        pruning_path = root / f"seed_{seed}" / "pruning_metadata.json"
        if pruning_path.exists():
            stats = json.loads(pruning_path.read_text(encoding="utf-8"))
            pruning_rows.append(
                f"| {seed} | {stats['total_expert_parameters']} | {stats['pruned_parameters']} | "
                f"{stats['surviving_parameters']} | {stats['realized_sparsity']:.4f} | "
                f"{stats['pruning_method']} | {stats['mask_hash'][:16]}... |"
            )

    gradient_rows = []
    for row in native:
        condition_dir = _condition_dir_from_record(root, row)
        means = _mean_gradient_norms(condition_dir)
        gradient_rows.append(
            f"| {row['reference_seed']} | {row['router_age_percent']} | {means['expert']:.4f} | "
            f"{means['router']:.4f} | {means['shared']:.4f} |"
        )

    markdown = f"""# Router-Age Recovery Experiment Results

Reference seeds: {", ".join(str(seed) for seed in seeds)}

Fixed 80%-magnitude-pruned expert state `E_T^{{80%}}` paired with router
checkpoints `R_t` from t in {{0, 10, 20, 40, 60, 80, 100}}% of the reference
training trajectory. Shared parameters, pruned expert weights, and the
pruning mask are byte-identical across every router-age condition within a
seed (verified via SHA-256 hashes recorded in each condition's
`metadata.json`). Only the router parameters differ across conditions.

## Pruning Summary (computed once per seed, applied to every router age)

| Seed | Total expert params | Pruned | Surviving | Realized sparsity | Method | Mask hash |
|---:|---:|---:|---:|---:|---|---|
{chr(10).join(pruning_rows)}

## Native-Confidence Router-Age Sweep

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Δ vs R100 | Δ vs R0 | T(5%) | T(10%) | Mean sel. prob | Entropy | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_table(native)}

## Confidence-Matched Control

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Δ vs R100 | Δ vs R0 | T(5%) | T(10%) | Mean sel. prob | Entropy | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_table(conf_matched)}

## Mean Gradient Norms During Recovery (native-confidence conditions)

| Seed | Router age % | Mean expert grad norm | Mean router grad norm | Mean shared grad norm |
|---:|---:|---:|---:|---:|
{chr(10).join(gradient_rows)}

Raw records: [router_age_recovery_all_records.json](router_age_recovery_all_records.json),
aggregate table: [router_age_recovery_aggregate.csv](router_age_recovery_aggregate.csv)
"""
    report_path = root / "router_age_recovery_results.md"
    report_path.write_text(markdown, encoding="utf-8")
    return report_path


CSV_COLUMNS = [
    "reference_seed",
    "router_step",
    "router_age_percent",
    "pruning_sparsity",
    "pruning_method",
    "confidence_control",
    "temperature",
    "initial_validation_loss",
    "early_auc",
    "final_validation_loss",
    "time_to_threshold_5pct",
    "time_to_threshold_10pct",
    "mean_selected_probability",
    "routing_entropy",
    "assignment_agreement_with_final_router",
    "mean_expert_gradient_norm",
    "mean_router_gradient_norm",
    "mean_shared_gradient_norm",
]


def _mean_gradient_norms(condition_dir: Path) -> dict[str, float]:
    path = condition_dir / "gradient_stats" / "gradient_stats.jsonl"
    if not path.exists():
        return {"expert": 0.0, "router": 0.0, "shared": 0.0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"expert": 0.0, "router": 0.0, "shared": 0.0}
    return {
        "expert": sum(row["expert_grad_norm"] for row in rows) / len(rows),
        "router": sum(row["router_grad_norm"] for row in rows) / len(rows),
        "shared": sum(row["shared_grad_norm"] for row in rows) / len(rows),
    }


def _write_partial_csv(records: list[dict], root: Path) -> None:
    csv_path = root / "router_age_recovery_aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for record in records:
            gradient_means = _mean_gradient_norms(_condition_dir_from_record(root, record))
            writer.writerow(
                [
                    record["reference_seed"],
                    record["router_step"],
                    record["router_age_percent"],
                    record["sparsity"],
                    record["pruning_method"],
                    record["confidence_control"],
                    record["temperature"],
                    record["initial_validation_loss"],
                    record["early_auc"],
                    record["final_validation_loss"],
                    record["time_to_threshold"]["within_5pct"],
                    record["time_to_threshold"]["within_10pct"],
                    record["mean_selected_probability_final"],
                    record["routing_entropy_final"],
                    record["assignment_agreement_with_final_router_final"],
                    gradient_means["expert"],
                    gradient_means["router"],
                    gradient_means["shared"],
                ]
            )


def _condition_dir_from_record(root: Path, record: dict) -> Path:
    suffix = "confmatched" if record["confidence_control"] else "native"
    return root / f"seed_{record['reference_seed']}" / f"age_{record['router_age_percent']:03d}pct_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Router-age recovery experiment for fixed pruned experts.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY)
    parser.add_argument("--recovery-steps", type=int, default=None)
    args = parser.parse_args()
    result = run_router_age_recovery(args.configs, args.output_dir, args.sparsity, args.recovery_steps)
    print(json.dumps({"output_dir": result["output_dir"], "num_records": len(result["records"])}, indent=2))


if __name__ == "__main__":
    main()
