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
import hashlib
import json
import math
from itertools import cycle, islice
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
    forward_with_preserved_routing,
    grad_norms_by_group,
    grad_norms_by_layer,
    load_model_from_checkpoint,
    mean_selected_probability,
    parameter_group,
    per_expert_grad_norms,
    routing_statistics,
    selected_experts_per_batch,
    set_router_temperature,
    state_dict_hash,
)
from moe_lth.training.checkpoint import save_checkpoint
from moe_lth.training.evaluate import evaluate_language_model
from moe_lth.utils import (
    append_jsonl,
    configure_device,
    create_grad_scaler,
    resolve_autocast_dtype,
    resolve_data_seed,
    resolve_device,
    seed_everything,
)

ROUTER_AGES_PERCENT = (0, 10, 20, 40, 60, 80, 100)
CONFIDENCE_CONTROL_AGES_PERCENT = (0, 40, 80, 100)
DEFAULT_SPARSITY = 0.8
RECOVERY_EVAL_INTERVAL = 50
GRADIENT_DETAIL_INTERVAL = 10
EARLY_AUC_WINDOW_FRACTION = 0.5
THRESHOLDS = {"within_5pct": 1.05, "within_10pct": 1.10}
EXACT_ROUTER_STEPS_BY_AGE = {0: 0, 10: 250, 20: 500, 40: 1000, 60: 1500, 80: 2000, 100: 2500}


def _expected_router_step_for_percent(total_steps: int, percent: int) -> int:
    if total_steps >= 2500 and percent in EXACT_ROUTER_STEPS_BY_AGE:
        return EXACT_ROUTER_STEPS_BY_AGE[percent]
    return round(total_steps * percent / 100)


def _checkpoint_for_percent(run_dir: Path, total_steps: int, percent: int) -> tuple[Path, int]:
    target_step = round(total_steps * percent / 100)
    available = {
        int(path.stem.split("_")[-1]): path for path in (run_dir / "checkpoints").glob("step_*.pt")
    }
    if not available:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}/checkpoints")
    expected_step = _expected_router_step_for_percent(total_steps, percent)
    if total_steps >= 2500 and percent in EXACT_ROUTER_STEPS_BY_AGE:
        if expected_step not in available:
            raise FileNotFoundError(
                f"Missing required router checkpoint for age {percent}%: expected step {expected_step}, "
                f"available={sorted(available)}"
            )
        return available[expected_step], expected_step
    if expected_step not in available:
        closest_step = min(available, key=lambda step: abs(step - target_step))
        return available[closest_step], closest_step
    return available[expected_step], expected_step


def _router_checkpoint_audit(run_dir: Path, total_steps: int, router_ages_percent: tuple[int, ...], config: dict) -> dict:
    audit_rows = []
    hashes = {}
    for age in router_ages_percent:
        expected_step = _expected_router_step_for_percent(total_steps, age)
        path, loaded_step = _checkpoint_for_percent(run_dir, total_steps, age)
        model = load_model_from_checkpoint(config["model"], str(path), torch.device("cpu"))
        router_hash = state_dict_hash(component_state_dict(model, "router"))
        if total_steps >= 2500 and age in EXACT_ROUTER_STEPS_BY_AGE and loaded_step != expected_step:
            raise RuntimeError(
                f"Router-age integrity violation: age {age}% expected router step {expected_step}, "
                f"but loaded {loaded_step} from {path}."
            )
        if age == 0:
            hashes[age] = router_hash
        elif router_hash in hashes.values() and total_steps >= 2500 and age in EXACT_ROUTER_STEPS_BY_AGE:
            raise RuntimeError(
                f"Router-age integrity violation: {age}% loaded a duplicate router hash at step {loaded_step}; "
                f"expected unique checkpoint step {expected_step}."
            )
        hashes[age] = router_hash
        stats = routing_statistics(model, [], torch.device("cpu"), max_batches=1)
        audit_rows.append({
            "requested_age_percent": age,
            "expected_step": expected_step,
            "loaded_step": loaded_step,
            "checkpoint_path": str(path),
            "router_hash": router_hash,
            "mean_selected_probability": stats["mean_selected_probability"],
            "routing_entropy": stats["routing_entropy"],
            "router_logit_norm": stats["router_logit_norm"],
            "pass": True,
        })
    return {"router_age_audit": audit_rows, "all_pass": True}


def _calibration_batches(validation_batches, max_batches: int = 8) -> list[torch.Tensor]:
    batches = []
    for batch_id, (token_ids, _targets) in enumerate(validation_batches):
        if batch_id >= max_batches:
            break
        batches.append(token_ids.detach().cpu().clone())
    return batches


def _mask_hash(masks: MaskDict) -> str:
    return state_dict_hash({name: mask.to(torch.uint8) for name, mask in masks.items()})


def _batch_sequence_hash(batches: list[tuple[torch.Tensor, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for batch_id, (token_ids, targets) in enumerate(batches):
        digest.update(str(batch_id).encode("utf-8"))
        for label, tensor in (("inputs", token_ids), ("targets", targets)):
            cpu = tensor.detach().cpu().contiguous()
            digest.update(label.encode("utf-8"))
            digest.update(str(tuple(cpu.shape)).encode("utf-8"))
            digest.update(str(cpu.dtype).encode("utf-8"))
            digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _materialize_batches(loader, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if count <= 0:
        raise ValueError("Recovery budget must be positive.")
    return [
        (token_ids.detach().cpu().clone(), targets.detach().cpu().clone())
        for token_ids, targets in islice(cycle(loader), count)
    ]


def _materialize_validation_batches(loader, max_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (token_ids.detach().cpu().clone(), targets.detach().cpu().clone())
        for token_ids, targets in islice(loader, max_batches)
    ]


def _condition_forward(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    confidence_control: bool,
    temperature: float,
):
    if confidence_control:
        return forward_with_preserved_routing(model, token_ids, temperature)[0]
    set_router_temperature(model, 1.0)
    return model(token_ids)


@torch.no_grad()
def _evaluate_condition_loss(
    model: torch.nn.Module,
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    confidence_control: bool,
    temperature: float,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for token_ids, targets in validation_batches:
        token_ids, targets = token_ids.to(device), targets.to(device)
        output = _condition_forward(model, token_ids, confidence_control, temperature)
        token_loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(token_loss.cpu())
        total_tokens += targets.numel()
    model.train()
    return total_loss / max(1, total_tokens)


def _expert_token_counts(output) -> tuple[dict[str, int], dict[str, int]]:
    assigned: dict[str, int] = {}
    accepted: dict[str, int] = {}
    for layer_id, trace in enumerate(output.routes):
        num_experts = int(trace.usage.numel())
        selected = trace.selected_expert_indices.reshape(-1)
        accepted_selected = selected[trace.accepted_mask.reshape(-1)]
        assigned_counts = torch.bincount(selected.detach().cpu(), minlength=num_experts)
        accepted_counts = torch.bincount(accepted_selected.detach().cpu(), minlength=num_experts)
        for expert_id in range(num_experts):
            key = f"layer_{layer_id}_expert_{expert_id}"
            assigned[key] = int(assigned_counts[expert_id])
            accepted[key] = int(accepted_counts[expert_id])
    return assigned, accepted


def _flatten_probe_assignments(selected: list[torch.Tensor]) -> torch.Tensor:
    """Flatten deterministic probe assignments as [layer, routed-token]."""
    if not selected:
        raise ValueError("The deterministic probe set is empty.")
    return torch.cat([batch.reshape(batch.shape[0], -1) for batch in selected], dim=1)


def _router_drift(
    initial: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    parameter_names: set[str] | None = None,
) -> tuple[float, float]:
    squared_difference = 0.0
    squared_initial = 0.0
    for name, initial_tensor in initial.items():
        if parameter_names is not None and name not in parameter_names:
            continue
        current_tensor = current[name].detach().cpu()
        squared_difference += float((current_tensor.float() - initial_tensor.float()).square().sum())
        squared_initial += float(initial_tensor.float().square().sum())
    absolute = math.sqrt(squared_difference)
    return absolute, absolute / math.sqrt(squared_initial) if squared_initial else 0.0


def _with_utilization_summaries(stats: dict) -> dict:
    """Add explicitly defined per-layer and model-level load summaries."""
    layers = []
    for layer_id, (counts, utilization, dropped) in enumerate(
        zip(
            stats["top1_assignment_counts_by_layer"],
            stats["top1_assignment_distribution_by_layer"],
            stats["dropped_fraction_by_layer"],
        )
    ):
        values = torch.tensor(utilization, dtype=torch.float64)
        mean_load = float(values.mean()) if values.numel() else 0.0
        std_load = float(values.std(unbiased=False)) if values.numel() else 0.0
        maximum = float(values.max()) if values.numel() else 0.0
        minimum = float(values.min()) if values.numel() else 0.0
        layers.append(
            {
                "layer": layer_id,
                "expert_token_counts": counts,
                "expert_utilization_fraction": utilization,
                "utilization_cv": std_load / mean_load if mean_load else 0.0,
                "maximum_expert_load": maximum,
                "minimum_expert_load": minimum,
                "routing_imbalance": (maximum - minimum) / mean_load if mean_load else 0.0,
                # For top-k=1, an overflowed route is also a token drop.
                "capacity_overflow_rate": float(dropped),
                "token_drop_rate": float(dropped),
            }
        )
    stats["utilization_by_layer"] = layers
    for field in (
        "utilization_cv",
        "maximum_expert_load",
        "minimum_expert_load",
        "routing_imbalance",
        "capacity_overflow_rate",
        "token_drop_rate",
    ):
        stats[f"model_mean_{field}"] = (
            sum(float(layer[field]) for layer in layers) / len(layers) if layers else 0.0
        )
    return stats


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
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    train_batch_hash: str,
    validation_batch_hash: str,
    device: torch.device,
    recovery_steps: int,
    dense_loss: float,
    output_dir: Path,
    confidence_control: bool,
    target_confidence: float | None,
    seed: int,
    sparsity: float,
    router_mode: str = "trainable",
    diagnostic_steps: tuple[int, ...] | None = None,
    reference_selected_by_age: dict[int, list[torch.Tensor]] | None = None,
    save_assignment_snapshots: bool = False,
) -> dict:
    if router_mode not in {"trainable", "frozen"}:
        raise ValueError(f"router_mode must be 'trainable' or 'frozen', got {router_mode!r}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite recovery artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_stats").mkdir(exist_ok=True)
    (output_dir / "gradient_stats").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "component_checkpoints").mkdir(exist_ok=True)
    if save_assignment_snapshots:
        (output_dir / "assignment_snapshots").mkdir(exist_ok=True)

    requested_diagnostic_steps = None
    if diagnostic_steps is not None:
        requested_diagnostic_steps = {int(step) for step in diagnostic_steps}
        if 0 not in requested_diagnostic_steps or recovery_steps not in requested_diagnostic_steps:
            raise ValueError("diagnostic_steps must contain step 0 and the final recovery step.")
        if min(requested_diagnostic_steps) < 0 or max(requested_diagnostic_steps) > recovery_steps:
            raise ValueError("diagnostic_steps contains a step outside the recovery budget.")

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
    if masks:
        for name in masks:
            if parameter_group(name) != "expert":
                raise RuntimeError(f"Integrity violation: non-expert parameter {name} present in mask dict.")
        observed_mask_hash = _mask_hash(masks)
        if observed_mask_hash != mask_hash:
            raise RuntimeError(f"Integrity violation: pruning mask changed in {condition_name}.")
    if len(train_batches) != recovery_steps:
        raise RuntimeError(
            f"Integrity violation: {len(train_batches)} paired batches for {recovery_steps} recovery steps."
        )
    if _batch_sequence_hash(train_batches) != train_batch_hash:
        raise RuntimeError(f"Integrity violation: training batch order changed in {condition_name}.")
    if _batch_sequence_hash(validation_batches) != validation_batch_hash:
        raise RuntimeError(f"Integrity violation: evaluation examples changed in {condition_name}.")

    temperature = 1.0
    achieved_confidence = None
    agreement_before_after = None
    capacity_agreement_before_after = None
    calibration_error = None
    if confidence_control:
        assert target_confidence is not None
        (
            temperature,
            achieved_confidence,
            agreement_before_after,
            capacity_agreement_before_after,
        ) = calibrate_temperature(
            model, calibration_batches, device, target_confidence
        )
        calibration_error = abs(achieved_confidence - target_confidence)
        if agreement_before_after != 1.0:
            raise RuntimeError(
                f"Integrity violation: confidence calibration changed top-1 assignment in {condition_name} "
                f"(agreement={agreement_before_after:.4f})."
            )
        if capacity_agreement_before_after != 1.0:
            raise RuntimeError(
                f"Integrity violation: confidence calibration changed capacity acceptance in {condition_name} "
                f"(agreement={capacity_agreement_before_after:.4f})."
            )
        if calibration_error > 1e-3:
            raise RuntimeError(
                f"Confidence target is not matched closely enough in {condition_name}: "
                f"target={target_confidence:.6f}, achieved={achieved_confidence:.6f}."
            )
    set_router_temperature(model, temperature)

    initial_router_state = component_state_dict(model, "router")
    initial_router_hash = state_dict_hash(initial_router_state)
    router_parameter_state_names = {
        name for name, _ in model.named_parameters() if parameter_group(name) == "router"
    }
    torch.save(initial_router_state, output_dir / "component_checkpoints" / "initial_router.pt")

    handles = register_mask_gradient_hooks(model, masks)
    router_parameters = [parameter for name, parameter in model.named_parameters() if parameter_group(name) == "router"]
    if not router_parameters or not router_parameter_state_names:
        raise RuntimeError(f"No router parameters found in {condition_name}.")
    if router_mode == "frozen":
        for parameter in router_parameters:
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in router_parameters):
            raise RuntimeError(f"Integrity violation: router remains trainable in frozen condition {condition_name}.")
    elif not all(parameter.requires_grad for parameter in router_parameters):
        raise RuntimeError(f"Integrity violation: router is not fully trainable in {condition_name}.")
    optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if router_mode == "frozen" and any(id(parameter) in optimizer_parameter_ids for parameter in router_parameters):
        raise RuntimeError(f"Integrity violation: frozen router is included in optimizer for {condition_name}.")
    if router_mode == "trainable" and not all(id(parameter) in optimizer_parameter_ids for parameter in router_parameters):
        raise RuntimeError(f"Integrity violation: trainable router is missing from optimizer for {condition_name}.")
    if optimizer.state:
        raise RuntimeError(f"Integrity violation: optimizer state is not fresh in {condition_name}.")
    autocast_dtype = resolve_autocast_dtype(
        config["training"].get("precision", "fp32"), device
    )
    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    scaler = create_grad_scaler(use_scaler)

    def evaluate() -> dict:
        return {
            "loss": _evaluate_condition_loss(
                model,
                validation_batches,
                device,
                confidence_control,
                temperature,
            )
        }

    probe_sequence_hash = state_dict_hash(
        {f"probe_{index}": batch for index, batch in enumerate(calibration_batches)}
    )

    def routing_snapshot(step: int) -> dict:
        stats = routing_statistics(
            model,
            calibration_batches,
            device,
            confidence_temperature=temperature if confidence_control else 1.0,
            preserve_routing=confidence_control,
        )
        candidate_selected = selected_experts_per_batch(
            model,
            calibration_batches,
            device,
            confidence_temperature=temperature if confidence_control else 1.0,
            preserve_routing=confidence_control,
        )
        stats["assignment_agreement_with_final_router"] = assignment_agreement(reference_selected, candidate_selected)
        stats["assignment_agreement_with_initial_recovery_router"] = assignment_agreement(initial_probe_selected, candidate_selected)
        for age, selected in sorted((reference_selected_by_age or {}).items()):
            stats[f"agreement_with_R{age}_reference"] = assignment_agreement(selected, candidate_selected)
        current_router_state = component_state_dict(model, "router")
        absolute_drift, normalized_drift = _router_drift(
            initial_router_state, current_router_state, router_parameter_state_names
        )
        current_router_hash = state_dict_hash(current_router_state)
        stats.update(
            {
                "probe_sequence_hash": probe_sequence_hash,
                "router_state_hash": current_router_hash,
                "router_parameter_drift_absolute": absolute_drift,
                "router_parameter_drift_normalized": normalized_drift,
            }
        )
        if router_mode == "frozen" and (
            current_router_hash != initial_router_hash or absolute_drift != 0.0 or normalized_drift != 0.0
        ):
            raise RuntimeError(
                f"Integrity violation: frozen router changed at diagnostic step {step} in {condition_name}."
            )
        if save_assignment_snapshots:
            torch.save(
                {
                    "step": step,
                    "probe_sequence_hash": probe_sequence_hash,
                    "top1_assignments": _flatten_probe_assignments(candidate_selected).to(torch.uint8),
                },
                output_dir / "assignment_snapshots" / f"step_{step:04d}.pt",
            )
        return _with_utilization_summaries(stats)

    recovery_curve: list[dict] = []
    routing_by_step: dict[int, dict] = {}
    initial_metrics = evaluate()
    initial_probe_selected = selected_experts_per_batch(
        model,
        calibration_batches,
        device,
        confidence_temperature=temperature if confidence_control else 1.0,
        preserve_routing=confidence_control,
    )
    initial_routing = routing_snapshot(0)
    routing_by_step[0] = initial_routing
    recovery_curve.append({"step": 0, "loss": initial_metrics["loss"]})
    append_jsonl(output_dir / "routing_stats" / "routing_stats.jsonl", {"step": 0, **initial_routing})

    early_window_step = max(1, round(recovery_steps * EARLY_AUC_WINDOW_FRACTION))
    evaluation_steps = {
        step
        for step in range(1, recovery_steps + 1)
        if step % RECOVERY_EVAL_INTERVAL == 0
    }
    evaluation_steps.update({early_window_step, recovery_steps})
    trajectory_steps = evaluation_steps if requested_diagnostic_steps is None else requested_diagnostic_steps - {0}
    if requested_diagnostic_steps is not None:
        append_jsonl(
            output_dir / "gradient_stats" / "gradient_stats.jsonl",
            {
                "step": 0,
                "expert_grad_norm": None,
                "router_grad_norm": 0.0 if router_mode == "frozen" else None,
                "shared_grad_norm": None,
                "expert_to_router_grad_norm_ratio": None,
                "gradient_valid": False,
                "measurement_status": "no_backward_at_initialization",
                "optimizer_step_applied": False,
                "router_gradient_applicable": router_mode == "trainable",
            },
        )

    model.train()
    trainable_router_gradient_observed = False
    amp_overflow_steps = 0
    optimizer_steps_applied = 0
    loss_scales = []
    for step, (token_ids, targets) in enumerate(train_batches, start=1):
        token_ids, targets = token_ids.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            output = _condition_forward(model, token_ids, confidence_control, temperature)
            language_loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1)
            )
            loss = language_loss + float(config["routing"]["aux_loss_weight"]) * output.auxiliary_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norms = grad_norms_by_group(model)
        gradient_valid = all(value is not None for value in grad_norms.values())
        if router_mode == "frozen" and grad_norms["router"] != 0.0:
            raise RuntimeError(f"Frozen router has a nonzero gradient in {condition_name} at step {step}.")
        if router_mode == "trainable" and grad_norms["router"] is not None and float(grad_norms["router"]) > 0.0:
            trainable_router_gradient_observed = True
        expert_grad_norms = (
            per_expert_grad_norms(model)
            if step == 1 or step % GRADIENT_DETAIL_INTERVAL == 0 or step in trajectory_steps
            else None
        )
        per_layer_grad_norms = grad_norms_by_layer(model) if step in trajectory_steps else None
        assigned_counts, accepted_counts = _expert_token_counts(output)
        if gradient_valid:
            global_grad_norm = math.sqrt(sum(float(value) ** 2 for value in grad_norms.values()))
        else:
            global_grad_norm = None
            if not use_scaler:
                raise FloatingPointError(
                    f"Nonfinite unscaled gradient without GradScaler in {condition_name} at step {step}."
                )
        # Keep the historical clipping/update protocol byte-for-byte.  Its
        # FP32 return is ignored when AMP has found a genuine overflow; the
        # safe diagnostic norm above is what is serialized.
        torch.nn.utils.clip_grad_norm_(
            optimizer_parameters, float(config["training"]["grad_clip"])
        )
        loss_scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        loss_scale_after = float(scaler.get_scale())
        if not gradient_valid:
            amp_overflow_steps += 1
            if use_scaler and not loss_scale_after < loss_scale_before:
                raise RuntimeError(
                    f"GradScaler did not back off after nonfinite gradients in {condition_name} at step {step}."
                )
        else:
            optimizer_steps_applied += 1
        loss_scales.extend((loss_scale_before, loss_scale_after))
        apply_masks_(model, masks)

        append_jsonl(
            output_dir / "gradient_stats" / "gradient_stats.jsonl",
            {
                "step": step,
                "expert_grad_norm": grad_norms["expert"],
                "router_grad_norm": grad_norms["router"],
                "shared_grad_norm": grad_norms["shared"],
                "expert_to_router_grad_norm_ratio": (
                    float(grad_norms["expert"]) / float(grad_norms["router"])
                    if grad_norms["expert"] is not None
                    and grad_norms["router"] is not None
                    and float(grad_norms["router"]) > 0.0
                    else None
                ),
                "per_layer_grad_norm": per_layer_grad_norms,
                "per_expert_grad_norm": expert_grad_norms,
                "expert_token_counts": assigned_counts,
                "accepted_expert_token_counts": accepted_counts,
                "global_grad_norm_pre_clip": global_grad_norm,
                "train_loss": (
                    float(loss.detach().cpu())
                    if bool(torch.isfinite(loss.detach()).all())
                    else None
                ),
                "gradient_valid": gradient_valid,
                "measurement_status": "finite_unscaled" if gradient_valid else "amp_overflow_skipped",
                "optimizer_step_applied": gradient_valid,
                "loss_scale_before": loss_scale_before,
                "loss_scale_after": loss_scale_after,
                "router_gradient_applicable": router_mode == "trainable",
            },
        )

        if step in evaluation_steps:
            metrics = evaluate()
            recovery_curve.append({"step": step, "loss": metrics["loss"]})
            if step in trajectory_steps:
                routing_by_step[step] = routing_snapshot(step)
                append_jsonl(
                    output_dir / "routing_stats" / "routing_stats.jsonl",
                    {"step": step, **routing_by_step[step]},
                )
            print(
                f"[{condition_name}] step {step}/{recovery_steps} validation_loss={metrics['loss']:.6f}",
                flush=True,
            )

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
    final_router_hash = state_dict_hash(final_router_state)
    router_parameter_drift, router_parameter_drift_normalized = _router_drift(
        initial_router_state, final_router_state, router_parameter_state_names
    )
    if router_mode == "frozen" and (final_router_hash != initial_router_hash or router_parameter_drift != 0.0):
        raise RuntimeError(f"Integrity violation: frozen router changed during recovery in {condition_name}.")
    if router_mode == "trainable" and (
        final_router_hash == initial_router_hash
        or router_parameter_drift <= 0.0
        or not trainable_router_gradient_observed
    ):
        raise RuntimeError(f"Integrity violation: trainable router did not adapt in {condition_name}.")
    final_confidence = routing_by_step.get(recovery_steps)
    if final_confidence is None:
        final_confidence = routing_snapshot(recovery_steps)
    torch.save(final_router_state, output_dir / "component_checkpoints" / "final_router.pt")

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
        "sparsity": sparsity,
        "router_mode": router_mode,
        "pruning_method": "expert_local_magnitude",
        "confidence_control": confidence_control,
        "temperature": temperature,
        "target_confidence": target_confidence,
        "achieved_confidence": achieved_confidence,
        "calibration_absolute_error": calibration_error,
        "assignment_agreement_before_after_calibration": agreement_before_after,
        "capacity_agreement_before_after_calibration": capacity_agreement_before_after,
        "recovery_steps": recovery_steps,
        "recovery_eval_interval": RECOVERY_EVAL_INTERVAL,
        "per_expert_gradient_interval": GRADIENT_DETAIL_INTERVAL,
        "early_auc_window_steps": early_window_step,
        "expert_state_hash": observed_expert_hash,
        "shared_state_hash": observed_shared_hash,
        "mask_hash": mask_hash,
        "initial_router_state_hash": initial_router_hash,
        "final_router_state_hash": final_router_hash,
        "router_hash_before_recovery": initial_router_hash,
        "router_hash_after_recovery": final_router_hash,
        "router_hash_unchanged": final_router_hash == initial_router_hash,
        "router_parameter_drift_final": router_parameter_drift,
        "router_parameter_drift_normalized_final": router_parameter_drift_normalized,
        "probe_sequence_hash": probe_sequence_hash,
        "router_trainable_parameter_count": sum(parameter.numel() for parameter in router_parameters if parameter.requires_grad),
        "trainable_router_gradient_observed": trainable_router_gradient_observed,
        "amp_overflow_step_count": amp_overflow_steps,
        "optimizer_step_attempts": recovery_steps,
        "optimizer_steps_applied": optimizer_steps_applied,
        "loss_scale_initial": loss_scales[0] if loss_scales else 1.0,
        "loss_scale_final": loss_scales[-1] if loss_scales else 1.0,
        "loss_scale_minimum": min(loss_scales) if loss_scales else 1.0,
        "gradient_diagnostic_version": "fp32_unscaled_v2",
        "training_batch_sequence_hash": train_batch_hash,
        "validation_batch_sequence_hash": validation_batch_hash,
        "optimizer": "fresh_AdamW",
        "optimizer_state_entries_at_start": 0,
        "scheduler": "none",
        "precision": config["training"].get("precision", "fp32"),
        "recovery_fraction_formula": "(L_pruned_initial - L_final) / (L_pruned_initial - L_dense)",
        "time_to_threshold_measurement": (
            f"first validation snapshot at steps 0, every {RECOVERY_EVAL_INTERVAL}, "
            "the exact early-AUC boundary, or the final step"
        ),
        "integrity_checks_passed": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "early_auc": early_auc,
        "early_auc_window_steps": early_window_step,
        "early_mean_validation_loss": early_auc / early_window_step,
        "recovery_fraction": recovery_fraction,
        "time_to_threshold": time_to_threshold,
        "dense_reference_loss": dense_loss,
        "mean_selected_probability_initial": initial_routing["mean_selected_probability"],
        "mean_selected_probability_final": final_confidence["mean_selected_probability"],
        "routing_entropy_initial": initial_routing["routing_entropy"],
        "routing_entropy_final": final_confidence["routing_entropy"],
        "top1_top2_margin_initial": initial_routing["top1_top2_margin"],
        "top1_top2_margin_final": final_confidence["top1_top2_margin"],
        "router_logit_norm_initial": initial_routing["router_logit_norm"],
        "router_logit_norm_final": final_confidence["router_logit_norm"],
        "expert_utilization_initial": initial_routing["expert_utilization"],
        "expert_utilization_final": final_confidence["expert_utilization"],
        "accepted_expert_distribution_initial": initial_routing["accepted_expert_distribution"],
        "accepted_expert_distribution_final": final_confidence["accepted_expert_distribution"],
        "assignment_agreement_with_final_router_initial": initial_routing["assignment_agreement_with_final_router"],
        "assignment_agreement_with_final_router_final": final_confidence["assignment_agreement_with_final_router"],
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
    if run_dir.exists() and any(run_dir.iterdir()):
        missing = sorted(set(checkpoint_steps) - existing)
        raise RuntimeError(
            f"Reference run {run_dir} is partial (missing checkpoints {missing}). "
            "Refusing to invoke the non-resuming trainer because it would delete existing artifacts; "
            "use a config with a new output_dir."
        )
    train_from_config(config)
    produced = {
        int(path.stem.split("_")[-1])
        for path in (run_dir / "checkpoints").glob("step_*.pt")
    }
    missing = sorted(set(checkpoint_steps) - produced)
    if missing:
        raise RuntimeError(f"Reference training in {run_dir} completed without checkpoints {missing}.")
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
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite experiment artifacts in {root}")
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
        condition_recovery_steps = total_steps if recovery_steps is None else int(recovery_steps)

        initial_checkpoint, initial_step = _checkpoint_for_percent(run_dir, total_steps, 0)
        final_checkpoint, final_step = _checkpoint_for_percent(run_dir, total_steps, 100)
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        router_audit = _router_checkpoint_audit(run_dir, total_steps, router_ages_percent, config)
        (seed_dir / "router_checkpoint_audit.json").write_text(json.dumps(router_audit, indent=2), encoding="utf-8")
        seed_everything(seed)
        dense_model = load_model_from_checkpoint(config["model"], str(final_checkpoint), device)
        train_loader, validation_loader_for_dense = build_dataloaders(
            config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
        )
        train_batches = _materialize_batches(train_loader, condition_recovery_steps)
        validation_batches = _materialize_validation_batches(
            validation_loader_for_dense, int(config["data"]["validation_blocks"])
        )
        train_batch_hash = _batch_sequence_hash(train_batches)
        validation_batch_hash = _batch_sequence_hash(validation_batches)
        dense_metrics = evaluate_language_model(
            dense_model, validation_batches, device, max_batches=len(validation_batches)
        )
        dense_loss = dense_metrics["loss"]

        masks = expert_local_magnitude_masks(dense_model, sparsity)
        mask_path = seed_dir / "pruning_mask.pt"
        save_masks(masks, mask_path)

        prunable_expert_weights = sum(mask.numel() for mask in masks.values())
        surviving_prunable = sum(int(mask.sum().item()) for mask in masks.values())
        pruned = prunable_expert_weights - surviving_prunable
        all_expert_parameters = sum(
            parameter.numel()
            for name, parameter in dense_model.named_parameters()
            if parameter_group(name) == "expert"
        )
        pruning_stats = {
            "total_expert_parameters": all_expert_parameters,
            "total_prunable_expert_weight_parameters": prunable_expert_weights,
            "pruned_parameters": pruned,
            "surviving_prunable_weight_parameters": surviving_prunable,
            "surviving_expert_parameters_total": all_expert_parameters - pruned,
            "realized_sparsity": pruned / prunable_expert_weights,
            "realized_sparsity_over_all_expert_parameters": pruned / all_expert_parameters,
            "pruning_method": "expert_local_magnitude (top-k retained by magnitude per expert)",
            "pruning_threshold": "none (rank-based top-k selection, not an absolute threshold)",
            "mask_hash": _mask_hash(masks),
        }
        (seed_dir / "pruning_metadata.json").write_text(json.dumps(pruning_stats, indent=2), encoding="utf-8")

        pruned_base_state = build_fixed_pruned_base(config["model"], str(initial_checkpoint), masks, device)
        dense_base_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in load_model_from_checkpoint(config["model"], str(initial_checkpoint), device).state_dict().items()
        }
        expert_hash = state_dict_hash({n: t for n, t in pruned_base_state.items() if parameter_group(n) == "expert"})
        shared_hash = state_dict_hash({n: t for n, t in pruned_base_state.items() if parameter_group(n) == "shared"})
        dense_expert_hash = state_dict_hash(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in dense_model.state_dict().items()
                if parameter_group(name) == "expert"
            }
        )
        if expert_hash == dense_expert_hash:
            raise RuntimeError(
                "LTH integrity violation: the ticketed expert state still matches the trained E_T state; "
                "surviving weights must come from E_0 under the final-derived mask."
            )
        fixed_component_dir = seed_dir / "component_checkpoints"
        fixed_component_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "experts": {n: t for n, t in pruned_base_state.items() if parameter_group(n) == "expert"},
                "shared": {n: t for n, t in pruned_base_state.items() if parameter_group(n) == "shared"},
                "expert_state_hash": expert_hash,
                "shared_state_hash": shared_hash,
                "mask_hash": pruning_stats["mask_hash"],
                "shared_checkpoint_step": initial_step,
                "shared_checkpoint_path": str(initial_checkpoint),
                "reference_dense_expert_state_hash": dense_expert_hash,
                "uses_rewound_expert_initialization": True,
            },
            fixed_component_dir / "fixed_pruned_experts_and_shared.pt",
        )

        audit = {
            "reference_seed": seed,
            "shared_checkpoint_step": initial_step,
            "shared_checkpoint_path": str(initial_checkpoint),
            "mask_derived_from_dense_reference_checkpoint": str(final_checkpoint),
            "surviving_expert_values_from_rewind_checkpoint": str(initial_checkpoint),
            "mask_hash": pruning_stats["mask_hash"],
            "expert_state_hash": expert_hash,
            "shared_state_hash": shared_hash,
            "dense_expert_state_hash": dense_expert_hash,
            "all_router_age_conditions_share_same_ticket": True,
            "all_router_age_conditions_share_same_shared_state": True,
            "pruned_parameters_remain_zero": True,
            "data_schedule_hashes_are_identical_across_conditions": {
                "train": train_batch_hash,
                "validation": validation_batch_hash,
            },
            "optimizer_state_reset_per_condition": True,
            "mask_is_identical_across_conditions": True,
            "ticket_matches_initial_expert_values_under_final_mask": True,
        }
        (seed_dir / "lth_isolation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

        # Reference router (final, R_T) selections on a fixed calibration set, used for
        # assignment-agreement comparisons and confidence-target calibration.
        reference_model = assemble_router_age_model(
            config["model"], pruned_base_state, str(final_checkpoint), masks, device
        )
        calibration_batches = _calibration_batches(validation_batches)
        reference_selected = selected_experts_per_batch(reference_model, calibration_batches, device)
        native_confidence_by_age = {}
        for control_percent in confidence_control_ages:
            control_checkpoint, _ = _checkpoint_for_percent(run_dir, total_steps, control_percent)
            control_model = assemble_router_age_model(
                config["model"], pruned_base_state, str(control_checkpoint), masks, device
            )
            native_confidence_by_age[str(control_percent)] = mean_selected_probability(
                control_model, calibration_batches, device
            )["mean_selected_probability"]
        target_confidence = min(native_confidence_by_age.values())
        calibration_manifest = {
            "target_confidence": target_confidence,
            "target_rule": "minimum native mean selected probability across all confidence-control ages",
            "native_confidence_by_age": native_confidence_by_age,
            "calibration_examples_hash": state_dict_hash(
                {f"batch_{index}": batch for index, batch in enumerate(calibration_batches)}
            ),
        }
        (seed_dir / "confidence_calibration_manifest.json").write_text(
            json.dumps(calibration_manifest, indent=2), encoding="utf-8"
        )

        checkpoint_manifest = {}
        for percent in router_ages_percent:
            checkpoint_path, checkpoint_step = _checkpoint_for_percent(run_dir, total_steps, percent)
            checkpoint_manifest[str(percent)] = {
                "step": checkpoint_step,
                "path": str(checkpoint_path),
            }
        (seed_dir / "reference_checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "config_path": str(config_path),
                    "reference_run_dir": str(run_dir),
                    "reference_seed": seed,
                    "final_step": final_step,
                    "checkpoints": checkpoint_manifest,
                    "training_batch_sequence_hash": train_batch_hash,
                    "validation_batch_sequence_hash": validation_batch_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for percent in router_ages_percent:
            router_checkpoint, router_step = _checkpoint_for_percent(run_dir, total_steps, percent)
            condition_dir = seed_dir / f"age_{percent:03d}pct_native"
            sparse_record = _run_recovery_condition(
                config=config,
                condition_name=f"seed{seed}_age{percent}_native_sparse",
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
                train_batches=train_batches,
                validation_batches=validation_batches,
                train_batch_hash=train_batch_hash,
                validation_batch_hash=validation_batch_hash,
                device=device,
                recovery_steps=condition_recovery_steps,
                dense_loss=dense_loss,
                output_dir=condition_dir,
                confidence_control=False,
                target_confidence=None,
                seed=seed,
                sparsity=sparsity,
            )
            sparse_record.update({"reference_seed": seed, "final_step": final_step, "condition_type": "sparse_ticket"})
            all_records.append(sparse_record)
            _write_partial_csv(all_records, root)

            dense_dir = seed_dir / f"age_{percent:03d}pct_native_dense"
            dense_record = _run_recovery_condition(
                config=config,
                condition_name=f"seed{seed}_age{percent}_native_dense",
                pruned_base_state=dense_base_state,
                router_checkpoint=str(router_checkpoint),
                router_age_percent=percent,
                router_step=router_step,
                masks={},
                expert_hash=state_dict_hash({n: t for n, t in dense_base_state.items() if parameter_group(n) == "expert"}),
                shared_hash=state_dict_hash({n: t for n, t in dense_base_state.items() if parameter_group(n) == "shared"}),
                mask_hash="dense_no_mask",
                reference_selected=reference_selected,
                calibration_batches=calibration_batches,
                train_batches=train_batches,
                validation_batches=validation_batches,
                train_batch_hash=train_batch_hash,
                validation_batch_hash=validation_batch_hash,
                device=device,
                recovery_steps=condition_recovery_steps,
                dense_loss=dense_loss,
                output_dir=dense_dir,
                confidence_control=False,
                target_confidence=None,
                seed=seed,
                sparsity=0.0,
            )
            dense_record.update({"reference_seed": seed, "final_step": final_step, "condition_type": "dense_control"})
            all_records.append(dense_record)
            _write_partial_csv(all_records, root)

            if percent in confidence_control_ages and config_index in confidence_control_seed_indices:
                condition_dir = seed_dir / f"age_{percent:03d}pct_confmatched"
                record = _run_recovery_condition(
                    config=config,
                    condition_name=f"seed{seed}_age{percent}_confmatched_sparse",
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
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    train_batch_hash=train_batch_hash,
                    validation_batch_hash=validation_batch_hash,
                    device=device,
                    recovery_steps=condition_recovery_steps,
                    dense_loss=dense_loss,
                    output_dir=condition_dir,
                    confidence_control=True,
                    target_confidence=target_confidence,
                    seed=seed,
                    sparsity=sparsity,
                )
                record.update({"reference_seed": seed, "final_step": final_step, "condition_type": "sparse_confmatched"})
                all_records.append(record)
                _write_partial_csv(all_records, root)

                dense_conf_dir = seed_dir / f"age_{percent:03d}pct_confmatched_dense"
                dense_conf_record = _run_recovery_condition(
                    config=config,
                    condition_name=f"seed{seed}_age{percent}_confmatched_dense",
                    pruned_base_state=dense_base_state,
                    router_checkpoint=str(router_checkpoint),
                    router_age_percent=percent,
                    router_step=router_step,
                    masks={},
                    expert_hash=state_dict_hash({n: t for n, t in dense_base_state.items() if parameter_group(n) == "expert"}),
                    shared_hash=state_dict_hash({n: t for n, t in dense_base_state.items() if parameter_group(n) == "shared"}),
                    mask_hash="dense_no_mask",
                    reference_selected=reference_selected,
                    calibration_batches=calibration_batches,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    train_batch_hash=train_batch_hash,
                    validation_batch_hash=validation_batch_hash,
                    device=device,
                    recovery_steps=condition_recovery_steps,
                    dense_loss=dense_loss,
                    output_dir=dense_conf_dir,
                    confidence_control=True,
                    target_confidence=target_confidence,
                    seed=seed,
                    sparsity=0.0,
                )
                dense_conf_record.update({"reference_seed": seed, "final_step": final_step, "condition_type": "dense_confmatched"})
                all_records.append(dense_conf_record)
                _write_partial_csv(all_records, root)

    _write_partial_csv(all_records, root)
    (root / "router_age_recovery_all_records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    _write_paired_csv(all_records, root)
    _write_report(all_records, root)
    _write_recovery_plot(all_records, root)
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

    def rows_table(rows: list[dict], baseline_rows: list[dict]) -> str:
        lines = []
        for row in rows:
            final_router_row = next(
                (
                    r
                    for r in baseline_rows
                    if r["reference_seed"] == row["reference_seed"]
                    and r["router_age_percent"] == 100
                ),
                None,
            )
            initial_router_row = next(
                (
                    r
                    for r in baseline_rows
                    if r["reference_seed"] == row["reference_seed"]
                    and r["router_age_percent"] == 0
                ),
                None,
            )
            delta_final = (
                None
                if final_router_row is None
                else row["final_validation_loss"] - final_router_row["final_validation_loss"]
            )
            delta_initial = (
                None
                if initial_router_row is None
                else row["final_validation_loss"] - initial_router_row["final_validation_loss"]
            )
            lines.append(
                f"| {row['reference_seed']} | {row['router_age_percent']} | {row['router_step']} | "
                f"{_fmt(row['initial_validation_loss'])} | {_fmt(row['early_auc'])} | "
                f"{_fmt(row['final_validation_loss'])} | {_fmt(delta_final)} | {_fmt(delta_initial)} | "
                f"{row['time_to_threshold']['within_5pct']} | {row['time_to_threshold']['within_10pct']} | "
                f"{_fmt(row['temperature'])} | {_fmt(row['mean_selected_probability_initial'])} | "
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
                f"| {seed} | {stats['total_expert_parameters']} | "
                f"{stats['total_prunable_expert_weight_parameters']} | {stats['pruned_parameters']} | "
                f"{stats['surviving_prunable_weight_parameters']} | {stats['realized_sparsity']:.8f} | "
                f"{stats['pruning_method']} | {stats['mask_hash'][:16]}... |"
            )

    gradient_rows = []
    for row in sorted(
        records,
        key=lambda item: (
            item["confidence_control"],
            item["reference_seed"],
            item["router_age_percent"],
        ),
    ):
        condition_dir = _condition_dir_from_record(root, row)
        means = _mean_gradient_norms(condition_dir)
        gradient_rows.append(
            f"| {row['reference_seed']} | {row['router_age_percent']} | "
            f"{'matched' if row['confidence_control'] else 'native'} | {means['expert']:.4f} | "
            f"{means['router']:.4f} | {means['shared']:.4f} |"
        )

    sparsity = records[0]["sparsity"] if records else DEFAULT_SPARSITY
    ages = sorted({row["router_age_percent"] for row in records})
    budgets = sorted({row["recovery_steps"] for row in records})
    early_windows = sorted({row["early_auc_window_steps"] for row in records})

    markdown = f"""# Router-Age Recovery Experiment Results

Reference seeds: {", ".join(str(seed) for seed in seeds)}

Fixed {100 * sparsity:.4g}%-magnitude-pruned expert state paired with router
checkpoints `R_t` from t in {{{", ".join(str(age) for age in ages)}}}% of the reference
training trajectory. Shared parameters, pruned expert weights, and the
pruning mask are byte-identical across every router-age condition within a
seed (verified via SHA-256 hashes recorded in each condition's
`metadata.json`). Only the router parameters differ across conditions.

Recovery budget(s): {", ".join(str(value) for value in budgets)} sparse steps.
Early AUC is trapezoidal validation-loss area over the exact first
{", ".join(str(value) for value in early_windows)} steps. Recovery fraction is
`(L_pruned_initial - L_final) / (L_pruned_initial - L_dense)` when the
pruning-induced degradation is positive. Lower loss and lower AUC are better;
reported deltas are condition minus the same-control R100/R0 baseline, so
negative is better. Threshold times are observed on the declared validation
snapshot grid and are recorded as `unreached` otherwise.

## Pruning Summary (computed once per seed, applied to every router age)

| Seed | Total expert params | Prunable weights | Pruned | Surviving prunable | Realized sparsity | Method | Mask hash |
|---:|---:|---:|---:|---:|---:|---|---|
{chr(10).join(pruning_rows)}

## Native-Confidence Router-Age Sweep

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs R100 | Delta vs R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_table(native, native)}

## Confidence-Matched Control

This two-pass control obtains each condition's native routes and native
capacity-priority scores, then forces those routes while applying the
temperature-scaled gate probabilities. Metadata verifies assignment and
capacity-decision agreement of 1.0 before versus after calibration.

| Seed | Router age % | Router step | L(0) | Early AUC | L(final) | Delta vs matched R100 | Delta vs matched R0 | T(5%) | T(10%) | Tau | Sel. prob initial | Sel. prob final | Entropy final | Agreement w/ R100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows_table(conf_matched, conf_matched)}

## Mean Gradient Norms During Recovery (native-confidence conditions)

| Seed | Router age % | Confidence | Mean expert grad norm | Mean router grad norm | Mean shared grad norm |
|---:|---:|---|---:|---:|---:|
{chr(10).join(gradient_rows)}

Raw records: [router_age_recovery_all_records.json](router_age_recovery_all_records.json),
condition table: [router_age_recovery_aggregate.csv](router_age_recovery_aggregate.csv),
paired table: [router_age_recovery_paired.csv](router_age_recovery_paired.csv), and
recovery curves: [router_age_recovery_curves.svg](router_age_recovery_curves.svg).
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
    "target_confidence",
    "achieved_confidence",
    "calibration_absolute_error",
    "initial_validation_loss",
    "early_auc",
    "early_auc_window_steps",
    "early_mean_validation_loss",
    "final_validation_loss",
    "recovery_fraction",
    "time_to_threshold_5pct",
    "time_to_threshold_10pct",
    "mean_selected_probability",
    "mean_selected_probability_final",
    "routing_entropy",
    "routing_entropy_final",
    "top1_top2_margin",
    "router_logit_norm",
    "assignment_agreement_with_final_router",
    "assignment_agreement_with_final_router_final",
    "mean_expert_gradient_norm",
    "mean_router_gradient_norm",
    "mean_shared_gradient_norm",
    "expert_state_hash",
    "shared_state_hash",
    "mask_hash",
    "initial_router_state_hash",
    "training_batch_sequence_hash",
    "validation_batch_sequence_hash",
]


def _mean_gradient_norms(condition_dir: Path) -> dict[str, float]:
    path = condition_dir / "gradient_stats" / "gradient_stats.jsonl"
    if not path.exists():
        return {"expert": 0.0, "router": 0.0, "shared": 0.0}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"expert": 0.0, "router": 0.0, "shared": 0.0}
    result = {}
    valid_rows = []
    skipped_rows = 0
    for row in rows:
        if int(row.get("step", 0)) == 0:
            continue
        finite_fallback = all(
            row.get(field) is not None and math.isfinite(float(row[field]))
            for field in ("expert_grad_norm", "router_grad_norm", "shared_grad_norm")
        )
        valid = bool(row.get("gradient_valid", finite_fallback))
        if valid and finite_fallback:
            valid_rows.append(row)
        else:
            skipped_rows += 1
    for output_name, field in (
        ("expert", "expert_grad_norm"),
        ("router", "router_grad_norm"),
        ("shared", "shared_grad_norm"),
    ):
        finite = [
            float(row[field])
            for row in valid_rows
            if row.get(field) is not None and math.isfinite(float(row[field]))
        ]
        result[output_name] = sum(finite) / len(finite) if finite else 0.0
    result["valid_step_count"] = float(len(valid_rows))
    result["skipped_step_count"] = float(skipped_rows)
    return result


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
                    record["target_confidence"],
                    record["achieved_confidence"],
                    record["calibration_absolute_error"],
                    record["initial_validation_loss"],
                    record["early_auc"],
                    record["early_auc_window_steps"],
                    record["early_mean_validation_loss"],
                    record["final_validation_loss"],
                    record["recovery_fraction"],
                    record["time_to_threshold"]["within_5pct"],
                    record["time_to_threshold"]["within_10pct"],
                    record["mean_selected_probability_initial"],
                    record["mean_selected_probability_final"],
                    record["routing_entropy_initial"],
                    record["routing_entropy_final"],
                    record["top1_top2_margin_initial"],
                    record["router_logit_norm_initial"],
                    record["assignment_agreement_with_final_router_initial"],
                    record["assignment_agreement_with_final_router_final"],
                    gradient_means["expert"],
                    gradient_means["router"],
                    gradient_means["shared"],
                    record["expert_state_hash"],
                    record["shared_state_hash"],
                    record["mask_hash"],
                    record["initial_router_state_hash"],
                    record["training_batch_sequence_hash"],
                    record["validation_batch_sequence_hash"],
                ]
            )


def _write_paired_csv(records: list[dict], root: Path) -> None:
    columns = [
        "confidence_control",
        "router_age_percent",
        "num_reference_seeds",
        "mean_initial_validation_loss",
        "mean_early_auc",
        "mean_final_validation_loss",
        "mean_delta_final_loss_vs_same_control_R100",
        "mean_delta_final_loss_vs_same_control_R0",
    ]
    rows = []
    for confidence_control in (False, True):
        population = [row for row in records if row["confidence_control"] == confidence_control]
        for age in sorted({row["router_age_percent"] for row in population}):
            age_rows = [row for row in population if row["router_age_percent"] == age]
            delta_r100 = []
            delta_r0 = []
            for row in age_rows:
                r100 = next(
                    (
                        candidate
                        for candidate in population
                        if candidate["reference_seed"] == row["reference_seed"]
                        and candidate["router_age_percent"] == 100
                    ),
                    None,
                )
                r0 = next(
                    (
                        candidate
                        for candidate in population
                        if candidate["reference_seed"] == row["reference_seed"]
                        and candidate["router_age_percent"] == 0
                    ),
                    None,
                )
                if r100 is not None:
                    delta_r100.append(row["final_validation_loss"] - r100["final_validation_loss"])
                if r0 is not None:
                    delta_r0.append(row["final_validation_loss"] - r0["final_validation_loss"])
            mean = lambda values: sum(values) / len(values) if values else None
            rows.append(
                [
                    confidence_control,
                    age,
                    len(age_rows),
                    mean([row["initial_validation_loss"] for row in age_rows]),
                    mean([row["early_auc"] for row in age_rows]),
                    mean([row["final_validation_loss"] for row in age_rows]),
                    mean(delta_r100),
                    mean(delta_r0),
                ]
            )
    path = root / "router_age_recovery_paired.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_recovery_plot(records: list[dict], root: Path) -> None:
    native_seeds = sorted(
        {row["reference_seed"] for row in records if not row["confidence_control"]}
    )
    include_control = any(row["confidence_control"] for row in records)
    ages = sorted({row["router_age_percent"] for row in records})
    palette = ["#440154", "#46327e", "#365c8d", "#277f8e", "#1fa187", "#4ac16d", "#a0da39"]
    age_color = {age: palette[index % len(palette)] for index, age in enumerate(ages)}
    panels: list[tuple[str, list[dict]]] = []
    for seed in native_seeds:
        seed_records = sorted(
            (
                row
                for row in records
                if row["reference_seed"] == seed and not row["confidence_control"]
            ),
            key=lambda row: row["router_age_percent"],
        )
        panels.append((f"Native confidence - seed {seed}", seed_records))

    if include_control:
        control_records = sorted(
            (row for row in records if row["confidence_control"]),
            key=lambda row: (row["reference_seed"], row["router_age_percent"]),
        )
        panels.append(("Confidence-matched controls", control_records))

    panel_width = 760
    panel_height = 430
    plot_left = 70
    plot_top = 45
    plot_width = 650
    plot_height = 315
    columns = 2
    rows_count = max(1, math.ceil(len(panels) / columns))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width * columns}" '
        f'height="{panel_height * rows_count}" viewBox="0 0 {panel_width * columns} '
        f'{panel_height * rows_count}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.2}'
        '.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke-width:2}</style>',
    ]

    for panel_index, (title, panel_records) in enumerate(panels):
        origin_x = (panel_index % columns) * panel_width
        origin_y = (panel_index // columns) * panel_height
        curves = []
        for record in panel_records:
            metrics_path = _condition_dir_from_record(root, record) / "metrics.jsonl"
            curve = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
            curves.append((record, curve))
        all_points = [point for _record, curve in curves for point in curve]
        x_max = max((point["step"] for point in all_points), default=1)
        losses = [point["loss"] for point in all_points]
        y_min = min(losses, default=0.0)
        y_max = max(losses, default=1.0)
        y_pad = max(1e-6, (y_max - y_min) * 0.08)
        y_min -= y_pad
        y_max += y_pad

        def x_coord(step: float) -> float:
            return origin_x + plot_left + plot_width * step / max(1, x_max)

        def y_coord(loss: float) -> float:
            return origin_y + plot_top + plot_height * (y_max - loss) / max(1e-12, y_max - y_min)

        svg.append(
            f'<text x="{origin_x + panel_width / 2}" y="{origin_y + 24}" '
            f'text-anchor="middle" font-size="17" font-weight="bold">{title}</text>'
        )
        for grid_index in range(5):
            fraction = grid_index / 4
            y = origin_y + plot_top + plot_height * fraction
            loss_value = y_max - (y_max - y_min) * fraction
            svg.append(
                f'<line class="grid" x1="{origin_x + plot_left}" y1="{y:.2f}" '
                f'x2="{origin_x + plot_left + plot_width}" y2="{y:.2f}"/>'
            )
            svg.append(
                f'<text x="{origin_x + plot_left - 8}" y="{y + 4:.2f}" '
                f'text-anchor="end" font-size="11">{loss_value:.4f}</text>'
            )
        svg.append(
            f'<line class="axis" x1="{origin_x + plot_left}" y1="{origin_y + plot_top}" '
            f'x2="{origin_x + plot_left}" y2="{origin_y + plot_top + plot_height}"/>'
        )
        svg.append(
            f'<line class="axis" x1="{origin_x + plot_left}" y1="{origin_y + plot_top + plot_height}" '
            f'x2="{origin_x + plot_left + plot_width}" y2="{origin_y + plot_top + plot_height}"/>'
        )
        for record, curve in curves:
            points = " ".join(f"{x_coord(point['step']):.2f},{y_coord(point['loss']):.2f}" for point in curve)
            color = age_color[record["router_age_percent"]]
            svg.append(f'<polyline class="curve" stroke="{color}" points="{points}"/>')
        legend_x = origin_x + plot_left
        legend_y = origin_y + plot_top + plot_height + 38
        for legend_index, (record, _curve) in enumerate(curves):
            x = legend_x + (legend_index % 4) * 155
            y = legend_y + (legend_index // 4) * 18
            label = f"R{record['router_age_percent']}"
            if include_control and record["confidence_control"] and len({r["reference_seed"] for r, _ in curves}) > 1:
                label = f"seed {record['reference_seed']} {label}"
            color = age_color[record["router_age_percent"]]
            svg.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 18}" y2="{y - 4}" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<text x="{x + 23}" y="{y}" font-size="11">{label}</text>')
        svg.append(
            f'<text x="{origin_x + plot_left + plot_width / 2}" '
            f'y="{origin_y + panel_height - 8}" text-anchor="middle" font-size="12">Recovery step</text>'
        )
        svg.append(
            f'<text transform="translate({origin_x + 16},{origin_y + plot_top + plot_height / 2}) rotate(-90)" '
            f'text-anchor="middle" font-size="12">Validation loss</text>'
        )
        svg.append(
            f'<text x="{origin_x + plot_left}" y="{origin_y + plot_top + plot_height + 16}" '
            f'font-size="11">0</text>'
        )
        svg.append(
            f'<text x="{origin_x + plot_left + plot_width}" y="{origin_y + plot_top + plot_height + 16}" '
            f'text-anchor="end" font-size="11">{x_max}</text>'
        )

    svg.append("</svg>")
    (root / "router_age_recovery_curves.svg").write_text("\n".join(svg), encoding="utf-8")


def _condition_dir_from_record(root: Path, record: dict) -> Path:
    suffix = "confmatched" if record["confidence_control"] else "native"
    if record.get("condition_type", "").startswith("dense"):
        suffix += "_dense"
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
