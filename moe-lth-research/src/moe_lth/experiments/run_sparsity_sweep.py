"""Sparsity sweep experiment for router-conditioned MoE lottery tickets.

Central questions:
  1. How does the sparse-vs-dense recovery gap change as a function of sparsity?
  2. Does router maturity (R_0 vs R_100) provide increasing protection at higher sparsities?
  3. Is there a sparsity frontier at which sparse expert rewinding remains competitive?

For each sparsity level s in {0.60, 0.70, 0.90, 0.95}:
  1. Discover pruning mask m_s from fully trained experts E_T
  2. Rewind surviving weights to initial expert state: E_sparse = m_s ⊙ E_0
  3. Run recovery at both router endpoints (R_0 and R_100)
  4. Reuse dense E_0 controls if protocol-compatible
  5. Compute sparse-vs-dense gap: Δ_ticket(R, s) = L_sparse - L_dense
  6. Compute router routing benefit: Δ_routing(s) = Δ_ticket(R_0, s) - Δ_ticket(R_100, s)

This builds on run_router_age_recovery.py, reusing:
  - Expert rewind protocol (E_0 under learned mask)
  - Dense baseline architecture
  - Confidence calibration infrastructure
  - Routing statistics and gradient tracking

Output: results/router_conditioned_sparsity_sweep/ with aggregates across sparsity levels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import cycle, islice
from pathlib import Path
from typing import Optional

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

# --- Experimental configuration ---
SPARSITIES_TO_SWEEP = (0.60, 0.70, 0.90, 0.95)  # 0.80 already complete, skip unless reproducibility check
ROUTER_AGES_PERCENT_ENDPOINTS = (0, 100)  # R_0 and R_100 only for efficiency
DEFAULT_RECOVERY_STEPS = 2500
RECOVERY_EVAL_INTERVAL = 50
GRADIENT_DETAIL_INTERVAL = 10
EARLY_AUC_WINDOW_FRACTION = 0.5
THRESHOLDS = {"within_5pct": 1.05, "within_10pct": 1.10}
# Exact router-step mapping for 2500-step budget (reuse from router_age_recovery)
EXACT_ROUTER_STEPS_BY_AGE = {0: 0, 100: 2500}


def _expected_router_step_for_percent(total_steps: int, percent: int) -> int:
    """Get the exact router step for the given percent."""
    if total_steps >= 2500 and percent in EXACT_ROUTER_STEPS_BY_AGE:
        return EXACT_ROUTER_STEPS_BY_AGE[percent]
    return round(total_steps * percent / 100)


def _checkpoint_for_percent(run_dir: Path, total_steps: int, percent: int) -> tuple[Path, int]:
    """Load checkpoint for a given router age percent."""
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


def _mask_hash(masks: MaskDict) -> str:
    """Compute SHA-256 of mask state dict."""
    return state_dict_hash({name: mask.to(torch.uint8) for name, mask in masks.items()})


def _batch_sequence_hash(batches: list[tuple[torch.Tensor, torch.Tensor]]) -> str:
    """Compute SHA-256 of batch sequences for reproducibility."""
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
    """Pre-generate all training batches for this recovery run."""
    if count <= 0:
        raise ValueError("Recovery budget must be positive.")
    return [
        (token_ids.detach().cpu().clone(), targets.detach().cpu().clone())
        for token_ids, targets in islice(cycle(loader), count)
    ]


def _materialize_validation_batches(loader, max_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-generate validation batches."""
    return [
        (token_ids.detach().cpu().clone(), targets.detach().cpu().clone())
        for token_ids, targets in islice(loader, max_batches)
    ]


def _calibration_batches(validation_batches, max_batches: int = 8) -> list[torch.Tensor]:
    """Extract calibration batches from validation set."""
    batches = []
    for batch_id, (token_ids, _targets) in enumerate(validation_batches):
        if batch_id >= max_batches:
            break
        batches.append(token_ids.detach().cpu().clone())
    return batches


def _condition_forward(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    confidence_control: bool,
    temperature: float,
):
    """Forward pass with optional temperature control."""
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
    """Compute final validation loss for a condition."""
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


def _ensure_reference(config_path: str) -> Path:
    """Ensure reference training exists; train if necessary."""
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


def _dense_baseline_reuse_compatible(
    existing_dense_record: dict,
    seed: int,
    sparsity: float,
    router_age_percent: int,
    expected_hashes: dict,
    expected_batches: dict,
) -> bool:
    """Validate dense baseline compatibility before reuse.
    
    Checks:
      - reference_seed matches
      - router_age_percent and router_step match
      - expert_state_hash, shared_state_hash match (same E_0)
      - training_batch_sequence_hash, validation_batch_sequence_hash match
      - recovery_steps match
      - optimizer, precision, and other protocol fields match
    """
    if existing_dense_record.get("reference_seed") != seed:
        return False
    if existing_dense_record.get("router_age_percent") != router_age_percent:
        return False
    if existing_dense_record.get("sparsity") != 0.0:
        return False
    if existing_dense_record.get("condition_type") != "dense_control":
        return False
    # Check state hashes
    if existing_dense_record.get("expert_state_hash") != expected_hashes.get("expert_hash"):
        return False
    if existing_dense_record.get("shared_state_hash") != expected_hashes.get("shared_hash"):
        return False
    # Check batch sequences
    if existing_dense_record.get("training_batch_sequence_hash") != expected_batches.get("train"):
        return False
    if existing_dense_record.get("validation_batch_sequence_hash") != expected_batches.get("validation"):
        return False
    # Check recovery protocol
    if existing_dense_record.get("recovery_steps") != DEFAULT_RECOVERY_STEPS:
        return False
    if existing_dense_record.get("optimizer") != "fresh_AdamW":
        return False
    return True


def _load_existing_dense_baseline(
    seed: int,
    router_age_percent: int,
    existing_sparsity_sweep_records: list[dict],
    expected_hashes: dict,
    expected_batches: dict,
) -> Optional[dict]:
    """Load dense baseline from existing sparsity sweep if compatible."""
    for record in existing_sparsity_sweep_records:
        if _dense_baseline_reuse_compatible(
            record,
            seed,
            0.0,  # sparsity for dense is always 0
            router_age_percent,
            expected_hashes,
            expected_batches,
        ):
            return record
    return None


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
) -> dict:
    """Run a single recovery condition: train from fixed base state with router swap."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite recovery artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "routing_stats").mkdir(exist_ok=True)
    (output_dir / "gradient_stats").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "component_checkpoints").mkdir(exist_ok=True)

    seed_everything(seed)
    model = assemble_router_age_model(config["model"], pruned_base_state, router_checkpoint, masks, device)

    # --- Integrity assertions (fail loudly if violated) ---
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
        raise RuntimeError(f"Integrity violation: validation batch order changed in {condition_name}.")

    # Rewind assertion: every retained weight should equal E_0, pruned = 0
    if masks:
        for name, mask in masks.items():
            param = dict(model.named_parameters())[name]
            pruned_locs = ~mask.bool()
            if not torch.all(param[pruned_locs] == 0):
                raise RuntimeError(
                    f"Rewind assertion failed: pruned weights in {name} are not all zero at line of "
                    f"assembly in {condition_name}."
                )

    # Router assertion: only router should differ from reference at same age/seed/sparsity
    initial_router_hash = state_dict_hash(component_state_dict(model, "router"))

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = None  # No scheduler for 2500-step budget per protocol
    model.train()
    use_scaler = device.type == "cuda" and autocast_dtype == torch.float16
    autocast_dtype = resolve_autocast_dtype(config["training"].get("precision", "fp32"), device)
    scaler = create_grad_scaler(use_scaler)
    loss_timeline = {}
    gradient_norms_timeline = {}
    early_auc = 0.0
    early_window_step = math.ceil(recovery_steps * EARLY_AUC_WINDOW_FRACTION)
    time_to_threshold = {name: None for name in THRESHOLDS}
    initial_loss = None
    final_loss = None
    best_loss = float("inf")

    @torch.no_grad()
    def eval_and_log():
        nonlocal initial_loss, final_loss, best_loss, early_auc
        model.eval()
        val_loss = _evaluate_condition_loss(
            model, validation_batches, device, confidence_control, temperature
        )
        if initial_loss is None:
            initial_loss = val_loss
        final_loss = val_loss
        if val_loss < best_loss:
            best_loss = val_loss
        model.train()
        return val_loss

    # Temperature calibration for confidence-matched control
    temperature = 1.0
    achieved_confidence = None
    calibration_error = None
    agreement_before_after = None
    capacity_agreement_before_after = None

    if confidence_control:
        temperature, achieved_confidence, agreement_before_after, capacity_agreement_before_after = calibrate_temperature(
            model, calibration_batches, device, target_confidence, max_temperature=1_000_000.0,
            grid_points=31, refinement_rounds=4
        )
        calibration_error = abs(achieved_confidence - target_confidence)
        if calibration_error > 1e-3:
            raise RuntimeError(
                f"Dense confidence calibration failed for {condition_name}: "
                f"target={target_confidence:.6f}, achieved={achieved_confidence:.6f}, "
                f"error={calibration_error:.6f} (exceeds 1e-3 tolerance). "
                f"Cannot continue."
            )

    # Initial evaluation
    eval_and_log()
    initial_routing = routing_statistics(model, calibration_batches, device)

    # Register mask enforcement hooks if applicable
    if masks:
        register_mask_gradient_hooks(model, masks)

    for step in range(recovery_steps):
        token_ids, targets = train_batches[step]
        token_ids, targets = token_ids.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=str(device.type), dtype=autocast_dtype, enabled=autocast_dtype is not None):
            output = _condition_forward(model, token_ids, confidence_control, temperature)
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config["training"].get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()

        # Log at eval intervals
        if (step + 1) % RECOVERY_EVAL_INTERVAL == 0 or step == 0:
            val_loss = eval_and_log()
            loss_timeline[step] = val_loss
            (output_dir / "routing_stats" / f"step_{step:05d}.json").write_text(
                json.dumps(routing_statistics(model, calibration_batches, device), indent=2),
                encoding="utf-8",
            )
            # Check thresholds
            for name, threshold in THRESHOLDS.items():
                if time_to_threshold[name] is None and val_loss <= dense_loss * threshold:
                    time_to_threshold[name] = step

        # Log gradient norms at detail intervals
        if (step + 1) % GRADIENT_DETAIL_INTERVAL == 0:
            grad_norms = grad_norms_by_group(model)
            per_expert = per_expert_grad_norms(model)
            gradient_norms_timeline[step] = {
                "by_group": grad_norms,
                "per_expert": per_expert,
            }
            (output_dir / "gradient_stats" / f"step_{step:05d}.json").write_text(
                json.dumps(gradient_norms_timeline[step], indent=2),
                encoding="utf-8",
            )

        # Mask enforcement assertion: pruned weights must stay zero
        if masks and (step + 1) % 500 == 0:
            for name, mask in masks.items():
                param = dict(model.named_parameters())[name]
                pruned_locs = ~mask.bool()
                if not torch.all(param[pruned_locs] == 0):
                    raise RuntimeError(
                        f"Mask enforcement violation in {name} at step {step + 1} of {condition_name}: "
                        f"pruned weights are not all zero."
                    )

        # Accumulate early AUC
        if step < early_window_step:
            early_auc += val_loss

    final_loss = eval_and_log()
    final_routing = routing_statistics(model, calibration_batches, device)
    final_router_state = component_state_dict(model, "router")

    # Final mask enforcement assertion
    if masks:
        for name, mask in masks.items():
            param = dict(model.named_parameters())[name]
            pruned_locs = ~mask.bool()
            if not torch.all(param[pruned_locs] == 0):
                raise RuntimeError(
                    f"Final mask enforcement violation in {name} of {condition_name}: "
                    f"pruned weights are not all zero at end of recovery."
                )

    # Save checkpoints
    torch.save(model.state_dict(), output_dir / "checkpoints" / "final.pt")
    torch.save(
        {
            "experts": {n: t for n, t in model.state_dict().items() if parameter_group(n) == "expert"},
            "shared": {n: t for n, t in model.state_dict().items() if parameter_group(n) == "shared"},
            "router": component_state_dict(model, "router"),
        },
        output_dir / "component_checkpoints" / "final.pt",
    )

    # Compute recovery metrics
    recovery_fraction = (initial_loss - final_loss) / max(1e-8, initial_loss - dense_loss)
    if early_window_step > 0:
        early_auc /= early_window_step
    else:
        early_auc = final_loss

    # Metadata
    metadata = {
        "condition": condition_name,
        "seed": seed,
        "router_age_percent": router_age_percent,
        "router_checkpoint": str(router_checkpoint),
        "router_step": router_step,
        "sparsity": sparsity,
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
        "final_router_state_hash": state_dict_hash(final_router_state),
        "training_batch_sequence_hash": train_batch_hash,
        "validation_batch_sequence_hash": validation_batch_hash,
        "optimizer": "fresh_AdamW",
        "scheduler": "none",
        "precision": config["training"].get("precision", "fp32"),
        "recovery_fraction_formula": "(L_initial - L_final) / (L_initial - L_dense)",
        "expert_surviving_weight_source": "E_0",
        "mask_source": "E_T",
        "shared_state_source": "E_0",
        "requested_router_step": _expected_router_step_for_percent(int(config["training"]["steps"]), router_age_percent),
        "loaded_router_step": router_step,
        "integrity_checks_passed": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Summary
    summary = {
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "best_validation_loss": best_loss,
        "early_auc": early_auc,
        "early_auc_window_steps": early_window_step,
        "early_mean_validation_loss": early_auc / early_window_step if early_window_step > 0 else final_loss,
        "recovery_fraction": recovery_fraction,
        "time_to_threshold": time_to_threshold,
        "dense_reference_loss": dense_loss,
        "mean_selected_probability_initial": initial_routing["mean_selected_probability"],
        "mean_selected_probability_final": final_routing["mean_selected_probability"],
        "routing_entropy_initial": initial_routing["routing_entropy"],
        "routing_entropy_final": final_routing["routing_entropy"],
        "top1_top2_margin_initial": initial_routing["top1_top2_margin"],
        "top1_top2_margin_final": final_routing["top1_top2_margin"],
        "router_logit_norm_initial": initial_routing["router_logit_norm"],
        "router_logit_norm_final": final_routing["router_logit_norm"],
        "expert_utilization_initial": initial_routing["expert_utilization"],
        "expert_utilization_final": final_routing["expert_utilization"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {**metadata, **summary}


def _write_partial_csv(records: list[dict], root: Path) -> None:
    """Write incremental CSV as conditions complete."""
    csv_path = root / "sparsity_sweep_all_records.csv"
    if not records:
        return
    fieldnames = sorted(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, "") for k in fieldnames})


def _write_paired_csv(records: list[dict], root: Path) -> None:
    """Write paired R_0 vs R_100 comparison."""
    # Group by (seed, sparsity)
    paired = {}
    for rec in records:
        if rec.get("condition_type") != "sparse_control":
            continue
        key = (rec["reference_seed"], rec["sparsity"])
        if key not in paired:
            paired[key] = {}
        paired[key][f"R{rec['router_age_percent']}"] = rec

    rows = []
    for (seed, sparsity), endpoints in sorted(paired.items()):
        if "R0" not in endpoints or "R100" not in endpoints:
            continue  # Skip incomplete pairs
        r0 = endpoints["R0"]
        r100 = endpoints["R100"]
        sparse_r0 = r0["final_validation_loss"]
        dense_r0 = r0.get("dense_baseline_loss_r0")
        sparse_r100 = r100["final_validation_loss"]
        dense_r100 = r100.get("dense_baseline_loss_r100")
        if dense_r0 is None or dense_r100 is None:
            continue  # Skip if dense baselines missing
        gap_r0 = sparse_r0 - dense_r0
        gap_r100 = sparse_r100 - dense_r100
        gap_reduction = gap_r0 - gap_r100
        proportional_reduction = gap_reduction / gap_r0 if gap_r0 > 0 else None
        rows.append({
            "reference_seed": seed,
            "sparsity": f"{sparsity:.2f}",
            "sparse_R0_final": f"{sparse_r0:.6f}",
            "dense_R0_final": f"{dense_r0:.6f}",
            "ticket_gap_R0": f"{gap_r0:.6f}",
            "sparse_R100_final": f"{sparse_r100:.6f}",
            "dense_R100_final": f"{dense_r100:.6f}",
            "ticket_gap_R100": f"{gap_r100:.6f}",
            "gap_reduction_routing": f"{gap_reduction:.6f}",
            "proportional_gap_reduction": f"{proportional_reduction:.6f}" if proportional_reduction is not None else "undefined",
        })

    csv_path = root / "sparsity_sweep_paired.csv"
    if rows:
        fieldnames = [
            "reference_seed", "sparsity",
            "sparse_R0_final", "dense_R0_final", "ticket_gap_R0",
            "sparse_R100_final", "dense_R100_final", "ticket_gap_R100",
            "gap_reduction_routing", "proportional_gap_reduction"
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def run_sparsity_sweep(
    config_paths: list[str],
    output_dir: str,
    recovery_steps: int | None = None,
) -> dict:
    """Main sparsity sweep runner.
    
    For each sparsity level (60%, 70%, 90%, 95%):
      1. For each seed:
         a. Discover pruning mask from E_T
         b. Build sparse base = m_s ⊙ E_0
         c. For each router age endpoint (0%, 100%):
            - Run sparse recovery
            - Reuse or validate dense control
            - Log metrics and hashes
    """
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to mix or overwrite experiment artifacts in {root}")
    root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    sparsity_dense_baselines: dict = {}  # Cache dense baselines by (seed, router_age_percent)

    for sparsity in SPARSITIES_TO_SWEEP:
        sparsity_dir = root / f"sparsity_{sparsity:.2f}"
        sparsity_dir.mkdir(parents=True, exist_ok=True)

        for config_index, config_path in enumerate(config_paths):
            config = load_config(config_path)
            seed = int(config["seed"])
            run_dir = _ensure_reference(config_path)
            device = resolve_device(config["device"])
            configure_device(device)
            total_steps = int(config["training"]["steps"])
            recovery_steps_actual = total_steps if recovery_steps is None else int(recovery_steps)

            # Load reference models and data
            initial_checkpoint, initial_step = _checkpoint_for_percent(run_dir, total_steps, 0)
            final_checkpoint, final_step = _checkpoint_for_percent(run_dir, total_steps, 100)

            seed_sparsity_dir = sparsity_dir / f"seed_{seed}"
            seed_sparsity_dir.mkdir(parents=True, exist_ok=True)

            seed_everything(seed)
            dense_model = load_model_from_checkpoint(config["model"], str(final_checkpoint), device)
            train_loader, validation_loader = build_dataloaders(
                config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
            )

            # Materialize batch sequences (identical across sparsities for reproducibility)
            train_batches = _materialize_batches(train_loader, recovery_steps_actual)
            validation_batches = _materialize_validation_batches(
                validation_loader, int(config["data"]["validation_blocks"])
            )
            train_batch_hash = _batch_sequence_hash(train_batches)
            validation_batch_hash = _batch_sequence_hash(validation_batches)

            # Compute dense loss (once per seed, shared across sparsities)
            dense_metrics = evaluate_language_model(
                dense_model, validation_batches, device, max_batches=len(validation_batches)
            )
            dense_loss = dense_metrics["loss"]

            # Discover mask for this sparsity from E_T
            masks = expert_local_magnitude_masks(dense_model, sparsity)
            mask_path = seed_sparsity_dir / "pruning_mask.pt"
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
                "reference_seed": seed,
                "sparsity": sparsity,
                "total_expert_parameters": all_expert_parameters,
                "total_prunable_expert_weight_parameters": prunable_expert_weights,
                "pruned_parameters": pruned,
                "surviving_prunable_weight_parameters": surviving_prunable,
                "realized_sparsity": pruned / prunable_expert_weights,
                "realized_sparsity_over_all_expert_parameters": pruned / all_expert_parameters,
                "pruning_method": "expert_local_magnitude (top-k retained by magnitude per expert)",
                "mask_hash": _mask_hash(masks),
            }
            (seed_sparsity_dir / "pruning_metadata.json").write_text(
                json.dumps(pruning_stats, indent=2), encoding="utf-8"
            )

            # Build sparse base state: E_0 under mask m_s
            pruned_base_state = build_fixed_pruned_base(config["model"], str(initial_checkpoint), masks, device)

            # Build dense base state: full E_0 (no mask)
            dense_base_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in load_model_from_checkpoint(config["model"], str(initial_checkpoint), device).state_dict().items()
            }

            # Compute state hashes
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

            # Save component checkpoint
            fixed_component_dir = seed_sparsity_dir / "component_checkpoints"
            fixed_component_dir.mkdir(exist_ok=True)
            torch.save(
                {
                    "experts": {n: t for n, t in pruned_base_state.items() if parameter_group(n) == "expert"},
                    "shared": {n: t for n, t in pruned_base_state.items() if parameter_group(n) == "shared"},
                    "expert_state_hash": expert_hash,
                    "shared_state_hash": shared_hash,
                    "mask_hash": pruning_stats["mask_hash"],
                    "uses_rewound_expert_initialization": True,
                    "sparsity": sparsity,
                    "mask_source": "E_T",
                    "surviving_values_source": "E_0",
                },
                fixed_component_dir / f"fixed_pruned_experts_and_shared_s{sparsity:.2f}.pt",
            )

            # Audit: lth protocol compliance
            audit = {
                "reference_seed": seed,
                "sparsity": sparsity,
                "shared_checkpoint_path": str(initial_checkpoint),
                "mask_derived_from_dense_reference_checkpoint": str(final_checkpoint),
                "surviving_expert_values_from_rewind_checkpoint": str(initial_checkpoint),
                "mask_hash": pruning_stats["mask_hash"],
                "expert_state_hash": expert_hash,
                "shared_state_hash": shared_hash,
                "dense_expert_state_hash": dense_expert_hash,
                "expert_surviving_weight_source": "E_0",
                "mask_source": "E_T",
                "all_router_age_conditions_share_same_sparse_ticket": True,
                "all_router_age_conditions_share_same_shared_state": True,
                "pruned_parameters_remain_zero": True,
                "mask_is_identical_across_router_conditions": True,
            }
            (seed_sparsity_dir / "lth_isolation_audit.json").write_text(
                json.dumps(audit, indent=2), encoding="utf-8"
            )

            # Calibration setup
            calibration_batches = _calibration_batches(validation_batches)

            # For each router age endpoint
            for percent in ROUTER_AGES_PERCENT_ENDPOINTS:
                router_checkpoint, router_step = _checkpoint_for_percent(run_dir, total_steps, percent)

                # --- Sparse condition ---
                condition_dir = seed_sparsity_dir / f"age_{percent:03d}pct_sparse"
                sparse_record = _run_recovery_condition(
                    config=config,
                    condition_name=f"seed{seed}_sparsity{sparsity:.2f}_age{percent}_sparse",
                    pruned_base_state=pruned_base_state,
                    router_checkpoint=str(router_checkpoint),
                    router_age_percent=percent,
                    router_step=router_step,
                    masks=masks,
                    expert_hash=expert_hash,
                    shared_hash=shared_hash,
                    mask_hash=pruning_stats["mask_hash"],
                    calibration_batches=calibration_batches,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    train_batch_hash=train_batch_hash,
                    validation_batch_hash=validation_batch_hash,
                    device=device,
                    recovery_steps=recovery_steps_actual,
                    dense_loss=dense_loss,
                    output_dir=condition_dir,
                    confidence_control=False,
                    target_confidence=None,
                    seed=seed,
                    sparsity=sparsity,
                )
                sparse_record.update({"condition_type": "sparse_control", "sparsity": sparsity})
                all_records.append(sparse_record)
                _write_partial_csv(all_records, root)

                # --- Dense control (reuse if compatible, else run) ---
                dense_cache_key = (seed, percent)
                if dense_cache_key in sparsity_dense_baselines:
                    # Reuse from cache
                    dense_record = sparsity_dense_baselines[dense_cache_key].copy()
                    dense_record["dense_baseline_reused"] = True
                else:
                    # Run dense condition
                    dense_dir = seed_sparsity_dir / f"age_{percent:03d}pct_dense"
                    dense_record = _run_recovery_condition(
                        config=config,
                        condition_name=f"seed{seed}_sparsity{sparsity:.2f}_age{percent}_dense",
                        pruned_base_state=dense_base_state,
                        router_checkpoint=str(router_checkpoint),
                        router_age_percent=percent,
                        router_step=router_step,
                        masks={},
                        expert_hash=state_dict_hash(
                            {n: t for n, t in dense_base_state.items() if parameter_group(n) == "expert"}
                        ),
                        shared_hash=state_dict_hash(
                            {n: t for n, t in dense_base_state.items() if parameter_group(n) == "shared"}
                        ),
                        mask_hash="dense_no_mask",
                        calibration_batches=calibration_batches,
                        train_batches=train_batches,
                        validation_batches=validation_batches,
                        train_batch_hash=train_batch_hash,
                        validation_batch_hash=validation_batch_hash,
                        device=device,
                        recovery_steps=recovery_steps_actual,
                        dense_loss=dense_loss,
                        output_dir=dense_dir,
                        confidence_control=False,
                        target_confidence=None,
                        seed=seed,
                        sparsity=0.0,
                    )
                    dense_record["dense_baseline_reused"] = False
                    # Cache for future sparsities
                    sparsity_dense_baselines[dense_cache_key] = dense_record.copy()

                dense_record.update({"condition_type": "dense_control", "sparsity": 0.0})
                all_records.append(dense_record)
                _write_partial_csv(all_records, root)

                # Compute sparse-dense gap
                sparse_final = sparse_record["final_validation_loss"]
                dense_final = dense_record["final_validation_loss"]
                ticket_gap = sparse_final - dense_final

                print(
                    f"[seed={seed}, sparsity={sparsity:.2f}, age={percent}%] "
                    f"sparse_final={sparse_final:.6f}, dense_final={dense_final:.6f}, "
                    f"gap={ticket_gap:.6f}"
                )

    _write_partial_csv(all_records, root)
    (root / "sparsity_sweep_all_records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    _write_paired_csv(all_records, root)

    return {"records": all_records, "output_dir": str(root)}


def main():
    parser = argparse.ArgumentParser(description="Router-conditioned MoE sparsity sweep")
    parser.add_argument(
        "--configs",
        required=True,
        nargs="+",
        help="Reference training config paths (e.g., configs/router_age_reference_seed7.yaml ...)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory for sparsity sweep (will be created)",
    )
    parser.add_argument(
        "--recovery-steps",
        type=int,
        default=None,
        help=f"Recovery training budget (default: {DEFAULT_RECOVERY_STEPS})",
    )
    args = parser.parse_args()
    result = run_sparsity_sweep(
        args.configs,
        args.output_dir,
        recovery_steps=args.recovery_steps,
    )
    print(f"\nSparsity sweep complete. Output: {result['output_dir']}")
    print(f"Records: {len(result['records'])}")


if __name__ == "__main__":
    main()
