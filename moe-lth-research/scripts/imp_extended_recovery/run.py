"""CLI entry point for the sequential matched IMP extension matrix."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/fixed_budget_dataset")]

import numpy as np
import torch
import common
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.pruning.masks import apply_masks_, expert_weight_parameters, load_masks
from moe_lth.pruning.router_age import (
    component_state_dict,
    load_model_from_checkpoint,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)


START_STEP = 2500
TARGET_STEP = 5000
EXTENSION_STEPS = TARGET_STEP - START_STEP
EVAL_INTERVAL = 50
CHECKPOINT_STEPS = (3000, 4000, 5000)
RUN_DIR = ROOT / "results/imp_80_85/trajectories/e8/seed_7/final_s0.85"
SOURCE_CHECKPOINT = RUN_DIR / "checkpoints/final_recovered.pt"
SOURCE_STAGE = RUN_DIR / "stage.json"
MASK_PATH = RUN_DIR.parent / "mask_09.pt"
OUTPUT_ROOT = ROOT / "results/imp_extended_recovery"
OUTPUT_DIR = OUTPUT_ROOT / "e8/seed_7/final_s0.85"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source() -> dict:
    for path in (SOURCE_CHECKPOINT, SOURCE_STAGE, MASK_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required IMP source artifact is missing: {path}")
    stage = _json(SOURCE_STAGE)
    checkpoint_sha = _sha256(SOURCE_CHECKPOINT)
    if checkpoint_sha != stage["checkpoint_sha256"]:
        raise RuntimeError("The saved 85% recovery checkpoint hash differs from stage.json")
    checkpoint = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    masks = load_masks(MASK_PATH)
    mask_hash = recovery._mask_hash(masks)
    if mask_hash != stage["mask_hash"]:
        raise RuntimeError("The saved 85% IMP mask hash differs from stage.json")
    missing_state = [
        name
        for name in ("optimizer", "scaler", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "batch_position")
        if name not in checkpoint
    ]
    return {
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_checkpoint_step": checkpoint.get("step"),
        "source_checkpoint_keys": sorted(checkpoint),
        "model_parameters_present": "model" in checkpoint,
        "optimizer_state_present": "optimizer" in checkpoint,
        "amp_grad_scaler_state_present": "scaler" in checkpoint,
        "rng_state_present": not any(key.endswith("_rng") for key in missing_state),
        "training_data_position_present": "batch_position" not in missing_state,
        "missing_training_state": missing_state,
        "training_and_validation_curves_present": (RUN_DIR / "metrics.jsonl").is_file(),
        "gradient_curve_present": (RUN_DIR / "gradient_stats/gradient_stats.jsonl").is_file(),
        "attempted_updates": stage["optimizer_step_attempts"],
        "applied_updates": stage["optimizer_steps_applied"],
        "mask_path": str(MASK_PATH),
        "mask_hash": mask_hash,
        "mask_sparsity": stage["actual_sparsity"],
        "final_validation_cross_entropy": stage["final_validation_loss"],
        "router_trainable": stage["router_mode"] == "trainable",
        "router_gradient_observed": stage["trainable_router_gradient_observed"],
        "router_parameter_drift": stage["router_parameter_drift_final"],
    }


def _annotate_jsonl(path: Path, offset: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            row["absolute_step"] = offset + int(row["step"])
            if "loss" in row:
                row["validation_cross_entropy"] = row["loss"]
            if "train_cross_entropy" in row:
                row["training_cross_entropy"] = row["train_cross_entropy"]
            target.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _mask_intact(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> bool:
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        parameter = parameters[name].detach()
        pruned = ~mask.to(parameter.device)
        if pruned.any() and torch.count_nonzero(parameter[pruned]).item():
            return False
    return True


def _save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    absolute_step: int,
    mask_hash: str,
) -> None:
    payload = {
        "step": absolute_step,
        "continuation_step": absolute_step - START_STEP,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "training_data_position": absolute_step,
        "mask_hash": mask_hash,
        "optimizer_continuation": "fresh_adamw_from_step_2500_weights",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_extension() -> dict:
    if not torch.__version__.startswith("2.7.0"):
        raise RuntimeError("The saved IMP protocol requires the verified torch 2.7.0 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The source recovery used CUDA FP16; refusing a CPU fallback that changes the protocol")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite extension artifacts in {OUTPUT_DIR}")

    audit = audit_source()
    source_stage = _json(SOURCE_STAGE)
    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    config = source["config"]
    seed = int(source_stage["seed"])
    masks = load_masks(MASK_PATH)
    mask_hash = recovery._mask_hash(masks)

    recovery.seed_everything(seed)
    train_loader, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), seed
    )
    full_training_stream = recovery._materialize_batches(train_loader, TARGET_STEP)
    original_prefix_hash = recovery._batch_sequence_hash(full_training_stream[:START_STEP])
    if original_prefix_hash != source_stage["training_batch_sequence_hash"]:
        raise RuntimeError("Reconstructed first 2,500 training batches do not match the completed IMP run")
    training_batches = full_training_stream[START_STEP:TARGET_STEP]
    del full_training_stream
    validation_batches = recovery._materialize_validation_batches(
        validation_loader, int(config["data"]["validation_blocks"])
    )
    validation_hash = recovery._batch_sequence_hash(validation_batches)
    if validation_hash != source_stage["validation_batch_sequence_hash"]:
        raise RuntimeError("Reconstructed validation batches do not match the completed IMP run")
    continuation_hash = recovery._batch_sequence_hash(training_batches)

    initial_model = load_model_from_checkpoint(config["model"], str(SOURCE_CHECKPOINT), torch.device("cpu"))
    initial_state = {name: tensor.detach().clone() for name, tensor in initial_model.state_dict().items()}
    for name, mask in masks.items():
        if name not in initial_state or torch.count_nonzero(initial_state[name][~mask]).item():
            raise RuntimeError(f"Source checkpoint does not satisfy fixed-mask integrity for {name}")
    expert_hash = state_dict_hash(component_state_dict(initial_model, "expert"))
    shared_hash = state_dict_hash(component_state_dict(initial_model, "shared"))
    initial_router_state = component_state_dict(initial_model, "router")
    router_parameter_names = {
        name for name, _ in initial_model.named_parameters() if parameter_group(name) == "router"
    }
    reference_model = load_model_from_checkpoint(config["model"], str(SOURCE_CHECKPOINT), torch.device("cuda"))
    device = next(reference_model.parameters()).device
    calibration_batches = recovery._calibration_batches(validation_batches)
    reference_selected = selected_experts_per_batch(reference_model, calibration_batches, device)
    del reference_model, initial_model

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    state_log = OUTPUT_DIR / "integrity.jsonl"

    def state_snapshot(model, optimizer, scaler, relative_step: int) -> None:
        absolute_step = START_STEP + relative_step
        if relative_step % EVAL_INTERVAL == 0:
            intact = _mask_intact(model, masks)
            if not intact:
                raise RuntimeError(f"Fixed IMP mask was violated at step {absolute_step}")
            current_router_state = component_state_dict(model, "router")
            drift, normalized_drift = recovery._router_drift(
                initial_router_state, current_router_state, router_parameter_names
            )
            with state_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "step": absolute_step,
                    "mask_hash": mask_hash,
                    "mask_integrity": "passed",
                    "router_parameter_drift_from_step_2500": drift,
                    "router_parameter_drift_normalized_from_step_2500": normalized_drift,
                    "router_state_hash": state_dict_hash(current_router_state),
                }, sort_keys=True) + "\n")
        if absolute_step in CHECKPOINT_STEPS:
            _save_training_checkpoint(
                OUTPUT_DIR / "checkpoints" / f"step_{absolute_step}.pt",
                model,
                optimizer,
                scaler,
                absolute_step,
                mask_hash,
            )

    record = recovery._run_recovery_condition(
        config=config,
        condition_name="e8_seed7_imp85_extended_fresh_adamw",
        pruned_base_state=initial_state,
        router_checkpoint=str(SOURCE_CHECKPOINT),
        router_age_percent=0,
        router_step=START_STEP,
        masks=masks,
        expert_hash=expert_hash,
        shared_hash=shared_hash,
        mask_hash=mask_hash,
        reference_selected=reference_selected,
        calibration_batches=calibration_batches,
        train_batches=training_batches,
        validation_batches=validation_batches,
        train_batch_hash=continuation_hash,
        validation_batch_hash=validation_hash,
        device=device,
        recovery_steps=EXTENSION_STEPS,
        dense_loss=float(source_stage["dense_reference_loss"]),
        output_dir=OUTPUT_DIR,
        confidence_control=False,
        target_confidence=None,
        seed=seed,
        sparsity=float(source_stage["sparsity"]),
        router_mode="trainable",
        state_snapshot_callback=state_snapshot,
    )

    for relative_path in (
        Path("metrics.jsonl"),
        Path("gradient_stats/gradient_stats.jsonl"),
        Path("routing_stats/routing_stats.jsonl"),
    ):
        _annotate_jsonl(OUTPUT_DIR / relative_path, START_STEP)

    result = {
        "protocol": "sparse convergence diagnostic; no matched dense continuation",
        "continuation_type": "fresh_optimizer_non_exact",
        "source_step": START_STEP,
        "target_step": TARGET_STEP,
        "attempted_steps": record["optimizer_step_attempts"],
        "applied_optimizer_updates": record["optimizer_steps_applied"],
        "source_validation_cross_entropy": source_stage["final_validation_loss"],
        "extension_initial_validation_cross_entropy": record["initial_validation_loss"],
        "extension_final_validation_cross_entropy": record["final_validation_loss"],
        "validation_cross_entropy_change": record["final_validation_loss"] - record["initial_validation_loss"],
        "mask_hash": mask_hash,
        "router_trainable": record["router_mode"] == "trainable",
        "router_gradient_observed": record["trainable_router_gradient_observed"],
        "router_parameter_drift": record["router_parameter_drift_final"],
        "training_prefix_hash_verified": original_prefix_hash,
        "continued_training_batch_hash": continuation_hash,
        "validation_batch_hash": validation_hash,
        "missing_source_state": audit["missing_training_state"],
        "dense_comparison_performed": False,
        "duration_seconds": time.perf_counter() - start_time,
        "trainer_record": record,
    }
    (OUTPUT_DIR / "extension_manifest.json").write_text(json.dumps({
        "source_audit": audit,
        "methodological_note": (
            "This is not an exact continuation: the 2,500-step source checkpoint contains model weights only. "
            "AdamW moments, GradScaler state, runtime RNG states, and a saved data cursor are absent. "
            "A new AdamW and scaler are initialized from the step-2,500 weights. The deterministic input stream "
            "is reconstructed, its first 2,500 batches are hash-checked, and the next 2,500 batches are used."
        ),
        **result,
    }, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "extended_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    if __package__:
        from .pipeline import main as matrix_main
    else:
        from pipeline import main as matrix_main
    matrix_main()


if __name__ == "__main__":
    main()