"""Coordinator for matched IMP and dense continuation diagnostics."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import random
import subprocess
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
from moe_lth.pruning.masks import load_masks
from moe_lth.pruning.router_age import (
    component_state_dict,
    load_model_from_checkpoint,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)


EXPERT_COUNTS = (8, 16, 32)
SEEDS = (7, 17, 29, 43, 61)
SPARSITIES = (0.80, 0.85)
START_STEP = 2500
TARGET_STEP = 5000
EXTENSION_STEPS = TARGET_STEP - START_STEP
EVAL_INTERVAL = 50
CHECKPOINT_STEPS = (3000, 4000, 5000)
CANONICAL_INDEX = ROOT / "results/final_dataset_a/canonical/dataset_a_canonical.csv"
IMP_ROOT = ROOT / "results/imp_80_85"
OUTPUT_ROOT = ROOT / "results/imp_extended_recovery"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pairs() -> dict[tuple[int, int, float], dict]:
    with CANONICAL_INDEX.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        selected = [
            row for row in rows
            if row["router_age"] == "0" and float(row["sparsity"]) in SPARSITIES
            and int(row["num_experts"]) in EXPERT_COUNTS and int(row["seed"]) in SEEDS
        ]
    pairs = {
        (int(row["num_experts"]), int(row["seed"]), float(row["sparsity"])): row
        for row in selected
    }
    expected = len(EXPERT_COUNTS) * len(SEEDS) * len(SPARSITIES)
    if len(pairs) != expected:
        raise RuntimeError(f"Expected {expected} canonical matched rows, found {len(pairs)}")
    return pairs


def _source_paths(experts: int, seed: int, sparsity: float) -> tuple[Path, Path, Path]:
    pair_root = IMP_ROOT / "trajectories" / f"e{experts}" / f"seed_{seed}"
    stage_dir = pair_root / f"final_s{sparsity:.2f}"
    checkpoint = stage_dir / "checkpoints/final_recovered.pt"
    mask = pair_root / f"mask_{8 if sparsity == 0.80 else 9:02d}.pt"
    return stage_dir, checkpoint, mask


def _source_ready(experts: int, seed: int, sparsity: float) -> bool:
    stage_dir, checkpoint, mask_path = _source_paths(experts, seed, sparsity)
    if not all(path.is_file() for path in (stage_dir / "stage.json", checkpoint, mask_path)):
        return False
    stage = _json(stage_dir / "stage.json")
    if _sha256(checkpoint) != stage.get("checkpoint_sha256") or int(stage.get("optimizer_step_attempts", -1)) != START_STEP:
        raise RuntimeError(f"IMP source checkpoint integrity/protocol mismatch at {stage_dir}")
    if recovery._mask_hash(load_masks(mask_path)) != stage.get("mask_hash"):
        raise RuntimeError(f"IMP mask integrity mismatch at {mask_path}")
    return True


def _dense_source(pair: dict, experts: int, seed: int) -> Path:
    path = Path(pair["dense_artifact_path"]) / "checkpoints/final_recovered.pt"
    if not path.is_file() or _sha256(path) != pair["dense_final_checkpoint_sha256"]:
        raise RuntimeError(f"Matched dense baseline missing or hash-mismatched for E{experts} seed{seed}: {path}")
    return path


def _extension_dir(experts: int, seed: int, condition: str, sparsity: float | None = None) -> Path:
    pair_root = OUTPUT_ROOT / f"e{experts}" / f"seed_{seed}"
    return pair_root / (f"final_s{sparsity:.2f}" if condition == "sparse" else "dense")


def _identity(experts, seed, condition, source_path, mask_hash, train_hash, validation_hash, sparsity) -> dict:
    return {
        "num_experts": experts,
        "seed": seed,
        "condition": condition,
        "sparsity": sparsity,
        "source_checkpoint_sha256": _sha256(source_path),
        "mask_hash": mask_hash,
        "training_batch_hash": train_hash,
        "validation_batch_hash": validation_hash,
        "source_step": START_STEP,
        "target_step": TARGET_STEP,
        "optimizer_protocol": "fresh_adamw_and_grad_scaler_from_step_2500",
        "evaluation_interval": EVAL_INTERVAL,
    }


def _mask_intact(model: torch.nn.Module, masks: dict[str, torch.Tensor]) -> bool:
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        parameter = parameters[name].detach()
        pruned = ~mask.to(parameter.device)
        if pruned.any() and torch.count_nonzero(parameter[pruned]).item():
            return False
    return True


def _write_absolute_jsonl(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as input_stream, temporary.open("w", encoding="utf-8") as output_stream:
        for line in input_stream:
            row = json.loads(line)
            row["absolute_step"] = START_STEP + int(row["step"])
            if "loss" in row:
                row["validation_cross_entropy"] = row["loss"]
            if "train_cross_entropy" in row:
                row["training_cross_entropy"] = row["train_cross_entropy"]
            output_stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(destination)


def _save_training_checkpoint(path, model, optimizer, scaler, absolute_step, mask_hash, identity):
    payload = {
        "step": absolute_step,
        "continuation_step": absolute_step - START_STEP,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "training_data_position": absolute_step,
        "mask_hash": mask_hash,
        "identity": identity,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _is_complete(output_dir: Path, identity: dict) -> bool:
    manifest_path = output_dir / "extension_manifest.json"
    checkpoint_path = output_dir / "checkpoints/step_5000.pt"
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        return False
    manifest = _json(manifest_path)
    saved_identity = manifest.get("identity")
    if saved_identity is None:
        # Preserve the finished E8/seed7/85% run made by the original single-case runner.
        legacy_case = (
            identity["num_experts"] == 8 and identity["seed"] == 7
            and identity["condition"] == "sparse" and identity["sparsity"] == 0.85
        )
        old_audit = manifest.get("source_audit", {})
        if not legacy_case:
            return False
        if old_audit.get("source_checkpoint_sha256") != identity["source_checkpoint_sha256"]:
            raise RuntimeError(f"Existing extension source differs at {output_dir}")
        if manifest.get("mask_hash") != identity["mask_hash"]:
            raise RuntimeError(f"Existing extension mask differs at {output_dir}")
        if manifest.get("continued_training_batch_hash") != identity["training_batch_hash"]:
            raise RuntimeError(f"Existing extension training sequence differs at {output_dir}")
        if manifest.get("validation_batch_hash") != identity["validation_batch_hash"]:
            raise RuntimeError(f"Existing extension validation sequence differs at {output_dir}")
    elif saved_identity != identity:
        raise RuntimeError(f"Existing extension identity differs at {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("step") != TARGET_STEP or checkpoint.get("mask_hash") != identity["mask_hash"]:
        raise RuntimeError(f"Existing step-5,000 checkpoint failed integrity validation at {checkpoint_path}")
    if saved_identity is not None and checkpoint.get("identity") != identity:
        raise RuntimeError(f"Existing step-5,000 checkpoint identity mismatch at {checkpoint_path}")
    expected_checkpoint_hash = manifest.get("step_5000_checkpoint_sha256")
    if expected_checkpoint_hash and _sha256(checkpoint_path) != expected_checkpoint_hash:
        raise RuntimeError(f"Existing step-5,000 checkpoint hash mismatch at {checkpoint_path}")
    metrics_path = output_dir / "metrics_absolute.jsonl"
    if not metrics_path.is_file():
        metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return False
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    return bool(rows and rows[-1].get("absolute_step") == TARGET_STEP)


def _run_extension(
    experts: int,
    seed: int,
    condition: str,
    source_path: Path,
    source_stage: dict | None,
    config: dict,
    masks: dict[str, torch.Tensor],
    training_batches: list,
    validation_batches: list,
    train_hash: str,
    validation_hash: str,
    baseline_dense_ce: float,
    sparsity: float | None,
) -> dict:
    output_dir = _extension_dir(experts, seed, condition, sparsity)
    mask_hash = recovery._mask_hash(masks)
    identity = _identity(experts, seed, condition, source_path, mask_hash, train_hash, validation_hash, sparsity)
    if _is_complete(output_dir, identity):
        print(f"[skip] E{experts} seed{seed} {condition} s={sparsity}: complete through step 5000", flush=True)
        return _json(output_dir / "extension_manifest.json")
    if output_dir.exists() and any(output_dir.iterdir()) and not (output_dir / "interruption.pt").is_file():
        raise RuntimeError(f"Incomplete extension has no resumable training snapshot; preserving {output_dir}")

    source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if int(source_payload.get("step", -1)) != START_STEP:
        raise RuntimeError(f"Expected a step-2,500 source checkpoint: {source_path}")
    model_cpu = load_model_from_checkpoint(config["model"], str(source_path), torch.device("cpu"))
    initial_state = {name: tensor.detach().clone() for name, tensor in model_cpu.state_dict().items()}
    if masks and not _mask_intact(model_cpu, masks):
        raise RuntimeError(f"Source model violates its fixed IMP mask: {source_path}")
    expert_hash = state_dict_hash(component_state_dict(model_cpu, "expert"))
    shared_hash = state_dict_hash(component_state_dict(model_cpu, "shared"))
    initial_router_state = component_state_dict(model_cpu, "router")
    router_parameter_names = {
        name for name, _ in model_cpu.named_parameters() if parameter_group(name) == "router"
    }
    del model_cpu

    device = torch.device("cuda")
    reference_model = load_model_from_checkpoint(config["model"], str(source_path), device)
    reference_selected = selected_experts_per_batch(
        reference_model, recovery._calibration_batches(validation_batches), device
    )
    del reference_model
    output_dir.mkdir(parents=True, exist_ok=True)
    state_log = output_dir / "integrity.jsonl"

    def state_snapshot(model, optimizer, scaler, relative_step: int) -> None:
        absolute_step = START_STEP + relative_step
        if relative_step % EVAL_INTERVAL == 0:
            if not _mask_intact(model, masks):
                raise RuntimeError(f"Fixed-mask integrity failed at absolute step {absolute_step}")
            router_state = component_state_dict(model, "router")
            drift, normalized_drift = recovery._router_drift(
                initial_router_state, router_state, router_parameter_names
            )
            with state_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "step": absolute_step,
                    "mask_hash": mask_hash,
                    "mask_integrity": "passed",
                    "router_parameter_drift_from_step_2500": drift,
                    "router_parameter_drift_normalized_from_step_2500": normalized_drift,
                    "router_state_hash": state_dict_hash(router_state),
                }, sort_keys=True) + "\n")
        if absolute_step in CHECKPOINT_STEPS:
            _save_training_checkpoint(
                output_dir / "checkpoints" / f"step_{absolute_step}.pt",
                model, optimizer, scaler, absolute_step, mask_hash, identity,
            )

    started = time.perf_counter()
    record = recovery._run_recovery_condition(
        config=config,
        condition_name=f"e{experts}_seed{seed}_{condition}_{sparsity}",
        pruned_base_state=initial_state,
        router_checkpoint=str(source_path),
        router_age_percent=0,
        router_step=START_STEP,
        masks=masks,
        expert_hash=expert_hash,
        shared_hash=shared_hash,
        mask_hash=mask_hash,
        reference_selected=reference_selected,
        calibration_batches=recovery._calibration_batches(validation_batches),
        train_batches=training_batches,
        validation_batches=validation_batches,
        train_batch_hash=train_hash,
        validation_batch_hash=validation_hash,
        device=device,
        recovery_steps=EXTENSION_STEPS,
        dense_loss=baseline_dense_ce,
        output_dir=output_dir,
        confidence_control=False,
        target_confidence=None,
        seed=seed,
        sparsity=0.0 if sparsity is None else sparsity,
        router_mode="trainable",
        state_snapshot_callback=state_snapshot,
        interruption_checkpoint=output_dir / "interruption.pt",
        interruption_interval=100,
    )

    _write_absolute_jsonl(output_dir / "metrics.jsonl", output_dir / "metrics_absolute.jsonl")
    _write_absolute_jsonl(output_dir / "gradient_stats/gradient_stats.jsonl", output_dir / "gradient_stats/gradient_stats_absolute.jsonl")
    _write_absolute_jsonl(output_dir / "routing_stats/routing_stats.jsonl", output_dir / "routing_stats/routing_stats_absolute.jsonl")
    result = {
        "identity": identity,
        "num_experts": experts,
        "seed": seed,
        "condition": condition,
        "sparsity": sparsity,
        "continuation_type": "fresh_optimizer_non_exact",
        "source_step": START_STEP,
        "target_step": TARGET_STEP,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "optimizer": "fresh_adamw",
        "scheduler": "none",
        "attempted_steps": record["optimizer_step_attempts"],
        "applied_optimizer_updates": record["optimizer_steps_applied"],
        "extension_initial_validation_cross_entropy": record["initial_validation_loss"],
        "extension_final_validation_cross_entropy": record["final_validation_loss"],
        "validation_cross_entropy_change": record["final_validation_loss"] - record["initial_validation_loss"],
        "mask_hash": mask_hash,
        "mask_integrity": "passed",
        "router_trainable": record["router_mode"] == "trainable",
        "router_gradient_observed": record["trainable_router_gradient_observed"],
        "router_parameter_drift": record["router_parameter_drift_final"],
        "training_batch_hash": train_hash,
        "validation_batch_hash": validation_hash,
        "dense_baseline_validation_cross_entropy_at_2500": (
            record["initial_validation_loss"] if condition == "dense" else baseline_dense_ce
        ),
        "canonical_dense_validation_cross_entropy_at_2500": baseline_dense_ce,
        "source_sparse_validation_cross_entropy_at_2500": source_stage["final_validation_loss"] if source_stage else None,
        "duration_seconds": time.perf_counter() - started,
        "trainer_record": record,
    }
    checkpoint_path = output_dir / "checkpoints/step_5000.pt"
    result["step_5000_checkpoint_sha256"] = _sha256(checkpoint_path)
    for name, value in (("extension_manifest.json", result), ("extended_summary.json", result)):
        temporary = output_dir / f"{name}.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_dir / name)
    return result


def _prepare_sources(groups: list[tuple[int, int]]) -> None:
    missing = [
        (experts, seed) for experts, seed in groups
        if not all(_source_ready(experts, seed, sparsity) for sparsity in SPARSITIES)
    ]
    if not missing:
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/imp_80_85/run.py"),
        "--execute",
        "--resume",
        "--groups",
        *(f"{experts}:{seed}" for experts, seed in missing),
    ]
    print("Preparing missing 2,500-step IMP sources (verified stages are reused):", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _materialize_group_stream(experts: int, seed: int, pair: dict) -> tuple[dict, list, list, str, str]:
    _, source_checkpoint, _ = _source_paths(experts, seed, 0.80)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    recovery.seed_everything(seed)
    train_loader, validation_loader = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), seed
    )
    stream = recovery._materialize_batches(train_loader, TARGET_STEP)
    if recovery._batch_sequence_hash(stream[:START_STEP]) != pair["data_hash"]:
        raise RuntimeError(f"Saved IMP batch prefix differs for E{experts} seed{seed}")
    training_batches = stream[START_STEP:TARGET_STEP]
    del stream
    validation_batches = recovery._materialize_validation_batches(
        validation_loader, int(config["data"]["validation_blocks"])
    )
    validation_hash = recovery._batch_sequence_hash(validation_batches)
    if validation_hash != pair["validation_hash"]:
        raise RuntimeError(f"Canonical validation sequence differs for E{experts} seed{seed}")
    return config, training_batches, validation_batches, recovery._batch_sequence_hash(training_batches), validation_hash


def _write_comparisons(pairs: dict, groups: list[tuple[int, int]]) -> None:
    rows = []
    for experts, seed in groups:
        dense_dir = _extension_dir(experts, seed, "dense")
        if not (dense_dir / "extension_manifest.json").is_file():
            continue
        dense_manifest = _json(dense_dir / "extension_manifest.json")
        if not _is_complete(dense_dir, dense_manifest.get("identity", {})):
            continue
        dense_2500 = float(dense_manifest["dense_baseline_validation_cross_entropy_at_2500"])
        dense_5000 = float(dense_manifest["extension_final_validation_cross_entropy"])
        for sparsity in SPARSITIES:
            sparse_dir = _extension_dir(experts, seed, "sparse", sparsity)
            manifest_path = sparse_dir / "extension_manifest.json"
            if not manifest_path.is_file():
                continue
            sparse_manifest = _json(manifest_path)
            _, sparse_source, sparse_mask_path = _source_paths(experts, seed, sparsity)
            sparse_identity = _identity(
                experts,
                seed,
                "sparse",
                sparse_source,
                recovery._mask_hash(load_masks(sparse_mask_path)),
                dense_manifest["identity"]["training_batch_hash"],
                dense_manifest["identity"]["validation_batch_hash"],
                sparsity,
            )
            if not _is_complete(sparse_dir, sparse_identity):
                continue
            source_ce = sparse_manifest.get("source_sparse_validation_cross_entropy_at_2500")
            if source_ce is None:
                source_ce = sparse_manifest.get("source_validation_cross_entropy")
            sparse_5000 = float(sparse_manifest.get("extension_final_validation_cross_entropy", sparse_manifest.get("final_validation_loss")))
            rows.append({
                "num_experts": experts,
                "seed": seed,
                "sparsity": sparsity,
                "continuation_protocol": "fresh_optimizer_for_sparse_and_dense",
                "dense_validation_cross_entropy_2500": dense_2500,
                "sparse_validation_cross_entropy_2500": float(source_ce),
                "sparse_minus_dense_gap_2500": float(source_ce) - dense_2500,
                "dense_validation_cross_entropy_5000": dense_5000,
                "sparse_validation_cross_entropy_5000": sparse_5000,
                "sparse_minus_dense_gap_5000": sparse_5000 - dense_5000,
                "winning_ticket_claim": "not_inferred_from_unequal_step_budgets",
            })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "sparse_dense_comparison.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = [
        "num_experts", "seed", "sparsity", "continuation_protocol",
        "dense_validation_cross_entropy_2500", "sparse_validation_cross_entropy_2500",
        "sparse_minus_dense_gap_2500", "dense_validation_cross_entropy_5000",
        "sparse_validation_cross_entropy_5000", "sparse_minus_dense_gap_5000",
        "winning_ticket_claim",
    ]
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_matrix(groups: list[tuple[int, int]]) -> list[dict]:
    if not torch.__version__.startswith("2.7.0"):
        raise RuntimeError("The saved IMP protocol requires the verified torch 2.7.0 environment")
    if not torch.cuda.is_available():
        raise RuntimeError("The source recovery used CUDA FP16; refusing a CPU fallback that changes the protocol")
    _prepare_sources(groups)
    pairs = _canonical_pairs()
    results = []
    for experts, seed in groups:
        pair = pairs[(experts, seed, 0.80)]
        config, training_batches, validation_batches, train_hash, validation_hash = _materialize_group_stream(experts, seed, pair)
        dense_path = _dense_source(pair, experts, seed)
        dense_payload = torch.load(dense_path, map_location="cpu", weights_only=False)
        dense_config = dense_payload["config"]
        if any(dense_config.get(section) != config.get(section) for section in ("model", "data", "routing")):
            raise RuntimeError(f"Matched dense model/data/routing config differs for E{experts} seed{seed}")
        optimizer_fields = ("learning_rate", "weight_decay", "grad_clip", "precision", "batch_size", "steps")
        if any(dense_config.get("training", {}).get(field) != config.get("training", {}).get(field) for field in optimizer_fields):
            raise RuntimeError(f"Matched dense optimizer/training config differs for E{experts} seed{seed}")
        dense_ce = float(pair["dense_final_validation_loss"])
        _run_extension(
            experts, seed, "dense", dense_path, None, config, {},
            training_batches, validation_batches, train_hash, validation_hash, dense_ce, None,
        )
        del dense_payload
        for sparsity in SPARSITIES:
            stage_dir, source_path, mask_path = _source_paths(experts, seed, sparsity)
            stage = _json(stage_dir / "stage.json")
            if stage["training_batch_sequence_hash"] != pair["data_hash"]:
                raise RuntimeError(f"IMP training sequence hash mismatch at {stage_dir}")
            masks = load_masks(mask_path)
            source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
            result = _run_extension(
                experts, seed, "sparse", source_path, stage, source_payload["config"], masks,
                training_batches, validation_batches, train_hash, validation_hash, dense_ce, sparsity,
            )
            results.append(result)
            del source_payload
        del config, training_batches, validation_batches
        _write_comparisons(pairs, groups)
    return results


def _plan(groups: list[tuple[int, int]]) -> dict:
    pairs = _canonical_pairs()
    entries = []
    for experts, seed in groups:
        source_status = {str(sparsity): _source_ready(experts, seed, sparsity) for sparsity in SPARSITIES}
        for sparsity in SPARSITIES:
            _dense_source(pairs[(experts, seed, sparsity)], experts, seed)
        entries.append({
            "num_experts": experts,
            "seed": seed,
            "source_recovery_ready": source_status,
            "matched_dense_baselines_verified": True,
            "discovery_steps_per_stage": START_STEP,
            "total_final_recovery_steps": TARGET_STEP,
            "extensions": ["dense", "sparse_0.80", "sparse_0.85"],
        })
    return {"execution_started": False, "groups": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="prepare only missing IMP sources, then run sparse and dense extensions sequentially")
    parser.add_argument(
        "--groups", nargs="*", default=[f"{experts}:{seed}" for experts in EXPERT_COUNTS for seed in SEEDS],
        help="architecture:seed pairs; defaults to all six requested pairs",
    )
    args = parser.parse_args()
    try:
        groups = [tuple(map(int, value.split(":"))) for value in args.groups]
    except ValueError:
        parser.error("groups must use architecture:seed notation")
    if not groups or len(set(groups)) != len(groups) or any(
        experts not in EXPERT_COUNTS or seed not in SEEDS for experts, seed in groups
    ):
        parser.error("groups must be unique pairs selected from E8/E16/E32 and seeds 7/17")
    if not args.execute:
        print(json.dumps(_plan(groups), indent=2))
        return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / ".run.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another extended recovery is using {OUTPUT_ROOT}") from error
        results = run_matrix(groups)
        print(json.dumps({
            "completed_or_skipped_sparse_extensions": len(results),
            "comparison_csv": str(OUTPUT_ROOT / "sparse_dense_comparison.csv"),
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()