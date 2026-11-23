"""Audit immutable sources and register the post-pruning factorial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

from common import (
    ATOL,
    CE_ATOL,
    CONDITIONS,
    GLOBAL_STEPS,
    MODEL_TYPES,
    OFFSETS,
    OUT,
    ROOT,
    SEEDS,
    TRANSITION,
    RTOL,
    csv_save,
    data_manifest,
    group_for_parameter,
    json_read,
    json_save,
    materialize_data,
    sha256,
    torch_load,
    unique_parameter_rows,
)

sys.path.insert(0, str(ROOT / "src"))

import torch
from moe_lth.pruning.masks import load_masks
from moe_lth.pruning.router_age import load_model_from_checkpoint


CODE_PATHS = (
    "src/moe_lth/models/transformer.py",
    "src/moe_lth/models/moe_layer.py",
    "src/moe_lth/models/experts.py",
    "src/moe_lth/models/router.py",
    "src/moe_lth/training/checkpoint.py",
    "src/moe_lth/training/evaluate.py",
    "src/moe_lth/utils.py",
    "src/moe_lth/data.py",
    "src/moe_lth/pruning/masks.py",
    "src/moe_lth/pruning/rewind.py",
    "src/moe_lth/pruning/router_age.py",
    "src/moe_lth/experiments/run_router_age_recovery.py",
    "scripts/imp_mask_transition_v1/common.py",
    "scripts/imp_mask_transition_v1/engine.py",
    "scripts/imp_mask_transition_v1/prepare.py",
    "scripts/imp_mask_transition_v1/worker.py",
    "scripts/imp_80_85/run.py",
    "scripts/imp_extended_recovery/pipeline.py",
    "scripts/imp_adaptation_factorial_v1/af_common.py",
    "scripts/imp_adaptation_factorial_v1/af_engine.py",
    "scripts/imp_adaptation_factorial_v1/af_native.py",
    "scripts/imp_adaptation_factorial_v1/af_diagnostics.py",
    "scripts/imp_postpruning_adaptation_v1/common.py",
    "scripts/imp_postpruning_adaptation_v1/diagnostics.py",
    "scripts/imp_postpruning_adaptation_v1/engine.py",
    "scripts/imp_postpruning_adaptation_v1/prepare.py",
    "scripts/imp_postpruning_adaptation_v1/worker.py",
    "scripts/imp_postpruning_adaptation_v1/analyze.py",
    "scripts/imp_postpruning_adaptation_v1/finalize_step1500.py",
)

from diagnostics import _fixed_positions, probe_identity, probe_indices


def variants() -> list[dict]:
    return json_read(TRANSITION / "manifest.json")["variants"]


def abrupt_attempt(seed: int) -> Path:
    attempts = sorted((TRANSITION / "attempts" / f"seed{seed}_abrupt").glob("attempt_*"))
    if len(attempts) != 1:
        raise RuntimeError(f"Expected one immutable abrupt attempt for seed {seed}; found {len(attempts)}.")
    return attempts[0]


def result_metric(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json_read(path)


def transition_ce_by_id_and_step() -> dict[tuple[str, int], float]:
    rows = []
    with (TRANSITION / "trajectory_metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        result[(row["id"], int(row["step"]))] = float(row["ce"])
    return result


def refresh_historical_contrasts() -> None:
    path = OUT / "historical_transition_contrasts.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ce = transition_ce_by_id_and_step()
    for row in rows:
        seed = int(row["seed"])
        abrupt = ce[(f"seed{seed}_abrupt", 1000)]
        fixed = ce[(f"seed{seed}_fixed", 1000)]
        row["abrupt_minus_fixed_at_1000"] = abrupt - fixed
        row["fixed_step_1000_ce"] = fixed
    csv_save(path, rows)
    values = [float(row["abrupt_minus_fixed_at_1000"]) for row in rows]
    mean_gap = sum(values) / len(values)
    audit_path = OUT / "audit.md"
    audit = audit_path.read_text(encoding="utf-8")
    audit = audit.replace(
        "mean **nan**.",
        f"mean **{mean_gap:.6f}**.",
    )
    audit_path.write_text(audit, encoding="utf-8")


def mask_rows(seed: int, mask_path: Path) -> tuple[list[dict], list[dict]]:
    masks = load_masks(mask_path)
    model = load_model_from_checkpoint(
        next(v for v in variants() if v["seed"] == seed)["config"]["model"],
        str(next(v for v in variants() if v["seed"] == seed)["initial"]),
        torch.device("cpu"),
    )
    parameters = dict(model.named_parameters())
    rows = []
    for name, mask in sorted(masks.items()):
        kept = int(mask.sum())
        rows.append(
            {
                "seed": seed,
                "parameter": name,
                "shape": json.dumps(list(mask.shape), separators=(",", ":")),
                "parameter_count": mask.numel(),
                "retained_count": kept,
                "pruned_count": mask.numel() - kept,
                "sparsity": 1 - kept / mask.numel(),
                "group": group_for_parameter(name),
            }
        )
        if name not in parameters or parameters[name].shape != mask.shape:
            raise RuntimeError(f"Mask/model mismatch: seed {seed} parameter {name}.")
    partition = unique_parameter_rows(model, masks, seed)
    del model
    return rows, partition


def source_paths(variant: dict) -> dict[str, Path]:
    seed = int(variant["seed"])
    abrupt = abrupt_attempt(seed)
    dense = Path(variant["references"]["dense"]["snapshots"]["1000"])
    fixed = Path(variant["references"]["fixed"]["snapshots"]["1000"])
    return {
        "initialization": Path(variant["initial"]),
        "imp80_mask": Path(variant["mask"]),
        "abrupt_pre_pruning_dense_equivalent_candidate": dense,
        "abrupt_post_pruning": abrupt / "snapshots/step_1000.pt",
        "unpruned_step_1000": dense,
        "fixed_sparse_comparator": fixed,
        "abrupt_event": abrupt / "segment_0/event_1000.json",
        "abrupt_validation_pre": abrupt / "segment_0/grid_1000_pre.json",
        "abrupt_validation_post": abrupt / "segment_0/grid_1000.json",
        "abrupt_validation_1250": abrupt / "segment_0/grid_1250.json",
        "abrupt_validation_1500": abrupt / "segment_0/grid_1500.json",
    }


def build_audit() -> None:
    out = OUT
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing study output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "prefixes").mkdir()
    (out / "attempts").mkdir()
    (out / "status").mkdir()
    manifest = variants()
    if tuple(v["seed"] for v in manifest) != SEEDS:
        raise RuntimeError("Transition manifest seed order/cohort differs from registered cohort.")

    artifacts: list[dict] = []
    compatibility: list[dict] = []
    coverage: list[dict] = []
    partitions: list[dict] = []
    masks_inventory: list[dict] = []
    input_hashes: dict[str, str] = {}
    batch_rows: list[dict] = []
    references: dict[int, dict] = {}
    run_rows: list[dict] = []

    for variant in manifest:
        seed = int(variant["seed"])
        paths = source_paths(variant)
        references[seed] = paths
        for kind, path in paths.items():
            if not path.is_file():
                if kind in {"fixed_sparse_comparator"}:
                    continue
                raise FileNotFoundError(f"Missing required {kind} artifact for seed {seed}: {path}")
            digest = sha256(path)
            input_hashes[str(path)] = digest
            artifacts.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "path": str(path),
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                }
            )

        mask_path = paths["imp80_mask"]
        per_tensor, partition = mask_rows(seed, mask_path)
        masks_inventory.extend(per_tensor)
        partitions.extend(partition)

        payload = torch_load(paths["initialization"])
        if payload.get("step") != 0:
            raise RuntimeError(f"Seed {seed} initialization checkpoint is not step zero.")
        init_keys = sorted(payload)
        del payload

        dense_payload = torch_load(paths["unpruned_step_1000"])
        dense_model_keys = sorted(dense_payload)
        dense_payload_model = dense_payload.get("model", {})
        dense_model_only = "optimizer" not in dense_payload
        dense_step = dense_payload.get("step")
        del dense_payload, dense_payload_model

        sparse_payload = torch_load(paths["abrupt_post_pruning"])
        sparse_model_only = "optimizer" not in sparse_payload
        sparse_step = sparse_payload.get("step")
        sparse_payload_keys = sorted(sparse_payload)
        del sparse_payload

        compatibility.append(
            {
                "seed": seed,
                "initial_checkpoint": str(paths["initialization"]),
                "initial_step": 0,
                "initial_payload_keys": json.dumps(init_keys),
                "initial_has_full_optimizer_state": False,
                "dense_step_1000_candidate": str(paths["unpruned_step_1000"]),
                "dense_step": dense_step,
                "dense_payload_keys": json.dumps(dense_model_keys),
                "dense_has_optimizer_scaler_rng_data_position": not dense_model_only,
                "abrupt_pre_event_candidate": str(paths["abrupt_pre_pruning_dense_equivalent_candidate"]),
                "abrupt_pre_event_is_verified_same_as_dense": "pending numerical check",
                "abrupt_post_event_checkpoint": str(paths["abrupt_post_pruning"]),
                "abrupt_post_step": sparse_step,
                "abrupt_post_payload_keys": json.dumps(sparse_payload_keys),
                "abrupt_post_has_optimizer_scaler_rng_data_position": not sparse_model_only,
                "exact_branch_state_available_at_1000": False,
                "required_action": "reconstruct common 1000-update dense prefix; verify model/loss; apply exact mask and zero masked Adam moments",
                "tolerance": f"model allclose atol={ATOL}, rtol={RTOL}; validation CE abs diff <= {CE_ATOL}; bitwise hash also recorded",
            }
        )

        train, validation = materialize_data(variant, count=5000)
        dm = data_manifest(variant, train, validation)
        batch_rows.extend(
            {
                "seed": seed,
                **row,
            }
            for row in dm["next_500_batches"]
        )
        input_hashes[f"training-seed-{seed}-next500"] = dm["next_500_batches_sha256"]
        input_hashes[f"validation-seed-{seed}"] = dm["validation_sha256"]

        abrupt_root = abrupt_attempt(seed)
        dense_snapshots = variant["references"]["dense"]["snapshots"]
        required_source = {
            1000: abrupt_root / "segment_0/grid_1000.json",
            1025: Path(dense_snapshots.get("1025", "")),
            1100: Path(dense_snapshots.get("1100", "")),
            1250: abrupt_root / "segment_0/grid_1250.json",
            1500: abrupt_root / "segment_0/grid_1500.json",
        }
        dense_by_step = {
            int(step): Path(path)
            for step, path in dense_snapshots.items()
        }
        for model_type in MODEL_TYPES:
            for offset, global_step in zip(OFFSETS, GLOBAL_STEPS):
                if model_type == "sparse":
                    path = abrupt_root / "segment_0" / f"grid_{global_step:04d}.json"
                else:
                    path = dense_by_step.get(global_step)
                    if path is None:
                        path = Path("")
                available = path.is_file() if str(path) else False
                coverage.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "branch_offset": offset,
                        "global_step": global_step,
                        "required_eval_present": available,
                        "existing_artifact": str(path) if available else "",
                        "existing_diagnostic_compatible": False,
                        "note": "existing runs do not contain the complete factorial diagnostic grid" if not available else "historical full-model snapshot; re-evaluate/verify before reuse",
                    }
                )

        event = json_read(paths["abrupt_event"])
        pre = json_read(paths["abrupt_validation_pre"])
        post = json_read(paths["abrupt_validation_post"])
        step1500 = json_read(paths["abrupt_validation_1500"])
        transition_ce = transition_ce_by_id_and_step()
        fixed_ce = transition_ce[(f"seed{seed}_fixed", 1000)]
        run_rows.append(
            {
                "seed": seed,
                "paired_pre_post_validation_damage": post["ce"] - pre["ce"],
                "paired_pre_post_probe_damage": event["damage"],
                "abrupt_minus_fixed_at_1000": post["ce"] - fixed_ce,
                "abrupt_absolute_ce_improvement_1000_to_1500": post["ce"] - step1500["ce"],
                "pre_event_ce": pre["ce"],
                "post_event_ce": post["ce"],
                "fixed_step_1000_ce": fixed_ce,
                "abrupt_step_1500_ce": step1500["ce"],
                "paired_validation_target_tokens": pre["target_tokens"],
                "pre_event_path": str(paths["abrupt_validation_pre"]),
                "post_event_path": str(paths["abrupt_validation_post"]),
            }
        )
        del train, validation

    # Identify already available full-control endpoints without asserting they
    # supply the missing offset-25/100 measures or intervention diagnostics.
    reuse_candidates = []
    for v in manifest:
        seed = int(v["seed"])
        for model_type, condition in (("unpruned", "dense"), ("sparse", "abrupt")):
            if condition == "dense":
                path = Path(v["references"]["dense"]["snapshots"]["1500"])
            else:
                path = abrupt_attempt(seed) / "snapshots/step_1000.pt"
            reuse_candidates.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "condition": "full",
                    "candidate_endpoint": str(path),
                    "candidate_exists": path.is_file(),
                    "reusable_for_primary_endpoint": "verify after exact continuation",
                    "full_trajectory_reusable": False,
                    "reason": "missing required offset-25/100 checkpoints and new diagnostic measurements",
                }
            )

    transition_resource = json_read(TRANSITION / "resource_estimate.json")
    update_rates = [
        float(item["wall_seconds"]) / int(item["attempts"])
        for item in transition_resource["benchmark"]
    ]
    median_seconds_per_update = statistics.median(update_rates)
    new_branch_updates = 40 * 500
    prefix_updates = len(SEEDS) * 1000
    diagnostic_allowance = 0.20
    estimate = {
        "status": "pre_dispatch_estimate",
        "planned_endpoints": 40,
        "planned_new_branch_continuations": 40,
        "planned_reusable_full_endpoint_candidates": 10,
        "reusable_full_endpoint_claim": "candidate only; requires numerical verification; trajectories are not reusable because offset-25/100 and the requested diagnostics are absent",
        "new_branch_attempted_updates": new_branch_updates,
        "prefix_reconstruction_attempted_updates": prefix_updates,
        "engineering_pilot_branch_updates": 8 * 500,
        "engineering_checks_excluded_from_core_cohort": "GPU equivalence, interruption/resume, and measurement probes; accounted after host benchmarks",
        "benchmark_source": str(TRANSITION / "resource_estimate.json"),
        "benchmark_seconds_per_update_by_host": update_rates,
        "median_seconds_per_update": median_seconds_per_update,
        "diagnostic_overhead_allowance": diagnostic_allowance,
        "estimated_branch_gpu_hours": new_branch_updates * median_seconds_per_update * (1 + diagnostic_allowance) / 3600,
        "estimated_prefix_gpu_hours": prefix_updates * median_seconds_per_update * (1 + diagnostic_allowance) / 3600,
        "estimated_core_gpu_hours_total": (new_branch_updates + prefix_updates) * median_seconds_per_update * (1 + diagnostic_allowance) / 3600,
        "minimum_free_gpu_bytes": 24 * 1024**3,
        "minimum_free_disk_bytes": 200 * 1024**3,
        "maximum_gpu_utilization_percent": 90,
        "workers_per_host": 1,
        "storage_free_bytes_at_registration": __import__("shutil").disk_usage(OUT).free,
        "estimated_storage_bytes": int(40 * 5 * 822_000_000 + 5 * 2 * 822_000_000 + 10_000_000_000),
        "storage_estimate_note": "upper bound assumes five full-state resumable checkpoints per 500-update endpoint plus dense/sparse prefix states and diagnostics; verify against actual filesystem before launch",
        "dispatch_status": "not dispatched",
    }

    condition_manifest = []
    for seed in SEEDS:
        for model_type in MODEL_TYPES:
            for condition in CONDITIONS:
                condition_manifest.append(
                    {
                        "job_id": f"seed{seed}_{model_type}_{condition}",
                        "seed": seed,
                        "model_type": model_type,
                        "condition": condition,
                        "router_trainable": condition in ("full", "experts_frozen"),
                        "experts_trainable": condition in ("full", "router_frozen"),
                        "shared_trainable": True,
                        "start_global_step": 1000,
                        "end_global_step": 1500,
                        "attempted_updates": 500,
                        "eval_offsets": json.dumps(OFFSETS),
                        "source_prefix": str(OUT / "prefixes" / f"seed{seed}"),
                        "source_full_state_available": False,
                        "prefix_reconstruction_required": True,
                        "full_endpoint_candidate": condition == "full",
                        "planned_reuse": "endpoint candidate only after validation" if condition == "full" else "no",
                    }
                )

    input_hashes[str(TRANSITION / "manifest.json")] = sha256(TRANSITION / "manifest.json")
    input_hashes[str(TRANSITION / "seal.json")] = sha256(TRANSITION / "seal.json")
    input_hashes[str(TRANSITION / "input_hashes.json")] = sha256(TRANSITION / "input_hashes.json")
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation source missing: {path}")
        input_hashes[str(path)] = sha256(path)

    protocol = f"""# Post-pruning router × expert adaptation experiment

## Status and selection

This is a prospective intervention motivated by observed trajectories, not a
preregistered overall study. The global-step 1000–1500 window was selected after
examining prior trajectories; it is therefore not outcome-independent. The mask
cohort is existing, native 32E IMP80, with seeds 7, 17, 29, 43, and 61. Seed 17
is the predetermined engineering pilot. Every seed is retained regardless of
historical or new matching status. No new mask search, source training, sparsity,
architecture, pruning schedule, or training beyond global step 1500 is allowed.

## Conditions and state

At step 1000, compare the abrupt IMP80 post-deletion state with its compatible
unpruned state. Reconstruct the common original native prefix only because saved
step-1000 artifacts do not contain the complete optimizer/scaler/RNG/data state.
Verify reconstructed weights and validation loss against historical checkpoints
using the tolerances below. The abrupt pre-event model is expected to equal the
dense model because both share the same unpruned prefix; test this explicitly.
If this reconstruction cannot be established, stop before production dispatch.

For each seed and model type, all four branches clone one identical complete
branch-start state, including parameters/buffers, AdamW groups/steps/moments,
GradScaler, Python/NumPy/Torch CPU/CUDA RNG, exact next-batch index, event count,
and masks. No optimizer or RNG reset is permitted at this boundary.

| Condition | Router | Experts | Shared |
|---|---|---|---|
| full | train | train | train |
| router_frozen | frozen | train | train |
| experts_frozen | train | frozen | train |
| router_experts_frozen | frozen | frozen | train |

The partition is complete and disjoint. Experts include all routed `fc1`/`fc2`
weights and biases. Tied embedding/output parameters are counted once. All shared
parameters remain trainable. Freezing uses `requires_grad=False`, keeps forward
computation and upstream gradient propagation, preserves optimizer parameter
groups and state, and leaves frozen gradients absent so AdamW neither updates nor
decays those parameters. For sparse models, keep the exact IMP80 mask, masked
gradient hooks, masked-weight enforcement, and native coordinate-shaped moment
zeroing. Native clipping and resulting optimizer responses are part of the
training-policy intervention.

## Schedule and measurements

Each endpoint is exactly 500 attempted updates, global steps 1001–1500. AMP skips
do not extend duration. Keep native learning rate, AdamW betas/epsilon/decay,
gradient clipping, FP16/GradScaler, auxiliary objective, full-softmax/selected
gates, capacity rules, dropout and model modes. All branch starts within a
seed/type receive the same RNG and data position. The exact next 500 original
training batches are used.

Evaluate task-only CE at offsets 0, 25, 100, 250, 500 (global steps 1000, 1025,
1100, 1250, 1500), FP32, TF32 disabled, eval/inference, over identical complete
validation batches including the original partial final batch, target-token
weighted. Require equal branch baseline losses within `atol={CE_ATOL}`.
Primary endpoint penalties are `P_R=L_router_frozen-L_full`,
`P_E=L_experts_frozen-L_full`, and `P_RE=L_router_experts_frozen-L_full`, within
each model type. Sparse-excess penalties are `E_R=P_R_sparse-P_R_unpruned` and
`E_E=P_E_sparse-P_E_unpruned`. CE-scale factorial interaction is
`I=L_11-L_10-L_01+L_00`, with 1 indicating adaptation; report sparse, unpruned,
and sparse-minus-unpruned. Also report absolute CE reduction and trapezoidal
integrated excess CE and normalized average excess over the fixed observed grid.
No interpolation beyond linear trapezoidal integration. No fraction-of-pruning-
damage recovery score or primary winner label.

The deterministic diagnostic probe is four complete batches selected by
SHA256 ordering of batch indices from the verified training stream before any
branch outcome is measured. It may overlap training data and is not an
independent test set. Use the same batch structure across times. At every
registered offset record proposed/accepted route loads, dropped assignments,
assignment agreement, accepted-set Jaccard (empty union undefined), selected
gate distributions and gate changes for all tokens, unchanged proposed experts,
and separately proposed/accepted populations. Record expert functions on fixed
branch-start and current native layer-input populations with expert and position
identity; bounded token positions and zero-norm/low-count exclusions are
reported. Record router-input displacement/similarity, component displacement
(attention, normalization, tied embedding/head where applicable), per-update
parameter-change norms by component, unscaled/pre-clip gradient norms, clipping
coefficients, AMP skips/scaler, and path-length versus net displacement.
Diagnostics may not change RNG, modes, gradients, buffers, optimizer/scaler, masks,
or data position. Store streaming summaries and bounded caches, not all gradients
or deltas; report overhead.

## Numerical checks and stopping

Before production verify native continuation equivalence, model restoration,
identical branch-start forwards, bitwise frozen parameters and Adam state,
trainable-state preservation, gradient propagation through frozen modules,
sparse-mask persistence, tied-parameter handling, interruption/resume, batch
sequence and attempted/applied accounting, weighted validation and precision,
and diagnostic side-effect preservation. Run the seed-17 engineering pilot, then
dispatch all remaining seeds after checks pass regardless of scientific effect.
No effect-size or significance gate controls completion. Record model equivalence
as bitwise or tolerance-based; locked tolerances are model allclose
`atol={ATOL}, rtol={RTOL}` and CE absolute difference `<= {CE_ATOL}`. Use one
worker per Viper/Adder host, existing minimum-free-memory admission, no dispatch
above 90% GPU utilization, shared exclusive claims, atomic statuses/heartbeats, unique resumable attempts and
validated completion hashes. Do not kill unrelated jobs.

## Inference

The source seed is the independent unit (`n=5`). Show individual seed effects
first, then mean, median, range, sample SD, and explicitly descriptive seed-level
intervals. No token/expert/batch/checkpoint pseudoreplication, winner classifier,
feature search, or pooled correlation fishing. Freezing estimates the effect of
restricting adaptation under this training policy. Frozen router weights do not
freeze assignments; unchanged assignments do not imply unchanged gates. Component
movement is not a causal contribution to CE. Shared-only improvement does not
show frozen experts are functionally irrelevant. Routing diagnostics do not
isolate assignment, gate, or capacity mechanisms. Results do not establish
pre-pruning recoverability prediction, universality, a pruning algorithm, a
token-routing mechanism, or an explanation for late sparse outperformance.
Late dense deterioration is a separate analysis-only audit. Recommend at most one
falsifiable follow-up; do not launch it.
"""
    (out / "protocol.md").write_text(protocol, encoding="utf-8")
    registration = {
        "status": "protocol prepared; numerical equivalence and pilot pending",
        "label": "Prospective intervention motivated by observed trajectories; not an overall preregistration.",
        "window_selection": "1000-1500 was selected after examining previous trajectories.",
        "seeds": list(SEEDS),
        "model_types": list(MODEL_TYPES),
        "conditions": list(CONDITIONS),
        "endpoints": len(condition_manifest),
        "offsets": list(OFFSETS),
        "global_steps": list(GLOBAL_STEPS),
        "pilot_seed": 17,
        "cohort_dispatch_gate": "engineering/resource validation only; no effect-size gate",
        "sparse_condition": "existing native 32E IMP80 abrupt event at global step 1000",
        "control": "verified compatible unpruned trajectory at global step 1000",
        "reconstruction_required": True,
        "tolerances": {"model_atol": ATOL, "model_rtol": RTOL, "validation_ce_atol": CE_ATOL},
        "primary_contrasts": [
            "P_R=L_router_frozen-L_full",
            "P_E=L_experts_frozen-L_full",
            "P_RE=L_router_experts_frozen-L_full",
            "E_R=P_R_sparse-P_R_unpruned",
            "E_E=P_E_sparse-P_E_unpruned",
            "I=L11-L10-L01+L00",
        ],
        "uncertainty": "descriptive seed-level mean, median, range, sample SD, and intervals",
        "observed_window_acknowledged": True,
        "sealed_before_new_branch_outcomes": True,
    }
    json_save(out / "registration.json", registration)
    json_save(out / "resource_estimate.json", estimate)
    json_save(out / "input_code_configuration_hashes.json", input_hashes)
    csv_save(out / "artifact_inventory.csv", artifacts)
    csv_save(out / "checkpoint_compatibility.csv", compatibility)
    csv_save(out / "measurement_coverage.csv", coverage)
    csv_save(out / "parameter_partition.csv", partitions)
    csv_save(out / "imp80_tensor_counts.csv", masks_inventory)
    csv_save(out / "condition_manifest.csv", condition_manifest)
    csv_save(out / "next_500_batch_hashes.csv", batch_rows)
    csv_save(out / "historical_full_endpoint_reuse_candidates.csv", reuse_candidates)
    csv_save(out / "historical_transition_contrasts.csv", run_rows)

    damage = sum(row["paired_pre_post_validation_damage"] for row in run_rows) / len(run_rows)
    gap = sum(row["abrupt_minus_fixed_at_1000"] for row in run_rows) / len(run_rows)
    improvement = sum(
        row["abrupt_absolute_ce_improvement_1000_to_1500"] for row in run_rows
    ) / len(run_rows)
    audit = f"""# Artifact and diagnostic audit

## Provenance and full-state compatibility

The source artifacts, paths, SHA-256 hashes, and sizes are recorded in
`artifact_inventory.csv`. The original initialization checkpoints are step zero
and model-only. The available dense step-1000 checkpoint is a historical
model-only snapshot. The abrupt post-event step-1000 checkpoint is also
model-only. No exact complete branch state at the event boundary is available
(no optimizer, scaler, RNG, or global data cursor in those saved step-1000
snapshots). The initial-optimizer snapshots from older audits do not include the
complete state at step 1000. Therefore production must reconstruct the common
prefix with the original native implementation, init, mask schedule, RNG, and
deterministic stream; compare the reconstructed model and validation loss to
historical checkpoints under the registered tolerances; then save a complete
prefix checkpoint. If it fails, stop production.

For abrupt pre-deletion state, the dense step-1000 snapshot is the candidate
common-prefix reference. The equality between abrupt pre-event and dense is
expected from identical unpruned training but remains to be explicitly
verified; it is not assumed. Abrupt post-deletion state is the retained
step-1000 model snapshot in the transition attempt. Per-tensor IMP80 mask counts
are in `imp80_tensor_counts.csv`. Branch state must additionally include the
native optimizer's scalar step counters and coordinate moments, GradScaler,
Python/NumPy/Torch CPU/CUDA RNG states and `sampler_next_batch=1000`.

Training stream and exact attempted updates 1001–1500 are hashed for each seed
in `next_500_batch_hashes.csv`. Validation hash, batch count, per-batch target
counts and total targets are included in `artifact_inventory.csv` and the input
hash file. The validation set includes the original partial final batch. Existing
complete-native continuation outcomes and the missing factorial grid are
summarized in `measurement_coverage.csv`. The existing full controls have
candidate endpoint data, but do not provide required offset-25/100 model
snapshots or all new diagnostics. Consequently the pre-dispatch estimate
conservatively plans 40 exact 500-attempt continuations (20,000 branch updates),
plus five necessary 1,000-update common-prefix reconstructions (5,000 updates).
Ten historical full-endpoint values are candidates for reuse only after exact
continuation verification; they do not replace the missing trajectories or
diagnostics. This is not a claim that 40 endpoint rows inherently require 40
new jobs; the additional full continuations are planned because of the missing
registered timepoints/diagnostics.

## Existing trajectory review

The historical 1000–1500 window was selected after previous results had been
examined. This experiment is not an overall preregistration. For clarity, the
three quantities below remain distinct:

* Paired pre/post-deletion validation CE change at step 1000: mean
  **{damage:.6f}** over five seeds (positive means CE worsened). Per-seed paired
  values are in `historical_transition_contrasts.csv`; they use the same full
  validation batches before and after the abrupt deletion.
* Abrupt-minus-fixed step-1000 trajectory gap: mean **{gap:.6f}**. This is a
  comparison of different training trajectories, not immediate pruning damage.
  The approximately 0.498 gap is not described as deletion damage.
* Absolute abrupt CE improvement from global step 1000 to 1500: mean
  **{improvement:.6f}** (positive means CE fell during continuation).

The paired pre/post measure is also present in the abrupt attempt's `grid_1000_pre`
and post-event `grid_1000`; the event diagnostic separately records a fixed
training-probe damage measure. These two paired readouts are not interchangeable.
Existing routing/gating/capacity and update logs are listed as input artifacts.
Their coverage limitations are explicit: they describe the prior full adaptation
trajectory and do not identify a routing mechanism or the causal contribution of
parameter movement.

## Output and scope controls

The output namespace is isolated at `results/imp_postpruning_adaptation_v1/`.
No source experiment, mask, checkpoint, manifest or report was modified. No new
training has been dispatched by this preparation step. No causal claim about
late sparse outperformance, an online pruning algorithm, a token-routing
mechanism, or a pre-pruning predictor is authorized by these measurements.
"""
    (out / "audit.md").write_text(audit, encoding="utf-8")


def refresh_probe_seal() -> None:
    """Freeze data-derived probe identities and current source hashes pre-dispatch."""
    registration_path = OUT / "registration.json"
    if not registration_path.is_file():
        raise FileNotFoundError("Study registration is missing; run --audit first.")
    status_dir = OUT / "status"
    branch_statuses = [
        path
        for path in status_dir.glob("*.json")
        if path.name != "last_resource_admission.json"
        and not path.name.endswith("_prefix.json")
    ]
    if branch_statuses or (OUT / "pilot_validation.json").exists():
        raise RuntimeError("Refusing to change probe registration after branch outcomes exist.")
    failed_prefixes = []
    for path in status_dir.glob("seed*_prefix.json"):
        status = json_read(path)
        error = str(status.get("error", ""))
        if (
            status.get("status") != "FAILED"
            or "prefix validation CE mismatch:" not in error
        ):
            raise RuntimeError(
                f"Refusing to reseal after a non-reference prefix failure: {path}"
            )
        failed_prefixes.append(path)
    registration = json_read(registration_path)
    previous_registration_sha256 = sha256(registration_path)
    old_probe_path = OUT / "diagnostic_probe_manifest.json"
    archive_probe_path = None
    if failed_prefixes:
        if not old_probe_path.is_file() or sha256(old_probe_path) != registration.get(
            "diagnostic_probe_manifest_sha256"
        ):
            raise RuntimeError("Failed-prefix reseal requires the intact prior probe registration.")
        revision = int(registration.get("registration_revision", 1))
        archive_probe_path = OUT / f"diagnostic_probe_manifest_revision_{revision}.json"
        if archive_probe_path.exists():
            raise FileExistsError(f"Prior probe registration archive already exists: {archive_probe_path}")
        archive_probe_path.write_bytes(old_probe_path.read_bytes())
    manifest = variants()
    probe_rows = []
    input_hashes = json_read(OUT / "input_code_configuration_hashes.json")
    for variant in manifest:
        seed = int(variant["seed"])
        train, _validation = materialize_data(variant, count=2500)
        indices = probe_indices(seed, len(train))
        batches = [train[index] for index in indices]
        identity = probe_identity(seed, indices, batches)
        token_count = sum(inputs.numel() for inputs, _targets in batches)
        batch_hashes = []
        for index, (inputs, targets) in zip(indices, batches, strict=True):
            digest = hashlib.sha256()
            for tensor in (inputs, targets):
                value = tensor.detach().cpu().contiguous()
                digest.update(str(tuple(value.shape)).encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.numpy().tobytes())
            batch_hashes.append({"batch_index": index, "sha256": digest.hexdigest()})
        probe_rows.append(
            {
                "seed": seed,
                "selection": "lowest SHA256 batch-index ranks from first 2500 verified training batches",
                "batch_indices": indices,
                "batch_count": len(batches),
                "probe_identity_sha256": identity,
                "batch_hashes": batch_hashes,
                "fixed_token_position_indices": _fixed_positions(seed, token_count),
                "token_count": token_count,
                "independent_test_set": False,
                "may_overlap_training": True,
            }
        )
        input_hashes[f"diagnostic-probe-seed-{seed}"] = identity
        del train, batches

    probe_path = OUT / "diagnostic_probe_manifest.json"
    json_save(probe_path, {"selection_rule": "SHA256 ordering fixed in diagnostics.py", "seeds": probe_rows})
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation source missing: {path}")
        input_hashes[str(path)] = sha256(path)
    trajectory_metrics = TRANSITION / "trajectory_metrics.csv"
    if not trajectory_metrics.is_file():
        raise FileNotFoundError(f"Historical trajectory metric table is missing: {trajectory_metrics}")
    input_hashes[str(trajectory_metrics)] = sha256(trajectory_metrics)
    json_save(OUT / "input_code_configuration_hashes.json", input_hashes)

    registration["diagnostic_probe_manifest"] = str(probe_path)
    registration["diagnostic_probe_manifest_sha256"] = sha256(probe_path)
    registration["diagnostic_probe_sealed_before_branch_outcomes"] = True
    registration["probe_identities"] = {
        str(row["seed"]): row["probe_identity_sha256"] for row in probe_rows
    }
    if failed_prefixes:
        registration["registration_revision"] = int(
            registration.get("registration_revision", 1)
        ) + 1
        registration["supersedes_registration_sha256"] = previous_registration_sha256
        registration["supersedes_probe_manifest"] = str(archive_probe_path)
        registration["probe_reseal_reason"] = (
            "Corrected the dense-prefix verification reference from the post-IMP80 "
            "readout to the historical dense step-1000 readout after a pre-outcome "
            "validation-only prefix failure."
        )
    json_save(registration_path, registration)
    resource_path = OUT / "resource_estimate.json"
    resource = json_read(resource_path)
    resource["minimum_free_gpu_bytes"] = 24 * 1024**3
    resource["minimum_free_disk_bytes"] = 200 * 1024**3
    resource["maximum_gpu_utilization_percent"] = 90
    resource["workers_per_host"] = 1
    resource["dispatch_status"] = "not dispatched"
    json_save(resource_path, resource)
    protocol_path = OUT / "protocol.md"
    protocol = protocol_path.read_text(encoding="utf-8")
    protocol = protocol.replace(
        "existing minimum-free-memory admission, shared",
        "existing minimum-free-memory admission, no dispatch above 90% GPU utilization, shared",
    )
    protocol_path.write_text(protocol, encoding="utf-8")


def register_engineering_revision() -> None:
    """Register the approved checkpoint-only finalization repair."""
    import af_native
    import engine

    registration_path = OUT / "registration.json"
    original_registration_path = OUT / "registration_revision_2.json"
    original_hashes_path = OUT / "input_code_configuration_hashes_revision_2.json"
    original_probe_path = OUT / "diagnostic_probe_manifest_revision_2.json"
    if not all(path.is_file() for path in (registration_path, original_registration_path, original_hashes_path, original_probe_path)):
        raise FileNotFoundError("Revision-2 registration archives are incomplete.")
    if (OUT / "registration_revision_3.json").exists():
        raise FileExistsError("Engineering registration revision 3 already exists.")
    if (OUT / "engineering_revision_3.json").exists():
        raise FileExistsError("Engineering revision record already exists.")

    original_registration = json_read(original_registration_path)
    if int(original_registration.get("registration_revision", 0)) != 2:
        raise RuntimeError("Only registration revision 2 can be amended by this repair.")
    if sha256(registration_path) != sha256(original_registration_path):
        raise RuntimeError("Current registration differs from its preserved revision-2 copy.")
    old_hashes = json_read(original_hashes_path)
    current_hashes = json_read(OUT / "input_code_configuration_hashes.json")
    if sha256(OUT / "input_code_configuration_hashes.json") != sha256(original_hashes_path):
        raise RuntimeError("Current source-hash table differs from its preserved revision-2 copy.")
    if sha256(OUT / "diagnostic_probe_manifest.json") != sha256(original_probe_path):
        raise RuntimeError("Probe manifest changed after registration revision 2.")

    status_dir = OUT / "status"
    branch_status_path = status_dir / "seed17_unpruned_full.json"
    prefix_status_path = status_dir / "seed17_prefix.json"
    failure_path = OUT / "pilot_validation.json"
    if not branch_status_path.is_file() or not prefix_status_path.is_file() or not failure_path.is_file():
        raise FileNotFoundError("Failed pilot status or verified prefix status is missing.")
    branch_status = json_read(branch_status_path)
    prefix_status = json_read(prefix_status_path)
    pilot_failure = json_read(failure_path)
    expected_error = "component_checkpoints does not exist"
    prefix_verification_path = OUT / "prefixes/seed17/prefix_verification.json"
    prefix_verification = json_read(prefix_verification_path)
    if prefix_status.get("status") == "RUNNING":
        try:
            os.kill(int(prefix_status["pid"]), 0)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError("Cannot verify whether the stale prefix process is still alive.") from exc
        else:
            raise RuntimeError("A prefix process is still running; refusing registration revision.")
        if prefix_verification.get("status") != "verified":
            raise RuntimeError("Stale prefix status has no verified step-1000 reconstruction.")
        stale_prefix_archive = status_dir / "seed17_prefix_revision_2_stale_running.json"
        if stale_prefix_archive.exists():
            raise FileExistsError(f"Stale prefix status archive already exists: {stale_prefix_archive}")
        shutil.copy2(prefix_status_path, stale_prefix_archive)
        prefix_status.update(
            status="COMPLETE",
            prefix_verification=str(prefix_verification_path),
            prefix_sha256=sha256(OUT / "prefixes/seed17/reconstruction_interruption.pt"),
            reconciled_stale_status_archive=str(stale_prefix_archive),
        )
        json_save(prefix_status_path, prefix_status)
    if (
        branch_status.get("status") != "FAILED"
        or expected_error not in str(branch_status.get("error", ""))
        or pilot_failure.get("status") != "FAIL"
        or expected_error not in str(pilot_failure.get("error", ""))
        or prefix_status.get("status") != "COMPLETE"
    ):
        raise RuntimeError("The recorded failure does not match the approved engineering repair.")
    branch_statuses = [
        path
        for path in status_dir.glob("seed*_*.json")
        if "_prefix" not in path.stem
    ]
    if branch_statuses != [branch_status_path]:
        raise RuntimeError(
            "Registration revision 3 is allowed only before any other pilot branch starts."
        )
    attempt_dir = Path(branch_status["attempt"]).resolve()
    if Path(pilot_failure.get("attempt", attempt_dir)).resolve() != attempt_dir:
        raise RuntimeError("Pilot failure and branch status refer to different attempts.")
    if (attempt_dir / "result.json").exists():
        raise FileExistsError("Failed attempt already has a result; refusing revision.")

    interruption = af_native.decode_numpy(torch_load(attempt_dir / "interruption.pt"))
    if (
        int(interruption.get("step", -1)) != 1500
        or int(interruption.get("global_step", -1)) != 1500
        or int(interruption.get("sampler_next_batch", -1)) != 1500
        or interruption.get("study_condition") != "full"
    ):
        raise RuntimeError("Failed pilot does not contain a valid global-step-1500 state.")
    updates = engine._read_jsonl(attempt_dir / "study_optimizer_and_updates.jsonl")
    update_keys = [(int(row["global_step"]), row["group"]) for row in updates]
    expected_keys = {
        (step, group)
        for step in range(1001, 1501)
        for group in ("router", "experts", "shared")
    }
    if len(updates) != 1500 or len(set(update_keys)) != 1500 or set(update_keys) != expected_keys:
        raise RuntimeError("Failed pilot update rows are incomplete or duplicated.")
    gradient_rows = engine._read_jsonl(attempt_dir / "gradient_stats/gradient_stats.jsonl")
    if (
        len(gradient_rows) != 500
        or [int(row["step"]) for row in gradient_rows] != list(range(1001, 1501))
    ):
        raise RuntimeError("Failed pilot native gradient rows are incomplete or duplicated.")
    if (attempt_dir / "checkpoints/final_recovered.pt").exists() or (
        attempt_dir / "component_checkpoints/final_router.pt"
    ).exists():
        raise FileExistsError("Native finalization artifacts partially exist; inspect manually.")

    for source, destination in (
        (branch_status_path, status_dir / "seed17_unpruned_full_revision_2_failed.json"),
        (failure_path, OUT / "pilot_validation_revision_2_failed.json"),
    ):
        if destination.exists():
            raise FileExistsError(f"Failure archive already exists: {destination}")
        shutil.copy2(source, destination)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"Failure archive differs from source: {destination}")

    allowed_changes = {
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/engine.py"),
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/diagnostics.py"),
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/prepare.py"),
    }
    for path_string, expected_hash in old_hashes.items():
        path = Path(path_string)
        if path.is_absolute() and path.is_file() and path_string not in allowed_changes:
            if sha256(path) != expected_hash:
                raise RuntimeError(f"Unexpected source change since revision 2: {path}.")

    new_hashes = dict(current_hashes)
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation source missing: {path}")
        new_hashes[str(path)] = sha256(path)
    json_save(OUT / "input_code_configuration_hashes.json", new_hashes)
    new_hash_table_sha256 = sha256(OUT / "input_code_configuration_hashes.json")
    code_hash_changes = {}
    for path_string, new_hash in new_hashes.items():
        old_hash = old_hashes.get(path_string)
        if old_hash != new_hash:
            code_hash_changes[path_string] = {"old": old_hash, "new": new_hash}
    revision_record = {
        "revision": 3,
        "failure": str(branch_status["error"]),
        "failed_attempt": str(attempt_dir),
        "failed_attempt_status_archive": str(status_dir / "seed17_unpruned_full_revision_2_failed.json"),
        "failed_pilot_status_archive": str(OUT / "pilot_validation_revision_2_failed.json"),
        "saved_checkpoint_step": 1500,
        "attempted_updates": 500,
        "unique_per_group_update_rows": len(update_keys),
        "training_updates_after_checkpoint": 0,
        "repair_scope": "idempotent creation of native output directories before fresh/resumed work and checkpoint-only finalization",
        "scientific_conditions_unchanged": True,
        "probe_manifest_sha256_unchanged": sha256(OUT / "diagnostic_probe_manifest.json"),
        "supersedes_registration": str(original_registration_path),
        "supersedes_registration_sha256": sha256(original_registration_path),
        "supersedes_source_hash_manifest": str(original_hashes_path),
        "supersedes_source_hash_manifest_sha256": sha256(original_hashes_path),
        "new_source_hash_manifest_sha256": new_hash_table_sha256,
        "source_hash_changes": code_hash_changes,
    }
    revision_path = OUT / "engineering_revision_3.json"
    json_save(revision_path, revision_record)
    registration = json_read(registration_path)
    registration["registration_revision"] = 3
    registration["supersedes_registration"] = str(original_registration_path)
    registration["supersedes_registration_sha256"] = sha256(original_registration_path)
    registration["supersedes_source_hash_manifest"] = str(original_hashes_path)
    registration["supersedes_source_hash_manifest_sha256"] = sha256(original_hashes_path)
    registration["source_hash_manifest"] = str(OUT / "input_code_configuration_hashes.json")
    registration["source_hash_manifest_sha256"] = new_hash_table_sha256
    registration["engineering_revision_record"] = str(revision_path)
    registration["engineering_revision_record_sha256"] = sha256(revision_path)
    registration["diagnostic_probe_manifest_unchanged_across_revision"] = True
    registration["scientific_conditions_changed_across_revision"] = False
    json_save(registration_path, registration)
    shutil.copy2(registration_path, OUT / "registration_revision_3.json")
    print(json.dumps(revision_record, indent=2))


def register_finalization_revision() -> None:
    """Register the follow-up correction to the checkpoint-only finalizer."""
    import af_native
    import engine

    registration_path = OUT / "registration.json"
    original_registration_path = OUT / "registration_revision_3.json"
    original_hashes_path = OUT / "input_code_configuration_hashes_revision_3.json"
    original_probe_path = OUT / "diagnostic_probe_manifest_revision_3.json"
    if not all(
        path.is_file()
        for path in (
            registration_path,
            original_registration_path,
            original_hashes_path,
            original_probe_path,
        )
    ):
        raise FileNotFoundError("Revision-3 registration archives are incomplete.")
    if (OUT / "registration_revision_4.json").exists() or (
        OUT / "engineering_revision_4.json"
    ).exists():
        raise FileExistsError("Engineering registration revision 4 already exists.")
    registration = json_read(registration_path)
    if int(registration.get("registration_revision", 0)) != 3:
        raise RuntimeError("Only registration revision 3 can be amended by this repair.")
    if sha256(registration_path) != sha256(original_registration_path):
        raise RuntimeError("Current registration differs from its preserved revision-3 copy.")
    if sha256(OUT / "input_code_configuration_hashes.json") != sha256(original_hashes_path):
        raise RuntimeError("Current source-hash table differs from its preserved revision-3 copy.")
    if sha256(OUT / "diagnostic_probe_manifest.json") != sha256(original_probe_path):
        raise RuntimeError("Probe manifest changed after registration revision 3.")

    status_dir = OUT / "status"
    branch_status_path = status_dir / "seed17_unpruned_full.json"
    finalizer_failure_path = status_dir / "seed17_finalization_revision_3_failed.json"
    branch_status = json_read(branch_status_path)
    finalizer_failure = json_read(finalizer_failure_path)
    if (
        branch_status.get("status") != "FAILED"
        or "component_checkpoints does not exist" not in str(branch_status.get("error", ""))
        or finalizer_failure.get("status") != "FAILED_BEFORE_OUTPUT_WRITES"
        or finalizer_failure.get("optimizer_updates_after_checkpoint") != 0
        or finalizer_failure.get("training_batches_consumed") != 0
    ):
        raise RuntimeError("Revision-3 finalizer failure record differs from the approved follow-up.")
    prefix_status = json_read(status_dir / "seed17_prefix.json")
    if prefix_status.get("status") != "COMPLETE":
        raise RuntimeError("Verified seed-17 prefix status is not complete.")
    branch_statuses = [
        path
        for path in status_dir.glob("seed*_*.json")
        if "_prefix" not in path.stem
        and "_finalization_" not in path.stem
        and "_revision_" not in path.stem
    ]
    if branch_statuses != [branch_status_path]:
        raise RuntimeError("Revision 4 is blocked because another pilot branch has started.")
    attempt_dir = Path(branch_status["attempt"]).resolve()
    if (attempt_dir / "result.json").exists():
        raise FileExistsError("Failed attempt already has a result; refusing revision 4.")
    interruption = af_native.decode_numpy(torch_load(attempt_dir / "interruption.pt"))
    if (
        int(interruption.get("step", -1)) != 1500
        or int(interruption.get("sampler_next_batch", -1)) != 1500
    ):
        raise RuntimeError("The step-1500 interruption checkpoint changed.")
    updates = engine._read_jsonl(attempt_dir / "study_optimizer_and_updates.jsonl")
    update_keys = [(int(row["global_step"]), row["group"]) for row in updates]
    expected_keys = {
        (step, group)
        for step in range(1001, 1501)
        for group in ("router", "experts", "shared")
    }
    if len(updates) != 1500 or len(set(update_keys)) != 1500 or set(update_keys) != expected_keys:
        raise RuntimeError("Saved pilot update rows changed during finalizer failure.")
    for relative in (
        "checkpoints/final_recovered.pt",
        "component_checkpoints/final_router.pt",
        "optimizer_and_updates.csv",
        "trajectory_metrics.csv",
        "diagnostic_recomputation.json",
        "result.json",
    ):
        if (attempt_dir / relative).exists():
            raise FileExistsError(f"Unexpected finalizer output exists: {attempt_dir / relative}")

    old_hashes = json_read(original_hashes_path)
    allowed_changes = {
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/finalize_step1500.py"),
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/prepare.py"),
    }
    for path_string, expected_hash in old_hashes.items():
        path = Path(path_string)
        if path.is_absolute() and path.is_file() and path_string not in allowed_changes:
            if sha256(path) != expected_hash:
                raise RuntimeError(f"Unexpected source change since revision 3: {path}.")

    new_hashes = dict(old_hashes)
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation source missing: {path}")
        new_hashes[str(path)] = sha256(path)
    json_save(OUT / "input_code_configuration_hashes.json", new_hashes)
    new_hash_table_sha256 = sha256(OUT / "input_code_configuration_hashes.json")
    code_hash_changes = {
        path_string: {
            "old": old_hashes.get(path_string),
            "new": new_hash,
        }
        for path_string, new_hash in new_hashes.items()
        if old_hashes.get(path_string) != new_hash
    }
    revision_path = OUT / "engineering_revision_4.json"
    revision_record = {
        "revision": 4,
        "failure": finalizer_failure["error"],
        "failed_attempt": str(attempt_dir),
        "failed_finalization_record": str(finalizer_failure_path),
        "training_batches_consumed": 0,
        "optimizer_updates_after_checkpoint": 0,
        "repair_scope": "restore the exact saved step-1000 optimizer/scaler state into a diagnostic-only clone so preservation can be verified without a training loop",
        "scientific_conditions_changed": False,
        "probe_manifest_sha256_unchanged": sha256(OUT / "diagnostic_probe_manifest.json"),
        "supersedes_registration": str(original_registration_path),
        "supersedes_registration_sha256": sha256(original_registration_path),
        "supersedes_source_hash_manifest": str(original_hashes_path),
        "supersedes_source_hash_manifest_sha256": sha256(original_hashes_path),
        "new_source_hash_manifest_sha256": new_hash_table_sha256,
        "source_hash_changes": code_hash_changes,
    }
    json_save(revision_path, revision_record)
    registration["registration_revision"] = 4
    registration["supersedes_registration"] = str(original_registration_path)
    registration["supersedes_registration_sha256"] = sha256(original_registration_path)
    registration["supersedes_source_hash_manifest"] = str(original_hashes_path)
    registration["supersedes_source_hash_manifest_sha256"] = sha256(original_hashes_path)
    registration["source_hash_manifest"] = str(OUT / "input_code_configuration_hashes.json")
    registration["source_hash_manifest_sha256"] = new_hash_table_sha256
    registration["engineering_revision_record"] = str(revision_path)
    registration["engineering_revision_record_sha256"] = sha256(revision_path)
    registration["scientific_conditions_changed_across_revision"] = False
    json_save(registration_path, registration)
    shutil.copy2(registration_path, OUT / "registration_revision_4.json")
    print(json.dumps(revision_record, indent=2))


def register_frozen_state_revision() -> None:
    """Register the device-independent frozen-state integrity check."""
    import af_native
    import engine

    registration_path = OUT / "registration.json"
    original_registration_path = OUT / "registration_revision_4.json"
    original_hashes_path = OUT / "input_code_configuration_hashes_revision_4.json"
    original_probe_path = OUT / "diagnostic_probe_manifest_revision_4.json"
    if not all(
        path.is_file()
        for path in (
            registration_path,
            original_registration_path,
            original_hashes_path,
            original_probe_path,
        )
    ):
        raise FileNotFoundError("Revision-4 registration archives are incomplete.")
    if (OUT / "registration_revision_5.json").exists() or (
        OUT / "engineering_revision_5.json"
    ).exists():
        raise FileExistsError("Engineering registration revision 5 already exists.")
    registration = json_read(registration_path)
    if int(registration.get("registration_revision", 0)) != 4:
        raise RuntimeError("Only registration revision 4 can be amended by this repair.")
    if sha256(registration_path) != sha256(original_registration_path):
        raise RuntimeError("Current registration differs from its preserved revision-4 copy.")
    if sha256(OUT / "input_code_configuration_hashes.json") != sha256(original_hashes_path):
        raise RuntimeError("Current source-hash table differs from its preserved revision-4 copy.")
    if sha256(OUT / "diagnostic_probe_manifest.json") != sha256(original_probe_path):
        raise RuntimeError("Probe manifest changed after registration revision 4.")

    status_dir = OUT / "status"
    completed_path = status_dir / "seed17_unpruned_full.json"
    failed_path = status_dir / "seed17_unpruned_router_frozen.json"
    failure_record_path = status_dir / "seed17_unpruned_router_frozen_revision_4_failed.json"
    pilot_path = OUT / "pilot_validation.json"
    archived_pilot_path = OUT / "pilot_validation_revision_4_failed.json"
    completed = json_read(completed_path)
    failed = json_read(failed_path)
    failure_record = json_read(failure_record_path)
    pilot = json_read(pilot_path)
    if (
        completed.get("status") != "COMPLETE"
        or failed.get("status") != "FAILED"
        or "Expected all tensors to be on the same device" not in str(failed.get("error", ""))
        or failure_record.get("status") != "FAILED"
        or "Expected all tensors to be on the same device" not in str(failure_record.get("error", ""))
        or pilot.get("status") != "FAIL"
        or int(pilot.get("completed_branches", -1)) != 1
        or json_read(archived_pilot_path).get("status") != "FAIL"
    ):
        raise RuntimeError("The recorded frozen-state failure differs from the approved repair.")
    branch_statuses = [
        path
        for path in status_dir.glob("seed*_*.json")
        if "_prefix" not in path.stem
        and "_finalization_" not in path.stem
        and "_revision_" not in path.stem
    ]
    if set(branch_statuses) != {completed_path, failed_path}:
        raise RuntimeError("Revision 5 is blocked because another pilot branch has started.")
    attempt_dir = Path(failed["attempt"]).resolve()
    if Path(failure_record["attempt"]).resolve() != attempt_dir:
        raise RuntimeError("Frozen-state failure records point to different attempts.")
    checkpoint = af_native.decode_numpy(torch_load(attempt_dir / "interruption.pt"))
    if (
        int(checkpoint.get("step", -1)) != 1000
        or int(checkpoint.get("sampler_next_batch", -1)) != 1000
    ):
        raise RuntimeError("Failed frozen-state branch did not remain at its step-1000 boundary.")
    update_path = attempt_dir / "study_optimizer_and_updates.jsonl"
    if update_path.exists() and any(line.strip() for line in update_path.read_text().splitlines()):
        raise RuntimeError("The failed frozen-state branch contains optimizer update rows.")
    if (attempt_dir / "result.json").exists():
        raise FileExistsError("Failed frozen-state attempt already has a result.")

    for source, destination in (
        (failed_path, status_dir / "seed17_unpruned_router_frozen_revision_4_failed.json"),
        (pilot_path, OUT / "pilot_validation_revision_5_pre_repair.json"),
    ):
        if destination.exists():
            if source.read_bytes() != destination.read_bytes():
                raise RuntimeError(f"Existing failure archive differs from source: {destination}")
        else:
            shutil.copy2(source, destination)
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"Failure archive differs from source: {destination}")

    old_hashes = json_read(original_hashes_path)
    allowed_changes = {
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/engine.py"),
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/finalize_step1500.py"),
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/prepare.py"),
    }
    for path_string, expected_hash in old_hashes.items():
        path = Path(path_string)
        if path.is_absolute() and path.is_file() and path_string not in allowed_changes:
            if sha256(path) != expected_hash:
                raise RuntimeError(f"Unexpected source change since revision 4: {path}.")

    new_hashes = dict(old_hashes)
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation source missing: {path}")
        new_hashes[str(path)] = sha256(path)
    json_save(OUT / "input_code_configuration_hashes.json", new_hashes)
    new_hash_table_sha256 = sha256(OUT / "input_code_configuration_hashes.json")
    code_hash_changes = {
        path_string: {"old": old_hashes.get(path_string), "new": new_hash}
        for path_string, new_hash in new_hashes.items()
        if old_hashes.get(path_string) != new_hash
    }
    revision_path = OUT / "engineering_revision_5.json"
    revision_record = {
        "revision": 5,
        "failure": failed["error"],
        "failed_attempt": str(attempt_dir),
        "saved_checkpoint_step": 1000,
        "optimizer_updates_in_failed_attempt": 0,
        "repair_scope": "compare frozen optimizer tensors exactly after mapping to CPU only when their devices differ",
        "scientific_conditions_changed": False,
        "probe_manifest_sha256_unchanged": sha256(OUT / "diagnostic_probe_manifest.json"),
        "supersedes_registration": str(original_registration_path),
        "supersedes_registration_sha256": sha256(original_registration_path),
        "supersedes_source_hash_manifest": str(original_hashes_path),
        "supersedes_source_hash_manifest_sha256": sha256(original_hashes_path),
        "new_source_hash_manifest_sha256": new_hash_table_sha256,
        "source_hash_changes": code_hash_changes,
        "completed_unpruned_full_attempt_preserved": True,
        "failed_router_frozen_attempt_preserved": True,
    }
    json_save(revision_path, revision_record)
    registration["registration_revision"] = 5
    registration["supersedes_registration"] = str(original_registration_path)
    registration["supersedes_registration_sha256"] = sha256(original_registration_path)
    registration["supersedes_source_hash_manifest"] = str(original_hashes_path)
    registration["supersedes_source_hash_manifest_sha256"] = sha256(original_hashes_path)
    registration["source_hash_manifest"] = str(OUT / "input_code_configuration_hashes.json")
    registration["source_hash_manifest_sha256"] = new_hash_table_sha256
    registration["engineering_revision_record"] = str(revision_path)
    registration["engineering_revision_record_sha256"] = sha256(revision_path)
    registration["scientific_conditions_changed_across_revision"] = False
    json_save(registration_path, registration)
    shutil.copy2(registration_path, OUT / "registration_revision_5.json")
    print(json.dumps(revision_record, indent=2))


def register_analysis_revision() -> None:
    registration_path = OUT / "registration.json"
    original_registration_path = OUT / "registration_revision_5.json"
    original_hashes_path = OUT / "input_code_configuration_hashes_revision_5.json"
    probe_path = OUT / "diagnostic_probe_manifest.json"
    revision_path = OUT / "engineering_revision_6.json"
    archived_registration_path = OUT / "registration_revision_6.json"
    if revision_path.exists() or archived_registration_path.exists():
        raise FileExistsError("Engineering revision 6 has already been recorded.")
    registration = json_read(registration_path)
    original_registration = json_read(original_registration_path)
    if int(registration.get("registration_revision", -1)) != 5:
        raise RuntimeError("Analysis revision 6 requires active registration revision 5.")
    if (
        int(original_registration.get("registration_revision", -1)) != 5
        or sha256(registration_path) != sha256(original_registration_path)
    ):
        raise RuntimeError("Active registration differs from its preserved revision-5 copy.")
    active_hashes_path = OUT / "input_code_configuration_hashes.json"
    if not original_hashes_path.exists():
        if sha256(active_hashes_path) != registration.get("source_hash_manifest_sha256"):
            raise RuntimeError("Cannot archive revision-5 hashes: active manifest is not registered.")
        shutil.copy2(active_hashes_path, original_hashes_path)
    old_hashes = json_read(original_hashes_path)
    if (
        sha256(active_hashes_path)
        != registration.get("source_hash_manifest_sha256")
        or sha256(active_hashes_path) != sha256(original_hashes_path)
    ):
        raise RuntimeError("Active source hashes differ from the preserved revision-5 manifest.")
    if sha256(probe_path) != registration.get("diagnostic_probe_manifest_sha256"):
        raise RuntimeError("Probe manifest changed after registration revision 5.")
    if json_read(OUT / "pilot_validation.json").get("status") != "PASS":
        raise RuntimeError("The registered seed-17 pilot is not PASS.")

    expected_jobs = {
        f"seed{seed}_{model_type}_{condition}"
        for seed in SEEDS
        for model_type in MODEL_TYPES
        for condition in CONDITIONS
    }
    status_dir = OUT / "status"
    actual_jobs = {
        path.stem
        for path in status_dir.glob("seed*_*.json")
        if "_prefix" not in path.stem and "_revision_" not in path.stem
    }
    if actual_jobs != expected_jobs:
        raise RuntimeError("Cohort status files do not exactly match the registered 40 branches.")
    for job_id in sorted(expected_jobs):
        status = json_read(status_dir / f"{job_id}.json")
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"Analysis revision requires every cohort job COMPLETE: {job_id}.")

    allowed_changes = {
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/analyze.py"),
        str(ROOT / "scripts/imp_postpruning_adaptation_v1/prepare.py"),
    }
    for path_string, expected_hash in old_hashes.items():
        path = Path(path_string)
        if path.is_absolute() and path.is_file() and path_string not in allowed_changes:
            if sha256(path) != expected_hash:
                raise RuntimeError(f"Unexpected source change since revision 5: {path}.")

    new_hashes = dict(old_hashes)
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation source missing: {path}")
        new_hashes[str(path)] = sha256(path)
    json_save(OUT / "input_code_configuration_hashes.json", new_hashes)
    new_hash_table_sha256 = sha256(OUT / "input_code_configuration_hashes.json")
    code_hash_changes = {
        path_string: {"old": old_hashes.get(path_string), "new": new_hash}
        for path_string, new_hash in new_hashes.items()
        if old_hashes.get(path_string) != new_hash
    }
    revision_record = {
        "revision": 6,
        "failure": "AttributeError: 'float' object has no attribute 'values'",
        "failure_context": "The SVG trajectory renderer treated the flat tuple-keyed trajectory map as nested mappings.",
        "repair_scope": "flattened values are read directly from the tuple-keyed trajectory map",
        "completed_registered_branches": len(expected_jobs),
        "training_updates_after_cohort_completion": 0,
        "scientific_conditions_changed": False,
        "probe_manifest_sha256_unchanged": sha256(probe_path),
        "supersedes_registration": str(original_registration_path),
        "supersedes_registration_sha256": sha256(original_registration_path),
        "supersedes_source_hash_manifest": str(original_hashes_path),
        "supersedes_source_hash_manifest_sha256": sha256(original_hashes_path),
        "new_source_hash_manifest_sha256": new_hash_table_sha256,
        "source_hash_changes": code_hash_changes,
    }
    json_save(revision_path, revision_record)
    registration["registration_revision"] = 6
    registration["supersedes_registration"] = str(original_registration_path)
    registration["supersedes_registration_sha256"] = sha256(original_registration_path)
    registration["supersedes_source_hash_manifest"] = str(original_hashes_path)
    registration["supersedes_source_hash_manifest_sha256"] = sha256(original_hashes_path)
    registration["source_hash_manifest"] = str(OUT / "input_code_configuration_hashes.json")
    registration["source_hash_manifest_sha256"] = new_hash_table_sha256
    registration["engineering_revision_record"] = str(revision_path)
    registration["engineering_revision_record_sha256"] = sha256(revision_path)
    registration["status"] = "registered cohort complete; analysis pending"
    registration["scientific_conditions_changed_across_revision"] = False
    json_save(registration_path, registration)
    shutil.copy2(registration_path, archived_registration_path)
    print(json.dumps(revision_record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--refresh-contrasts", action="store_true")
    parser.add_argument("--seal-probes", action="store_true")
    parser.add_argument("--engineering-revision", action="store_true")
    parser.add_argument("--engineering-revision-4", action="store_true")
    parser.add_argument("--engineering-revision-5", action="store_true")
    parser.add_argument("--engineering-revision-6", action="store_true")
    args = parser.parse_args()
    if args.audit:
        build_audit()
    elif args.refresh_contrasts:
        refresh_historical_contrasts()
    elif args.seal_probes:
        refresh_probe_seal()
    elif args.engineering_revision:
        register_engineering_revision()
    elif args.engineering_revision_4:
        register_finalization_revision()
    elif args.engineering_revision_5:
        register_frozen_state_revision()
    elif args.engineering_revision_6:
        register_analysis_revision()
    else:
        parser.error(
            "Use --audit, --refresh-contrasts, --seal-probes, --engineering-revision, "
            "--engineering-revision-4, --engineering-revision-5, or "
            "--engineering-revision-6."
        )
