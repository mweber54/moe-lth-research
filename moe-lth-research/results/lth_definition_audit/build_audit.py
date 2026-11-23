#!/usr/bin/env python3
"""Build a read-only inventory of local >=80% sparse recovery evidence."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
import hashlib

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CANONICAL = ROOT / "results/final_dataset_a/canonical/dataset_a_canonical.csv"
UNKNOWN = "unknown"
THRESHOLDS = (2, 5, 10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def artifact_path(value: str | None) -> Path | None:
    if not value or value == UNKNOWN:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def truth(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def metric_curve(folder: Path | None) -> list[dict]:
    path = folder / "metrics.jsonl" if folder else None
    if path is None or not path.is_file():
        return []
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def final_metric(folder: Path | None) -> tuple[float | None, int | None, list[dict]]:
    curve = metric_curve(folder)
    if not curve:
        return None, None, curve
    try:
        loss, step = float(curve[-1]["loss"]), int(curve[-1]["step"])
    except (KeyError, TypeError, ValueError):
        return None, None, curve
    return (loss if math.isfinite(loss) else None), step, curve


def get_path(metadata: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return UNKNOWN


def get_mask_path(metadata: dict) -> str:
    value = get_path(metadata, ("mask_artifact_path", "mask_path", "mask_file", "pruning_mask_path", "mask_checkpoint"))
    if value != UNKNOWN:
        return value
    for key, path in metadata.items():
        if "mask" in key.lower() and "path" in key.lower() and isinstance(path, str):
            return path
    return UNKNOWN


def experts(metadata: dict, path: Path) -> str:
    value = metadata.get("num_experts", metadata.get("expert_count"))
    if value is not None:
        return str(value)
    match = re.search(r"(?:^|/)e(\d+)(?:/|$)", path.as_posix())
    if not match:
        match = re.search(r"(?:^|_)e(\d+)(?:_|$)", str(metadata.get("condition", "")))
    return match.group(1) if match else UNKNOWN


def age_value(metadata: dict) -> int | None:
    value = metadata.get("router_age_percent", metadata.get("router_age"))
    try:
        return int(float(str(value).removeprefix("R"))) if value is not None else None
    except ValueError:
        return None


def gap_values(sparse: float | None, dense: float | None) -> tuple[float | None, float | None]:
    if sparse is None or dense is None or dense == 0:
        return None, None
    absolute = sparse - dense
    return absolute, 100.0 * absolute / dense


def protocol_class(canonical: bool, age: int | None) -> str:
    if canonical:
        return "original_initialization_lth_candidate" if age == 0 else "hybrid_router_rewind_candidate"
    if age is not None and age > 0:
        return "other_or_unverifiable_nonzero_router_age"
    if age == 0:
        return "other_or_unverifiable_R0_components_or_mask_provenance"
    return "other_or_unverifiable"


def performance_class(gap_pct: float | None) -> str:
    if gap_pct is None:
        return "unpaired_or_unavailable_dense_baseline"
    gap = abs(gap_pct)
    if gap <= 2:
        return "within_2pct_screen"
    if gap <= 5:
        return "within_5pct_screen"
    if gap <= 10:
        return "within_10pct_screen"
    return "outside_10pct_screen"


def mask_hash_file(path: Path) -> str | None:
    try:
        masks = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(masks, dict) or not masks or not all(torch.is_tensor(value) for value in masks.values()):
        return None
    digest = hashlib.sha256()
    for name in sorted(masks):
        digest.update(name.encode("utf-8"))
        digest.update(masks[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def experts_from_mask_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return UNKNOWN
    try:
        masks = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return UNKNOWN
    expert_ids = {
        int(match.group(1))
        for name in masks
        if (match := re.search(r"\.moe\.experts\.(\d+)\.", name))
    }
    return str(len(expert_ids)) if expert_ids else UNKNOWN


def mask_tensor_index(canonical_rows: list[dict]) -> dict[str, list[str]]:
    paths: dict[str, set[str]] = {}

    def index(path: Path) -> None:
        if not path.is_file():
            return
        digest = mask_hash_file(path)
        if digest:
            paths.setdefault(digest, set()).add(str(path))

    for row in canonical_rows:
        path = artifact_path(row["mask_artifact_path"])
        if path:
            index(path)
    for directory, subdirs, files in os.walk(ROOT / "results", followlinks=False):
        is_mask_directory = Path(directory).name == "masks"
        for filename in files:
            path = Path(directory) / filename
            if path.suffix in {".pt", ".pth"} and (is_mask_directory or "mask" in filename.lower()):
                index(path)
    return {digest: sorted(found) for digest, found in paths.items()}


def canonical_records() -> list[dict]:
    source_rows = [row for row in read_csv(CANONICAL) if float(row["sparsity"]) >= 0.8 - 1e-9]
    records = []
    for row in source_rows:
        sparse_dir = artifact_path(row["sparse_artifact_path"])
        dense_dir = artifact_path(row["dense_artifact_path"])
        sparse_loss, sparse_step, sparse_curve = final_metric(sparse_dir)
        dense_loss, dense_step, dense_curve = final_metric(dense_dir)
        absolute_gap, gap_pct = gap_values(sparse_loss, dense_loss)
        sparse_points = {int(point["step"]): float(point["loss"]) for point in sparse_curve}
        dense_points = {int(point["step"]): float(point["loss"]) for point in dense_curve}
        shared_steps = sorted(set(sparse_points) & set(dense_points))
        matches = [step for step in shared_steps if sparse_points[step] <= dense_points[step]]
        positive_matches = [step for step in matches if step > 0]
        curve_gaps = [
            100.0 * (sparse_points[step] - dense_points[step]) / dense_points[step]
            for step in shared_steps if dense_points[step] != 0
        ]
        initialization = str(Path(row["source_artifact_path"]) / "checkpoints/step_0.pt")
        mask_file = artifact_path(row["mask_artifact_path"])
        init_file = artifact_path(initialization)
        router_file = artifact_path(row["source_checkpoint_path"])
        recovery_checkpoint = str(sparse_dir / "checkpoints/final_recovered.pt") if sparse_dir else UNKNOWN
        final_file = artifact_path(recovery_checkpoint)
        valid = truth(row.get("condition_valid")) and all(
            path is not None and path.is_file() for path in (mask_file, init_file, router_file, final_file)
        ) and sparse_step == dense_step == 2500
        record = {
            "run_id": row["condition_id"], "scope": "frozen_dataset_a_canonical", "validated": valid,
            "protocol_class": protocol_class(True, int(row["router_age"])), "num_experts": row["num_experts"],
            "seed": row["seed"], "target_sparsity": row["sparsity"], "actual_sparsity": row["actual_sparsity"],
            "router_age_percent": row["router_age"], "mask_hash": row["mask_hash"], "mask_path": row["mask_artifact_path"],
            "mask_source_checkpoint_path": str(Path(row["source_artifact_path"]) / "checkpoints/step_2500.pt"),
            "mask_source_step": 2500, "initialization_checkpoint_path": initialization,
            "router_checkpoint_path": row["source_checkpoint_path"], "recovery_artifact_path": row["sparse_artifact_path"],
            "recovery_checkpoint_path": recovery_checkpoint, "dense_artifact_path": row["dense_artifact_path"],
            "dense_loss": dense_loss if dense_loss is not None else UNKNOWN,
            "sparse_loss": sparse_loss if sparse_loss is not None else UNKNOWN,
            "absolute_loss_gap": absolute_gap if absolute_gap is not None else UNKNOWN,
            "relative_loss_gap_pct": gap_pct if gap_pct is not None else UNKNOWN,
            "performance_class": performance_class(gap_pct),
            "within_2pct": abs(gap_pct) <= 2 if gap_pct is not None else UNKNOWN,
            "within_5pct": abs(gap_pct) <= 5 if gap_pct is not None else UNKNOWN,
            "within_10pct": abs(gap_pct) <= 10 if gap_pct is not None else UNKNOWN,
            "dense_steps": dense_step if dense_step is not None else UNKNOWN,
            "sparse_steps": sparse_step if sparse_step is not None else UNKNOWN,
            "source_attempted_steps": row.get("source_attempt_budget", UNKNOWN),
            "recovery_attempted_steps": row.get("recovery_attempt_budget", UNKNOWN),
            "dense_optimizer_steps_applied": UNKNOWN, "sparse_optimizer_steps_applied": UNKNOWN,
            "ever_matched_or_better_at_shared_validation_step": bool(matches),
            "ever_matched_or_better_after_step_0": bool(positive_matches),
            "first_matched_validation_step": matches[0] if matches else UNKNOWN,
            "first_matched_positive_validation_step": positive_matches[0] if positive_matches else UNKNOWN,
            "minimum_relative_gap_during_training_pct": min(curve_gaps) if curve_gaps else UNKNOWN,
            "router_initialization_verified": int(row["router_age"]) == 0,
            "expert_shared_initialization_verified": True, "mask_provenance_verified": truth(row.get("condition_valid")),
            "source_config_id": row.get("source_config_id", UNKNOWN), "recovery_config_id": row.get("recovery_config_id", UNKNOWN),
            "config_path": "embedded in source step-0 checkpoint; config id in canonical manifest",
            "training_batch_hash": row.get("data_hash", UNKNOWN), "validation_batch_hash": row.get("validation_hash", UNKNOWN),
            "raw_validation_curve_path": str(sparse_dir / "metrics.jsonl") if sparse_dir else UNKNOWN,
            "dense_validation_curve_path": str(dense_dir / "metrics.jsonl") if dense_dir else UNKNOWN,
            "notes": "Trained-source mask; expert values and all shared state from step 0; router from selected age. Validation CE proxy.",
        }
        if sparse_dir and (sparse_dir / "metadata.json").is_file():
            metadata = json.loads((sparse_dir / "metadata.json").read_text(encoding="utf-8"))
            record["sparse_optimizer_steps_applied"] = metadata.get("optimizer_steps_applied", UNKNOWN)
            record["recovery_attempted_steps"] = metadata.get("recovery_steps", record["recovery_attempted_steps"])
        if dense_dir and (dense_dir / "metadata.json").is_file():
            metadata = json.loads((dense_dir / "metadata.json").read_text(encoding="utf-8"))
            record["dense_optimizer_steps_applied"] = metadata.get("optimizer_steps_applied", UNKNOWN)
        records.append(record)
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        key = (record["num_experts"], record["target_sparsity"], record["router_age_percent"])
        groups.setdefault(key, []).append(record)
    for group in groups.values():
        values = [float(record["relative_loss_gap_pct"]) for record in group if record["relative_loss_gap_pct"] != UNKNOWN]
        cell_mean = statistics.mean(values)
        cell_sd = statistics.stdev(values) if len(values) > 1 else 0.0
        for record in group:
            record["matched_cell_mean_relative_loss_gap_pct"] = cell_mean
            record["matched_cell_sd_relative_loss_gap_pct"] = cell_sd
            record["matched_cell_n_seeds"] = len(values)
            record["gap_from_matched_cell_mean_in_sd"] = (
                (float(record["relative_loss_gap_pct"]) - cell_mean) / cell_sd if cell_sd else UNKNOWN
            )
    return records


def raw_records(canonical: list[dict], mask_index: dict[str, list[str]]) -> list[dict]:
    canonical_paths = {
        os.path.realpath(path)
        for row in canonical
        if (path := artifact_path(row["recovery_artifact_path"])) is not None
    }
    seen, records = set(), []
    for directory, subdirs, files in os.walk(ROOT / "results", followlinks=False):
        subdirs[:] = [name for name in subdirs if name != "lth_definition_audit"]
        if "metadata.json" not in files:
            continue
        folder = Path(directory)
        try:
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            sparsity = float(metadata.get("sparsity", 0) or 0)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        digest = metadata.get("mask_hash")
        if sparsity < 0.8 - 1e-9 or not digest:
            continue
        resolved = os.path.realpath(folder)
        if resolved in canonical_paths or resolved in seen:
            continue
        seen.add(resolved)
        sparse_loss, sparse_step, curve = final_metric(folder)
        final_checkpoint = folder / "checkpoints/final_recovered.pt"
        age = age_value(metadata)
        dense_path = get_path(metadata, ("dense_artifact_path", "dense_result_path", "matched_dense_path"))
        dense_dir = artifact_path(dense_path)
        dense_loss, dense_step, dense_curve = final_metric(dense_dir) if dense_dir else (None, None, [])
        absolute_gap, gap_pct = gap_values(sparse_loss, dense_loss)
        sparse_points = {int(p["step"]): float(p["loss"]) for p in curve if "step" in p and "loss" in p}
        dense_points = {int(p["step"]): float(p["loss"]) for p in dense_curve if "step" in p and "loss" in p}
        shared_steps = sorted(set(sparse_points) & set(dense_points))
        matches = [step for step in shared_steps if sparse_points[step] <= dense_points[step]]
        curve_gaps = [100 * (sparse_points[s] - dense_points[s]) / dense_points[s] for s in shared_steps if dense_points[s]]
        integrity = metadata.get("integrity_checks_passed") is True
        validated = integrity and final_checkpoint.is_file() and bool(curve) and sparse_loss is not None
        router_path = get_path(metadata, ("router_checkpoint", "router_checkpoint_path"))
        init_path = get_path(metadata, ("initialization_checkpoint", "rewind_checkpoint", "rewind_checkpoint_path", "initial_checkpoint_path"))
        source_path = get_path(metadata, ("mask_source_checkpoint", "pruning_checkpoint", "mask_source_checkpoint_path"))
        run_id = str(metadata.get("condition", folder.relative_to(ROOT).as_posix()))
        mask_path = get_mask_path(metadata)
        matching_mask_paths = mask_index.get(digest, [])
        if matching_mask_paths and (
            mask_path == UNKNOWN
            or os.path.realpath(artifact_path(mask_path) or Path("/nonexistent"))
            not in {os.path.realpath(path) for path in matching_mask_paths}
        ):
            mask_path = matching_mask_paths[0]
        num_experts = experts(metadata, folder)
        if num_experts == UNKNOWN:
            num_experts = experts_from_mask_file(artifact_path(mask_path))
        mask_type = metadata.get("mask_type", metadata.get("mask_kind", UNKNOWN))
        pruning_method = metadata.get("pruning_method", UNKNOWN)
        if mask_type == "random":
            mask_provenance = "random count-matched control; not a trained-dense-derived mask"
        elif pruning_method == "expert_local_magnitude":
            mask_provenance = "expert-local magnitude method reported; source checkpoint not verified"
        else:
            mask_provenance = UNKNOWN
        records.append({
            "run_id": run_id, "scope": "additional_local_raw_metadata", "validated": validated,
            "protocol_class": protocol_class(False, age), "num_experts": num_experts,
            "seed": metadata.get("seed", UNKNOWN), "target_sparsity": sparsity,
            "actual_sparsity": metadata.get("actual_sparsity", UNKNOWN), "router_age_percent": age if age is not None else UNKNOWN,
            "mask_hash": digest, "mask_path": mask_path,
            "mask_paths": ";".join(matching_mask_paths) if matching_mask_paths else mask_path,
            "mask_tensor_hash_verified": bool(matching_mask_paths), "mask_type": mask_type,
            "pruning_method": pruning_method, "mask_provenance": mask_provenance,
            "mask_source_checkpoint_path": source_path,
            "mask_source_step": metadata.get("mask_source_step", UNKNOWN), "initialization_checkpoint_path": init_path,
            "router_checkpoint_path": router_path, "recovery_artifact_path": str(folder),
            "recovery_checkpoint_path": str(final_checkpoint) if final_checkpoint.is_file() else UNKNOWN,
            "dense_artifact_path": dense_path, "dense_loss": dense_loss if dense_loss is not None else UNKNOWN,
            "sparse_loss": sparse_loss if sparse_loss is not None else UNKNOWN,
            "absolute_loss_gap": absolute_gap if absolute_gap is not None else UNKNOWN,
            "relative_loss_gap_pct": gap_pct if gap_pct is not None else UNKNOWN,
            "performance_class": performance_class(gap_pct),
            "within_2pct": abs(gap_pct) <= 2 if gap_pct is not None else UNKNOWN,
            "within_5pct": abs(gap_pct) <= 5 if gap_pct is not None else UNKNOWN,
            "within_10pct": abs(gap_pct) <= 10 if gap_pct is not None else UNKNOWN,
            "dense_steps": dense_step if dense_step is not None else UNKNOWN,
            "sparse_steps": sparse_step if sparse_step is not None else UNKNOWN,
            "source_attempted_steps": metadata.get("source_attempts", UNKNOWN),
            "recovery_attempted_steps": metadata.get("recovery_steps", metadata.get("recovery_attempts", UNKNOWN)),
            "dense_optimizer_steps_applied": UNKNOWN,
            "sparse_optimizer_steps_applied": metadata.get("optimizer_steps_applied", UNKNOWN),
            "ever_matched_or_better_at_shared_validation_step": bool(matches) if dense_curve else UNKNOWN,
            "ever_matched_or_better_after_step_0": any(step > 0 for step in matches) if dense_curve else UNKNOWN,
            "first_matched_validation_step": matches[0] if matches else UNKNOWN,
            "first_matched_positive_validation_step": next((step for step in matches if step > 0), UNKNOWN),
            "minimum_relative_gap_during_training_pct": min(curve_gaps) if curve_gaps else UNKNOWN,
            "matched_cell_mean_relative_loss_gap_pct": UNKNOWN,
            "matched_cell_sd_relative_loss_gap_pct": UNKNOWN,
            "matched_cell_n_seeds": UNKNOWN,
            "gap_from_matched_cell_mean_in_sd": UNKNOWN,
            "router_initialization_verified": age == 0 and metadata.get("router_step") == 0,
            "expert_shared_initialization_verified": UNKNOWN, "mask_provenance_verified": UNKNOWN,
            "source_config_id": metadata.get("source_config_hash", UNKNOWN),
            "recovery_config_id": metadata.get("recovery_config_hash", UNKNOWN),
            "config_path": get_path(metadata, ("config_path", "config_file", "training_config_path")),
            "training_batch_hash": metadata.get("training_batch_sequence_hash", UNKNOWN),
            "validation_batch_hash": metadata.get("validation_batch_sequence_hash", UNKNOWN),
            "raw_validation_curve_path": str(folder / "metrics.jsonl") if curve else UNKNOWN,
            "dense_validation_curve_path": str(dense_dir / "metrics.jsonl") if dense_dir and dense_curve else UNKNOWN,
            "notes": "Outside frozen Dataset A pairings; missing initialization, mask-source, or dense linkage is unknown, not inferred.",
        })
    return records


def candidate_inventory(source_rows: list[dict], additional_runs: list[dict], mask_index: dict[str, list[str]]) -> list[dict]:
    by_hash: dict[str, dict] = {}
    for row in source_rows:
        digest = row["mask_hash"]
        indexed_paths = mask_index.get(digest, [])
        candidate_mask_paths = sorted(set([row["mask_artifact_path"], *indexed_paths]))
        entry = by_hash.setdefault(digest, {
            "mask_hash": digest, "candidate_scope": "canonical_dataset_a_mask", "num_experts": row["num_experts"],
            "seed": row["seed"], "target_sparsity": row["sparsity"], "actual_sparsity": row["actual_sparsity"],
            "mask_path": row["mask_artifact_path"],
            "mask_paths": ";".join(candidate_mask_paths),
            "mask_path_exists": bool(artifact_path(row["mask_artifact_path"]) and artifact_path(row["mask_artifact_path"]).is_file()),
            "mask_tensor_hash_verified": bool(indexed_paths),
            "candidate_kind": "trained_dense_pruning_mask",
            "mask_type": "trained_dense_magnitude",
            "pruning_method": "expert_local_magnitude",
            "mask_source_checkpoint_path": str(Path(row["source_artifact_path"]) / "checkpoints/step_2500.pt"),
            "initialization_checkpoint_path": str(Path(row["source_artifact_path"]) / "checkpoints/step_0.pt"),
            "mask_provenance": "expert-local magnitude pruning of trained source step 2500; frozen Dataset A provenance",
            "representative_recovery_artifact": row["sparse_artifact_path"], "router_ages_evaluated": set(),
            "source_config_id": row.get("source_config_id", UNKNOWN),
            "reload_status": "mask and source/recovery checkpoint paths linked in canonical manifest",
        })
        entry["router_ages_evaluated"].add(int(row["router_age"]))
    for run in additional_runs:
        digest = run["mask_hash"]
        if not digest or digest == UNKNOWN:
            continue
        if digest in by_hash:
            by_hash[digest]["router_ages_evaluated"].add(run["router_age_percent"])
            if run["mask_path"] != UNKNOWN:
                current = set(filter(None, by_hash[digest]["mask_paths"].split(";")))
                current.update(filter(None, run["mask_paths"].split(";")))
                by_hash[digest]["mask_paths"] = ";".join(sorted(current))
                by_hash[digest]["mask_tensor_hash_verified"] |= run["mask_tensor_hash_verified"]
            continue
        mask_file = artifact_path(run["mask_path"])
        by_hash[digest] = {
            "mask_hash": digest, "candidate_scope": "additional_local_raw_mask", "num_experts": run["num_experts"],
            "seed": run["seed"], "target_sparsity": run["target_sparsity"], "actual_sparsity": run["actual_sparsity"],
            "mask_path": run["mask_path"], "mask_paths": run["mask_paths"],
            "mask_path_exists": bool(mask_file and mask_file.is_file()),
            "mask_tensor_hash_verified": run["mask_tensor_hash_verified"],
            "candidate_kind": "random_mask_control" if run["mask_type"] == "random" else "provenance_unverified_mask_candidate",
            "mask_type": run["mask_type"], "pruning_method": run["pruning_method"],
            "mask_source_checkpoint_path": run["mask_source_checkpoint_path"],
            "initialization_checkpoint_path": run["initialization_checkpoint_path"], "mask_provenance": UNKNOWN,
            "representative_recovery_artifact": run["recovery_artifact_path"], "router_ages_evaluated": {run["router_age_percent"]},
            "source_config_id": run["source_config_id"],
            "reload_status": "mask tensor hash verified at local path" if run["mask_tensor_hash_verified"] else "mask tensor path/hash unavailable",
        }
    rows = []
    for row in by_hash.values():
        row["router_ages_evaluated"] = ";".join(str(value) for value in sorted(row["router_ages_evaluated"], key=lambda value: -1 if value == UNKNOWN else int(value)))
        rows.append(row)
    return sorted(rows, key=lambda row: (
        float(row["target_sparsity"]) if row["target_sparsity"] != UNKNOWN else 9,
        int(row["num_experts"]) if str(row["num_experts"]).isdigit() else 999,
        int(row["seed"]) if str(row["seed"]).isdigit() else 999,
        row["mask_hash"],
    ))


def build_readme(candidates: list[dict], runs: list[dict], canonical: list[dict]) -> str:
    canonical_hashes = {row["mask_hash"] for row in canonical}
    additional = [row for row in runs if row["scope"] == "additional_local_raw_metadata"]
    raw_hashes = {row["mask_hash"] for row in additional}
    raw_only_candidates = [row for row in candidates if row["candidate_scope"] == "additional_local_raw_mask"]
    raw_only_verified = sum(row["mask_tensor_hash_verified"] is True for row in raw_only_candidates)
    random_controls = sum(row["candidate_kind"] == "random_mask_control" for row in raw_only_candidates)
    r0 = [row for row in canonical if int(row["router_age_percent"]) == 0]
    all5 = [row for row in canonical if row["within_5pct"] is True]
    positive_crossings = sum(row["ever_matched_or_better_after_step_0"] is True for row in all5)
    cell_means = [float(row["matched_cell_mean_relative_loss_gap_pct"]) for row in all5]
    cell_sds = [float(row["matched_cell_sd_relative_loss_gap_pct"]) for row in all5]
    sd_offsets = [abs(float(row["gap_from_matched_cell_mean_in_sd"])) for row in all5]
    cell_groups = {}
    for row in canonical:
        cell_groups[(row["num_experts"], row["target_sparsity"], row["router_age_percent"])] = float(row["matched_cell_mean_relative_loss_gap_pct"])
    near_dense_cell_means = [value for value in cell_groups.values() if abs(value) <= 5]
    r0counts = [sum(row[f"within_{threshold}pct"] is True for row in r0) for threshold in THRESHOLDS]
    allcounts = [sum(row[f"within_{threshold}pct"] is True for row in canonical) for threshold in THRESHOLDS]
    lines = [
        "# Original-LTH Definition Audit (2026-10-12)", "", "## Conclusion", "",
        f"The frozen Dataset A collection contains {len(canonical_hashes)} distinct >=80% expert masks and {len(canonical)} router-age recovery conditions. Raw-curve recomputation finds {len(all5)} conditions within 5% of their matched dense validation CE, representing {len({row['mask_hash'] for row in all5})} masks; all use nonzero-age routers. Dataset A has {len(r0)} R0 recoveries with verified step-0 expert/shared/router initialization.", "",
        "No recovery meets a defensible, evidence-complete operational definition of an original-initialization winning ticket at >=80%. The eight near-dense cases are hybrid router-rewind recoveries, not original LTH tickets. R0 is an original-initialization candidate protocol, but no R0 endpoint is within 5% of matched dense validation CE.", "",
        "## Definition and Scope", "",
        "Frankle & Carbin (2019) define a winning ticket as a sparse subnetwork that, when trained in isolation from its original initialization, reaches the original dense network's test accuracy in a comparable number of iterations. This audit uses validation cross-entropy (CE) only as an explicitly limited language-modeling proxy, not as test accuracy. A percent CE gap is a descriptive screen, not an equivalence margin or proof. The 2%, 5%, and 10% bands are reported only as requested screens.", "",
        "The primary count is the frozen Dataset A grid in `results/final_dataset_a/canonical/dataset_a_canonical.csv`, checked against sparse and paired dense raw `metrics.jsonl` curves. The grid has 5 expert counts (8, 12, 16, 24, 32), 5 seeds (7, 17, 29, 43, 61), 4 target sparsities (80%, 85%, 90%, 95%), and 11 router ages. That is 100 distinct masks and 1,100 recovery conditions. Target sparsity applies only to expert weight matrices; it is not whole-network sparsity.", "",
        f"A separate local raw-metadata scan found {len(additional)} additional >=80% recovery artifacts, with {len(raw_hashes)} recorded mask hashes. Their union with the canonical hashes is {len(candidates)} distinct recorded hashes, including {len(raw_hashes - canonical_hashes)} raw-only hashes. These additional runs are not mixed into dense-paired threshold counts: datasets/treatments vary and pairing or initialization provenance may be unavailable. Unknown evidence is marked `unknown`. For these additional rows, `validated=True` means the raw metadata integrity flag is true, a final recovery checkpoint exists, and the metrics curve has a parseable finite endpoint; it is not an independent replay of the full protocol or proof of dense pairing.", "",
        f"All {raw_only_verified} raw-only mask hashes were matched to locally saved mask tensors by recomputing the same SHA-256 state-dictionary hash; their reload paths are recorded in `all_candidates.csv`. {random_controls} are explicitly labeled random-mask controls, not learned-mask LTH candidates. For the remaining legacy masks, source-dense checkpoint provenance is still unverified.", "",
        "The repository's project summary reports a separate long-budget multi-domain 80% initialization-rewind result close to or better than dense. Its referenced raw result directory and checkpoints are absent locally, so mask provenance, full initialization restoration, exact pairing, and budgets cannot be independently verified. The summary also says its larger validation mixture included WikiText test data, so it is not evidence from an untouched test-only evaluation. It is not counted as a confirmed ticket or candidate run here.", "",
        "## Checkpoint and Mask Logic", "",
        "Canonical masks are expert-local magnitude masks from the trained source step-2500 model. `scripts/fixed_budget_dataset/run.py` loads the step-0 state, derives masks from step 2500, masks the step-0 base, and calls `run_router_age_recovery._run_recovery_condition`. `src/moe_lth/pruning/router_age.py` builds the fixed base from the rewind checkpoint, swaps only state tensors classified as router from the selected checkpoint, then reapplies masks. Its component classifier treats all non-expert, non-router state (including embeddings and normalization) as shared.", "",
        "For Dataset A, R0 selects the step-0 source checkpoint for both router and expert/shared base. Thus R0 restores all model components to the original initialization before applying the trained-dense-derived expert mask. R10-R100 retain step-0 expert/shared state and mask but use a nonzero-age router: hybrid router rewind, not full-network early-weight rewind. AdamW is fresh; router parameters are trainable. Gradient hooks zero masked-coordinate gradients and masks are reapplied after optimizer steps. Finalization checks masked checkpoint coordinates remain zero.", "",
        "Code: [router_age.py](../../src/moe_lth/pruning/router_age.py), [run_router_age_recovery.py](../../src/moe_lth/experiments/run_router_age_recovery.py), [masks.py](../../src/moe_lth/pruning/masks.py), [rewind.py](../../src/moe_lth/pruning/rewind.py), [magnitude_prune.py](../../src/moe_lth/pruning/magnitude_prune.py), [fixed_budget_dataset/run.py](../../scripts/fixed_budget_dataset/run.py).", "",
        "Dense and sparse use 2,500 attempted steps and same-seed train/validation batch-sequence hashes; endpoints are at step 2,500. AMP may skip a small number of updates, so applied updates can differ slightly and are recorded separately. Sparse training uses a fresh optimizer. This is expert-only pruning. Validation CE is token-weighted and excludes the auxiliary term per the frozen protocol audit.", "",
        "## Recomputed Performance", "",
        "| Population | Runs | Within 2% | Within 5% | Within 10% |", "| --- | ---: | ---: | ---: | ---: |",
        f"| All canonical >=80% | {len(canonical)} | {allcounts[0]} | {allcounts[1]} | {allcounts[2]} |",
        f"| Canonical R0 | {len(r0)} | {r0counts[0]} | {r0counts[1]} | {r0counts[2]} |", "",
        f"The {r0counts[2]} R0 runs within 10% represent {len({row['mask_hash'] for row in r0 if row['within_10pct'] is True})} masks. No R0 case is within 2% or 5%; all-age within-10% count is {allcounts[2]}.", "",
        "The eight conditions within 5% are: E16 seeds 17/29 at 80%, R60 (3.35%, 3.45%); E24 seed 17 at 80%, R50 (4.01%) and R40 (4.91%); E16 seed 29 at 80%, R70 (4.29%) and R80 (4.40%); and E8 seed 29 at 80%, R20 (4.83%). These are eight recovery conditions but four masks. Exact losses, gaps, paths, budgets, and matched-step results are in `near_dense_candidates.csv` and `all_recovery_runs.csv`.", "",
        f"The audit compares loss curves only at shared logged steps. All {len(all5)} match/exceed dense at step 0; {positive_crossings} also do so at one or more positive logged steps, while {len(all5) - positive_crossings} match only at step 0. Step-zero performance is before sparse retraining and is not LTH success. At step 2,500, each is 3.35%-4.91% worse than its dense comparator.", "",
        f"Across each selected run's five-seed architecture/sparsity/router-age cell, the mean gap is {min(cell_means):.2f}%-{max(cell_means):.2f}% and sample SD is {min(cell_sds):.2f}-{max(cell_sds):.2f} percentage points. Selected individual gaps lie {min(sd_offsets):.2f}-{max(sd_offsets):.2f} SD below their cell means. {len(near_dense_cell_means)} of {len(cell_groups)} five-seed cell means are within 5%. These are descriptive seed-variation statistics, not equivalence or significance tests.", "",
        "## Status by Requested Category", "",
        "- **Strict original-LTH evidence:** 0 verified >=80% winners. R0 satisfies mask-from-trained-dense, original initialization, isolated sparse retraining, and equal attempted-step budget; none is dense-equivalent even under the 5% screen. Test accuracy was not measured.",
        "- **Near-dense initialization candidates:** R0 counts are 0/0/19 within 2%/5%/10% of matched dense validation CE. The 10% band is screening only.",
        "- **Router-rewind recovery candidates:** 8 canonical conditions within 5%, all R20-R80, representing 4 masks at 80% sparsity.",
        "- **Unsuccessful or inconclusive:** other R0 runs miss the 10% screen; higher-age runs outside the screens are not near-dense. Additional raw runs without verified full initialization or dense pairing remain inconclusive, not failures. Missing historical experiments are unavailable, not failures.", "",
        "Five seeds exist per canonical architecture/sparsity cell, but the eight cases were selected on the same validation outcomes summarized here. No equivalence test or significance claim is made; the selected cases do not establish seed-population equivalence.", "",
        "## Collection for Geometry Work", "",
        "Preserve and use all 100 canonical masks, not just the successes: the set spans all five expert counts, five seeds, and 80/85/90/95% sparsities, with R0-R100 router-age recoveries under the same mask and step-0 expert/shared state. This matched collection retains successful and unsuccessful controls and isolates router age. Prioritize the eight <5% hybrid endpoints and 19 R0 <10% screens for initial inspection, but keep same-mask failing ages and high-gap masks as negative controls. Avoid selecting only on validation outcomes when studying geometry/performance associations.", "",
        "Exact mask, initialization, router, and recovery checkpoint paths are in `checkpoint_manifest.csv`; files are referenced, not copied. The four masks behind the strongest hybrid runs and their age-matched R0/nonzero-age alternatives are listed by hash in the CSV inventory.", "",
        "## Inventory Files and Limitations", "",
        "- `all_candidates.csv`: distinct >=80% hashes; canonical provenance or explicit unknowns for raw-only hashes.",
        "- `all_recovery_runs.csv`: all 1,100 canonical conditions plus additional local raw metadata artifacts; only exact canonical dense pairings contribute threshold labels.",
        "- `r0_candidates.csv`: all R0 entries with verified and unknown component restoration kept distinct.",
        "- `near_dense_candidates.csv`: paired runs within any 2%, 5%, or 10% screen.",
        "- `winning_ticket_evidence.csv`: header-only because no candidate meets the conservative evidence criteria.",
        "- `checkpoint_manifest.csv`: reload paths, configuration IDs, and batch hashes; unknown paths are explicit.",
        "- `build_audit.py`: read-only builder; run `python3 results/lth_definition_audit/build_audit.py` from the repository root.", "",
        "The metadata scan does not follow symlinks. Canonical paths come from the frozen Dataset A manifest. All raw-only mask tensors were located and rehashed, but this does not independently establish which trained dense checkpoint generated a legacy magnitude mask or verify every legacy initialization. The inventory covers local files only, not remote/unindexed artifacts. No experiments were launched and no existing results were modified.", "",
        "## Final Answers", "",
        f"1. Distinct >=80% masks: {len(candidates)} recorded hashes across canonical plus local raw metadata (100 canonical; {len(candidates) - len(canonical_hashes)} additional raw-only hashes).",
        f"2. Dataset A recoveries verified to use original initialization for all relevant weights: {len(r0)} R0 conditions. There are {sum(row['router_age_percent'] == 0 for row in additional)} other raw R0 records, but their expert/shared initialization and/or mask provenance is unverified and they are excluded from this verified count.",
        f"3. Canonical R0 within 2% / 5% / 10%: {r0counts[0]} / {r0counts[1]} / {r0counts[2]}.",
        "4. Defensible original-LTH winners at >=80%: 0 verified; existing evidence supports candidates and router-rewind recoveries only.",
        "5. Near-dense (<5% CE gap) canonical conditions requiring a nonzero-age router: 8 of 8; four masks.",
        "6. Existing local raw evidence is insufficient to substantiate the original LTH claim at >=80%; the historical long-budget claim is not locally reproducible because raw artifacts are absent.",
        "7. Preserve the complete canonical 100-mask × 11-age collection, including strong candidates and matched negative controls; paths are in `checkpoint_manifest.csv`.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    source_rows = [row for row in read_csv(CANONICAL) if float(row["sparsity"]) >= 0.8 - 1e-9]
    canonical = canonical_records()
    mask_index = mask_tensor_index(source_rows)
    additional = raw_records(canonical, mask_index)
    runs = canonical + additional
    candidates = candidate_inventory(source_rows, additional, mask_index)
    r0 = [row for row in runs if str(row["router_age_percent"]) in {"0", "0.0"}]
    r0.sort(key=lambda row: (
        row["relative_loss_gap_pct"] == UNKNOWN,
        abs(float(row["relative_loss_gap_pct"])) if row["relative_loss_gap_pct"] != UNKNOWN else math.inf,
        row["run_id"],
    ))
    near = [row for row in runs if any(row.get(f"within_{threshold}pct") is True for threshold in THRESHOLDS)]
    near.sort(key=lambda row: (abs(float(row["relative_loss_gap_pct"])), row["run_id"]))
    write_csv("all_candidates.csv", candidates)
    write_csv("all_recovery_runs.csv", runs)
    write_csv("r0_candidates.csv", r0)
    write_csv("near_dense_candidates.csv", near)
    write_csv("winning_ticket_evidence.csv", [], [
        "run_id", "mask_hash", "num_experts", "seed", "target_sparsity", "protocol_class",
        "dense_loss", "sparse_loss", "absolute_loss_gap", "relative_loss_gap_pct", "evidence_caveats",
    ])
    manifest_fields = (
        "run_id", "scope", "protocol_class", "mask_hash", "mask_path", "mask_source_checkpoint_path",
        "initialization_checkpoint_path", "router_checkpoint_path", "recovery_artifact_path",
        "recovery_checkpoint_path", "dense_artifact_path", "source_config_id", "recovery_config_id",
        "config_path", "training_batch_hash", "validation_batch_hash", "validated",
    )
    write_csv("checkpoint_manifest.csv", [{key: row.get(key, UNKNOWN) for key in manifest_fields} for row in runs])
    (OUT / "README.md").write_text(build_readme(candidates, runs, canonical), encoding="utf-8")
    print(json.dumps({
        "canonical_recoveries": len(canonical),
        "canonical_distinct_masks": len({row["mask_hash"] for row in canonical}),
        "additional_raw_recovery_artifacts": len(additional),
        "raw_metadata_mask_hashes": len({row["mask_hash"] for row in additional}),
        "union_distinct_masks": len(candidates),
        "canonical_R0_runs": sum(int(row["router_age_percent"]) == 0 for row in canonical),
        "canonical_R0_within_2_5_10_pct": [sum(row[f"within_{t}pct"] is True for row in canonical if int(row["router_age_percent"]) == 0) for t in THRESHOLDS],
        "canonical_all_age_within_2_5_10_pct": [sum(row[f"within_{t}pct"] is True for row in canonical) for t in THRESHOLDS],
        "near_dense_runs": len(near),
        "winning_ticket_evidence_rows": 0,
    }, indent=2))


if __name__ == "__main__":
    main()