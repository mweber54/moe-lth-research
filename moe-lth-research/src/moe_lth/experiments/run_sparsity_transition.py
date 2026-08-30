"""Run the focused 75/85% standard-MoE sparsity transition experiment.

This runner deliberately preserves the historical recovery protocol: ``R_t``
denotes the router *initialization checkpoint*.  Router parameters continue to
train during recovery, exactly as they did in the completed standard-MoE runs.

Production runs fail closed unless the prior seed-level R0/R20/R100 dataset and
the exact reference checkpoint lineage that produced it are available.  This
prevents a numerically regenerated training trajectory from being mixed with
historical cells merely because its checkpoint filenames and seeds agree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

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


PROTOCOL_VERSION = "standard_moe_sparsity_transition_v1"
REFERENCE_SEEDS = (7, 17, 29)
NEW_SPARSITIES = (0.75, 0.85)
HISTORICAL_SPARSITIES = (0.60, 0.70, 0.80, 0.90, 0.95)
ROUTER_AGES = (0, 20, 100)
ROUTER_STEPS = {0: 0, 20: 500, 100: 2500}
RECOVERY_STEPS = 2500
RESULT_FILENAME = "transition_result.json"

CANONICAL_LONG_FIELDS = (
    "reference_seed",
    "sparsity",
    "achieved_sparsity",
    "router_age",
    "router_training_step",
    "router_checkpoint_identifier",
    "router_checkpoint_hash",
    "mask_hash",
    "sparse_final_validation_loss",
    "matched_dense_final_validation_loss",
    "ticket_gap",
    "recovery_steps",
    "training_evaluation_config_id",
    "completion_status",
    "audit_passed",
    "expert_state_hash",
    "shared_state_hash",
    "training_sequence_hash",
    "validation_sequence_hash",
    "dense_baseline_record_id",
    "mask_source",
    "expert_surviving_weight_source",
    "protocol_version",
    "result_path",
)


class TransitionPreflightError(RuntimeError):
    """Raised before scientific output is written when provenance is incomplete."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _same_float(left: Any, right: float, tolerance: float = 1e-9) -> bool:
    return _finite(left) and math.isclose(float(left), right, rel_tol=0.0, abs_tol=tolerance)


def _expert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "expert"}


def _shared_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in state.items() if parameter_group(name) == "shared"}


def _normalized_reference_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "output_dir"}


def _protocol_payload(config: dict[str, Any], recovery_steps: int, train_hash: str, validation_hash: str) -> dict[str, Any]:
    training = config["training"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model": config["model"],
        "data": config["data"],
        "routing": config["routing"],
        "reference_seed": int(config["seed"]),
        "optimizer": "AdamW",
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip": float(training["grad_clip"]),
        "scheduler": "none",
        "precision": training.get("precision", "fp32"),
        "recovery_steps": int(recovery_steps),
        "recovery_eval_interval": recovery.RECOVERY_EVAL_INTERVAL,
        "early_auc_window_fraction": recovery.EARLY_AUC_WINDOW_FRACTION,
        "training_sequence_hash": train_hash,
        "validation_sequence_hash": validation_hash,
        "router_checkpoint_semantics": "initialization_state_trainable_during_recovery",
    }


def _load_checkpoint_identity(config: dict[str, Any], checkpoint: Path, expected_step: int) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != expected_step:
        raise TransitionPreflightError(
            f"Checkpoint payload step mismatch for {checkpoint}: expected {expected_step}, got {payload.get('step')}."
        )
    payload_config = payload.get("config")
    if not isinstance(payload_config, dict):
        raise TransitionPreflightError(f"Checkpoint {checkpoint} has no reconstructable config payload.")
    if int(payload_config.get("seed", -1)) != int(config["seed"]):
        raise TransitionPreflightError(
            f"Checkpoint seed mismatch for {checkpoint}: expected {config['seed']}, got {payload_config.get('seed')}."
        )
    if _sha256_json(_normalized_reference_config(payload_config)) != _sha256_json(_normalized_reference_config(config)):
        raise TransitionPreflightError(f"Checkpoint config mismatch for {checkpoint}.")
    model = load_model_from_checkpoint(config["model"], str(checkpoint), torch.device("cpu"))
    return {
        "path": str(checkpoint),
        "identifier": f"step_{expected_step}.pt",
        "payload_step": expected_step,
        "file_sha256": _sha256_file(checkpoint),
        "router_hash": state_dict_hash(component_state_dict(model, "router")),
    }


def _required_checkpoint_paths(config: dict[str, Any]) -> dict[int, Path]:
    run_dir = Path(config["output_dir"])
    return {age: run_dir / "checkpoints" / f"step_{ROUTER_STEPS[age]}.pt" for age in ROUTER_AGES}


def _canonical_historical_row(row: dict[str, Any]) -> dict[str, Any]:
    def first(*names: str) -> Any:
        for name in names:
            if name in row and row[name] not in (None, ""):
                return row[name]
        return None

    return {
        **row,
        "reference_seed": int(first("reference_seed", "seed")),
        "sparsity": float(first("sparsity", "requested_sparsity")),
        "router_age": int(first("router_age", "router_age_percent")),
        "router_training_step": int(first("router_training_step", "router_step", "loaded_router_step")),
        "sparse_final_validation_loss": float(first("sparse_final_validation_loss", "sparse_final_loss", "final_validation_loss")),
        "matched_dense_final_validation_loss": float(first("matched_dense_final_validation_loss", "dense_final_loss", "dense_baseline_final_loss")),
        "ticket_gap": float(first("ticket_gap")),
        "mask_hash": first("mask_hash", "mask_sha256"),
        "router_checkpoint_hash": first("router_checkpoint_hash", "router_hash", "initial_router_state_hash"),
        "shared_state_hash": first("shared_state_hash"),
        "training_sequence_hash": first("training_sequence_hash", "training_batch_sequence_hash"),
        "validation_sequence_hash": first("validation_sequence_hash", "validation_batch_sequence_hash"),
        "audit_passed": _as_bool(first("audit_passed", "integrity_checks_passed")),
    }


def _load_historical_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise TransitionPreflightError(f"Historical seed-level file is missing: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            raw = list(csv.DictReader(handle))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("records", value.get("rows"))
        if not isinstance(value, list):
            raise TransitionPreflightError(f"Historical data in {path} is not a list of rows.")
        raw = value
    rows = [_canonical_historical_row(dict(row)) for row in raw]
    required_keys = {
        (seed, sparsity, age)
        for seed in REFERENCE_SEEDS
        for sparsity in HISTORICAL_SPARSITIES
        for age in ROUTER_AGES
    }
    filtered = [
        row
        for row in rows
        if row["reference_seed"] in REFERENCE_SEEDS
        and any(_same_float(row["sparsity"], sparsity) for sparsity in HISTORICAL_SPARSITIES)
        and row["router_age"] in ROUTER_AGES
    ]
    keys = [(row["reference_seed"], row["sparsity"], row["router_age"]) for row in filtered]
    if len(keys) != len(set(keys)):
        raise TransitionPreflightError("Historical seed-level data contains duplicate transition keys.")
    observed = set(keys)
    if observed != required_keys:
        missing = sorted(required_keys - observed)
        extra = sorted(observed - required_keys)
        raise TransitionPreflightError(
            f"Historical seed-level data is incomplete for R0/R20/R100: missing={missing}, extra={extra}."
        )
    for row in filtered:
        expected_step = ROUTER_STEPS[row["router_age"]]
        if row["router_training_step"] != expected_step:
            raise TransitionPreflightError(f"Historical router step mismatch in row {row}.")
        if not row["audit_passed"]:
            raise TransitionPreflightError(f"Historical row is not audited: {row}.")
        if not all(_finite(row[field]) for field in ("sparse_final_validation_loss", "matched_dense_final_validation_loss", "ticket_gap")):
            raise TransitionPreflightError(f"Historical row contains a nonfinite loss: {row}.")
        recomputed = row["sparse_final_validation_loss"] - row["matched_dense_final_validation_loss"]
        if not math.isclose(recomputed, row["ticket_gap"], rel_tol=0.0, abs_tol=1e-10):
            raise TransitionPreflightError(f"Historical ticket gap does not recompute exactly enough: {row}.")
        for field in ("mask_hash", "router_checkpoint_hash", "shared_state_hash", "training_sequence_hash", "validation_sequence_hash"):
            if not row.get(field):
                raise TransitionPreflightError(f"Historical row lacks required provenance field {field}: {row}.")
    return sorted(filtered, key=lambda row: (row["sparsity"], row["router_age"], row["reference_seed"]))


def _mask_statistics(masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = sum(mask.numel() for mask in masks.values())
    retained = sum(int(mask.sum().item()) for mask in masks.values())
    grouped: dict[str, list[torch.Tensor]] = {}
    for name, mask in masks.items():
        prefix, remainder = name.split(".moe.experts.", maxsplit=1)
        expert_id = remainder.split(".", maxsplit=1)[0]
        grouped.setdefault(f"{prefix}.moe.experts.{expert_id}", []).append(mask)
    per_expert = {}
    for expert, values in sorted(grouped.items()):
        count = sum(value.numel() for value in values)
        keep = sum(int(value.sum().item()) for value in values)
        per_expert[expert] = {
            "prunable_parameters": count,
            "retained_parameters": keep,
            "achieved_sparsity": (count - keep) / count,
        }
    return {
        "prunable_expert_parameters": total,
        "retained_expert_parameters": retained,
        "pruned_expert_parameters": total - retained,
        "achieved_sparsity": (total - retained) / total,
        "per_expert": per_expert,
    }


def _assert_rewound(
    initial_state: dict[str, torch.Tensor],
    ticket_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> None:
    for name, mask in masks.items():
        keep = mask.detach().cpu().bool()
        expected = initial_state[name].detach().cpu()
        actual = ticket_state[name].detach().cpu()
        if not torch.equal(actual[keep], expected[keep]):
            raise RuntimeError(f"Retained ticket coordinates are not rewound to E0 for {name}.")
        if not torch.equal(actual[~keep], torch.zeros_like(actual[~keep])):
            raise RuntimeError(f"Pruned ticket coordinates are nonzero for {name}.")


def _lineage_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return [row for row in rows if row["reference_seed"] == seed]


def audit_transition_prerequisites(
    config_paths: list[str],
    historical_seed_level: str | None,
    recovery_steps: int = RECOVERY_STEPS,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Read-only, fail-closed provenance audit performed before output creation."""
    configs = [load_config(path) for path in config_paths]
    seeds = tuple(int(config["seed"]) for config in configs)
    if smoke:
        if len(configs) != 1:
            raise TransitionPreflightError("Smoke mode requires exactly one reference config.")
        historical_rows: list[dict[str, Any]] = []
    else:
        if tuple(sorted(seeds)) != REFERENCE_SEEDS or len(set(seeds)) != len(REFERENCE_SEEDS):
            raise TransitionPreflightError(f"Production configs must contain seeds {REFERENCE_SEEDS} exactly once; got {seeds}.")
        if int(recovery_steps) != RECOVERY_STEPS:
            raise TransitionPreflightError(f"Production recovery budget must be {RECOVERY_STEPS}, got {recovery_steps}.")
        if not historical_seed_level:
            raise TransitionPreflightError("Production requires the prior seed-level R0/R20/R100 dataset.")
        historical_rows = _load_historical_rows(Path(historical_seed_level))

    missing = [
        str(path)
        for config in configs
        for path in _required_checkpoint_paths(config).values()
        if not path.exists()
    ]
    if missing:
        raise TransitionPreflightError(
            "Required verified reference checkpoints are missing; this runner will not regenerate them: "
            + ", ".join(missing)
        )

    audit: dict[str, Any] = {
        "all_pass": True,
        "protocol_version": PROTOCOL_VERSION,
        "smoke": smoke,
        "router_checkpoint_semantics": "initialization_state_trainable_during_recovery",
        "historical_seed_level": historical_seed_level,
        "seeds": {},
    }
    for config_path, config in zip(config_paths, configs):
        seed = int(config["seed"])
        paths = _required_checkpoint_paths(config)
        identities = {
            age: _load_checkpoint_identity(config, path, ROUTER_STEPS[age])
            for age, path in paths.items()
        }
        initial_model = load_model_from_checkpoint(config["model"], str(paths[0]), torch.device("cpu"))
        trained_model = load_model_from_checkpoint(config["model"], str(paths[100]), torch.device("cpu"))
        initial_state = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}
        shared_hash = state_dict_hash(_shared_state(initial_state))

        loader, validation_loader = build_dataloaders(
            config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
        )
        train_batches = recovery._materialize_batches(loader, int(recovery_steps))
        validation_batches = recovery._materialize_validation_batches(
            validation_loader, int(config["data"]["validation_blocks"])
        )
        train_hash = recovery._batch_sequence_hash(train_batches)
        validation_hash = recovery._batch_sequence_hash(validation_batches)

        anchor_masks = {}
        if not smoke:
            rows = _lineage_rows(historical_rows, seed)
            for age in ROUTER_AGES:
                expected_hashes = {row["router_checkpoint_hash"] for row in rows if row["router_age"] == age}
                if expected_hashes != {identities[age]["router_hash"]}:
                    raise TransitionPreflightError(
                        f"Seed {seed} R{age} router lineage mismatch: historical={sorted(expected_hashes)}, "
                        f"local={identities[age]['router_hash']}."
                    )
            for field, local in (
                ("shared_state_hash", shared_hash),
                ("training_sequence_hash", train_hash),
                ("validation_sequence_hash", validation_hash),
            ):
                expected = {row[field] for row in rows}
                if expected != {local}:
                    raise TransitionPreflightError(
                        f"Seed {seed} {field} lineage mismatch: historical={sorted(expected)}, local={local}."
                    )
            for sparsity in HISTORICAL_SPARSITIES:
                masks = expert_local_magnitude_masks(trained_model, sparsity)
                current_hash = recovery._mask_hash(masks)
                expected_hashes = {
                    row["mask_hash"] for row in rows if _same_float(row["sparsity"], sparsity)
                }
                if expected_hashes != {current_hash}:
                    raise TransitionPreflightError(
                        f"Seed {seed} s={sparsity:.2f} E_T/mask lineage mismatch: "
                        f"historical={sorted(expected_hashes)}, local={current_hash}."
                    )
                anchor_masks[f"{sparsity:.2f}"] = current_hash

        audit["seeds"][str(seed)] = {
            "config_path": config_path,
            "reference_run_dir": config["output_dir"],
            "checkpoints": {f"R{age}": identities[age] for age in ROUTER_AGES},
            "shared_state_hash": shared_hash,
            "training_sequence_hash": train_hash,
            "validation_sequence_hash": validation_hash,
            "historical_mask_anchors": anchor_masks,
        }
        del train_batches, validation_batches, trained_model, initial_model, initial_state
    return audit


def _load_record_documents(paths: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "records" in document:
            document = document["records"]
        if isinstance(document, dict):
            document = [document]
        if not isinstance(document, list):
            raise ValueError(f"Dense record document is not a record/list: {path}")
        for row in document:
            records.append({**row, "result_path": row.get("result_path", str(path))})
    return records


def _logical_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    if record.get("condition_type") == "dense_control":
        return ("dense", int(record["reference_seed"]), int(record["router_age"]))
    if record.get("condition_type") == "sparse_ticket":
        return (
            "sparse",
            int(record["reference_seed"]),
            round(float(record["sparsity"]), 8),
            int(record["router_age"]),
        )
    return None


def _scan_owned_records(root: Path) -> list[dict[str, Any]]:
    records = []
    if not root.exists():
        return records
    for path in root.rglob(RESULT_FILENAME):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row["result_path"] = str(path)
        records.append(row)
    return records


def _record_matches(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    if record.get("completion_status") != "complete" or record.get("protocol_version") != PROTOCOL_VERSION:
        return False
    if not _as_bool(record.get("audit_passed")):
        return False
    if not _finite(record.get("final_validation_loss")):
        return False
    for field, value in expected.items():
        observed = record.get(field)
        if isinstance(value, float):
            if not _same_float(observed, value):
                return False
        elif observed != value:
            return False
    return True


def _next_attempt_dir(base: Path) -> Path:
    if not base.exists() or not any(base.iterdir()):
        return base
    for index in range(1, 10_000):
        candidate = base.with_name(f"{base.name}_retry_{index:03d}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise RuntimeError(f"Could not allocate a non-overwriting retry directory for {base}.")


def _record_id(record: dict[str, Any]) -> str:
    return _sha256_json(
        {
            "condition_type": record["condition_type"],
            "reference_seed": record["reference_seed"],
            "sparsity": record["sparsity"],
            "router_age": record["router_age"],
            "router_checkpoint_hash": record["router_checkpoint_hash"],
            "mask_hash": record["mask_hash"],
            "training_evaluation_config_id": record["training_evaluation_config_id"],
        }
    )


def _write_result(condition_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    completed = {**record, "completion_status": "complete", "audit_passed": True}
    completed["result_path"] = str(condition_dir / RESULT_FILENAME)
    completed["record_id"] = _record_id(completed)
    _atomic_json(condition_dir / RESULT_FILENAME, completed)
    return completed


def _canonical_new_long(record: dict[str, Any]) -> dict[str, Any]:
    return {field: record.get(field) for field in CANONICAL_LONG_FIELDS}


def _historical_long(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_seed": row["reference_seed"],
        "sparsity": row["sparsity"],
        "achieved_sparsity": row.get("achieved_sparsity", row["sparsity"]),
        "router_age": row["router_age"],
        "router_training_step": row["router_training_step"],
        "router_checkpoint_identifier": row.get("router_checkpoint_identifier", f"historical:R{row['router_age']}"),
        "router_checkpoint_hash": row["router_checkpoint_hash"],
        "mask_hash": row["mask_hash"],
        "sparse_final_validation_loss": row["sparse_final_validation_loss"],
        "matched_dense_final_validation_loss": row["matched_dense_final_validation_loss"],
        "ticket_gap": row["ticket_gap"],
        "recovery_steps": row.get("recovery_steps", RECOVERY_STEPS),
        "training_evaluation_config_id": row.get("training_evaluation_config_id", "historical_protocol_verified"),
        "completion_status": "complete",
        "audit_passed": row["audit_passed"],
        "expert_state_hash": row.get("expert_state_hash"),
        "shared_state_hash": row["shared_state_hash"],
        "training_sequence_hash": row["training_sequence_hash"],
        "validation_sequence_hash": row["validation_sequence_hash"],
        "dense_baseline_record_id": row.get("dense_baseline_record_id"),
        "mask_source": row.get("mask_source", "ET"),
        "expert_surviving_weight_source": row.get("expert_surviving_weight_source", "E0"),
        "protocol_version": row.get("protocol_version", "historical_corrected_lth"),
        "result_path": row.get("result_path", "historical_seed_level"),
    }


def _write_incremental(
    root: Path,
    dense_records: list[dict[str, Any]],
    sparse_records: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    dense_sorted = sorted(dense_records, key=lambda row: (row["reference_seed"], row["router_age"]))
    sparse_sorted = sorted(
        sparse_records,
        key=lambda row: (row["sparsity"], row["reference_seed"], row["router_age"]),
    )
    _atomic_json(root / "transition_run_manifest.json", {
        "protocol_version": PROTOCOL_VERSION,
        "dense_records": dense_sorted,
        "sparse_records": sparse_sorted,
        "audit": audit,
    })
    _atomic_csv(root / "transition_new_seed_level.csv", [_canonical_new_long(row) for row in sparse_sorted], CANONICAL_LONG_FIELDS)


def run_sparsity_transition(
    config_paths: list[str],
    output_dir: str,
    historical_seed_level: str | None,
    dense_record_paths: list[str] | None = None,
    recovery_steps: int = RECOVERY_STEPS,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run/resume the exact transition matrix after a strict read-only preflight."""
    dense_record_paths = dense_record_paths or []
    preflight = audit_transition_prerequisites(
        config_paths, historical_seed_level, recovery_steps, smoke=smoke
    )
    historical_rows = [] if smoke else _load_historical_rows(Path(str(historical_seed_level)))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "transition_prerun_audit.json", preflight)

    owned = _scan_owned_records(root)
    external_dense = _load_record_documents(dense_record_paths)
    dense_records: list[dict[str, Any]] = []
    sparse_records: list[dict[str, Any]] = []
    run_audit = {
        "all_pass": True,
        "smoke": smoke,
        "new_dense_runs": 0,
        "dense_baselines_reused": [],
        "new_sparse_runs": 0,
        "sparse_runs_resumed": [],
        "non_overwriting_retries": [],
        "router_checkpoint_semantics": "initialization_state_trainable_during_recovery",
    }

    for config_path in config_paths:
        config = load_config(config_path)
        seed = int(config["seed"])
        paths = _required_checkpoint_paths(config)
        device = resolve_device(config["device"])
        configure_device(device)
        identities = {
            age: _load_checkpoint_identity(config, paths[age], ROUTER_STEPS[age])
            for age in ROUTER_AGES
        }

        seed_everything(seed)
        trained = load_model_from_checkpoint(config["model"], str(paths[100]), device)
        initial_model = load_model_from_checkpoint(config["model"], str(paths[0]), device)
        initial_state = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}
        dense_base = initial_state
        dense_expert_hash = state_dict_hash(_expert_state(dense_base))
        shared_hash = state_dict_hash(_shared_state(dense_base))

        loader, validation_loader = build_dataloaders(
            config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
        )
        train_batches = recovery._materialize_batches(loader, int(recovery_steps))
        validation_batches = recovery._materialize_validation_batches(
            validation_loader, int(config["data"]["validation_blocks"])
        )
        train_hash = recovery._batch_sequence_hash(train_batches)
        validation_hash = recovery._batch_sequence_hash(validation_batches)
        config_id = _sha256_json(_protocol_payload(config, recovery_steps, train_hash, validation_hash))
        calibration = recovery._calibration_batches(validation_batches)
        dense_reference_loss = recovery.evaluate_language_model(
            trained, validation_batches, device, max_batches=len(validation_batches)
        )["loss"]

        dense_by_age: dict[int, dict[str, Any]] = {}
        dense_reference_model = recovery.assemble_router_age_model(
            config["model"], dense_base, str(paths[100]), {}, device
        )
        dense_reference_selected = selected_experts_per_batch(dense_reference_model, calibration, device)
        for age in ROUTER_AGES:
            identity = identities[age]
            expected = {
                "condition_type": "dense_control",
                "reference_seed": seed,
                "sparsity": 0.0,
                "router_age": age,
                "router_training_step": ROUTER_STEPS[age],
                "router_checkpoint_hash": identity["router_hash"],
                "router_checkpoint_file_sha256": identity["file_sha256"],
                "mask_hash": "dense_no_mask",
                "expert_state_hash": dense_expert_hash,
                "shared_state_hash": shared_hash,
                "training_sequence_hash": train_hash,
                "validation_sequence_hash": validation_hash,
                "recovery_steps": int(recovery_steps),
                "training_evaluation_config_id": config_id,
            }
            candidates = owned + external_dense
            reusable = next((row for row in candidates if _record_matches(row, expected)), None)
            if reusable is not None:
                dense = dict(reusable)
                dense_by_age[age] = dense
                dense_records.append(dense)
                run_audit["dense_baselines_reused"].append(dense.get("record_id", dense.get("result_path")))
                continue

            base_dir = root / "dense" / f"seed_{seed}" / f"R{age}"
            condition_dir = _next_attempt_dir(base_dir)
            if condition_dir != base_dir:
                run_audit["non_overwriting_retries"].append(str(condition_dir))
            core = recovery._run_recovery_condition(
                config=config,
                condition_name=f"transition_seed{seed}_R{age}_dense",
                pruned_base_state=dense_base,
                router_checkpoint=str(paths[age]),
                router_age_percent=age,
                router_step=ROUTER_STEPS[age],
                masks={},
                expert_hash=dense_expert_hash,
                shared_hash=shared_hash,
                mask_hash="dense_no_mask",
                reference_selected=dense_reference_selected,
                calibration_batches=calibration,
                train_batches=train_batches,
                validation_batches=validation_batches,
                train_batch_hash=train_hash,
                validation_batch_hash=validation_hash,
                device=device,
                recovery_steps=int(recovery_steps),
                dense_loss=dense_reference_loss,
                output_dir=condition_dir,
                confidence_control=False,
                target_confidence=None,
                seed=seed,
                sparsity=0.0,
            )
            dense = _write_result(condition_dir, {
                **core,
                **expected,
                "router_checkpoint_identifier": identity["identifier"],
                "router_checkpoint_path": identity["path"],
                "requested_sparsity": 0.0,
                "achieved_sparsity": 0.0,
                "mask_identifier": "dense_no_mask",
                "matched_dense_final_validation_loss": core["final_validation_loss"],
                "ticket_gap": 0.0,
                "mask_source": "none",
                "expert_surviving_weight_source": "E0",
                "shared_state_source": "E0",
                "protocol_version": PROTOCOL_VERSION,
                "router_checkpoint_semantics": "initialization_state_trainable_during_recovery",
            })
            dense_by_age[age] = dense
            dense_records.append(dense)
            owned.append(dense)
            run_audit["new_dense_runs"] += 1
            _write_incremental(root, dense_records, sparse_records, run_audit)

        for sparsity in NEW_SPARSITIES:
            masks = expert_local_magnitude_masks(trained, sparsity)
            mask_hash = recovery._mask_hash(masks)
            mask_stats = _mask_statistics(masks)
            if not math.isclose(mask_stats["achieved_sparsity"], sparsity, rel_tol=0.0, abs_tol=1e-6):
                raise RuntimeError(
                    f"Achieved sparsity mismatch for seed {seed}, requested {sparsity}: {mask_stats['achieved_sparsity']}"
                )
            ticket = build_fixed_pruned_base(config["model"], str(paths[0]), masks, device)
            _assert_rewound(initial_state, ticket, masks)
            expert_hash = state_dict_hash(_expert_state(ticket))
            if expert_hash == state_dict_hash(_expert_state(trained.state_dict())):
                raise RuntimeError("Sparse ticket unexpectedly equals trained E_T expert state.")

            mask_dir = root / "masks" / f"sparsity_{sparsity:.2f}" / f"seed_{seed}" / mask_hash
            mask_path = mask_dir / "pruning_mask.pt"
            metadata_path = mask_dir / "pruning_metadata.json"
            if mask_path.exists() or metadata_path.exists():
                if not (mask_path.exists() and metadata_path.exists()):
                    raise RuntimeError(f"Partial content-addressed mask artifact at {mask_dir}.")
                prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if prior_metadata.get("mask_hash") != mask_hash:
                    raise RuntimeError(f"Mask metadata hash mismatch at {mask_dir}.")
            else:
                mask_dir.mkdir(parents=True, exist_ok=True)
                save_masks(masks, mask_path)
                _atomic_json(metadata_path, {
                    "reference_seed": seed,
                    "requested_sparsity": sparsity,
                    "mask_hash": mask_hash,
                    "mask_identifier": f"seed{seed}_s{sparsity:.2f}_{mask_hash[:16]}",
                    "mask_source": "ET",
                    "mask_source_checkpoint": str(paths[100]),
                    "mask_source_checkpoint_sha256": identities[100]["file_sha256"],
                    "pruning_method": "expert_local_magnitude",
                    **mask_stats,
                })

            sparse_reference_model = recovery.assemble_router_age_model(
                config["model"], ticket, str(paths[100]), masks, device
            )
            sparse_reference_selected = selected_experts_per_batch(
                sparse_reference_model, calibration, device
            )
            for age in ROUTER_AGES:
                identity = identities[age]
                dense = dense_by_age[age]
                expected = {
                    "condition_type": "sparse_ticket",
                    "reference_seed": seed,
                    "sparsity": sparsity,
                    "router_age": age,
                    "router_training_step": ROUTER_STEPS[age],
                    "router_checkpoint_hash": identity["router_hash"],
                    "router_checkpoint_file_sha256": identity["file_sha256"],
                    "mask_hash": mask_hash,
                    "expert_state_hash": expert_hash,
                    "shared_state_hash": shared_hash,
                    "training_sequence_hash": train_hash,
                    "validation_sequence_hash": validation_hash,
                    "recovery_steps": int(recovery_steps),
                    "training_evaluation_config_id": config_id,
                    "dense_baseline_record_id": dense["record_id"],
                }
                reusable = next((row for row in owned if _record_matches(row, expected)), None)
                if reusable is not None:
                    sparse_records.append(dict(reusable))
                    run_audit["sparse_runs_resumed"].append(reusable["record_id"])
                    continue

                base_dir = root / "sparse" / f"sparsity_{sparsity:.2f}" / f"seed_{seed}" / f"R{age}"
                condition_dir = _next_attempt_dir(base_dir)
                if condition_dir != base_dir:
                    run_audit["non_overwriting_retries"].append(str(condition_dir))
                core = recovery._run_recovery_condition(
                    config=config,
                    condition_name=f"transition_seed{seed}_s{sparsity:.2f}_R{age}_sparse",
                    pruned_base_state=ticket,
                    router_checkpoint=str(paths[age]),
                    router_age_percent=age,
                    router_step=ROUTER_STEPS[age],
                    masks=masks,
                    expert_hash=expert_hash,
                    shared_hash=shared_hash,
                    mask_hash=mask_hash,
                    reference_selected=sparse_reference_selected,
                    calibration_batches=calibration,
                    train_batches=train_batches,
                    validation_batches=validation_batches,
                    train_batch_hash=train_hash,
                    validation_batch_hash=validation_hash,
                    device=device,
                    recovery_steps=int(recovery_steps),
                    dense_loss=dense["final_validation_loss"],
                    output_dir=condition_dir,
                    confidence_control=False,
                    target_confidence=None,
                    seed=seed,
                    sparsity=sparsity,
                )
                ticket_gap = core["final_validation_loss"] - dense["final_validation_loss"]
                sparse = _write_result(condition_dir, {
                    **core,
                    **expected,
                    "router_checkpoint_identifier": identity["identifier"],
                    "router_checkpoint_path": identity["path"],
                    "requested_sparsity": sparsity,
                    "achieved_sparsity": mask_stats["achieved_sparsity"],
                    "mask_identifier": f"seed{seed}_s{sparsity:.2f}_{mask_hash[:16]}",
                    "mask_path": str(mask_path),
                    "sparse_final_validation_loss": core["final_validation_loss"],
                    "matched_dense_final_validation_loss": dense["final_validation_loss"],
                    "ticket_gap": ticket_gap,
                    "mask_source": "ET",
                    "mask_source_checkpoint": str(paths[100]),
                    "expert_surviving_weight_source": "E0",
                    "rewind_checkpoint": str(paths[0]),
                    "shared_state_source": "E0",
                    "protocol_version": PROTOCOL_VERSION,
                    "router_checkpoint_semantics": "initialization_state_trainable_during_recovery",
                })
                sparse_records.append(sparse)
                owned.append(sparse)
                run_audit["new_sparse_runs"] += 1
                _write_incremental(root, dense_records, sparse_records, run_audit)

            same_mask = {
                row["mask_hash"]
                for row in sparse_records
                if row["reference_seed"] == seed and _same_float(row["sparsity"], sparsity)
            }
            if same_mask != {mask_hash}:
                raise RuntimeError(f"Mask identity differs across routers for seed {seed}, sparsity {sparsity}.")

    expected_sparse = len(config_paths) * len(NEW_SPARSITIES) * len(ROUTER_AGES)
    expected_dense = len(config_paths) * len(ROUTER_AGES)
    unique_sparse = {_logical_key(row) for row in sparse_records}
    unique_dense = {_logical_key(row) for row in dense_records}
    if len(unique_sparse) != expected_sparse or len(unique_dense) != expected_dense:
        raise RuntimeError(
            f"Incomplete transition output: sparse={len(unique_sparse)}/{expected_sparse}, "
            f"dense={len(unique_dense)}/{expected_dense}."
        )
    _write_incremental(root, dense_records, sparse_records, run_audit)

    combined = [_historical_long(row) for row in historical_rows]
    combined.extend(_canonical_new_long(row) for row in sparse_records)
    combined.sort(key=lambda row: (float(row["sparsity"]), int(row["router_age"]), int(row["reference_seed"])))
    if not smoke:
        if len(combined) != 63:
            raise RuntimeError(f"Expected 63 combined rows, got {len(combined)}.")
        _atomic_csv(root / "standard_moe_transition_combined.csv", combined, CANONICAL_LONG_FIELDS)
    run_audit["all_pass"] = True
    run_audit["completed_sparse_records"] = expected_sparse
    run_audit["validated_dense_records"] = expected_dense
    _write_incremental(root, dense_records, sparse_records, run_audit)
    return {
        "output_dir": str(root),
        "audit": run_audit,
        "dense_records": dense_records,
        "sparse_records": sparse_records,
        "combined_rows": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--historical-seed-level")
    parser.add_argument("--dense-records", nargs="*", default=[])
    parser.add_argument("--recovery-steps", type=int, default=RECOVERY_STEPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-report")
    args = parser.parse_args()
    try:
        audit = audit_transition_prerequisites(
            args.configs,
            args.historical_seed_level,
            args.recovery_steps,
            smoke=args.smoke,
        )
    except TransitionPreflightError as error:
        report = {"all_pass": False, "error": str(error), "protocol_version": PROTOCOL_VERSION}
        if args.audit_report:
            _atomic_json(Path(args.audit_report), report)
        raise
    if args.audit_report:
        _atomic_json(Path(args.audit_report), audit)
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    result = run_sparsity_transition(
        args.configs,
        args.output_dir,
        args.historical_seed_level,
        args.dense_records,
        args.recovery_steps,
        smoke=args.smoke,
    )
    print(json.dumps({"output_dir": result["output_dir"], "audit": result["audit"]}, indent=2))


if __name__ == "__main__":
    main()
