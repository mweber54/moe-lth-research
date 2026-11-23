"""Shared, isolated utilities for the post-pruning adaptation experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/imp_postpruning_adaptation_v1"
TRANSITION = ROOT / "results/imp_mask_transition_v1"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from moe_lth.data import build_dataloaders
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import apply_masks_, load_masks
from moe_lth.pruning.router_age import load_model_from_checkpoint, state_dict_hash

SEEDS = (7, 17, 29, 43, 61)
MODEL_TYPES = ("unpruned", "sparse")
CONDITIONS = ("full", "router_frozen", "experts_frozen", "router_experts_frozen")
OFFSETS = (0, 25, 100, 250, 500)
GLOBAL_STEPS = (1000, 1025, 1100, 1250, 1500)
ATOL = 1e-6
RTOL = 1e-5
CE_ATOL = 1e-6
TRAINABLE_GROUPS = {
    "full": frozenset(("router", "experts", "shared")),
    "router_frozen": frozenset(("experts", "shared")),
    "experts_frozen": frozenset(("router", "shared")),
    "router_experts_frozen": frozenset(("shared",)),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_save(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def json_read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_save(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def torch_save_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=False)


def group_for_parameter(name: str) -> str:
    if ".moe.experts." in name:
        return "experts"
    if ".moe.router." in name:
        return "router"
    return "shared"


def batch_hash(batches: list[tuple[torch.Tensor, torch.Tensor]]) -> str:
    return recovery._batch_sequence_hash(batches)


def materialize_data(variant: dict, count: int = 5000) -> tuple[list, list]:
    seed = int(variant["seed"])
    recovery.seed_everything(seed)
    train_loader, validation_loader = build_dataloaders(
        variant["config"]["data"], 128, seed
    )
    train = recovery._materialize_batches(train_loader, count)
    validation = recovery._materialize_validation_batches(validation_loader, 32)
    if count >= 2500:
        if recovery._batch_sequence_hash(train[:2500]) != variant["canonical_pair"]["data_hash"]:
            raise RuntimeError(f"Training prefix hash mismatch for seed {seed}.")
    if count >= 5000:
        if recovery._batch_sequence_hash(train[2500:]) != variant["extension_identity"]["training_batch_hash"]:
            raise RuntimeError(f"Training suffix hash mismatch for seed {seed}.")
    if recovery._batch_sequence_hash(validation) != variant["canonical_pair"]["validation_hash"]:
        raise RuntimeError(f"Validation batch hash mismatch for seed {seed}.")
    return train, validation


def data_manifest(variant: dict, train: list, validation: list) -> dict:
    next_batches = train[1000:1500]
    per_batch = []
    for offset, (inputs, targets) in enumerate(next_batches, start=1001):
        digest = hashlib.sha256()
        for tensor in (inputs, targets):
            data = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(data.shape)).encode())
            digest.update(str(data.dtype).encode())
            digest.update(data.numpy().tobytes())
        per_batch.append(
            {
                "global_step": offset,
                "sha256": digest.hexdigest(),
                "target_tokens": targets.numel(),
            }
        )
    return {
        "seed": int(variant["seed"]),
        "training_prefix_0_2499_sha256": batch_hash(train[:2500]),
        "training_suffix_2500_4999_sha256": batch_hash(train[2500:5000]),
        "next_500_batches_sha256": batch_hash(next_batches),
        "next_500_batches": per_batch,
        "validation_sha256": batch_hash(validation),
        "validation_batches": len(validation),
        "validation_target_tokens": sum(targets.numel() for _, targets in validation),
        "validation_batch_target_counts": [targets.numel() for _, targets in validation],
    }


def unique_parameter_rows(
    model: torch.nn.Module, masks: dict[str, torch.Tensor], seed: int
) -> list[dict]:
    params = dict(model.named_parameters())
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    if model.token_embedding.weight is not model.lm_head.weight:
        raise RuntimeError("Expected tied input embedding/output head.")

    rows = []
    seen = set()
    for name, parameter in params.items():
        identity = id(parameter)
        if identity in seen:
            raise RuntimeError(f"Duplicate unique parameter enumeration: {name}")
        seen.add(identity)
        names = aliases[identity]
        groups = {group_for_parameter(alias) for alias in names}
        if len(groups) != 1:
            raise RuntimeError(f"Tied aliases cross parameter groups: {names}")
        mask = masks.get(name)
        kept = int(mask.sum()) if mask is not None else parameter.numel()
        rows.append(
            {
                "seed": seed,
                "parameter": name,
                "aliases": ";".join(names),
                "shape": json.dumps(list(parameter.shape), separators=(",", ":")),
                "parameter_count": parameter.numel(),
                "mask_membership": "IMP80_weight" if mask is not None else "unmasked",
                "retained_count": kept,
                "pruned_count": parameter.numel() - kept,
                "group": next(iter(groups)),
            }
        )
    if sum(row["parameter_count"] for row in rows) != sum(
        parameter.numel() for parameter in model.parameters()
    ):
        raise RuntimeError("Parameter partition is not complete.")
    if sum(row["group"] == "shared" for row in rows) == 0:
        raise RuntimeError("Shared partition is empty.")
    return rows


def model_from_state(config: dict, state: dict, device: torch.device) -> TinyMoELanguageModel:
    model = TinyMoELanguageModel(config["model"]).to(device)
    model.load_state_dict(state)
    return model


def preserve_rng_and_modes(model: torch.nn.Module):
    class Protected:
        def __enter__(self):
            self.python = random.getstate()
            self.numpy = np.random.get_state()
            self.cpu = torch.get_rng_state()
            self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            self.modes = [(module, module.training) for module in model.modules()]
            self.buffers = {
                name: value.detach().clone()
                for name, value in model.named_buffers()
            }
            return self

        def __exit__(self, exc_type, exc, tb):
            random.setstate(self.python)
            np.random.set_state(self.numpy)
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state_all(self.cuda)
            for module, mode in self.modes:
                module.training = mode
            for name, value in model.named_buffers():
                if not torch.equal(value, self.buffers[name]):
                    raise RuntimeError(f"Read-only measurement changed buffer {name}.")
            return False

    return Protected()
