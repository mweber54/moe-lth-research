from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "device": "auto",
    "data": {
        "path": None,
        "train_path": None,
        "validation_path": None,
        "seq_len": 64,
        "train_fraction": 0.9,
        "validation_blocks": 32,
        "reshuffle_each_epoch": False,
        "max_train_examples": None,
        "max_validation_examples": None,
        "tokenizer": "byte",
        "tokenizer_vocab_size": 256,
        "tokenizer_cache_path": None,
        "tokenizer_train_bytes": None,
        "tokenizer_max_ngram": 8,
        "tokenizer_min_frequency": 2,
    },
    "model": {
        "vocab_size": 256,
        "max_seq_len": 64,
        "num_layers": 4,
        "num_heads": 4,
        "d_model": 256,
        "num_experts": 8,
        "expert_hidden_size": 1024,
        "dropout": 0.0,
        "top_k": 1,
        "capacity_factor": 1.25,
    },
    "routing": {
        "mode": "learned",
        "aux_loss_weight": 0.01,
        "replay_path": None,
        "swap_pairs": [],
        "corruption_fraction": 0.0,
    },
    "training": {
        "steps": 1000,
        "batch_size": 8,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "eval_interval": 100,
        "log_interval": 10,
        "checkpoint_steps": [0, 10, 50, 100, 250, 500, 1000],
        "save_optimizer": True,
        "record_train_routes": False,
        "record_rich_routes": False,
        "data_seed": None,
        "precision": "fp32",
    },
    "pruning": {
        "sparsities": [0.5, 0.7, 0.8, 0.9, 0.95],
        "rewind_fractions": [0.0, 0.01, 0.05, 0.1],
        "rewind_protocol": "full_budget_restart",
    },
    "output_dir": "results/runs/baseline",
}


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    extends = loaded.pop("extends", None)
    base = load_config(source.parent / extends) if extends else DEFAULT_CONFIG
    config = _deep_merge(base, loaded)
    config["model"]["max_seq_len"] = config["data"]["seq_len"]
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
