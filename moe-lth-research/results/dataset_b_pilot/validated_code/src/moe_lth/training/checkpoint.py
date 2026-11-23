from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    validation_loss: float | None,
    config: dict,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "validation_loss": validation_loss,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    # Router temperature became persistent when the router-confidence control
    # was added.  Older checkpoints legitimately omit only these buffers and
    # should load at their native default temperature of 1.0.
    incompatible = model.load_state_dict(payload["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        name for name in incompatible.missing_keys if not name.endswith(".moe.router.temperature")
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
