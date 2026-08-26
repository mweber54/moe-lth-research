from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class RouteHistory:
    routes: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)

    def record(self, step: int, layer_id: int, selected_experts: torch.Tensor) -> None:
        cpu_routes = selected_experts.detach().to(device="cpu")
        dtype = torch.uint8 if int(cpu_routes.max()) <= 255 else torch.int16
        self.routes[(step, layer_id)] = cpu_routes.to(dtype=dtype).numpy()

    def get(self, step: int, layer_id: int, device: torch.device) -> torch.Tensor:
        key = (step, layer_id)
        if key not in self.routes:
            raise KeyError(f"No replay route for step={step}, layer={layer_id}.")
        return torch.from_numpy(self.routes[key].astype(np.int64)).to(device)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"step_{step}_layer_{layer}": values for (step, layer), values in self.routes.items()}
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "RouteHistory":
        history = cls()
        with np.load(path, allow_pickle=True) as archive:
            rich_trace_names = [
                name for name in archive.files
                if name.startswith("trace_") and name.endswith("selected_expert_ids")
            ]
            if rich_trace_names:
                for name in rich_trace_names:
                    parts = name.split("_")
                    if len(parts) < 4:
                        continue
                    step = int(parts[1])
                    layer = int(parts[2])
                    selected = np.asarray(archive[name])
                    primary_ids = selected[:, 0] if selected.ndim > 1 else selected
                    batch_key = f"trace_{step}_{layer}_batch_indices"
                    seq_key = f"trace_{step}_{layer}_seq_positions"
                    if batch_key in archive and seq_key in archive:
                        batch_indices = np.asarray(archive[batch_key]).astype(np.int64)
                        seq_positions = np.asarray(archive[seq_key]).astype(np.int64)
                        batch_size = int(batch_indices.max()) + 1 if batch_indices.size else 1
                        seq_len = int(seq_positions.max()) + 1 if seq_positions.size else 1
                        reconstructed = np.full((batch_size, seq_len), -1, dtype=np.int64)
                        reconstructed[batch_indices, seq_positions] = np.asarray(primary_ids).astype(np.int64)
                        history.routes[(step, layer)] = reconstructed
                    else:
                        history.routes[(step, layer)] = np.asarray(primary_ids).astype(np.int64)
                if history.routes:
                    return history

            for name in archive.files:
                if name.startswith("metadata") or name.startswith("trace_"):
                    continue
                parts = name.split("_")
                if len(parts) >= 4 and parts[0] == "step":
                    step = int(parts[1])
                    layer = int(parts[3])
                    history.routes[(step, layer)] = archive[name]
        return history


def save_validation_routes(
    path: str | Path,
    checkpoint_step: int,
    batches: list[list[torch.Tensor]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for batch_id, layer_routes in enumerate(batches):
        for layer_id, routes in enumerate(layer_routes):
            arrays[f"checkpoint_{checkpoint_step}_batch_{batch_id}_layer_{layer_id}"] = (
                routes.detach().cpu().to(torch.uint8).numpy()
            )
    np.savez_compressed(destination, **arrays)


def load_validation_route_batches(path: str | Path, device: torch.device) -> list[list[torch.Tensor]]:
    grouped: dict[int, dict[int, torch.Tensor]] = {}
    with np.load(path) as archive:
        for name in archive.files:
            parts = name.split("_")
            batch_id = int(parts[3])
            layer_id = int(parts[5])
            grouped.setdefault(batch_id, {})[layer_id] = torch.from_numpy(
                archive[name].astype(np.int64)
            ).to(device)
    return [
        [layers[layer_id] for layer_id in sorted(layers)]
        for _, layers in sorted(grouped.items())
    ]
