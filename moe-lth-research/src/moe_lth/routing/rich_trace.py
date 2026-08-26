from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from moe_lth.models.moe_layer import RouteTrace
from moe_lth.routing.route_history import RouteHistory


@dataclass
class RichRouteTrace:
    selected_expert_ids: np.ndarray  # [num_tokens, top_k], uint8/int16
    gate_values: np.ndarray          # [num_tokens, top_k], float16
    accepted_mask: np.ndarray        # [num_tokens, top_k], bool
    batch_indices: np.ndarray        # [num_tokens], int32
    seq_positions: np.ndarray        # [num_tokens], int16
    step: int
    layer_id: int


class RichRouteHistory:
    TRACE_SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.traces: dict[tuple[int, int], RichRouteTrace] = {}
        self.schema_version = self.TRACE_SCHEMA_VERSION
        self.metadata: dict[str, object] = {
            "schema_version": self.schema_version,
            "trace_schema_version": self.schema_version,
            "top_k": 1,
            "num_experts": 0,
            "num_steps": 0,
            "num_layers": 0,
            "compatibility": "legacy_route_history",
        }

    def record(self, step: int, layer_id: int, route_trace: RouteTrace, batch_size: int, seq_len: int) -> None:
        num_tokens = batch_size * seq_len
        top_k = int(route_trace.selected_expert_indices.shape[-1])
        self.metadata["top_k"] = top_k
        self.metadata["num_layers"] = max(int(self.metadata.get("num_layers", 0)), int(layer_id + 1))
        self.metadata["num_steps"] = max(int(self.metadata.get("num_steps", 0)), int(step + 1))

        cpu_indices = route_trace.selected_expert_indices.detach().cpu().reshape(num_tokens, top_k)
        dtype = np.uint8 if int(cpu_indices.max()) <= 255 else np.int16
        selected_expert_ids = cpu_indices.to(dtype=getattr(torch, dtype.__name__)).numpy()

        gate_values = route_trace.selected_probabilities.detach().cpu().reshape(num_tokens, top_k).to(torch.float16).numpy()
        accepted_mask = route_trace.accepted_mask.detach().cpu().reshape(num_tokens, top_k).numpy()

        batch_indices = np.repeat(np.arange(batch_size, dtype=np.int32), seq_len)
        seq_positions = np.tile(np.arange(seq_len, dtype=np.int16), batch_size)

        trace = RichRouteTrace(
            selected_expert_ids=selected_expert_ids,
            gate_values=gate_values,
            accepted_mask=accepted_mask,
            batch_indices=batch_indices,
            seq_positions=seq_positions,
            step=step,
            layer_id=layer_id,
        )
        self.traces[(step, layer_id)] = trace
        max_expert = int(trace.selected_expert_ids.max()) if trace.selected_expert_ids.size else 0
        self.metadata["num_experts"] = max(int(self.metadata.get("num_experts", 0)), max_expert + 1)

    def get(self, step: int, layer_id: int) -> RichRouteTrace:
        return self.traces[(step, layer_id)]

    def get_primary_ids(self, step: int, layer_id: int, device: torch.device) -> torch.Tensor:
        trace = self.get(step, layer_id)
        primary = trace.selected_expert_ids[:, 0]
        return torch.from_numpy(primary.astype(np.int64)).to(device)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for (step, layer), trace in self.traces.items():
            prefix = f"trace_{step}_{layer}_"
            arrays[prefix + "selected_expert_ids"] = trace.selected_expert_ids
            arrays[prefix + "gate_values"] = trace.gate_values
            arrays[prefix + "accepted_mask"] = trace.accepted_mask
            arrays[prefix + "batch_indices"] = trace.batch_indices
            arrays[prefix + "seq_positions"] = trace.seq_positions

        metadata_payload = {
            "schema_version": self.schema_version,
            "trace_schema_version": self.schema_version,
            "metadata": self.metadata,
        }
        arrays["metadata"] = np.array(json.dumps(metadata_payload), dtype=np.str_)

        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> RichRouteHistory:
        history = cls()
        with np.load(path, allow_pickle=True) as archive:
            metadata = archive.get("metadata")
            meta_dict: dict[str, object] = {}
            if metadata is not None:
                payload = metadata.item() if hasattr(metadata, "item") else metadata
                if isinstance(payload, str):
                    meta_dict = json.loads(payload)
                elif isinstance(payload, dict):
                    meta_dict = payload
            schema_version = int(meta_dict.get("schema_version", meta_dict.get("trace_schema_version", 1)))
            history.schema_version = schema_version
            history.metadata.update(meta_dict.get("metadata", meta_dict))
            history.metadata["schema_version"] = schema_version
            history.metadata["trace_schema_version"] = schema_version

            trace_keys = set()
            for name in archive.files:
                if name.startswith("trace_"):
                    parts = name.split("_")
                    step = int(parts[1])
                    layer = int(parts[2])
                    trace_keys.add((step, layer))

            for step, layer in trace_keys:
                prefix = f"trace_{step}_{layer}_"
                trace = RichRouteTrace(
                    selected_expert_ids=archive[prefix + "selected_expert_ids"],
                    gate_values=archive[prefix + "gate_values"],
                    accepted_mask=archive[prefix + "accepted_mask"],
                    batch_indices=archive[prefix + "batch_indices"],
                    seq_positions=archive[prefix + "seq_positions"],
                    step=step,
                    layer_id=layer,
                )
                history.traces[(step, layer)] = trace

        if history.traces:
            top_k = int(next(iter(history.traces.values())).selected_expert_ids.shape[1])
            history.metadata["top_k"] = top_k
            history.metadata["num_steps"] = max(step for step, _ in history.traces) + 1
            history.metadata["num_layers"] = max(layer for _, layer in history.traces) + 1

        return history

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        metadata_blob = json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h.update(metadata_blob)
        for step, layer in sorted(self.traces.keys()):
            trace = self.traces[(step, layer)]
            h.update(f"{step}_{layer}".encode("utf-8"))
            h.update(trace.selected_expert_ids.tobytes())
            h.update(trace.gate_values.tobytes())
            h.update(trace.accepted_mask.tobytes())
            h.update(trace.batch_indices.tobytes())
            h.update(trace.seq_positions.tobytes())
        return h.hexdigest()

    def verify_integrity(self) -> None:
        for (step, layer), trace in self.traces.items():
            assert trace.step == step
            assert trace.layer_id == layer
            num_tokens = trace.selected_expert_ids.shape[0]
            top_k = trace.selected_expert_ids.shape[1]
            assert trace.gate_values.shape == (num_tokens, top_k)
            assert trace.accepted_mask.shape == (num_tokens, top_k)
            assert trace.batch_indices.shape == (num_tokens,)
            assert trace.seq_positions.shape == (num_tokens,)
            assert trace.batch_indices.dtype == np.int32
            assert trace.seq_positions.dtype == np.int16
            assert np.array_equal(trace.batch_indices, np.repeat(np.arange(num_tokens // trace.seq_positions.size, dtype=np.int32), trace.seq_positions.size)) or trace.batch_indices.size == num_tokens
            assert np.array_equal(trace.seq_positions, np.tile(np.arange(trace.seq_positions.size, dtype=np.int16), num_tokens // trace.seq_positions.size)) or trace.seq_positions.size == num_tokens

            assert trace.selected_expert_ids.dtype in (np.uint8, np.int16)
            assert trace.gate_values.dtype == np.float16
            assert trace.accepted_mask.dtype == bool
            assert trace.selected_expert_ids.shape[0] == trace.batch_indices.shape[0]
            assert trace.selected_expert_ids.shape[0] == trace.seq_positions.shape[0]

            if self.metadata.get("schema_version") is not None:
                assert int(self.metadata.get("schema_version", 0)) >= 1


def upgrade_legacy_history(old: RouteHistory) -> RichRouteHistory:
    rich = RichRouteHistory()
    for (step, layer_id), routes in old.routes.items():
        batch_size = routes.shape[0]
        seq_len = routes.shape[1] if len(routes.shape) > 1 else 1
        num_tokens = batch_size * seq_len
        top_k = 1

        flat_routes = routes.reshape(num_tokens, top_k)
        gate_values = np.full((num_tokens, top_k), np.nan, dtype=np.float16)
        accepted_mask = np.full((num_tokens, top_k), True, dtype=bool)
        batch_indices = np.repeat(np.arange(batch_size, dtype=np.int32), seq_len)
        seq_positions = np.tile(np.arange(seq_len, dtype=np.int16), batch_size)

        trace = RichRouteTrace(
            selected_expert_ids=flat_routes,
            gate_values=gate_values,
            accepted_mask=accepted_mask,
            batch_indices=batch_indices,
            seq_positions=seq_positions,
            step=step,
            layer_id=layer_id,
        )
        rich.traces[(step, layer_id)] = trace

    rich.schema_version = RichRouteHistory.TRACE_SCHEMA_VERSION
    rich.metadata.update({
        "schema_version": rich.schema_version,
        "trace_schema_version": rich.schema_version,
        "compatibility": "legacy_route_history_upgrade",
        "top_k": 1,
    })
    return rich
