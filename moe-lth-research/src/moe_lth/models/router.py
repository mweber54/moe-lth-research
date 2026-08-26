from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RouteOverride:
    expert_ids: torch.Tensor
    gate_values: torch.Tensor | None = None


@dataclass
class RouterOutput:
    selected_experts: torch.Tensor
    probabilities: torch.Tensor
    selected_probability: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 1):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.projection = nn.Linear(d_model, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        # Positive logit-temperature for confidence calibration (router-age experiment);
        # dividing by a positive scalar never changes the top-1/top-k assignment.
        self.register_buffer("temperature", torch.tensor(1.0), persistent=False)

    def forward(
        self,
        inputs: torch.Tensor,
        override_ids: torch.Tensor | RouteOverride | None = None,
        override_gate_values: torch.Tensor | None = None,
    ) -> RouterOutput:
        if isinstance(override_ids, RouteOverride):
            override_gate_values = override_ids.gate_values if override_gate_values is None else override_gate_values
            override_ids = override_ids.expert_ids

        logits = self.projection(inputs) / self.temperature
        probabilities = torch.softmax(logits, dim=-1)
        if override_ids is None:
            selected = probabilities.topk(k=self.top_k, dim=-1).indices
            selected_probability = probabilities.gather(-1, selected)
        elif override_gate_values is not None:
            if override_ids.ndim == 1:
                selected = override_ids.unsqueeze(-1)
            else:
                selected = override_ids
            if selected.shape[-1] != self.top_k:
                raise ValueError(
                    f"Override shape {tuple(selected.shape)} is incompatible with top_k={self.top_k}."
                )
            selected_probability = override_gate_values.reshape_as(selected).to(dtype=probabilities.dtype)
        elif override_ids.ndim == 1:
            primary = override_ids.unsqueeze(-1)
            if self.top_k == 1:
                selected = primary
            else:
                remaining = probabilities.clone()
                remaining.scatter_(-1, primary, -1.0)
                additional = remaining.topk(k=self.top_k - 1, dim=-1).indices
                selected = torch.cat((primary, additional), dim=-1)
            selected_probability = probabilities.gather(-1, selected)
        elif override_ids.shape[-1] == self.top_k:
            selected = override_ids
            selected_probability = probabilities.gather(-1, selected)
        else:
            raise ValueError(
                f"Override shape {tuple(override_ids.shape)} is incompatible with top_k={self.top_k}."
            )
        entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
        top_two = probabilities.topk(k=min(2, self.num_experts), dim=-1).values
        margin = top_two[..., 0] - top_two[..., 1] if self.num_experts > 1 else top_two[..., 0]
        return RouterOutput(selected, probabilities, selected_probability, entropy, margin)


class Top1Router(TopKRouter):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__(d_model, num_experts, top_k=1)
