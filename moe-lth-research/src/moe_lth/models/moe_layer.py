from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .experts import ExpertFFN
from .router import RouteOverride, TopKRouter


@dataclass
class RouteTrace:
    selected_experts: torch.Tensor
    selected_probability: torch.Tensor
    selected_expert_indices: torch.Tensor
    selected_probabilities: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor
    usage: torch.Tensor
    dropped_fraction: torch.Tensor
    accepted_mask: torch.Tensor


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        dropout: float,
    ):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList(
            ExpertFFN(d_model, hidden_size, dropout) for _ in range(num_experts)
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def forward(
        self,
        inputs: torch.Tensor,
        override_ids: torch.Tensor | RouteOverride | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, RouteTrace]:
        batch_size, seq_len, d_model = inputs.shape
        flat_inputs = inputs.reshape(-1, d_model)
        flat_override = None
        flat_override_gate = None
        flat_capacity_scores = None
        if isinstance(override_ids, RouteOverride):
            flat_override = override_ids.expert_ids.reshape(-1, self.top_k)
            flat_override_gate = override_ids.gate_values.reshape(-1, self.top_k) if override_ids.gate_values is not None else None
            flat_capacity_scores = (
                override_ids.capacity_scores.reshape(-1, self.top_k)
                if override_ids.capacity_scores is not None
                else None
            )
        elif override_ids is not None:
            flat_override = override_ids.reshape(-1)
        router_output = self.router(flat_inputs, flat_override, flat_override_gate)
        selected = router_output.selected_experts
        selected_probability = router_output.selected_probability
        output = torch.zeros_like(flat_inputs)
        capacity = max(
            1,
            math.ceil(
                self.capacity_factor
                * flat_inputs.shape[0]
                * self.top_k
                / self.num_experts
            ),
        )
        accepted = torch.zeros_like(selected, dtype=torch.bool)

        for expert_id, expert in enumerate(self.experts):
            assignments = torch.nonzero(selected == expert_id, as_tuple=False)
            if assignments.numel() == 0:
                continue
            if assignments.shape[0] > capacity:
                priority = selected_probability if flat_capacity_scores is None else flat_capacity_scores
                scores = priority[assignments[:, 0], assignments[:, 1]]
                assignments = assignments[scores.topk(capacity).indices]
            token_positions = assignments[:, 0]
            route_slots = assignments[:, 1]
            accepted[token_positions, route_slots] = True
            expert_output = expert(flat_inputs[token_positions])
            weights = selected_probability[token_positions, route_slots, None]
            output = output.index_add(0, token_positions, expert_output * weights)

        usage = torch.bincount(selected.flatten(), minlength=self.num_experts).float()
        usage = usage / max(1, selected.numel())
        mean_probability = router_output.probabilities.mean(dim=0)
        auxiliary_loss = self.num_experts * torch.sum(usage.detach() * mean_probability)
        trace = RouteTrace(
            selected_experts=selected[:, 0].reshape(batch_size, seq_len),
            selected_probability=selected_probability[:, 0].reshape(batch_size, seq_len),
            selected_expert_indices=selected.reshape(batch_size, seq_len, self.top_k),
            selected_probabilities=selected_probability.reshape(batch_size, seq_len, self.top_k),
            entropy=router_output.entropy.reshape(batch_size, seq_len),
            margin=router_output.margin.reshape(batch_size, seq_len),
            usage=usage,
            dropped_fraction=1.0 - accepted.float().mean(),
            accepted_mask=accepted,
        )
        return output.reshape(batch_size, seq_len, d_model), auxiliary_loss, trace
