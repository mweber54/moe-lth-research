from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from moe_lth.routing.interventions import RoutingController


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    controller: RoutingController | None = None,
    override_batches: list[list[torch.Tensor]] | None = None,
    max_batches: int | None = None,
    route_step_offset: int = 0,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    expert_loss_sum: dict[tuple[int, int], float] = defaultdict(float)
    expert_token_count: dict[tuple[int, int], int] = defaultdict(int)
    routing_batches: list[list[torch.Tensor]] = []

    for batch_id, (token_ids, targets) in enumerate(data_loader):
        if max_batches is not None and batch_id >= max_batches:
            break
        token_ids, targets = token_ids.to(device), targets.to(device)
        if override_batches is not None:
            overrides = override_batches[batch_id]
        else:
            overrides = (
                None
                if controller is None
                else controller.overrides(token_ids, route_step_offset + batch_id)
            )
        output = model(token_ids, overrides)
        token_losses = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        total_loss += float(token_losses.sum().cpu())
        total_tokens += targets.numel()
        routing_batches.append([trace.selected_experts.detach().cpu() for trace in output.routes])

        for layer_id, trace in enumerate(output.routes):
            for expert_id in range(trace.usage.numel()):
                mask = (trace.selected_expert_indices == expert_id).any(dim=-1)
                count = int(mask.sum().cpu())
                if count:
                    expert_loss_sum[(layer_id, expert_id)] += float(token_losses[mask].sum().cpu())
                    expert_token_count[(layer_id, expert_id)] += count

    mean_loss = total_loss / max(1, total_tokens)
    expert_local_loss = {
        f"layer_{layer}_expert_{expert}": expert_loss_sum[(layer, expert)]
        / expert_token_count[(layer, expert)]
        for layer, expert in expert_token_count
    }
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "expert_local_loss": expert_local_loss,
        "routing_batches": routing_batches,
    }


@torch.no_grad()
def evaluate_expert_substitution_matrix(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    max_batches: int = 1,
) -> dict[str, list[list[float]]]:
    """Measure loss on D_source after replacing its expert with target expert."""
    model.eval()
    num_layers = len(model.blocks)
    num_experts = model.blocks[0].moe.num_experts
    sums = torch.zeros(num_layers, num_experts, num_experts, dtype=torch.float64)
    counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    for batch_id, (token_ids, targets) in enumerate(data_loader):
        if batch_id >= max_batches:
            break
        token_ids, targets = token_ids.to(device), targets.to(device)
        baseline = model(token_ids)
        original_routes = [trace.selected_experts.clone() for trace in baseline.routes]
        for layer_id in range(num_layers):
            for source_expert in range(num_experts):
                source_mask = original_routes[layer_id] == source_expert
                source_count = int(source_mask.sum().cpu())
                if not source_count:
                    continue
                counts[layer_id, source_expert] += source_count
                for target_expert in range(num_experts):
                    overrides = [routes.clone() for routes in original_routes]
                    overrides[layer_id][source_mask] = target_expert
                    output = model(token_ids, overrides)
                    token_losses = F.cross_entropy(
                        output.logits.reshape(-1, output.logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="none",
                    ).reshape_as(targets)
                    sums[layer_id, source_expert, target_expert] += float(token_losses[source_mask].sum().cpu())
    matrices = {}
    for layer_id in range(num_layers):
        denominator = counts[layer_id].clamp_min(1).to(torch.float64)[:, None]
        matrices[f"layer_{layer_id}"] = (sums[layer_id] / denominator).tolist()
    return matrices
