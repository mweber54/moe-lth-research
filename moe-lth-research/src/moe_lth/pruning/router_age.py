"""Utilities for the router-age recovery experiment.

These helpers let a fixed, once-pruned expert/shared parameter state be paired
with router parameters taken from a *different* point in the same reference
training trajectory, so that router age can be varied while expert maturity
and the pruning mask are held byte-identical across conditions.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from pathlib import Path

import torch

from moe_lth.models import TinyMoELanguageModel
from moe_lth.models.router import RouteOverride
from moe_lth.training.checkpoint import load_checkpoint

from .masks import MaskDict, apply_masks_


def parameter_group(name: str) -> str:
    """Classify a parameter name as 'expert', 'router', or 'shared'."""
    if ".moe.experts." in name:
        return "expert"
    if ".moe.router." in name:
        return "router"
    return "shared"


def router_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if parameter_group(name) == "router"]


def component_state_dict(model: torch.nn.Module, group: str) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if parameter_group(name) == group
    }


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_model_from_checkpoint(model_config: dict, checkpoint_path: str, device: torch.device) -> TinyMoELanguageModel:
    model = TinyMoELanguageModel(deepcopy(model_config)).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    return model


def build_fixed_pruned_base(
    model_config: dict,
    rewind_checkpoint: str,
    masks: MaskDict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build the fixed LTH ticket state.

    The expert pruning mask is derived from the fully trained experts (E_T), but
    the surviving values are taken from the original initialization (E_0). This
    produces the fixed sparse ticket m_80 ⊙ E_0 while preserving the same shared
    parameters and router checkpoint across age conditions.
    """
    model = load_model_from_checkpoint(model_config, rewind_checkpoint, device)
    apply_masks_(model, masks)
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def assemble_router_age_model(
    model_config: dict,
    pruned_base_state: dict[str, torch.Tensor],
    router_checkpoint: str,
    masks: MaskDict,
    device: torch.device,
) -> TinyMoELanguageModel:
    """Build a model with fixed shared+pruned-expert state and a swapped-in router."""
    model = TinyMoELanguageModel(deepcopy(model_config)).to(device)
    model.load_state_dict(pruned_base_state)

    router_source = TinyMoELanguageModel(deepcopy(model_config))
    load_checkpoint(router_checkpoint, router_source, map_location="cpu")
    router_state = {
        name: tensor
        for name, tensor in router_source.state_dict().items()
        if parameter_group(name) == "router"
    }
    current_state = model.state_dict()
    current_state.update({name: tensor.to(device) for name, tensor in router_state.items()})
    model.load_state_dict(current_state)

    # Re-apply the mask defensively: router swap must never touch expert weights.
    apply_masks_(model, masks)
    return model


def set_router_temperature(model: torch.nn.Module, temperature: float) -> None:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("Router temperature must be positive and finite.")
    for block in model.blocks:
        block.moe.router.temperature.fill_(float(temperature))


def forward_with_preserved_routing(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    temperature: float,
) -> tuple[object, dict[str, float]]:
    """Apply calibrated gate amplitudes while preserving native routing.

    A positive temperature preserves an argmax only for a fixed router input.
    In a multi-layer MoE, changing an early gate amplitude changes later hidden
    states and can therefore change later assignments.  The confidence control
    first obtains the full native-confidence route, then performs the gradient-
    bearing forward with those expert IDs and native capacity-priority scores
    forced at every layer.  Gate values still come from the temperature-scaled
    router, so router gradients and MoE branch amplitudes remain calibrated.
    """
    set_router_temperature(model, 1.0)
    with torch.no_grad():
        native = model(token_ids)
    overrides = [
        RouteOverride(
            expert_ids=trace.selected_expert_indices.detach(),
            capacity_scores=trace.selected_probabilities.detach(),
        )
        for trace in native.routes
    ]
    set_router_temperature(model, temperature)
    controlled = model(token_ids, overrides)

    assignment_total = 0
    assignment_equal = 0
    capacity_total = 0
    capacity_equal = 0
    for native_trace, controlled_trace in zip(native.routes, controlled.routes):
        assignment_total += native_trace.selected_expert_indices.numel()
        assignment_equal += int(
            (native_trace.selected_expert_indices == controlled_trace.selected_expert_indices)
            .sum()
            .item()
        )
        capacity_total += native_trace.accepted_mask.numel()
        capacity_equal += int((native_trace.accepted_mask == controlled_trace.accepted_mask).sum().item())
    return controlled, {
        "assignment_agreement_before_after": assignment_equal / max(1, assignment_total),
        "capacity_agreement_before_after": capacity_equal / max(1, capacity_total),
    }


@torch.no_grad()
def routing_statistics(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
    max_batches: int | None = None,
    *,
    confidence_temperature: float | None = None,
    preserve_routing: bool = False,
) -> dict:
    was_training = model.training
    model.eval()
    total_probability = 0.0
    total_entropy = 0.0
    total_margin = 0.0
    total_count = 0
    total_logit_norm = 0.0
    total_logit_count = 0
    agreement_weighted = 0.0
    capacity_agreement_weighted = 0.0
    agreement_count = 0
    num_layers = len(model.blocks)
    num_experts = model.blocks[0].moe.num_experts
    assignment_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    top1_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    accepted_counts = torch.zeros(num_layers, num_experts, dtype=torch.long)
    dropped_sum = torch.zeros(num_layers, dtype=torch.float64)
    dropped_batches = torch.zeros(num_layers, dtype=torch.long)
    for batch_id, token_ids in enumerate(batches):
        if max_batches is not None and batch_id >= max_batches:
            break
        token_ids = token_ids.to(device)
        if preserve_routing:
            if confidence_temperature is None:
                raise ValueError("confidence_temperature is required when preserve_routing=True")
            output, integrity = forward_with_preserved_routing(
                model, token_ids, confidence_temperature
            )
            batch_assignments = sum(trace.selected_expert_indices.numel() for trace in output.routes)
            agreement_weighted += integrity["assignment_agreement_before_after"] * batch_assignments
            capacity_agreement_weighted += integrity["capacity_agreement_before_after"] * batch_assignments
            agreement_count += batch_assignments
        else:
            if confidence_temperature is not None:
                set_router_temperature(model, confidence_temperature)
            output = model(token_ids)
        for layer_id, trace in enumerate(output.routes):
            total_probability += float(trace.selected_probability.sum().cpu())
            total_entropy += float(trace.entropy.sum().cpu())
            total_margin += float(trace.margin.sum().cpu())
            total_count += trace.selected_probability.numel()
            assignment_counts[layer_id] += torch.bincount(
                trace.selected_expert_indices.detach().cpu().flatten(), minlength=num_experts
            )
            top1_counts[layer_id] += torch.bincount(
                trace.selected_experts.detach().cpu().flatten(), minlength=num_experts
            )
            accepted_ids = trace.selected_expert_indices.reshape(-1)[trace.accepted_mask.reshape(-1)]
            accepted_counts[layer_id] += torch.bincount(
                accepted_ids.detach().cpu(), minlength=num_experts
            )
            dropped_sum[layer_id] += float(trace.dropped_fraction.detach().cpu())
            dropped_batches[layer_id] += 1

            hidden = output.pre_router_hidden_states[layer_id].reshape(
                -1, output.pre_router_hidden_states[layer_id].shape[-1]
            )
            logits = model.blocks[layer_id].moe.router.projection(hidden)
            logits = logits / model.blocks[layer_id].moe.router.temperature
            total_logit_norm += float(logits.float().norm(dim=-1).sum().cpu())
            total_logit_count += logits.shape[0]

    aggregate_assignments = assignment_counts.sum(dim=0)
    aggregate_top1 = top1_counts.sum(dim=0)
    aggregate_accepted = accepted_counts.sum(dim=0)

    def distributions(counts: torch.Tensor) -> list[list[float]]:
        denominator = counts.sum(dim=-1, keepdim=True).clamp_min(1)
        return (counts.float() / denominator).tolist()

    result = {
        "mean_selected_probability": total_probability / max(1, total_count),
        "routing_entropy": total_entropy / max(1, total_count),
        "top1_top2_margin": total_margin / max(1, total_count),
        "router_logit_norm": total_logit_norm / max(1, total_logit_count),
        "expert_assignment_counts_by_layer": assignment_counts.tolist(),
        "expert_utilization_by_layer": distributions(assignment_counts),
        "top1_assignment_counts_by_layer": top1_counts.tolist(),
        "top1_assignment_distribution_by_layer": distributions(top1_counts),
        "accepted_expert_token_counts_by_layer": accepted_counts.tolist(),
        "accepted_expert_distribution_by_layer": distributions(accepted_counts),
        "expert_assignment_counts": aggregate_assignments.tolist(),
        "expert_utilization": (
            aggregate_assignments.float() / aggregate_assignments.sum().clamp_min(1)
        ).tolist(),
        "top1_assignment_counts": aggregate_top1.tolist(),
        "top1_assignment_distribution": (
            aggregate_top1.float() / aggregate_top1.sum().clamp_min(1)
        ).tolist(),
        "accepted_expert_token_counts": aggregate_accepted.tolist(),
        "accepted_expert_distribution": (
            aggregate_accepted.float() / aggregate_accepted.sum().clamp_min(1)
        ).tolist(),
        "dropped_fraction_by_layer": (
            dropped_sum / dropped_batches.clamp_min(1)
        ).tolist(),
    }
    if preserve_routing:
        result["assignment_agreement_before_after"] = agreement_weighted / max(1, agreement_count)
        result["capacity_agreement_before_after"] = capacity_agreement_weighted / max(1, agreement_count)
    model.train(was_training)
    return result


@torch.no_grad()
def mean_selected_probability(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    return routing_statistics(model, batches, device, max_batches)


@torch.no_grad()
def selected_experts_per_batch(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
    *,
    confidence_temperature: float | None = None,
    preserve_routing: bool = False,
) -> list[torch.Tensor]:
    was_training = model.training
    model.eval()
    all_selected: list[torch.Tensor] = []
    for token_ids in batches:
        token_ids = token_ids.to(device)
        if preserve_routing:
            if confidence_temperature is None:
                raise ValueError("confidence_temperature is required when preserve_routing=True")
            output, _ = forward_with_preserved_routing(model, token_ids, confidence_temperature)
        else:
            if confidence_temperature is not None:
                set_router_temperature(model, confidence_temperature)
            output = model(token_ids)
        # concat across layers to a single per-token vector: (layers, batch, seq)
        stacked = torch.stack([trace.selected_experts for trace in output.routes], dim=0)
        all_selected.append(stacked.detach().cpu())
    model.train(was_training)
    return all_selected


def assignment_agreement(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    total = 0
    agree = 0
    for ref_batch, cand_batch in zip(reference, candidate):
        total += ref_batch.numel()
        agree += int((ref_batch == cand_batch).sum().item())
    return agree / max(1, total)


def calibrate_temperature(
    model: torch.nn.Module,
    calibration_batches: list[torch.Tensor],
    device: torch.device,
    target_confidence: float,
    max_temperature: float = 1000.0,
    grid_points: int = 25,
    refinement_rounds: int = 3,
) -> tuple[float, float, float, float]:
    """Match selected confidence while preserving routes and capacity decisions.

    The controlled multi-layer forward is not guaranteed to be monotone in a
    shared temperature, so calibration uses a deterministic log-grid search
    with local refinement rather than assuming binary-search monotonicity.

    Returns (temperature, achieved confidence, assignment agreement, capacity
    agreement), where both agreements compare the same router immediately
    before and after calibration.
    """
    if not 0.0 < float(target_confidence) <= 1.0:
        raise ValueError("target_confidence must be in (0, 1].")
    was_training = model.training
    model.eval()
    cache: dict[float, dict] = {}

    def stats_at(temperature: float) -> dict:
        key = float(temperature)
        if key not in cache:
            cache[key] = routing_statistics(
                model,
                calibration_batches,
                device,
                confidence_temperature=key,
                preserve_routing=True,
            )
        return cache[key]

    log_low = 0.0
    log_high = math.log(float(max_temperature))
    best_temperature = 1.0
    best_error = float("inf")
    for _ in range(refinement_rounds + 1):
        candidates = torch.linspace(log_low, log_high, steps=grid_points).exp().tolist()
        scored = []
        for temperature in candidates:
            error = abs(stats_at(temperature)["mean_selected_probability"] - target_confidence)
            scored.append((error, temperature))
            if error < best_error:
                best_error = error
                best_temperature = temperature
        best_index = min(range(len(scored)), key=lambda index: scored[index][0])
        lower_index = max(0, best_index - 1)
        upper_index = min(len(candidates) - 1, best_index + 1)
        log_low = math.log(candidates[lower_index])
        log_high = math.log(candidates[upper_index])
        if log_high - log_low < 1e-8:
            break

    best = stats_at(best_temperature)
    set_router_temperature(model, best_temperature)
    model.train(was_training)
    return (
        best_temperature,
        best["mean_selected_probability"],
        best["assignment_agreement_before_after"],
        best["capacity_agreement_before_after"],
    )


def grad_norms_by_group(model: torch.nn.Module) -> dict[str, float]:
    device = next(model.parameters()).device
    grouped: dict[str, list[torch.Tensor]] = {"expert": [], "router": [], "shared": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grouped[parameter_group(name)].append(parameter.grad.detach())
    totals = []
    for group in ("expert", "router", "shared"):
        gradients = grouped[group]
        if gradients:
            individual = torch._foreach_norm(gradients, 2.0)
            totals.append(torch.linalg.vector_norm(torch.stack([value.float() for value in individual])))
        else:
            totals.append(torch.tensor(0.0, device=device))
    values = torch.stack(totals).detach().cpu().tolist()
    return dict(zip(("expert", "router", "shared"), values))


def per_expert_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    device = next(model.parameters()).device
    keys: list[str] = []
    norm_tensors: list[torch.Tensor] = []
    for block_id, block in enumerate(model.blocks):
        for expert_id, expert in enumerate(block.moe.experts):
            gradients = [
                parameter.grad.detach()
                for parameter in expert.parameters()
                if parameter.grad is not None
            ]
            if gradients:
                individual = torch._foreach_norm(gradients, 2.0)
                total = torch.linalg.vector_norm(torch.stack([value.float() for value in individual]))
            else:
                total = torch.tensor(0.0, device=device)
            keys.append(f"layer_{block_id}_expert_{expert_id}")
            norm_tensors.append(total)
    values = torch.stack(norm_tensors).detach().cpu().tolist()
    return dict(zip(keys, values))
