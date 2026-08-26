"""Utilities for the router-age recovery experiment.

These helpers let a fixed, once-pruned expert/shared parameter state be paired
with router parameters taken from a *different* point in the same reference
training trajectory, so that router age can be varied while expert maturity
and the pruning mask are held byte-identical across conditions.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import torch

from moe_lth.models import TinyMoELanguageModel
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
    final_checkpoint: str,
    masks: MaskDict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load the final (T) checkpoint and apply the pruning mask once.

    Returns the full pruned state dict (shared + expert weights pruned,
    router weights present but to be overwritten per router-age condition).
    """
    model = load_model_from_checkpoint(model_config, final_checkpoint, device)
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
    for block in model.blocks:
        block.moe.router.temperature.fill_(float(temperature))


@torch.no_grad()
def mean_selected_probability(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_probability = 0.0
    total_entropy = 0.0
    total_margin = 0.0
    total_count = 0
    for batch_id, token_ids in enumerate(batches):
        if max_batches is not None and batch_id >= max_batches:
            break
        token_ids = token_ids.to(device)
        output = model(token_ids)
        for trace in output.routes:
            total_probability += float(trace.selected_probability.sum().cpu())
            total_entropy += float(trace.entropy.sum().cpu())
            total_margin += float(trace.margin.sum().cpu())
            total_count += trace.selected_probability.numel()
    return {
        "mean_selected_probability": total_probability / max(1, total_count),
        "routing_entropy": total_entropy / max(1, total_count),
        "top1_top2_margin": total_margin / max(1, total_count),
    }


@torch.no_grad()
def selected_experts_per_batch(
    model: torch.nn.Module,
    batches: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    model.eval()
    all_selected: list[torch.Tensor] = []
    for token_ids in batches:
        token_ids = token_ids.to(device)
        output = model(token_ids)
        # concat across layers to a single per-token vector: (layers, batch, seq)
        stacked = torch.stack([trace.selected_experts for trace in output.routes], dim=0)
        all_selected.append(stacked.detach().cpu())
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
    reference_selected: list[torch.Tensor],
    low: float = 0.05,
    high: float = 20.0,
    iterations: int = 25,
) -> tuple[float, float, float]:
    """Binary-search a positive logit temperature so mean selected probability
    approximately matches `target_confidence` on the calibration batches.

    Returns (temperature, achieved_mean_selected_probability, assignment_agreement_vs_native).
    Assignment agreement must be (numerically) 1.0 because dividing logits by a
    positive scalar never changes the argmax/top-k ordering.
    """
    set_router_temperature(model, 1.0)
    native_stats = mean_selected_probability(model, calibration_batches, device)
    if native_stats["mean_selected_probability"] <= target_confidence:
        # Native confidence already at/under target: temperature=1.0 (no sharpening needed
        # beyond native, since we only calibrate with temperature >= a small positive lower bound).
        pass

    def probability_at(temperature: float) -> float:
        set_router_temperature(model, temperature)
        return mean_selected_probability(model, calibration_batches, device)["mean_selected_probability"]

    # Probability is monotonically non-increasing in temperature (temperature > 0 flattens
    # the distribution), so binary search for the temperature hitting the target confidence.
    lo, hi = low, high
    prob_lo = probability_at(lo)
    prob_hi = probability_at(hi)
    if target_confidence >= prob_lo:
        best_temperature = lo
    elif target_confidence <= prob_hi:
        best_temperature = hi
    else:
        for _ in range(iterations):
            mid = (lo + hi) / 2.0
            prob_mid = probability_at(mid)
            if prob_mid > target_confidence:
                lo = mid
            else:
                hi = mid
        best_temperature = (lo + hi) / 2.0

    set_router_temperature(model, best_temperature)
    achieved = mean_selected_probability(model, calibration_batches, device)["mean_selected_probability"]
    candidate_selected = selected_experts_per_batch(model, calibration_batches, device)
    agreement = assignment_agreement(reference_selected, candidate_selected)
    return best_temperature, achieved, agreement


def grad_norms_by_group(model: torch.nn.Module) -> dict[str, float]:
    sums = {"expert": 0.0, "router": 0.0, "shared": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = parameter_group(name)
        sums[group] += float(parameter.grad.detach().float().pow(2).sum().cpu())
    return {group: value ** 0.5 for group, value in sums.items()}


def per_expert_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    for block_id, block in enumerate(model.blocks):
        for expert_id, expert in enumerate(block.moe.experts):
            total = 0.0
            for parameter in expert.parameters():
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().float().pow(2).sum().cpu())
            norms[f"layer_{block_id}_expert_{expert_id}"] = total ** 0.5
    return norms
