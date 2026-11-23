from __future__ import annotations

from collections import defaultdict

import torch

from .masks import MaskDict, expert_weight_parameters


def _expert_key(parameter_name: str) -> str:
    prefix, remainder = parameter_name.split(".moe.experts.", maxsplit=1)
    expert_id = remainder.split(".", maxsplit=1)[0]
    return f"{prefix}.moe.experts.{expert_id}"


def expert_local_magnitude_masks(model: torch.nn.Module, sparsity: float) -> MaskDict:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("Sparsity must be in [0, 1).")
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in expert_weight_parameters(model).items():
        grouped[_expert_key(name)].append((name, parameter))

    masks: MaskDict = {}
    for parameters in grouped.values():
        magnitudes = torch.cat([parameter.detach().abs().flatten().cpu() for _, parameter in parameters])
        keep_count = max(1, int(round(magnitudes.numel() * (1.0 - sparsity))))
        keep_indices = magnitudes.topk(keep_count, sorted=False).indices
        flat_mask = torch.zeros(magnitudes.numel(), dtype=torch.bool)
        flat_mask[keep_indices] = True
        offset = 0
        for name, parameter in parameters:
            count = parameter.numel()
            masks[name] = flat_mask[offset : offset + count].reshape_as(parameter)
            offset += count
    return masks


def global_expert_layer_magnitude_masks(model: torch.nn.Module, sparsity: float) -> MaskDict:
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = defaultdict(list)
    for name, parameter in expert_weight_parameters(model).items():
        layer = name.split(".moe.experts.", maxsplit=1)[0]
        grouped[layer].append((name, parameter))

    masks: MaskDict = {}
    for parameters in grouped.values():
        magnitudes = torch.cat([parameter.detach().abs().flatten().cpu() for _, parameter in parameters])
        keep_count = max(1, int(round(magnitudes.numel() * (1.0 - sparsity))))
        keep_indices = magnitudes.topk(keep_count, sorted=False).indices
        flat_mask = torch.zeros(magnitudes.numel(), dtype=torch.bool)
        flat_mask[keep_indices] = True
        offset = 0
        for name, parameter in parameters:
            count = parameter.numel()
            masks[name] = flat_mask[offset : offset + count].reshape_as(parameter)
            offset += count
    return masks

