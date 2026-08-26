from __future__ import annotations

from pathlib import Path

import torch


MaskDict = dict[str, torch.Tensor]


def expert_weight_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".moe.experts." in name and name.endswith(".weight")
    }


@torch.no_grad()
def apply_masks_(model: torch.nn.Module, masks: MaskDict) -> None:
    parameters = dict(model.named_parameters())
    for name, mask in masks.items():
        parameters[name].mul_(mask.to(parameters[name].device, dtype=parameters[name].dtype))


def random_masks_like(masks: MaskDict, seed: int) -> MaskDict:
    generator = torch.Generator().manual_seed(seed)
    random_masks: MaskDict = {}
    for name, mask in masks.items():
        keep = int(mask.sum().item())
        flat = torch.zeros(mask.numel(), dtype=torch.bool)
        indices = torch.randperm(mask.numel(), generator=generator)[:keep]
        flat[indices] = True
        random_masks[name] = flat.reshape_as(mask)
    return random_masks


def transfer_expert_masks(masks: MaskDict, source_expert: int, target_expert: int) -> MaskDict:
    transferred = {name: mask.clone() for name, mask in masks.items()}
    source_token = f".experts.{source_expert}."
    target_token = f".experts.{target_expert}."
    for name, mask in masks.items():
        if source_token in name:
            target_name = name.replace(source_token, target_token)
            if target_name in transferred and transferred[target_name].shape == mask.shape:
                transferred[target_name] = mask.clone()
    return transferred


def mask_jaccard(first: MaskDict, second: MaskDict) -> float:
    common = sorted(set(first) & set(second))
    if not common:
        return 0.0
    intersection = sum(torch.logical_and(first[name].bool(), second[name].bool()).sum().item() for name in common)
    union = sum(torch.logical_or(first[name].bool(), second[name].bool()).sum().item() for name in common)
    return float(intersection / union) if union else 1.0


def save_masks(masks: MaskDict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: mask.cpu().bool() for name, mask in masks.items()}, destination)


def load_masks(path: str | Path) -> MaskDict:
    return torch.load(path, map_location="cpu", weights_only=True)

