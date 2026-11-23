from __future__ import annotations

from copy import deepcopy

import torch

from moe_lth.models import TinyMoELanguageModel
from moe_lth.training.checkpoint import load_checkpoint

from .masks import MaskDict, apply_masks_


def prepare_rewound_model(
    model_config: dict,
    rewind_checkpoint: str,
    masks: MaskDict,
    device: torch.device,
    random_reinitialize_experts: bool = False,
) -> TinyMoELanguageModel:
    model = TinyMoELanguageModel(deepcopy(model_config)).to(device)
    load_checkpoint(rewind_checkpoint, model, map_location=device)
    if random_reinitialize_experts:
        for block in model.blocks:
            for expert in block.moe.experts:
                expert.apply(_reset_if_supported)
    apply_masks_(model, masks)
    return model


def _reset_if_supported(module: torch.nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


def register_mask_gradient_hooks(model: torch.nn.Module, masks: MaskDict) -> list[torch.utils.hooks.RemovableHandle]:
    parameters = dict(model.named_parameters())
    handles = []
    for name, mask in masks.items():
        device_mask = mask.to(parameters[name].device, dtype=parameters[name].dtype)
        handles.append(parameters[name].register_hook(lambda gradient, m=device_mask: gradient * m))
    return handles

