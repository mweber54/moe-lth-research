import torch

from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.masks import apply_masks_, mask_jaccard, random_masks_like

from test_model import model_config


def test_expert_local_masks_have_expected_density_and_apply():
    model = TinyMoELanguageModel(model_config())
    masks = expert_local_magnitude_masks(model, 0.8)
    density = sum(mask.sum().item() for mask in masks.values()) / sum(mask.numel() for mask in masks.values())
    assert abs(density - 0.2) < 0.01
    apply_masks_(model, masks)
    parameters = dict(model.named_parameters())
    assert all(torch.all(parameters[name][~mask] == 0) for name, mask in masks.items())


def test_random_mask_control_differs_but_has_same_density():
    model = TinyMoELanguageModel(model_config())
    masks = expert_local_magnitude_masks(model, 0.8)
    random_masks = random_masks_like(masks, 9)
    assert all(masks[name].sum() == random_masks[name].sum() for name in masks)
    assert mask_jaccard(masks, random_masks) < 1.0

