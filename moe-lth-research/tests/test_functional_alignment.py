import torch

from moe_lth.experiments.run_functional_alignment import (
    aligned_mask_jaccard,
    build_functional_alignment,
    linear_cka,
)
from moe_lth.models import TinyMoELanguageModel
from moe_lth.training.train import configure_router_trainability

from test_model import model_config


def test_fixed_random_router_is_frozen():
    model = TinyMoELanguageModel(model_config())
    configure_router_trainability(model, "fixed_random")
    assert all(
        not parameter.requires_grad
        for block in model.blocks
        for parameter in block.moe.router.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in model.blocks
        for expert in block.moe.experts
        for parameter in expert.parameters()
    )


def test_functional_alignment_recovers_expert_and_neuron_permutations():
    generator = torch.Generator().manual_seed(3)
    source_signatures = {}
    target_signatures = {}
    expert_mapping = {0: 1, 1: 0}
    permutations = {
        0: torch.tensor([1, 3, 0, 2]),
        1: torch.tensor([2, 0, 3, 1]),
    }
    for source_expert, target_expert in expert_mapping.items():
        hidden = torch.randn(64, 4, generator=generator)
        output = torch.randn(64, 3, generator=generator)
        source_signatures[(0, source_expert)] = {"hidden": hidden, "output": output}
        target_hidden = torch.empty_like(hidden)
        target_hidden[:, permutations[source_expert]] = hidden
        target_signatures[(0, target_expert)] = {"hidden": target_hidden, "output": output}

    alignment = build_functional_alignment(source_signatures, target_signatures, 1, 2)
    assert alignment["expert_mappings"]["layer_0"] == {"0": 1, "1": 0}
    assert alignment["matched_expert_output_cka"] > 0.999

    source_masks = {}
    target_masks = {}
    for source_expert, target_expert in expert_mapping.items():
        prefix = "blocks.0.moe.experts"
        fc1 = torch.rand(4, 3, generator=generator) > 0.5
        fc2 = torch.rand(3, 4, generator=generator) > 0.5
        source_masks[f"{prefix}.{source_expert}.fc1.weight"] = fc1
        source_masks[f"{prefix}.{source_expert}.fc2.weight"] = fc2
        target_fc1 = torch.empty_like(fc1)
        target_fc2 = torch.empty_like(fc2)
        target_fc1[permutations[source_expert]] = fc1
        target_fc2[:, permutations[source_expert]] = fc2
        target_masks[f"{prefix}.{target_expert}.fc1.weight"] = target_fc1
        target_masks[f"{prefix}.{target_expert}.fc2.weight"] = target_fc2

    assert aligned_mask_jaccard(source_masks, target_masks, alignment) == 1.0
    assert linear_cka(torch.eye(4), torch.eye(4)) > 0.999
