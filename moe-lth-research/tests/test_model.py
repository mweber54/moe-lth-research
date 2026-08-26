import torch

from moe_lth.models import TinyMoELanguageModel


def model_config():
    return {
        "vocab_size": 256,
        "max_seq_len": 8,
        "num_layers": 2,
        "num_heads": 2,
        "d_model": 16,
        "num_experts": 4,
        "expert_hidden_size": 32,
        "dropout": 0.0,
        "top_k": 1,
        "capacity_factor": 1.25,
    }


def test_model_returns_rich_route_traces():
    model = TinyMoELanguageModel(model_config())
    tokens = torch.randint(0, 256, (2, 8))
    output = model(tokens)
    assert output.logits.shape == (2, 8, 256)
    assert len(output.routes) == 2
    assert output.routes[0].selected_experts.shape == tokens.shape
    assert output.routes[0].usage.shape == (4,)
    assert torch.isfinite(output.auxiliary_loss)


def test_route_override_is_obeyed():
    model = TinyMoELanguageModel(model_config())
    tokens = torch.randint(0, 256, (2, 8))
    overrides = [torch.full_like(tokens, layer_id) for layer_id in range(2)]
    output = model(tokens, overrides)
    assert torch.all(output.routes[0].selected_experts == 0)
    assert torch.all(output.routes[1].selected_experts == 1)


def test_top2_routes_to_two_distinct_experts_and_backpropagates():
    config = model_config()
    config["top_k"] = 2
    model = TinyMoELanguageModel(config)
    tokens = torch.randint(0, 256, (2, 8))
    output = model(tokens)
    trace = output.routes[0]

    assert trace.selected_experts.shape == tokens.shape
    assert trace.selected_expert_indices.shape == (*tokens.shape, 2)
    assert trace.selected_probabilities.shape == (*tokens.shape, 2)
    assert torch.all(trace.selected_expert_indices[..., 0] != trace.selected_expert_indices[..., 1])
    assert torch.isclose(trace.usage.sum(), torch.tensor(1.0))

    output.logits.mean().backward()
    expert_gradients = [
        parameter.grad
        for block in model.blocks
        for expert in block.moe.experts
        for parameter in expert.parameters()
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in expert_gradients)


def test_top2_primary_override_is_obeyed():
    config = model_config()
    config["top_k"] = 2
    model = TinyMoELanguageModel(config)
    tokens = torch.randint(0, 256, (2, 8))
    overrides = [torch.full_like(tokens, layer_id) for layer_id in range(2)]
    output = model(tokens, overrides)

    assert torch.all(output.routes[0].selected_expert_indices[..., 0] == 0)
    assert torch.all(output.routes[1].selected_expert_indices[..., 0] == 1)
    assert torch.all(
        output.routes[0].selected_expert_indices[..., 0]
        != output.routes[0].selected_expert_indices[..., 1]
    )
