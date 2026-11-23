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


def test_capacity_policy_defaults_to_native_and_no_overflow_is_unchanged():
    torch.manual_seed(4)
    config = model_config()
    config["capacity_factor"] = 100.0
    model = TinyMoELanguageModel(config).eval()
    tokens = torch.randint(0, 256, (2, 8))

    default = model(tokens)
    native = model(tokens, capacity_enabled_layers=[True, True])
    no_drop = model(tokens, capacity_enabled_layers=[False, False])

    assert torch.equal(default.logits, native.logits)
    assert torch.equal(default.routes[0].accepted_mask, no_drop.routes[0].accepted_mask)
    assert torch.equal(default.logits, no_drop.logits)


def test_no_drop_preserves_routes_and_adds_only_overflow_expert_outputs():
    from moe_lth.models.moe_layer import MoEFeedForward

    torch.manual_seed(5)
    moe = MoEFeedForward(
        d_model=8,
        hidden_size=12,
        num_experts=4,
        top_k=1,
        capacity_factor=0.5,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        moe.router.projection.weight.zero_()
    inputs = torch.randn(1, 4, 8)

    native_output, _, native = moe(inputs)
    no_drop_output, _, no_drop = moe(inputs, capacity_enabled=False)
    chunked_output, _, chunked = moe(
        inputs, capacity_enabled=False, expert_chunk_size=1
    )
    expected_no_drop = torch.zeros_like(inputs.reshape(-1, inputs.shape[-1]))
    flat_inputs = inputs.reshape(-1, inputs.shape[-1])
    flat_ids = no_drop.selected_expert_indices.reshape(-1, 1)
    flat_gates = no_drop.selected_probabilities.reshape(-1, 1)
    for token_index, expert_id in enumerate(flat_ids[:, 0].tolist()):
        expert_output = moe.experts[expert_id](flat_inputs[token_index : token_index + 1])
        expected_no_drop[token_index] = expert_output[0] * flat_gates[token_index, 0]

    assert torch.equal(native.selected_expert_indices, no_drop.selected_expert_indices)
    assert torch.equal(native.selected_probabilities, no_drop.selected_probabilities)
    assert torch.equal(native.probabilities, no_drop.probabilities)
    assert native.accepted_mask.sum().item() == 1
    assert no_drop.accepted_mask.all()
    assert no_drop.dropped_fraction.item() == 0
    assert torch.allclose(no_drop_output.reshape_as(expected_no_drop), expected_no_drop, atol=1e-7, rtol=1e-6)
    assert torch.allclose(
        native_output.reshape_as(expected_no_drop)[native.accepted_mask[:, 0]],
        no_drop_output.reshape_as(expected_no_drop)[native.accepted_mask[:, 0]],
        atol=1e-7,
        rtol=1e-6,
    )
    rejected = ~native.accepted_mask[:, 0]
    assert torch.count_nonzero(native_output.reshape_as(expected_no_drop)[rejected]) == 0
    assert torch.allclose(
        no_drop_output.reshape_as(expected_no_drop)[rejected] - native_output.reshape_as(expected_no_drop)[rejected],
        expected_no_drop[rejected],
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(no_drop_output, chunked_output, atol=1e-6, rtol=1e-5)
    assert torch.equal(no_drop.accepted_mask, chunked.accepted_mask)


def test_layer_local_capacity_policy_does_not_change_upstream_layers():
    torch.manual_seed(6)
    model = TinyMoELanguageModel(model_config()).eval()
    tokens = torch.randint(0, 256, (2, 8))

    native = model(tokens)
    no_drop_layer_one = model(
        tokens, capacity_enabled_layers=[True, False]
    )

    assert torch.equal(
        native.pre_router_hidden_states[0],
        no_drop_layer_one.pre_router_hidden_states[0],
    )
    assert torch.equal(
        native.pre_router_hidden_states[1],
        no_drop_layer_one.pre_router_hidden_states[1],
    )
    assert torch.equal(
        native.routes[1].selected_expert_indices,
        no_drop_layer_one.routes[1].selected_expert_indices,
    )
    assert torch.equal(
        native.routes[1].selected_probabilities,
        no_drop_layer_one.routes[1].selected_probabilities,
    )
