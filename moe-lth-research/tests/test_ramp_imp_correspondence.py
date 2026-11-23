import pytest
import torch
from moe_lth.analysis.ramp import ContributionAccumulator, profile, neuron_mask_statistics, allocate_weight_mask
from moe_lth.models.transformer import TinyMoELanguageModel
from moe_lth.models.router import RouteOverride


def model(top_k=2, capacity=.4):
    torch.manual_seed(19)
    return TinyMoELanguageModel(dict(vocab_size=16, d_model=8, num_heads=2, num_layers=1,
        expert_hidden_size=5, num_experts=3, top_k=top_k, capacity_factor=capacity,
        dropout=0., max_seq_len=8))


def test_hand_calculation_and_down_columns():
    a = ContributionAccumulator(2)
    a.add(torch.tensor([[3., 4.], [0., 2.]]), torch.tensor([.5, 1.]))
    score = a.scores(torch.tensor([[2., 0.], [0., 3.]]))['ramp']
    torch.testing.assert_close(score, torch.tensor([3., 3 * 8 ** .5], dtype=torch.float64))


def test_zero_routing_and_streaming_and_gated_input():
    a = torch.nn.functional.silu(torch.tensor([[1., -2.], [3., 1.], [-1., 2.]])) * torch.tensor([[2., 1.], [4., -3.], [1., 2.]])
    p = torch.tensor([.2, 0., .7])
    full, streamed = ContributionAccumulator(2), ContributionAccumulator(2)
    full.add(a, p)
    for row, coefficient in zip(a, p):
        streamed.add(row[None], coefficient[None])
    torch.testing.assert_close(full.weighted_square, streamed.weighted_square)
    torch.testing.assert_close(full.weighted_square, (a.double()[[0, 2]] * p.double()[[0, 2], None]).square().sum(0))
    zero = ContributionAccumulator(2)
    zero.add(a, torch.zeros(3))
    assert zero.scores(torch.eye(2))['ramp'].sum() == 0


@pytest.mark.parametrize('top_k', [1, 2])
def test_native_capacity_topk_gelu_and_determinism(top_k):
    m = model(top_k)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    m.eval()
    with torch.inference_mode():
        before = m(tokens)
    result = profile(m, [tokens], chunk_size=1)
    again = profile(m, [tokens], chunk_size=3)
    trace = before.routes[0]
    ids = trace.selected_expert_indices.reshape(-1, top_k)
    p = trace.selected_probabilities.reshape_as(ids)
    accepted = trace.accepted_mask.reshape_as(ids)
    assert (~accepted).any()
    assert accepted.sum() > 0
    if top_k == 2:
        assert accepted[:, 1].any()
    hidden = before.pre_router_hidden_states[0].reshape(-1, 8)
    for e, expert in enumerate(m.blocks[0].moe.experts):
        positions, slots = ((ids == e) & accepted).nonzero(as_tuple=True)
        activation = torch.nn.functional.gelu(expert.fc1(hidden[positions]))
        direct = ((activation.double() * p[positions, slots, None].double()).square().sum(0)).sqrt() * expert.fc2.weight.double().norm(dim=0)
        torch.testing.assert_close(result['pooled', 0, e].scores(expert.fc2.weight)['ramp'], direct)
        torch.testing.assert_close(result['pooled', 0, e].weighted_square, again['pooled', 0, e].weighted_square)
        assert result['pooled', 0, e].count == len(positions)
    with torch.inference_mode():
        torch.testing.assert_close(before.logits, m(tokens).logits, rtol=0, atol=0)


def test_profile_matches_actual_fc2_input_hooks():
    m = model(top_k=2, capacity=2.).eval()
    # Constant router coefficients allow direct sum of native fc2 hook inputs.
    m.blocks[0].moe.router.projection.weight.data.zero_()
    native = {}
    hooks = [e.fc2.register_forward_pre_hook(lambda module, args, i=i: native.setdefault(i, []).append(args[0].detach().double().clone())) for i, e in enumerate(m.blocks[0].moe.experts)]
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        m(tokens)
    for hook in hooks:
        hook.remove()
    scores = profile(m, [tokens])
    for e, acts in native.items():
        direct = torch.cat(acts).square().sum(0) / 9
        torch.testing.assert_close(scores['pooled', 0, e].weighted_square, direct)


def test_domain_sums_and_restored_mode():
    m = model().train()
    x = torch.tensor([[1, 2, 3], [3, 2, 1]])
    a = profile(m, [x], [['tiny', 'wiki']])
    assert m.training
    for e in range(3):
        torch.testing.assert_close(a['pooled', 0, e].weighted_square, a['tiny', 0, e].weighted_square + a['wiki', 0, e].weighted_square)


def test_exact_neuron_mask_coordinates_and_no_mutation():
    prefix = 'blocks.0.moe.experts.0.'
    up, down = prefix + 'fc1.weight', prefix + 'fc2.weight'
    mask = {up: torch.tensor([[True, False], [True, True], [False, False]]), down: torch.tensor([[True, False, True], [False, False, True]])}
    state = {up: torch.arange(6.).reshape(3, 2), down: torch.arange(6.).reshape(2, 3)}
    old = {k: v.clone() for k, v in mask.items()}
    r = neuron_mask_statistics(mask, state, state)[0, 0]
    torch.testing.assert_close(r['surviving_weights'], torch.tensor([2, 2, 2]))
    torch.testing.assert_close(r['up_retention'], torch.tensor([.5, 1., 0.], dtype=torch.float64))
    torch.testing.assert_close(r['down_retention'], torch.tensor([.5, 0., 1.], dtype=torch.float64))
    assert all(torch.equal(mask[k], old[k]) for k in mask)
    with pytest.raises(ValueError):
        neuron_mask_statistics({up: mask[up]}, state, state)


@pytest.mark.parametrize('keep', [0, 1, 5, 12])
def test_h3_exact_budget_and_neuron_coordinate_ownership(keep):
    u = torch.tensor([[1., 8.], [7., 2.], [3., 6.]])
    d = torch.tensor([[4., 5., 9.], [2., 3., 1.]])
    a, b = allocate_weight_mask(torch.tensor([0., 2., 10.]), u, d, keep)
    assert int(a.sum() + b.sum()) == keep
    aa, bb = allocate_weight_mask(torch.tensor([0., 2., 10.]), u, d, keep)
    assert torch.equal(a, aa) and torch.equal(b, bb)


@pytest.mark.parametrize('scores,keep,expected', [([1.,2.,3.],5,[1,2,2]), ([100.,1.,1.],8,[4,2,2]), ([0.,0.,0.],5,[2,2,1])])
def test_h3_capped_largest_remainder_allocation(scores,keep,expected):
    up=torch.ones(3,2);down=torch.ones(2,3)
    u,d=allocate_weight_mask(torch.tensor(scores),up,down,keep)
    assert (u.sum(1)+d.sum(0)).tolist()==expected
