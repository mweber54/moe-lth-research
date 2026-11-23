"""Equation (3) RAMP and exact historical two-matrix neuron coordinates.

No model mutations, training, or token-by-neuron histories. The native forward
supplies post-capacity edges; deterministic eval activations are replayed in
bounded chunks. Unsupported expert architectures fail rather than approximate.
"""
from __future__ import annotations

import torch
from moe_lth.models.experts import ExpertFFN


class ContributionAccumulator:
    """Accumulate actual intermediate activations (also valid for gated FFNs)."""
    def __init__(self, width: int):
        self.weighted_square = torch.zeros(width, dtype=torch.float64)
        self.activation_square = torch.zeros(width, dtype=torch.float64)
        self.count = 0
        self.probability_square = 0.0
        self.probability_sum = 0.0

    def add(self, activation, coefficient):
        a = activation.detach().to(device='cpu', dtype=torch.float64)
        p = coefficient.detach().to(device='cpu', dtype=torch.float64).reshape(-1, 1)
        if a.ndim != 2 or a.shape != (len(p), len(self.weighted_square)):
            raise ValueError('Activation/coefficient shape mismatch')
        if not torch.isfinite(a).all() or not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError('Nonfinite activation or invalid routing coefficient')
        self.weighted_square += (a * p).square().sum(0)
        self.activation_square += a.square().sum(0)
        self.count += len(p)
        self.probability_square += p.square().sum().item()
        self.probability_sum += p.sum().item()

    def scores(self, down):
        if down.ndim != 2 or down.shape[1] != len(self.weighted_square):
            raise ValueError('Down projection must have one column per neuron')
        norm = down.detach().cpu().double().norm(dim=0)
        weighted = self.weighted_square.sqrt()
        return dict(ramp=weighted * norm, activation_only=self.activation_square.sqrt(),
                    down_norm=norm, routing_activation=weighted,
                    ramp_per_dispatch=(weighted * norm) / max(1, self.count) ** .5,
                    ramp_per_route_energy=(weighted * norm) / max(1e-300, self.probability_square) ** .5)


@torch.inference_mode()
def profile(model, batches, batch_domains=None, chunk_size=256):
    """CPU/GPU native inference; preserve model eval flags and RNG state.

    batch_domains labels each sequence; domain-specific sums use the *same*
    pooled dispatch, not a new capacity competition. Shared experts unsupported.
    """
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    modules = list(model.modules())
    flags = [m.training for m in modules]
    accumulator = {}
    device = next(model.parameters()).device
    if any(type(e) is not ExpertFFN for b in model.blocks for e in b.moe.experts):
        raise TypeError('Only audited ExpertFFN GELU adapter is supported; add and validate a gated adapter first')
    try:
        model.eval()
        for bi, tokens in enumerate(batches):
            tokens = tokens.to(device)
            output = model(tokens)
            labels = None if batch_domains is None else batch_domains[bi]
            for layer, (block, trace, hidden) in enumerate(zip(model.blocks, output.routes, output.pre_router_hidden_states)):
                ids = trace.selected_expert_indices.reshape(-1, block.moe.top_k)
                probabilities = trace.selected_probabilities.reshape_as(ids)
                accepted = trace.accepted_mask.reshape_as(ids)
                if accepted.dtype != torch.bool:
                    raise ValueError('Missing boolean post-capacity acceptance')
                flat = hidden.reshape(-1, hidden.shape[-1])
                for expert_id, expert in enumerate(block.moe.experts):
                    edges = ((ids == expert_id) & accepted).nonzero()
                    domains = ['pooled'] + ([] if labels is None else sorted(set(labels)))
                    for domain in domains:
                        accumulator.setdefault((domain, layer, expert_id), ContributionAccumulator(expert.fc2.in_features))
                    for start in range(0, len(edges), chunk_size):
                        edge = edges[start:start + chunk_size]
                        positions, slots = edge[:, 0], edge[:, 1]
                        # Exactly the eval-mode fc2 input, including fc1 bias and GELU.
                        activation = expert.dropout(expert.activation(expert.fc1(flat[positions])))
                        p = probabilities[positions, slots]
                        accumulator['pooled', layer, expert_id].add(activation, p)
                        if labels is not None:
                            sequence_ids = (positions // tokens.shape[1]).cpu().tolist()
                            for domain in domains[1:]:
                                keep = torch.tensor([labels[i] == domain for i in sequence_ids], device=device)
                                accumulator[domain, layer, expert_id].add(activation[keep], p[keep])
            del output
    finally:
        for module, flag in zip(modules, flags):
            module.training = flag
    return accumulator


def neuron_mask_statistics(masks, initial, final):
    """Arrays indexed by (layer, expert); only historically pruned coordinates.

    Final means refer to the explicitly supplied dense profiling checkpoint,
    not a sparse recovered state. Zero-survival means are NaN, not zero.
    """
    expected = {n for n in initial if '.moe.experts.' in n and n.endswith('.weight')}
    if set(masks) != expected or any(not n.endswith(('.fc1.weight', '.fc2.weight')) for n in masks):
        raise ValueError('Masks must cover exactly audited fc1/fc2 expert weight matrices')
    result = {}
    for up in sorted(n for n in masks if n.endswith('.fc1.weight')):
        prefix = up[:-len('fc1.weight')]
        down = prefix + 'fc2.weight'
        mu, md = masks[up], masks[down]
        if mu.dtype != torch.bool or md.dtype != torch.bool:
            raise ValueError('Masks must be boolean')
        if mu.shape != initial[up].shape or md.shape != initial[down].shape or mu.shape != md.T.shape:
            raise ValueError('Mask or neuron-coordinate shape mismatch')
        for name in (up, down):
            if final[name].shape != initial[name].shape:
                raise ValueError('Checkpoint coordinate mismatch')
        count = mu.sum(1) + md.sum(0)
        total = mu.shape[1] + md.shape[0]
        stats = dict(up_retention=mu.double().mean(1), down_retention=md.double().mean(0),
                     retention=count.double() / total, surviving_weights=count,
                     initial_magnitude=(initial[up].double().abs().sum(1) + initial[down].double().abs().sum(0)) / total,
                     final_magnitude=(final[up].double().abs().sum(1) + final[down].double().abs().sum(0)) / total)
        for label, state in [('initial', initial), ('final', final)]:
            u, d = state[up].double(), state[down].double()
            denominator = count.double().masked_fill(count == 0, float('nan'))
            stats[label + '_surviving_abs_mean'] = ((u.abs() * mu).sum(1) + (d.abs() * md).sum(0)) / denominator
            stats[label + '_surviving_rms'] = (((u.square() * mu).sum(1) + (d.square() * md).sum(0)) / denominator).sqrt()
            stats[label + '_surviving_positive_fraction'] = (((u > 0) & mu).sum(1) + ((d > 0) & md).sum(0)) / denominator
        result[int(up.split('.')[1]), int(up.split('.experts.')[1].split('.')[0])] = stats
    return result


def allocate_weight_mask(neuron_scores, up, down, keep):
    """Unexecuted H3 bridge: capped proportional allocation, magnitude within neuron.

    Largest-remainder deterministic apportionment across neurons, then stable
    descending magnitude among each fc1 row + fc2 column. Exact expert budget.
    """
    width, fan_in = up.shape
    if down.shape[1] != width or not 0 <= keep <= up.numel() + down.numel():
        raise ValueError('Invalid dimensions or keep budget')
    scores = neuron_scores.detach().cpu().double()
    if scores.shape != (width,) or not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Invalid scores')
    capacity = fan_in + down.shape[0]
    # Continuous capped proportional allocation, followed by one largest-
    # remainder rounding pass. Repeatedly rounding/reweighting the residual
    # would overallocate to the largest score and is not this rule.
    continuous = torch.zeros(width, dtype=torch.float64)
    active = torch.ones(width, dtype=torch.bool)
    remaining = float(keep)
    while remaining > 0 and active.any():
        weights = scores * active
        if weights.sum() == 0:
            weights = active.double()
        shares = remaining * weights / weights.sum()
        saturated = active & (shares >= capacity)
        if saturated.any():
            continuous[saturated] = capacity
            remaining -= capacity * int(saturated.sum())
            active[saturated] = False
        else:
            continuous[active] = shares[active]
            break
    quota = continuous.floor().long()
    remainder = int(keep) - int(quota.sum())
    fractions = continuous - quota
    fractions[quota >= capacity] = -1
    order = torch.argsort(fractions, descending=True, stable=True)
    quota[order[:remainder]] += 1
    if int(quota.sum()) != int(keep) or (quota > capacity).any():
        raise RuntimeError('Exact capped allocation failed')
    um = torch.zeros_like(up, device='cpu', dtype=torch.bool)
    dm = torch.zeros_like(down, device='cpu', dtype=torch.bool)
    for i, budget in enumerate(quota.tolist()):
        values = torch.cat([up[i].detach().cpu().abs(), down[:, i].detach().cpu().abs()])
        chosen = torch.zeros(len(values), dtype=torch.bool)
        chosen[torch.argsort(values, descending=True, stable=True)[:budget]] = True
        um[i], dm[:, i] = chosen[:fan_in], chosen[fan_in:]
    return um, dm
