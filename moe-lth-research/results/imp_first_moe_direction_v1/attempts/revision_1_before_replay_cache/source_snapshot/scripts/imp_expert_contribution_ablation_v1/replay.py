"""Isolated top-1 slot replay. No production model or sealed code is changed."""
import hashlib
import math
import types
import torch
from moe_lth.models.moe_layer import RouteTrace


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def capture(model, output, tokens, step, batch_hash):
    layers = []
    for index, (block, trace) in enumerate(zip(model.blocks, output.routes, strict=True)):
        ids = trace.selected_expert_indices.detach().cpu().reshape(-1, 1)
        gates = trace.selected_probabilities.detach().cpu().reshape(-1, 1)
        accepted = trace.accepted_mask.detach().cpu().reshape(-1, 1)
        capacity = max(1, math.ceil(block.moe.capacity_factor * ids.numel() / block.moe.num_experts))
        # Preserve native within-expert evaluation order, including topk ordering.
        order = []
        for expert in range(block.moe.num_experts):
            positions = torch.nonzero(ids[:, 0] == expert).flatten()
            if len(positions) > capacity:
                native_ids = trace.selected_expert_indices.detach().reshape(-1, 1)
                native_gates = trace.selected_probabilities.detach().reshape(-1, 1)
                native_positions = torch.nonzero(native_ids[:, 0] == expert).flatten()
                positions = native_positions[native_gates[native_positions, 0].topk(capacity).indices].cpu()
            if not accepted[positions].all() or int((accepted & (ids == expert)).sum()) != len(positions):
                raise RuntimeError('Donor acceptance mismatch')
            order.append(positions)
        layers.append(dict(layer=index, ids=ids, gates=gates, accepted=accepted,
                           capacity=capacity, order=order, token_positions=torch.arange(len(ids))))
    return dict(step=step, batch_hash=batch_hash, token_hash=digest(tokens), shape=list(tokens.shape), layers=layers)


def permute(layer, seed, source_seed, step):
    # Stateless keyed generator is independent of all training RNGs and variants.
    key = hashlib.sha256(f'{seed}:{source_seed}:{step}:{layer["layer"]}'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(key[:8], 'little') % (2**63-1))
    permutation = torch.randperm(len(layer['ids']), generator=generator)
    inverse = torch.argsort(permutation)
    result = dict(layer)
    for name in ('ids', 'gates', 'accepted', 'token_positions'):
        result[name] = layer[name][permutation]
    result['order'] = [inverse[positions] for positions in layer['order']]
    result['permutation_hash'] = digest(permutation)
    for expert in range(len(layer['order'])):
        for accepted_only in (False, True):
            def values(x):
                mask = x['ids'] == expert
                if accepted_only:
                    mask = mask & x['accepted']
                return x['gates'][mask].sort().values
            if not torch.equal(values(layer), values(result)):
                raise RuntimeError('Slot conservation failed')
    return result


def install(model):
    for block in model.blocks:
        moe = block.moe
        if moe.top_k != 1:
            raise ValueError('Registered cohort requires top-1')
        moe._native_forward = moe.forward
        moe._slot = None
        moe._replay_required = False
        moe.forward = types.MethodType(replay_forward, moe)


def replay_forward(self, inputs, override_ids=None, *, capacity_enabled=True, expert_chunk_size=None):
    if not self._replay_required:
        return self._native_forward(inputs, override_ids, capacity_enabled=capacity_enabled, expert_chunk_size=expert_chunk_size)
    if self._slot is None or override_ids is not None:
        raise RuntimeError('Replay requires an explicit slot payload; no native fallback')
    if not capacity_enabled or expert_chunk_size is not None:
        raise ValueError('Replay requires native capacity and unchunked computation')
    slot = self._slot
    self._slot = None  # a payload can be consumed exactly once
    batch, length, width = inputs.shape
    flat = inputs.reshape(-1, width)
    native = self.router(flat)
    ids = slot['ids'].to(flat.device)
    gates = slot['gates'].to(device=flat.device, dtype=native.probabilities.dtype).detach()
    accepted = slot['accepted'].to(flat.device)
    capacity = max(1, math.ceil(self.capacity_factor * len(flat) / self.num_experts))
    if ids.shape != (len(flat), 1) or gates.shape != ids.shape or accepted.shape != ids.shape:
        raise ValueError('Slot shape mismatch')
    if accepted.dtype != torch.bool or ids.dtype != torch.int64 or slot['capacity'] != capacity:
        raise ValueError('Slot dtype/capacity mismatch')
    if not torch.equal(slot['token_positions'].sort().values, torch.arange(len(flat))):
        raise ValueError('Token permutation is not bijective')
    if ids.min() < 0 or ids.max() >= self.num_experts or not torch.isfinite(gates).all() or (gates < 0).any() or (gates > 1).any():
        raise ValueError('Invalid expert or gate')
    output = torch.zeros_like(flat)
    seen = torch.zeros_like(accepted)
    for expert_id, expert in enumerate(self.experts):
        positions = slot['order'][expert_id].to(flat.device)
        if len(positions) > capacity or positions.unique().numel() != len(positions):
            raise ValueError('Invalid accepted positions')
        if len(positions):
            if positions.min() < 0 or positions.max() >= len(flat):
                raise ValueError('Token index out of range')
            if not (ids[positions] == expert_id).all() or not accepted[positions].all():
                raise ValueError('Acceptance/order mismatch')
            seen[positions] = True
            output = output.index_add(0, positions, expert(flat[positions]) * gates[positions])
    if not torch.equal(seen, accepted):
        raise ValueError('Missing accepted token')
    native_usage = torch.bincount(native.selected_experts.flatten(), minlength=self.num_experts).float() / native.selected_experts.numel()
    auxiliary = self.num_experts * torch.sum(native_usage.detach() * native.probabilities.mean(0))
    usage = torch.bincount(ids.flatten(), minlength=self.num_experts).float() / ids.numel()
    trace = RouteTrace(ids[:, 0].reshape(batch, length), gates[:, 0].reshape(batch, length),
                       native.probabilities.reshape(batch, length, self.num_experts),
                       ids.reshape(batch, length, 1), gates.reshape(batch, length, 1),
                       native.entropy.reshape(batch, length), native.margin.reshape(batch, length),
                       usage, 1-accepted.float().mean(), accepted)
    return output.reshape_as(inputs), auxiliary, trace


def forward(model, tokens, payload, step, batch_hash, permutation_seed=None, source_seed=None):
    if payload['step'] != step or payload['batch_hash'] != batch_hash or payload['token_hash'] != digest(tokens):
        raise ValueError('Trace step/batch identity mismatch')
    if payload['shape'] != list(tokens.shape) or len(payload['layers']) != len(model.blocks):
        raise ValueError('Trace layer/shape mismatch')
    try:
        for index, block in enumerate(model.blocks):
            layer = payload['layers'][index]
            if layer['layer'] != index:
                raise ValueError('Layer identity mismatch')
            block.moe._replay_required = True
            block.moe._slot = layer if permutation_seed is None else permute(layer, permutation_seed, source_seed, step)
        return model(tokens)
    finally:
        for block in model.blocks:
            block.moe._slot = None
            block.moe._replay_required = False
