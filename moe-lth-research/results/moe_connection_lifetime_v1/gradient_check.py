"""Compare routed task derivatives with native autograd on an intact batch."""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.with_name("moe_online_reinforcement_pruning_v2")))
import branch
from moe_lth.models.moe_layer import MoEFeedForward


def check(real):
    torch.manual_seed(7301)
    captured = {}
    if real:
        device = torch.device("cuda:0")
        saved, config, batches, _, model, _, scaler, dtype = branch.load_state(7, "A", device)
        layer = model.blocks[-1].moe
        tokens, labels = (a.to(device) for a in batches[249])
        scale = scaler.get_scale()
    else:
        layer = MoEFeedForward(7, 11, 3, 1, 0.6, 0.25).double().train()
        tokens = torch.randn(4, 8, 7, dtype=torch.float64)
        labels = torch.randint(7, (4, 8))
        scale = 1.0

    def remember(key):
        def capture(module, args, result):
            captured[key] = (args[0], result)
        return capture

    handles = [layer.register_forward_hook(remember("moe")),
               layer.router.register_forward_hook(remember("router"))]
    for i, expert in enumerate(layer.experts):
        handles.extend([expert.fc1.register_forward_hook(remember((i, "up"))),
                        expert.fc2.register_forward_hook(remember((i, "down")))])
    try:
        if real:
            with torch.autocast("cuda", dtype=dtype):
                out = branch.N.recovery._condition_forward(model, tokens, False, 1.)
                ce = F.cross_entropy(out.logits.flatten(0, 1), labels.flatten())
        else:
            logits, _, _ = layer(tokens)
            ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    finally:
        for handle in handles:
            handle.remove()

    _, (mixed, aux, trace) = captured["moe"]
    _, routing = captured["router"]
    gates = routing.selected_probability
    upstream, native_gate = torch.autograd.grad(ce * scale, (mixed, gates), retain_graph=True)
    dy = upstream.flatten(0, 1)
    gate_prediction = torch.zeros_like(gates)
    deviations = {}
    def compare(key, actual, expected):
        # Native mixed precision rounds GEMM inputs, products and reductions.
        a, b = actual.double(), expected.double()
        diff = (a-b).norm().item()
        relative = diff / max(a.norm().item(), b.norm().item(), 1e-30)
        deviations[key] = max(deviations.get(key, 0.), relative)
        if relative > (0.006 if real else 1e-12):
            raise RuntimeError(f"{key}: relative L2 error {relative}")

    count = 0
    for i, expert in enumerate(layer.experts):
        if (i, "up") not in captured:
            continue
        x, a = captured[i, "up"]
        h, v = captured[i, "down"]
        pairs = torch.nonzero(routing.selected_experts == i)
        n = h.shape[0]
        if n < len(pairs):
            pairs = pairs[gates[pairs[:, 0], pairs[:, 1]].topk(n).indices]
        pos, slot = pairs.unbind(1)
        assert trace.accepted_mask[pos, slot].all()
        assert torch.equal(x, captured["router"][0][pos])
        q = (dy[pos] * gates[pos, slot, None]).to(v.dtype).double()
        gate_prediction[pos, slot] = (dy[pos] * v).sum(-1).to(gates.dtype)
        ad = a.double()
        gelu = F.gelu(ad)
        d = (h != 0).double() / (1-expert.dropout.p) if expert.dropout.p else 1.
        compare("stored_activation", h, gelu*d)
        slope = 0.5*(1+torch.erf(ad/math.sqrt(2))) + ad*torch.exp(-0.5*ad.square())/math.sqrt(2*math.pi)
        feedback = q @ expert.fc2.weight.to(v.dtype).double()
        delta = feedback * d * slope
        params = (expert.fc2.weight, expert.fc2.bias, expert.fc1.weight, expert.fc1.bias)
        actual = torch.autograd.grad(ce * scale, params, retain_graph=True)
        expected = (q.T @ h.double(), q.sum(0), delta.T @ x.to(a.dtype).double(), delta.sum(0))
        for name, lhs, rhs in zip(("V","b_down","U","b_up"), actual, expected):
            compare(name, lhs, rhs)
        assert all(g is None for g in torch.autograd.grad(aux, params, retain_graph=True, allow_unused=True))
        count += n
    compare("selected_gate", native_gate, gate_prediction)
    dropped = ~trace.accepted_mask
    assert count == int(trace.accepted_mask.sum()) and dropped.any()
    assert torch.count_nonzero(native_gate[dropped]) == 0
    aux_grad, = torch.autograd.grad(aux, layer.router.projection.weight)
    assert aux_grad.isfinite().all() and aux_grad.norm() > 0
    return dict(relative_l2=deviations, accepted=count, dropped=int(dropped.sum()),
                task_ce=ce.item(), task_backward_scale=scale, aux_router_norm=aux_grad.norm().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["fixture","native"])
    kind = parser.parse_args().kind
    branch.save_json(HERE/f"gradient_{kind}.json", check(kind == "native"))
