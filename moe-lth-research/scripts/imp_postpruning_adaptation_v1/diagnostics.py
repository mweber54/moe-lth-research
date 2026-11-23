"""Bounded, RNG-preserving routing, function, representation, and update probes."""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch

from common import (
    OUT,
    group_for_parameter,
    json_save,
    preserve_rng_and_modes,
    state_dict_hash,
    torch_save_atomic,
)


def probe_indices(seed: int, batch_count: int, count: int = 4) -> list[int]:
    if batch_count < count:
        raise ValueError(f"Need {count} complete training probe batches.")
    return sorted(
        range(batch_count),
        key=lambda index: hashlib.sha256(
            f"imp-postpruning-adaptation-v1:{seed}:batch:{index}".encode()
        ).hexdigest(),
    )[:count]


def probe_identity(seed: int, indices: list[int], probe_batches: list) -> str:
    digest = hashlib.sha256()
    digest.update(f"imp-postpruning-adaptation-v1:{seed}".encode())
    digest.update(",".join(map(str, indices)).encode())
    for inputs, targets in probe_batches:
        for tensor in (inputs, targets):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _routes(model, batches: list, device: torch.device) -> dict:
    all_ids = []
    all_accepted = []
    all_gates = []
    all_inputs = [[] for _ in model.blocks]
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for inputs, _targets in batches:
                output = model(inputs.to(device))
                for layer, trace in enumerate(output.routes):
                    ids = trace.selected_expert_indices.detach().reshape(-1, trace.selected_expert_indices.shape[-1]).cpu()
                    accepted = trace.accepted_mask.detach().reshape_as(trace.selected_expert_indices).reshape_as(ids).cpu()
                    gates = trace.selected_probabilities.detach().reshape_as(trace.selected_expert_indices).reshape_as(ids).float().cpu()
                    if len(all_ids) == layer:
                        all_ids.append(ids)
                        all_accepted.append(accepted)
                        all_gates.append(gates)
                    else:
                        all_ids[layer] = torch.cat((all_ids[layer], ids))
                        all_accepted[layer] = torch.cat((all_accepted[layer], accepted))
                        all_gates[layer] = torch.cat((all_gates[layer], gates))
                    all_inputs[layer].append(
                        output.pre_router_hidden_states[layer].detach().reshape(-1, output.pre_router_hidden_states[layer].shape[-1]).cpu()
                    )
    finally:
        for module, training in previous_modes:
            module.training = training
    if len(all_ids) != len(model.blocks):
        raise RuntimeError(
            f"Expected route traces for {len(model.blocks)} layers, found {len(all_ids)}."
        )
    return {
        "ids": all_ids,
        "accepted": all_accepted,
        "gates": all_gates,
        "inputs": [torch.cat(items, dim=0) for items in all_inputs],
    }


def _fixed_positions(seed: int, token_count: int, count: int = 16) -> list[int]:
    return sorted(
        range(token_count),
        key=lambda index: hashlib.sha256(
            f"imp-postpruning-adaptation-v1:{seed}:position:{index}".encode()
        ).hexdigest(),
    )[: min(count, token_count)]


def _parameter_components(model: torch.nn.Module, start: dict[str, torch.Tensor]) -> list[dict]:
    names = dict(model.named_parameters())
    groups = {
        "attention": [],
        "normalization": [],
        "tied_embedding_head": [],
        "position_embedding": [],
        "other_shared": [],
    }
    for name in names:
        if group_for_parameter(name) != "shared":
            continue
        if name == "token_embedding.weight":
            component = "tied_embedding_head"
        elif "position_embedding" in name:
            component = "position_embedding"
        elif ".attention." in name:
            component = "attention"
        elif "norm" in name:
            component = "normalization"
        else:
            component = "other_shared"
        groups[component].append(name)
    rows = []
    with torch.no_grad():
        for component, members in groups.items():
            squared = 0.0
            reference_squared = 0.0
            parameter_count = 0
            for name in members:
                current = names[name].detach().double()
                initial = start[name].to(current.device).double()
                squared += float((current - initial).square().sum())
                reference_squared += float(initial.square().sum())
                parameter_count += current.numel()
            rows.append(
                {
                    "component": component,
                    "parameter_count": parameter_count,
                    "parameter_names": ";".join(members),
                    "net_displacement_l2": math.sqrt(squared),
                    "normalized_net_displacement": (
                        math.sqrt(squared / reference_squared)
                        if reference_squared > 0
                        else None
                    ),
                }
            )
    return rows


def _summary(values: torch.Tensor) -> dict:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "q10": None,
            "q50": None,
            "q90": None,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "q10": float(torch.quantile(values, 0.1)),
        "q50": float(torch.quantile(values, 0.5)),
        "q90": float(torch.quantile(values, 0.9)),
    }


@torch.inference_mode()
def create_start_cache(
    *,
    model: torch.nn.Module,
    seed: int,
    model_type: str,
    probe_batches: list,
    probe_indices_value: list[int],
    device: torch.device,
    path: Path,
) -> dict:
    model_state_hash = state_dict_hash(model.state_dict())
    if path.is_file():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["probe_identity"] != probe_identity(seed, probe_indices_value, probe_batches):
            raise RuntimeError("Diagnostic probe identity changed.")
        if cache["start_state_hash"] != model_state_hash:
            raise RuntimeError("Diagnostic cache belongs to a different branch-start state.")
        if cache["model_type"] != model_type:
            raise RuntimeError("Diagnostic cache belongs to a different model type.")
        return cache
    routes = _routes(model, probe_batches, device)
    positions = _fixed_positions(seed, routes["ids"][0].shape[0])
    fixed_inputs = [value[positions].clone() for value in routes["inputs"]]
    start_state = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if group_for_parameter(name) == "shared"
    }
    expert_outputs = []
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            for layer, layer_inputs in enumerate(fixed_inputs):
                layer_outputs = []
                device_inputs = layer_inputs.to(device)
                for expert in model.blocks[layer].moe.experts:
                    layer_outputs.append(expert(device_inputs).detach().float().cpu())
                expert_outputs.append(layer_outputs)
    finally:
        for module, training in previous_modes:
            module.training = training
    cache = {
        "seed": seed,
        "model_type": model_type,
        "probe_indices": probe_indices_value,
        "probe_identity": probe_identity(seed, probe_indices_value, probe_batches),
        "start_state_hash": model_state_hash,
        "positions": positions,
        "start_state": start_state,
        "routes": routes,
        "fixed_inputs": fixed_inputs,
        "baseline_expert_outputs": expert_outputs,
    }
    torch_save_atomic(path, cache)
    return cache


def _optimizer_signature(optimizer) -> tuple:
    result = []
    for parameter, state in optimizer.state.items():
        items = []
        for key, value in sorted(state.items()):
            if torch.is_tensor(value):
                items.append((key, value.data_ptr(), value._version, tuple(value.shape)))
            else:
                items.append((key, repr(value)))
        result.append((id(parameter), tuple(items)))
    return tuple(result)


def _diagnostic_state(model, optimizer, scaler):
    buffers = {
        name: (id(value), value.data_ptr(), value._version, tuple(value.shape))
        for name, value in model.named_buffers()
    }
    parameters = {
        name: (id(value), value.data_ptr(), value._version, tuple(value.shape))
        for name, value in model.named_parameters()
    }
    gradients = {
        name: None if parameter.grad is None else (
            parameter.grad.data_ptr(),
            parameter.grad._version,
            tuple(parameter.grad.shape),
        )
        for name, parameter in model.named_parameters()
    }
    return {
        "rng": (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        ),
        "modes": [(module, module.training) for module in model.modules()],
        "buffers": buffers,
        "parameters": parameters,
        "gradients": gradients,
        "optimizer": _optimizer_signature(optimizer),
        "scaler": scaler.state_dict(),
    }


def _assert_diagnostic_state(model, optimizer, scaler, before) -> None:
    after = _diagnostic_state(model, optimizer, scaler)
    b = before["rng"]
    a = after["rng"]
    if b[0] != a[0] or not _numpy_rng_equal(b[1], a[1]) or not torch.equal(b[2], a[2]):
        raise RuntimeError("Diagnostic changed a Python/NumPy/Torch CPU RNG state.")
    if b[3] is not None and (
        len(b[3]) != len(a[3])
        or any(not torch.equal(x, y) for x, y in zip(b[3], a[3]))
    ):
        raise RuntimeError("Diagnostic changed Torch CUDA RNG state.")
    if before["buffers"] != after["buffers"]:
        raise RuntimeError("Diagnostic changed model buffers.")
    if before["parameters"] != after["parameters"]:
        raise RuntimeError("Diagnostic changed model parameters.")
    if before["gradients"] != after["gradients"]:
        raise RuntimeError("Diagnostic changed existing gradients.")
    if before["optimizer"] != after["optimizer"]:
        raise RuntimeError("Diagnostic changed optimizer state.")
    if before["scaler"] != after["scaler"]:
        raise RuntimeError("Diagnostic changed GradScaler state.")
    for module, training in before["modes"]:
        if module.training != training:
            raise RuntimeError("Diagnostic failed to restore model mode.")


def _numpy_rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


@torch.inference_mode()
def collect_diagnostics(
    *,
    model: torch.nn.Module,
    optimizer,
    scaler,
    cache: dict,
    probe_batches: list,
    seed: int,
    model_type: str,
    condition: str,
    offset: int,
    device: torch.device,
    data_position: int,
    diagnostic_start_time: float | None = None,
    current_routes_override: dict | None = None,
) -> dict:
    before = _diagnostic_state(model, optimizer, scaler)
    parameter_start = cache["start_state"]
    start_routes = cache["routes"]
    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        current_routes = (
            current_routes_override
            if current_routes_override is not None
            else _routes(model, probe_batches, device)
        )
        route_rows = []
        expert_rows = []
        shared_rows = []
        for layer, (start_ids, current_ids) in enumerate(
            zip(start_routes["ids"], current_routes["ids"])
        ):
            start_acc = start_routes["accepted"][layer]
            current_acc = current_routes["accepted"][layer]
            start_gates = start_routes["gates"][layer]
            current_gates = current_routes["gates"][layer]
            agreement = float((start_ids == current_ids).float().mean())
            for expert_id in range(model.blocks[layer].moe.num_experts):
                proposed_start = (start_ids == expert_id).any(dim=-1)
                proposed_now = (current_ids == expert_id).any(dim=-1)
                accepted_start = proposed_start & start_acc.any(dim=-1)
                accepted_now = proposed_now & current_acc.any(dim=-1)
                union = int((accepted_start | accepted_now).sum())
                jaccard = (
                    float((accepted_start & accepted_now).sum()) / union
                    if union
                    else None
                )
                same_proposed = (start_ids == current_ids).all(dim=-1)
                for population, mask in (
                    ("proposed", proposed_now),
                    ("accepted", accepted_now),
                    ("same_proposed", proposed_now & same_proposed),
                ):
                    current_values = current_gates[mask]
                    baseline_values = start_gates[mask]
                    delta = current_values - baseline_values
                    stats = _summary(current_values)
                    delta_stats = _summary(delta)
                    route_rows.append(
                        {
                            "seed": seed,
                            "model_type": model_type,
                            "condition": condition,
                            "offset": offset,
                            "global_step": 1000 + offset,
                            "layer": layer,
                            "expert": expert_id,
                            "population": population,
                            "proposed_assignments": int(proposed_now.sum()),
                            "accepted_assignments": int(accepted_now.sum()),
                            "dropped_assignment_rate": 1
                            - int(current_acc.sum()) / max(1, current_acc.numel()),
                            "dropped_token_rate": float((~current_acc.all(dim=-1)).float().mean()),
                            "assignment_agreement_with_branch_start": agreement,
                            "accepted_set_jaccard": jaccard,
                            "gate_count": stats["count"],
                            "gate_mean": stats["mean"],
                            "gate_std": stats["std"],
                            "gate_q10": stats["q10"],
                            "gate_q50": stats["q50"],
                            "gate_q90": stats["q90"],
                            "gate_change_mean": delta_stats["mean"],
                            "gate_change_abs_mean": (
                                float(delta.abs().mean()) if delta.numel() else None
                            ),
                            "gate_change_rms": (
                                float(delta.square().mean().sqrt()) if delta.numel() else None
                            ),
                            "empty_union_jaccard_undefined": union == 0,
                        }
                    )

            positions = cache["positions"]
            fixed_inputs = cache["fixed_inputs"][layer].to(device)
            native_inputs = current_routes["inputs"][layer][positions].to(device)
            for expert_id, expert in enumerate(model.blocks[layer].moe.experts):
                current_fixed = expert(fixed_inputs).detach().float().cpu()
                baseline_fixed = cache["baseline_expert_outputs"][layer][expert_id]
                differences = (current_fixed - baseline_fixed).flatten(1)
                baseline_norm = baseline_fixed.flatten(1).norm(dim=1)
                current_norm = current_fixed.flatten(1).norm(dim=1)
                nonzero = (baseline_norm > 1e-12) & (current_norm > 1e-12)
                cosine = torch.nn.functional.cosine_similarity(
                    baseline_fixed, current_fixed, dim=-1
                )
                native_output = expert(native_inputs).detach().float().cpu()
                expert_rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": condition,
                        "offset": offset,
                        "global_step": 1000 + offset,
                        "layer": layer,
                        "expert": expert_id,
                        "input_population": "branch_start_fixed_inputs",
                        "position_count": len(positions),
                        "fixed_probe_token_positions": ",".join(map(str, positions)),
                        "output_norm_rms": float(current_fixed.square().mean().sqrt()),
                        "paired_output_change_l2_rms": float(differences.square().mean().sqrt()),
                        "paired_output_cosine_mean": float(cosine[nonzero].mean()) if nonzero.any() else None,
                        "zero_norm_exclusions": int((~nonzero).sum()),
                        "low_count_exclusions": 0,
                        "comparison": "current expert on branch-start fixed layer inputs vs branch-start expert outputs",
                    }
                )
                expert_rows.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "condition": condition,
                        "offset": offset,
                        "global_step": 1000 + offset,
                        "layer": layer,
                        "expert": expert_id,
                        "input_population": "current_native_inputs",
                        "position_count": len(positions),
                        "fixed_probe_token_positions": ",".join(map(str, positions)),
                        "output_norm_rms": float(native_output.square().mean().sqrt()),
                        "paired_output_change_l2_rms": None,
                        "paired_output_cosine_mean": None,
                        "zero_norm_exclusions": None,
                        "low_count_exclusions": 0,
                        "comparison": "current expert on current branch-native layer inputs; not treated as the fixed-input contrast",
                    }
                )

            base_input = start_routes["inputs"][layer][positions].to(device).float()
            current_input = current_routes["inputs"][layer][positions].to(device).float()
            delta_input = current_input - base_input
            input_cosine = torch.nn.functional.cosine_similarity(
                base_input, current_input, dim=-1
            )
            shared_rows.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "condition": condition,
                    "offset": offset,
                    "global_step": 1000 + offset,
                    "measurement": "router_input_representation",
                    "component": f"layer_{layer}",
                    "token_positions": len(positions),
                    "fixed_probe_token_positions": ",".join(map(str, positions)),
                    "displacement_l2_rms": float(delta_input.square().mean().sqrt()),
                    "cosine_similarity_mean": float(input_cosine.mean()),
                    "definition": "same fixed probe token positions; each model type has its own branch-start inputs",
                }
            )
        for component in _parameter_components(model, parameter_start):
            shared_rows.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "condition": condition,
                    "offset": offset,
                    "global_step": 1000 + offset,
                    "measurement": "shared_parameter_displacement",
                    **component,
                    "definition": "unique shared parameters relative to same model/seed branch start",
                }
            )
    finally:
        for module, training in previous_modes:
            module.training = training
    _assert_diagnostic_state(model, optimizer, scaler, before)
    return {
        "routing_rows": route_rows,
        "expert_rows": expert_rows,
        "shared_rows": shared_rows,
        "probe_identity": cache["probe_identity"],
        "probe_indices": cache["probe_indices"],
        "probe_position_indices": cache["positions"],
        "data_position_before": data_position,
        "data_position_after": data_position,
        "diagnostic_rng_modes_gradients_buffers_parameters_optimizer_scaler_preserved": True,
    }
