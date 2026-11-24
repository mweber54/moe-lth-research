"""Collect seed-7/order-A/step-250 cancellation pilot sufficient statistics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_lth.models import TinyMoELanguageModel

SOURCE = ROOT / "results/moe_group_gradient_consistency_v1"
CONFIG = json.loads((HERE / "configuration.json").read_text())
SEED = CONFIG["cell"]["seed"]
ORDER = CONFIG["cell"]["order"]
STEP = CONFIG["cell"]["step"]
CELL = f"seed{SEED}_{ORDER}"
DEVICE = torch.device("cuda")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(state) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(value.numpy().tobytes())
    return h.hexdigest()


def atomic_save(path: Path, value, *, allowlist: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def rng_equal(before, after) -> bool:
    return (
        before["python"] == after["python"]
        and before["numpy"][0] == after["numpy"][0]
        and np.array_equal(before["numpy"][1], after["numpy"][1])
        and before["numpy"][2:] == after["numpy"][2:]
        and torch.equal(before["cpu"], after["cpu"])
        and all(torch.equal(a, b) for a, b in zip(before["cuda"], after["cuda"]))
    )


def zero_gradient(parameter: torch.Tensor):
    return torch.zeros_like(parameter, dtype=torch.float64)


def model_sources(provenance):
    for relative in provenance["model"].values():
        path = Path(relative["path"])
        if sha(path) != relative["loaded_sha256"]:
            raise RuntimeError(f"native model source mismatch: {path}")


def select_inputs():
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    indices_path = SOURCE / f"indices_{CELL}.pt"
    alignment = next(
        row
        for row in provenance["batch_alignment"]
        if row["seed"] == SEED and row["order"] == ORDER
    )
    if sha(indices_path) != alignment["indices_sha256"]:
        raise RuntimeError("archived batch-index hash mismatch")
    indices = torch.load(indices_path, weights_only=True)[:STEP]
    seen = set()
    selected = []
    for archived_batch, row in enumerate(indices.tolist()):
        sequences = set(row)
        if not seen.intersection(sequences):
            seen.update(sequences)
            selected.append(archived_batch)
    if len(selected) < 180:
        raise RuntimeError(f"only {len(selected)} disjoint retained batches")
    selected = selected[:180]
    halves = [selected[:90], selected[90:]]
    frozen_paths = [
        SOURCE / f"split_half_gradients_{CELL}_{STEP}_h1_original.pt",
        SOURCE / f"split_half_gradients_{CELL}_{STEP}_h2_original.pt",
    ]
    frozen = [torch.load(path, weights_only=False) for path in frozen_paths]
    for expected, replay in zip(halves, frozen):
        if expected != replay["batch_ids"]:
            raise RuntimeError("probe half no longer matches the frozen split-half study")
    data_path = ROOT / "data/multidomain_balanced/multidomain_train.txt"
    label_path = SOURCE / "byte_domain_labels.npy"
    if sha(data_path) != provenance["corpus"]["training_sha256"]:
        raise RuntimeError("training corpus hash mismatch")
    if sha(label_path) != provenance["domain_labels_sha256"]:
        raise RuntimeError("domain-label hash mismatch")
    return provenance, indices, halves, np.frombuffer(data_path.read_bytes(), dtype=np.uint8), np.load(label_path)


def load_checkpoint(provenance):
    checkpoint = SOURCE.parents[0] / "moe_connection_lifetime_v1/prefixes" / CELL / f"state{STEP}.pt"
    if not checkpoint.exists():
        checkpoint = (
            SOURCE.parents[0]
            / "moe_online_reinforcement_pruning_v1/prefixes"
            / CELL
            / "native/interruption.pt"
        )
    expected = provenance["cells"][f"{CELL}_{STEP}"]["state_sha256"]
    if sha(checkpoint) != expected:
        raise RuntimeError("checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if payload.get("step") != STEP:
        raise RuntimeError("checkpoint step mismatch")
    return checkpoint, payload


def prepare_model(payload):
    manifest = json.loads(
        (ROOT / "results/imp_recovery_order_stability_v1/canonical_stream_manifest.json").read_text()
    )
    config = manifest["seeds"][str(SEED)]["config"]
    model = TinyMoELanguageModel(copy.deepcopy(config["model"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state = dict(model.named_parameters())
    names = sorted(
        name
        for name in state
        if ".moe.experts." in name
        and name.endswith((".fc1.weight", ".fc2.weight"))
    )
    parameters = [state[name] for name in names]
    if len(parameters) != 256 or sum(p.numel() for p in parameters) != 67_108_864:
        raise RuntimeError("expert-coordinate scope mismatch")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return model, names, parameters, config


def as_gradient(gradient, parameter):
    return zero_gradient(parameter) if gradient is None else gradient.detach().to(torch.float64)


def activation_measurement(model, output, domain):
    accumulators = {
        (layer, expert, d): {
            "activation_square": torch.zeros(1024, dtype=torch.float64, device=DEVICE),
            "routing_activation_square": torch.zeros(1024, dtype=torch.float64, device=DEVICE),
            "count": 0,
            "probability_square": torch.zeros((), dtype=torch.float64, device=DEVICE),
        }
        for layer in range(4)
        for expert in range(32)
        for d in (1, 2)
    }
    with torch.no_grad():
        for layer, (block, trace, hidden) in enumerate(
            zip(model.blocks, output.routes, output.pre_router_hidden_states)
        ):
            ids = trace.selected_expert_indices.reshape(-1, 1)
            probabilities = trace.selected_probabilities.reshape(-1, 1)
            accepted = trace.accepted_mask.reshape(-1, 1)
            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            flat_domain = domain.reshape(-1)
            for expert_id, expert in enumerate(block.moe.experts):
                edges = ((ids == expert_id) & accepted).nonzero()
                if edges.numel() == 0:
                    continue
                positions = edges[:, 0]
                slots = edges[:, 1]
                activation = expert.dropout(expert.activation(expert.fc1(flat_hidden[positions])))
                gate = probabilities[positions, slots]
                for d in (1, 2):
                    keep = flat_domain[positions] == d
                    count = int(keep.sum())
                    if count == 0:
                        continue
                    selected_activation = activation[keep].to(torch.float64)
                    selected_gate = gate[keep].to(torch.float64)
                    weighted = selected_activation * selected_gate[:, None]
                    value = accumulators[layer, expert_id, d]
                    value["activation_square"].add_(selected_activation.square().sum(0))
                    value["routing_activation_square"].add_(weighted.square().sum(0))
                    value["probability_square"].add_(selected_gate.square().sum())
                    value["count"] += count
    return accumulators


def merge_activation(total, current):
    for key, value in current.items():
        total[key]["activation_square"].add_(value["activation_square"])
        total[key]["routing_activation_square"].add_(value["routing_activation_square"])
        total[key]["probability_square"].add_(value["probability_square"])
        total[key]["count"] += value["count"]


def run_half(half_number, batch_ids, indices, data, labels, model, names, parameters, direct_validation):
    domain_target_sum = np.zeros(2, dtype=np.int64)
    unknown_target_sum = 0
    sum_gradient = [
        [torch.zeros_like(parameter, dtype=torch.float64, device=DEVICE) for parameter in parameters]
        for _ in range(2)
    ]
    sum_probe_square = [
        [torch.zeros_like(parameter, dtype=torch.float64, device=DEVICE) for parameter in parameters]
        for _ in range(2)
    ]
    accepted_occurrences = np.zeros((2, 4, 32), dtype=np.int64)
    proposed_occurrences = np.zeros((2, 4, 32), dtype=np.int64)
    contributing_batches = np.zeros((2, 4, 32), dtype=np.int64)
    accepted_gate_sum = np.zeros((2, 4, 32), dtype=np.float64)
    accepted_gate_square = np.zeros((2, 4, 32), dtype=np.float64)
    activation_total = {
        (layer, expert, d): {
            "activation_square": torch.zeros(1024, dtype=torch.float64, device=DEVICE),
            "routing_activation_square": torch.zeros(1024, dtype=torch.float64, device=DEVICE),
            "count": 0,
            "probability_square": torch.zeros((), dtype=torch.float64, device=DEVICE),
        }
        for layer in range(4)
        for expert in range(32)
        for d in (1, 2)
    }
    start = time.monotonic()
    direct_checks = []
    with torch.inference_mode(False):
        for position, batch_id in enumerate(batch_ids):
            offsets = indices[batch_id].numpy()[:, None] * 128 + np.arange(128)
            x = torch.tensor(data[offsets].astype(np.int64), device=DEVICE)
            y = torch.tensor(data[offsets + 1].astype(np.int64), device=DEVICE)
            domain = torch.tensor(labels[offsets + 1].astype(np.int64), device=DEVICE)
            output = model(x)
            losses = F.cross_entropy(
                output.logits.reshape(-1, 256),
                y.reshape(-1),
                reduction="none",
            ).reshape_as(y)
            known_counts = []
            domain_losses = []
            domain_masks = []
            for d in (1, 2):
                mask = domain == d
                count = int(mask.sum())
                known_counts.append(count)
                domain_target_sum[d - 1] += count
                if count == 0:
                    raise RuntimeError(f"batch {batch_id} has no targets for domain {d}")
                domain_masks.append(mask)
                domain_losses.append(losses[mask].mean())
            unknown_target_sum += int((~((domain == 1) | (domain == 2))).sum())
            validate_direct = position == 0 and direct_validation
            domain_gradients = []
            for domain_index, loss in enumerate(domain_losses):
                gradient = torch.autograd.grad(
                    loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                values = [as_gradient(g, p) for g, p in zip(gradient, parameters)]
                domain_gradients.append(values)
                count = known_counts[domain_index]
                for parameter_index, value in enumerate(values):
                    sum_gradient[domain_index][parameter_index].add_(value * count)
                    sum_probe_square[domain_index][parameter_index].add_(value.square() * count)
                del gradient, values
            direct_values = None
            if validate_direct:
                known = (domain == 1) | (domain == 2)
                direct_loss = losses[known].mean()
                direct = torch.autograd.grad(
                    direct_loss,
                    parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                direct_values = [as_gradient(g, p) for g, p in zip(direct, parameters)]
                del direct
            else:
                for loss in domain_losses:
                    loss.detach_()
            for layer, trace in enumerate(output.routes):
                ids = trace.selected_expert_indices.reshape(-1, 1)
                gates = trace.selected_probabilities.reshape_as(ids)
                accepted = trace.accepted_mask.reshape_as(ids)
                for d in (1, 2):
                    keep_domain = (domain.reshape(-1) == d).unsqueeze(1)
                    accepted_domain = keep_domain & accepted
                    proposed_domain = keep_domain & (ids >= 0)
                    for expert in range(32):
                        on_expert = ids == expert
                        count_accepted = int((accepted_domain & on_expert).sum())
                        count_proposed = int((proposed_domain & on_expert).sum())
                        accepted_occurrences[d - 1, layer, expert] += count_accepted
                        proposed_occurrences[d - 1, layer, expert] += count_proposed
                        contributing_batches[d - 1, layer, expert] += int(count_accepted > 0)
                        selected_gates = (
                            gates[accepted_domain & on_expert].detach().to(torch.float64)
                        )
                        if selected_gates.numel():
                            accepted_gate_sum[d - 1, layer, expert] += float(selected_gates.sum())
                            accepted_gate_square[d - 1, layer, expert] += float(selected_gates.square().sum())
            merge_activation(activation_total, activation_measurement(model, output, domain))
            if direct_values is not None:
                errors = []
                relative_errors = []
                for index, direct in enumerate(direct_values):
                    reconstructed = (
                        domain_gradients[0][index] * known_counts[0]
                        + domain_gradients[1][index] * known_counts[1]
                    ) / sum(known_counts)
                    difference = (direct - reconstructed).abs()
                    errors.append(float(difference.max()))
                    relative_errors.append(
                        float(difference.max() / torch.clamp(direct.abs().max(), min=1e-30))
                    )
                direct_checks.append(
                    {
                        "batch_id": int(batch_id),
                        "max_absolute_error": max(errors),
                        "max_relative_error": max(relative_errors),
                    }
                )
            if (position + 1) % 10 == 0 or position + 1 == len(batch_ids):
                atomic_json(
                    HERE / "progress.json",
                    {
                        "status": "RUNNING",
                        "half": half_number,
                        "probe": position + 1,
                        "total_probes": len(batch_ids),
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
                print(f"half {half_number}: {position + 1}/{len(batch_ids)}", flush=True)
            del x, y, domain, output, losses, domain_losses, domain_masks, domain_gradients, direct_values
    means = [
        [value.cpu() / int(domain_target_sum[d]) for value in sum_gradient[d]]
        for d in range(2)
    ]
    probe_moments = [
        [value.cpu() / int(domain_target_sum[d]) for value in sum_probe_square[d]]
        for d in range(2)
    ]
    activation_scores = {}
    for layer in range(4):
        for expert in range(32):
            down_norm = (
                model.blocks[layer].moe.experts[expert].fc2.weight.detach()
                .to(torch.float64)
                .norm(dim=0)
            )
            for d in (1, 2):
                value = activation_total[layer, expert, d]
                activation_scores[f"{layer}:{expert}:{d}"] = {
                    "activation_rms": (value["activation_square"] / max(1, value["count"])).sqrt().cpu(),
                    "routing_activation_rms": (
                        value["routing_activation_square"] / max(1, value["count"])
                    ).sqrt().cpu(),
                    "activation_l2": value["activation_square"].sqrt().cpu(),
                    "routing_activation_l2": value["routing_activation_square"].sqrt().cpu(),
                    "ramp": (value["routing_activation_square"].sqrt() * down_norm).cpu(),
                    "ramp_per_dispatch": (
                        value["routing_activation_square"].sqrt()
                        * down_norm
                        / max(1, value["count"]) ** 0.5
                    ).cpu(),
                    "ramp_per_route_energy": (
                        value["routing_activation_square"].sqrt()
                        * down_norm
                        / float(value["probability_square"].item()) ** 0.5
                        if value["probability_square"].item() > 0
                        else torch.zeros(1024, dtype=torch.float64)
                    ).cpu(),
                    "accepted_count": value["count"],
                }
    result = {
        "experiment": CONFIG["experiment"],
        "seed": SEED,
        "order": ORDER,
        "step": STEP,
        "half": half_number,
        "batch_ids": [int(i) for i in batch_ids],
        "names": names,
        "shapes": [tuple(parameter.shape) for parameter in parameters],
        "domain_target_count": domain_target_sum.tolist(),
        "unknown_target_count": int(unknown_target_sum),
        "accepted_occurrences": accepted_occurrences.tolist(),
        "proposed_occurrences": proposed_occurrences.tolist(),
        "contributing_batches": contributing_batches.tolist(),
        "accepted_gate_sum": accepted_gate_sum.tolist(),
        "accepted_gate_square": accepted_gate_square.tolist(),
        "domain_mean_unconditional": [[tensor.float() for tensor in side] for side in means],
        "domain_probe_second_moment": [[tensor.float() for tensor in side] for side in probe_moments],
        "activation_metrics": activation_scores,
        "direct_mixed_gradient_checks": direct_checks,
        "elapsed_seconds": time.monotonic() - start,
    }
    destination = HERE / f"coordinate_statistics_half{half_number}.pt"
    atomic_save(destination, result)
    return result


def artifact_audit(provenance, checkpoint, configuration, payload):
    lines = [
        "# Artifact audit",
        "",
        f"- Checkpoint: `{checkpoint.relative_to(ROOT)}`",
        f"- Checkpoint SHA-256: `{sha(checkpoint)}`",
        f"- Provenance cell: `seed{SEED}_{ORDER}_{STEP}`",
        f"- Configuration model-source hashes: verified from `native_provenance.json`.",
        f"- Frozen 180-batch split: verified independently against the two saved split-half files.",
        f"- Existing gradient artifact scope: 32 final-layer `fc2.weight` matrices only.",
        "- Existing sign sums, resolved counts, and absolute normalized-gradient sums cannot recover signed means, probe second moments, or `fc1` values.",
        "- No full per-probe gradient tensor archive was found.",
        "- Existing domain labels, corpus hash, batch indices, route replay, state hash, and deterministic pilot are reused.",
        "- Additional checkpoint replay is required for the 256-matrix pilot and for direct mixed-gradient reconstruction.",
        "",
        "## Checkpoint payload",
        "",
        f"- Payload keys: {', '.join(sorted(payload))}",
        f"- Optimizer instantiated: `False`",
        f"- Parameter updates: `0`",
        f"- Expert coordinate count: 67,108,864",
        f"- Matrices: 128 `fc1.weight` and 128 `fc2.weight`",
        "",
        "## Reused provenance",
        "",
        f"- Training corpus SHA-256: `{provenance['corpus']['training_sha256']}`",
        f"- Domain labels SHA-256: `{provenance['domain_labels_sha256']}`",
        f"- Batch-index SHA-256: `{next(x for x in provenance['batch_alignment'] if x['seed'] == SEED and x['order'] == ORDER)['indices_sha256']}`",
        f"- Configuration SHA-256: `{sha(HERE / 'configuration.json')}`",
        "",
        "No pre-existing signed aggregate is inferred from sign consistency.",
    ]
    (HERE / "artifact_audit.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--half", choices=(1, 2), type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen native replay")
    if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
        raise RuntimeError("less than 24 GiB GPU memory free")
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError("less than 30 GiB disk free")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provenance, indices, halves, data, labels = select_inputs()
    native = json.loads((ROOT / "results/moe_connection_lifetime_v1/native_provenance.json").read_text())
    model_sources(native)
    checkpoint, payload = load_checkpoint(provenance)
    model, names, parameters, model_config = prepare_model(payload)
    artifact_audit(provenance, checkpoint, model_config, payload)
    before_state = state_hash(model.state_dict())
    before_rng = rng_snapshot()
    before_model_modes = [module.training for module in model.modules()]
    requested_halves = [args.half] if args.half else [1, 2]
    results = {}
    for half_number in requested_halves:
        output_path = HERE / f"coordinate_statistics_half{half_number}.pt"
        if output_path.exists():
            results[half_number] = torch.load(output_path, weights_only=False)
            continue
        results[half_number] = run_half(
            half_number,
            halves[half_number - 1],
            indices,
            data,
            labels,
            model,
            names,
            parameters,
            direct_validation=half_number == 1,
        )
    after_rng = rng_snapshot()
    validation = {
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_model_hash_before": before_state,
        "checkpoint_model_hash_after": state_hash(model.state_dict()),
        "parameter_state_unchanged": before_state == state_hash(model.state_dict()),
        "optimizer_not_instantiated": True,
        "optimizer_updates": 0,
        "rng_unchanged": rng_equal(before_rng, after_rng),
        "module_modes_unchanged": before_model_modes == [module.training for module in model.modules()],
        "task_loss_only": True,
        "single_shared_forward_per_batch": True,
        "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32
        and not torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "halves": {
            str(half): {
                "batch_ids_sha256": hashlib.sha256(
                    np.asarray(value["batch_ids"], dtype=np.int64).tobytes()
                ).hexdigest(),
                "domain_target_count": value["domain_target_count"],
                "unknown_target_count": value["unknown_target_count"],
                "direct_mixed_gradient_checks": value["direct_mixed_gradient_checks"],
            }
            for half, value in results.items()
        },
    }
    atomic_json(HERE / "replay_validation.json", validation)
    if not all(
        validation[key]
        for key in (
            "parameter_state_unchanged",
            "optimizer_not_instantiated",
            "rng_unchanged",
            "module_modes_unchanged",
            "task_loss_only",
            "single_shared_forward_per_batch",
        )
    ):
        raise RuntimeError("pilot state-preservation validation failed")
    atomic_json(
        HERE / "progress.json",
        {
            "status": "COMPLETE",
            "halves": sorted(results),
            "numerical_replay_gate": "PASS",
            "optimizer_updates": 0,
        },
    )
    print(json.dumps({"status": "REPLAY_COMPLETE", "halves": list(results)}), flush=True)


if __name__ == "__main__":
    main()
