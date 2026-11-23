"""Exact common-prefix reconstruction and state-carrying factorial continuations."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import types
import socket
from pathlib import Path

from common import (
    ATOL,
    CE_ATOL,
    CONDITIONS,
    GLOBAL_STEPS,
    OUT,
    RTOL,
    TRAINABLE_GROUPS,
    TRANSITION,
    batch_hash,
    csv_save,
    group_for_parameter,
    json_read,
    json_save,
    materialize_data,
    preserve_rng_and_modes,
    sha256,
    torch_load,
    torch_save_atomic,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/imp_adaptation_factorial_v1"))

import numpy as np
import torch
import torch.nn.functional as F
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.masks import apply_masks_, load_masks
from moe_lth.pruning.rewind import register_mask_gradient_hooks
from moe_lth.pruning.router_age import (
    assemble_router_age_model,
    component_state_dict,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)
from moe_lth.utils import create_grad_scaler

import af_native
from diagnostics import collect_diagnostics, create_start_cache, probe_indices


class PrefixReached(Exception):
    """Internal stop after the saved full-state checkpoint at update 1000."""


class TorchProxy:
    def __init__(self, base, overrides: dict):
        self.base = base
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides[name] if name in self.overrides else getattr(self.base, name)


def variants() -> dict[int, dict]:
    return {
        int(row["seed"]): row
        for row in json_read(TRANSITION / "manifest.json")["variants"]
    }


def transition_attempt(seed: int) -> Path:
    paths = sorted((TRANSITION / "attempts" / f"seed{seed}_abrupt").glob("attempt_*"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one immutable abrupt attempt for seed {seed}.")
    return paths[0]


def historical_ce(seed: int, model_type: str, step: int) -> float:
    identifier = f"seed{seed}_{'dense' if model_type == 'unpruned' else 'abrupt'}"
    with (TRANSITION / "trajectory_metrics.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["id"] == identifier and int(row["step"]) == step:
                return float(row["ce"])
    raise RuntimeError(f"Historical CE is missing for {identifier} at step {step}.")


def precision_guard():
    class Guard:
        def __enter__(self):
            self.matmul = torch.backends.cuda.matmul.allow_tf32
            self.cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return self

        def __exit__(self, exc_type, exc, tb):
            torch.backends.cuda.matmul.allow_tf32 = self.matmul
            torch.backends.cudnn.allow_tf32 = self.cudnn
            return False

    return Guard()


@torch.inference_mode()
def evaluate(model: torch.nn.Module, batches: list, device: torch.device) -> dict:
    previous_mode = model.training
    model.eval()
    total_loss = 0.0
    token_count = 0
    try:
        with precision_guard(), torch.autocast(device_type=device.type, enabled=False):
            for inputs, targets in batches:
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                summed = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
                total_loss += float(summed.double().cpu())
                token_count += targets.numel()
    finally:
        model.train(previous_mode)
    if token_count == 0:
        raise RuntimeError("Validation set contains no target tokens.")
    return {
        "ce": total_loss / token_count,
        "loss_sum": total_loss,
        "target_tokens": token_count,
        "batches": len(batches),
        "precision": "FP32_TF32off",
    }


def native_invocation(
    *,
    config: dict,
    condition_name: str,
    model_state: dict[str, torch.Tensor],
    router_checkpoint: Path,
    masks: dict[str, torch.Tensor],
    reference_selected: list[torch.Tensor],
    calibration_batches: list[torch.Tensor],
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    validation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    seed: int,
    output_dir: Path,
    checkpoint: Path,
    condition: str,
    prefix_capture: bool = False,
    pre_update_callback=None,
    post_update_callback=None,
    clipping_callback=None,
) -> dict:
    """Call the repository's native optimizer loop with state-safe adapters."""
    original = recovery._run_recovery_condition
    state_hashes = {
        group: state_dict_hash(
            {
                name: tensor
                for name, tensor in model_state.items()
                if parameter_group(name) == group
            }
        )
        for group in ("expert", "shared")
    }
    mask_hash = recovery._mask_hash(masks)
    train_hash = recovery._batch_sequence_hash(train_batches)
    validation_hash = recovery._batch_sequence_hash(validation_batches)
    saved_checkpoint = str(checkpoint)

    def checkpoint_save(payload, path, *args, **kwargs):
        if isinstance(payload, dict) and "optimizer" in payload and "step" in payload:
            step = int(payload["step"])
            record = dict(payload)
            record["global_step"] = step
            record["sampler_next_batch"] = step
            record["study_condition"] = condition
            record["study_condition_name"] = condition_name
            record["trainability"] = {
                group: group in TRAINABLE_GROUPS[condition]
                for group in ("router", "experts", "shared")
            }
            record["parameter_partition_hash"] = parameter_partition_hash()
            record["mask_hash"] = mask_hash
            record["checkpoint_sha256_source"] = saved_checkpoint
            encoded = af_native.encode_numpy(record)
            torch_save_atomic(path, encoded)
            return None
        return torch.save(af_native.encode_numpy(payload), path, *args, **kwargs)

    def checkpoint_load(path, *args, **kwargs):
        kwargs["weights_only"] = True
        return af_native.decode_numpy(torch.load(path, *args, **kwargs))

    def native_evaluate(model, batches, eval_device, *args, **kwargs):
        return evaluate(model, batches, eval_device)["ce"]

    def clip_grad_norm(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        if clipping_callback is None:
            return torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, *args, **kwargs
            )
        grad_by_group = {g: 0.0 for g in ("router", "experts", "shared")}
        for parameter in parameters:
            if parameter.grad is None:
                continue
            name = parameter_names().get(id(parameter))
            if name is None:
                raise RuntimeError("Optimizer contains an unrecognized parameter.")
            grad_by_group[group_for_parameter(name)] += float(
                parameter.grad.detach().float().square().sum(dtype=torch.float64)
            )
        grad_by_group = {k: math.sqrt(v) for k, v in grad_by_group.items()}
        native_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, *args, **kwargs
        )
        norm_value = float(native_norm.detach().cpu())
        coefficient = float(
            torch.clamp(
                torch.as_tensor(max_norm, device=native_norm.device)
                / (native_norm + 1e-6),
                max=1.0,
            ).cpu()
        )
        if clipping_callback is not None:
            clipping_callback(grad_by_group, norm_value, coefficient)
        return native_norm

    def training_forward(model, token_ids, step):
        if pre_update_callback is not None:
            pre_update_callback(model, step)
        if prefix_capture and step > 1000:
            raise PrefixReached()
        return recovery._condition_forward(model, token_ids, False, 1.0)

    def state_callback(model, optimizer, scaler, step):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(group_for_parameter(name) in TRAINABLE_GROUPS[condition])
        if post_update_callback is not None:
            post_update_callback(model, optimizer, scaler, step)

    original_assemble = original.__globals__["assemble_router_age_model"]

    def assemble_with_names(*args, **kwargs):
        model = original_assemble(*args, **kwargs)
        _PARAMETER_NAMES.clear()
        _PARAMETER_NAMES.update(
            {id(parameter): name for name, parameter in model.named_parameters()}
        )
        return model

    # Every optimizer group retains every original unique parameter. The step-0
    # callback changes only requires_grad flags after AdamW has been constructed.
    proxy_nn_utils = TorchProxy(
        torch.nn.utils, {"clip_grad_norm_": clip_grad_norm}
    )
    proxy_nn = TorchProxy(torch.nn, {"utils": proxy_nn_utils})
    proxy = TorchProxy(
        torch,
        {
            "nn": proxy_nn,
            "save": checkpoint_save,
            "load": checkpoint_load,
        },
    )
    globals_copy = dict(
        original.__globals__,
        torch=proxy,
        register_mask_gradient_hooks=register_mask_gradient_hooks,
        assemble_router_age_model=assemble_with_names,
        _evaluate_condition_loss=native_evaluate,
        RECOVERY_EVAL_INTERVAL=1_000_000,
    )
    function = types.FunctionType(
        original.__code__,
        globals_copy,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    function.__kwdefaults__ = original.__kwdefaults__
    function(
        config=config,
        condition_name=condition_name,
        pruned_base_state=model_state,
        router_checkpoint=str(router_checkpoint),
        router_age_percent=0,
        router_step=1000,
        masks=masks,
        expert_hash=state_hashes["expert"],
        shared_hash=state_hashes["shared"],
        mask_hash=mask_hash,
        reference_selected=reference_selected,
        calibration_batches=calibration_batches,
        train_batches=train_batches,
        validation_batches=validation_batches,
        train_batch_hash=train_hash,
        validation_batch_hash=validation_hash,
        device=device,
        recovery_steps=1500,
        dense_loss=1.0,
        output_dir=output_dir,
        confidence_control=False,
        target_confidence=None,
        seed=seed,
        sparsity=0.0 if not masks else 0.8,
        router_mode=(
            "frozen"
            if condition in ("router_frozen", "router_experts_frozen")
            else "trainable"
        ),
        state_snapshot_callback=state_callback,
        training_forward_callback=training_forward,
        interruption_checkpoint=checkpoint,
        interruption_interval=100,
    )
    return {
        "mask_hash": mask_hash,
        "train_hash": train_hash,
        "validation_hash": validation_hash,
        "expert_hash": state_hashes["expert"],
        "shared_hash": state_hashes["shared"],
    }


def parameter_names() -> dict[int, str]:
    global _PARAMETER_NAMES
    return _PARAMETER_NAMES


_PARAMETER_NAMES: dict[int, str] = {}
_ACTIVE_OPTIMIZER = None
_ACTIVE_SCALER = None


def parameter_partition_hash() -> str:
    digest = hashlib.sha256()
    for name in sorted(_PARAMETER_NAMES.values()):
        digest.update(f"{name}:{group_for_parameter(name)}\n".encode())
    return digest.hexdigest()


def _hash_value(digest, value) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=repr):
            _hash_value(digest, key)
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())


def _structure_hash(value) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, value)
    return digest.hexdigest()


def _frozen_optimizer_hash(optimizer_state: dict, parameter_names: list[str], condition: str) -> str:
    parameter_ids = [
        int(parameter_id)
        for group in optimizer_state["param_groups"]
        for parameter_id in group["params"]
    ]
    if len(parameter_ids) != len(parameter_names):
        raise RuntimeError("Optimizer state parameter order differs from model parameter order.")
    frozen_names = {
        name for name in parameter_names
        if group_for_parameter(name) not in TRAINABLE_GROUPS[condition]
    }
    state_by_name = {
        name: optimizer_state["state"][parameter_id]
        for name, parameter_id in zip(parameter_names, parameter_ids, strict=True)
        if name in frozen_names and parameter_id in optimizer_state["state"]
    }
    return _structure_hash(state_by_name)


def _numpy_rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _rng_matches(payload: dict, device: torch.device) -> bool:
    current_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    expected_cuda = payload["cuda_rng"]
    return (
        payload["python_rng"] == random.getstate()
        and _numpy_rng_equal(payload["numpy_rng"], np.random.get_state())
        and torch.equal(payload["torch_rng"].cpu(), torch.get_rng_state())
        and len(expected_cuda) == len(current_cuda)
        and all(torch.equal(left.cpu(), right.cpu()) for left, right in zip(expected_cuda, current_cuda))
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ensure_native_output_directories(output_dir: Path) -> None:
    for name in ("routing_stats", "gradient_stats", "checkpoints", "component_checkpoints"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _truncate_jsonl_after_step(path: Path, step: int) -> None:
    rows = _read_jsonl(path)
    retained = [row for row in rows if int(row["global_step"]) <= step]
    if len(retained) == len(rows):
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in retained:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_variant_data(seed: int):
    variant = variants()[seed]
    train, validation = materialize_data(variant, 2500)
    return variant, train, validation


def build_reference_selected(
    config: dict,
    state: dict[str, torch.Tensor],
    validation: list,
    device: torch.device,
) -> list[torch.Tensor]:
    model = TinyMoELanguageModel(config["model"]).to(device)
    model.load_state_dict(state)
    model.eval()
    calibration = recovery._calibration_batches(validation)
    with preserve_rng_and_modes(model), torch.no_grad():
        selected = selected_experts_per_batch(model, calibration, device)
    del model
    return selected


def historical_snapshot_path(seed: int) -> Path:
    variant = variants()[seed]
    return Path(variant["references"]["dense"]["snapshots"]["1000"])


def reconstruct_prefix(seed: int, device: torch.device) -> dict:
    """Recreate global steps 1–1000 and save a complete native interruption state."""
    variant, train, validation = load_variant_data(seed)
    prefix_dir = OUT / "prefixes" / f"seed{seed}"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = prefix_dir / "reconstruction_interruption.pt"
    initial_payload = torch_load(variant["initial"])
    initial_state = initial_payload["model"]
    del initial_payload
    _PARAMETER_NAMES.clear()
    init_model = TinyMoELanguageModel(variant["config"]["model"])
    init_model.load_state_dict(initial_state)
    _PARAMETER_NAMES.update({id(parameter): name for name, parameter in init_model.named_parameters()})
    init_component_hashes = {
        group: state_dict_hash(
            {
                name: tensor
                for name, tensor in init_model.state_dict().items()
                if parameter_group(name) == group
            }
        )
        for group in ("expert", "shared")
    }
    init_router_path = Path(variant["initial"])
    selected = build_reference_selected(variant["config"], initial_state, validation, device)
    run_dir = prefix_dir / "native_1000"
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    prefix_status_path = OUT / "status" / f"seed{seed}_prefix.json"

    def prefix_progress(_model, _optimizer, _scaler, step):
        if step % 25 == 0 or step == 1000:
            json_save(
                prefix_status_path,
                {
                    "status": "RUNNING",
                    "seed": seed,
                    "global_step": step,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "heartbeat_unix": time.time(),
                },
            )

    try:
        native_invocation(
            config=variant["config"],
            condition_name=f"seed{seed}_prefix",
            model_state=initial_state,
            router_checkpoint=init_router_path,
            masks={},
            reference_selected=selected,
            calibration_batches=recovery._calibration_batches(validation),
            train_batches=train[:1500],
            validation_batches=validation,
            device=device,
            seed=seed,
            output_dir=run_dir,
            checkpoint=checkpoint,
            condition="full",
            prefix_capture=True,
            post_update_callback=prefix_progress,
        )
    except PrefixReached:
        pass
    else:
        raise RuntimeError("Prefix reconstruction did not stop at update 1000.")
    payload = af_native.decode_numpy(torch_load(checkpoint))
    if int(payload["step"]) != 1000 or int(payload["sampler_next_batch"]) != 1000:
        raise RuntimeError("Prefix checkpoint does not identify the exact step-1000 boundary.")
    if payload["identity"]["steps"] != 1500:
        raise RuntimeError("Prefix state is incompatible with the registered 1500-step continuation.")
    if payload["identity"]["seed"] != seed or payload["identity"]["config"] != variant["config"]:
        raise RuntimeError("Prefix checkpoint identity mismatch.")

    # Verify the reconstructed unpruned model against the historical dense
    # step-1000 model and common-readout validation loss.
    reference_path = historical_snapshot_path(seed)
    reference_payload = torch_load(reference_path)
    reference_state = reference_payload["model"]
    reconstructed_state = payload["model"]
    ref_hash = state_dict_hash(reference_state)
    reconstructed_hash = state_dict_hash(reconstructed_state)
    differences = [
        (name, float((reconstructed_state[name].float() - reference_state[name].float()).abs().max()))
        for name in reconstructed_state
    ]
    max_difference = max(value for _, value in differences)
    if not all(
        torch.allclose(reconstructed_state[name], reference_state[name], atol=ATOL, rtol=RTOL)
        for name in reconstructed_state
    ):
        raise RuntimeError(
            f"Seed {seed} prefix reconstruction failed model tolerance; max_abs={max_difference}."
        )
    reconstructed_model = TinyMoELanguageModel(variant["config"]["model"]).to(device)
    reconstructed_model.load_state_dict(reconstructed_state)
    reconstructed_ce = evaluate(reconstructed_model, validation, device)["ce"]
    historical_dense_ce = historical_ce(seed, "unpruned", 1000)
    if abs(reconstructed_ce - historical_dense_ce) > CE_ATOL:
        raise RuntimeError(
            f"Seed {seed} prefix validation CE mismatch: "
            f"{reconstructed_ce} vs historical dense {historical_dense_ce}."
        )

    prefix_record = {
        "seed": seed,
        "step": 1000,
        "sampler_next_batch": 1000,
        "state_checkpoint": str(checkpoint),
        "state_checkpoint_sha256": sha256(checkpoint),
        "state_keys": sorted(payload),
        "optimizer_parameter_groups": len(payload["optimizer"]["param_groups"]),
        "optimizer_state_entries": len(payload["optimizer"]["state"]),
        "scaler": payload["scaler"],
        "unpruned_model_hash": reconstructed_hash,
        "historical_dense_model_hash": ref_hash,
        "model_bitwise_equal": reconstructed_hash == ref_hash,
        "model_max_abs_difference": max_difference,
        "model_atol": ATOL,
        "model_rtol": RTOL,
        "validation_ce": reconstructed_ce,
        "historical_validation_ce": historical_dense_ce,
        "validation_ce_abs_difference": abs(reconstructed_ce - historical_dense_ce),
        "validation_ce_atol": CE_ATOL,
        "validation_hash": payload["identity"]["validation_hash"],
        "training_hash_steps_1_1500": payload["identity"]["train_hash"],
        "python_numpy_torch_cpu_cuda_rng_saved": all(
            key in payload
            for key in ("python_rng", "numpy_rng", "torch_rng", "cuda_rng")
        ),
        "optimizer_and_scaler_saved": "optimizer" in payload and "scaler" in payload,
        "elapsed_seconds": time.monotonic() - start,
        "status": "verified",
    }
    json_save(prefix_dir / "prefix_verification.json", prefix_record)
    # Save a conventional model checkpoint for native router assembly.
    router_path = prefix_dir / "router_step_1000.pt"
    torch_save_atomic(
        router_path,
        {
            "step": 1000,
            "model": reconstructed_state,
            "config": variant["config"],
            "validation_loss": reconstructed_ce,
        },
    )
    payload["source_optimizer_group_hashes"] = init_component_hashes
    torch_save_atomic(checkpoint, af_native.encode_numpy(payload))
    del reconstructed_model, init_model, payload, reference_payload
    return {
        **prefix_record,
        "router_checkpoint": str(router_path),
        "router_checkpoint_sha256": sha256(router_path),
    }


def masked_branch_state(
    prefix_path: Path,
    variant: dict,
    masks: dict[str, torch.Tensor],
    model_type: str,
    condition: str,
    router_path: Path,
    train: list,
    validation: list,
) -> dict:
    payload = af_native.decode_numpy(torch_load(prefix_path))
    if int(payload["step"]) != 1000:
        raise RuntimeError("Branch prefix is not at the registered global step 1000.")
    model = TinyMoELanguageModel(variant["config"]["model"])
    model.load_state_dict(payload["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant["config"]["training"]["learning_rate"]),
        weight_decay=float(variant["config"]["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    masked_scalar_state = {}
    if model_type == "sparse":
        apply_masks_(model, masks)
        for name, mask in masks.items():
            parameter = dict(model.named_parameters())[name]
            state = optimizer.state.get(parameter, {})
            masked_scalar_state[name] = {
                key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in state.items()
                if not torch.is_tensor(value) or value.shape != parameter.shape
            }
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.masked_fill_(~mask, 0)
            for key, previous in masked_scalar_state[name].items():
                current = state[key]
                if torch.is_tensor(previous):
                    if not torch.equal(current, previous):
                        raise RuntimeError(f"Masked optimizer scalar state changed: {name}/{key}.")
                elif current != previous:
                    raise RuntimeError(f"Masked optimizer scalar state changed: {name}/{key}.")
            for key, value in state.items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    if torch.count_nonzero(value[~mask]).item():
                        raise RuntimeError(f"Masked AdamW moment is nonzero: {name}/{key}.")
    elif masks:
        raise RuntimeError("Unpruned branch unexpectedly received sparse masks.")
    branch_model = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    historical_sparse_match = None
    if model_type == "sparse":
        historical_path = transition_attempt(int(variant["seed"])) / "snapshots/step_1000.pt"
        historical_payload = torch_load(historical_path)
        historical_state = historical_payload["model"]
        if set(branch_model) != set(historical_state):
            raise RuntimeError("Sparse branch and historical step-1000 state keys differ.")
        differences = [
            float((branch_model[name].float() - historical_state[name].float()).abs().max())
            for name in branch_model
        ]
        max_difference = max(differences)
        if not all(
            torch.allclose(
                branch_model[name],
                historical_state[name],
                atol=ATOL,
                rtol=RTOL,
            )
            for name in branch_model
        ):
            raise RuntimeError(
                f"IMP80 branch reconstruction failed historical model tolerance; "
                f"max_abs={max_difference}."
            )
        historical_sparse_match = {
            "artifact": str(historical_path),
            "artifact_sha256": sha256(historical_path),
            "model_hash": state_dict_hash(historical_state),
            "branch_model_hash": state_dict_hash(branch_model),
            "bitwise_equal": state_dict_hash(historical_state) == state_dict_hash(branch_model),
            "max_abs_difference": max_difference,
        }
        del historical_payload
    optimizer_state = optimizer.state_dict()
    payload.update(
        {
            "model": branch_model,
            "optimizer": optimizer_state,
            "step": 1000,
            "global_step": 1000,
            "sampler_next_batch": 1000,
            "recovery_curve": [],
            "routing_by_step": {},
            "loss_scales": [],
            "optimizer_steps_applied": 0,
            "amp_overflow_steps": 0,
            "router_gradient_observed": False,
            "log_sizes": {},
            "study_condition": condition,
            "study_condition_name": f"seed{variant['seed']}_{model_type}_{condition}",
            "trainability": {
                group: group in TRAINABLE_GROUPS[condition]
                for group in ("router", "experts", "shared")
            },
            "mask_hash": recovery._mask_hash(masks),
            "event": 10 if model_type == "sparse" else 0,
            "model_type": model_type,
            "parameter_partition_hash": parameter_partition_hash(),
            "masked_optimizer_moments_zero": model_type == "sparse",
            "masked_optimizer_scalar_state_preserved": model_type == "sparse",
        }
    )
    group_hashes = {
        group: state_dict_hash(
            {
                name: tensor
                for name, tensor in branch_model.items()
                if parameter_group(name) == group
            }
        )
        for group in ("expert", "shared", "router")
    }
    identity = {
        "config": variant["config"],
        "seed": int(variant["seed"]),
        "steps": 1500,
        "expert_hash": group_hashes["expert"],
        "shared_hash": group_hashes["shared"],
        "mask_hash": recovery._mask_hash(masks),
        "router_hash": group_hashes["router"],
        "train_hash": recovery._batch_sequence_hash(train[:1500]),
        "validation_hash": recovery._batch_sequence_hash(validation),
        "router_mode": (
            "frozen"
            if condition in ("router_frozen", "router_experts_frozen")
            else "trainable"
        ),
    }
    payload["identity"] = identity
    payload["study_identity"] = {
        "seed": int(variant["seed"]),
        "model_type": model_type,
        "condition": condition,
        "source_prefix_sha256": sha256(prefix_path),
        "router_checkpoint_sha256": sha256(router_path),
        "mask_hash": recovery._mask_hash(masks),
        "global_step": 1000,
        "training_batch_hash": recovery._batch_sequence_hash(train[1000:1500]),
        "validation_batch_hash": recovery._batch_sequence_hash(validation),
        "optimizer_reset": False,
        "historical_sparse_match": historical_sparse_match,
    }
    if model_type == "sparse":
        for name, mask in masks.items():
            if torch.count_nonzero(branch_model[name][~mask]).item():
                raise RuntimeError(f"Masked weight nonzero at branch start: {name}.")
    return payload


def run_condition(
    seed: int,
    model_type: str,
    condition: str,
    device: torch.device,
    attempt_dir: Path | None = None,
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown intervention condition {condition}.")
    prefix_dir = OUT / "prefixes" / f"seed{seed}"
    prefix_path = prefix_dir / "reconstruction_interruption.pt"
    if not prefix_path.is_file():
        raise FileNotFoundError(f"Verified prefix checkpoint is missing: {prefix_path}")
    prefix_status = json_read(prefix_dir / "prefix_verification.json")
    if prefix_status.get("status") != "verified":
        raise RuntimeError("Prefix reconstruction has not been numerically verified.")
    variant, train, validation = load_variant_data(seed)
    masks = (
        load_masks(variant["mask"])
        if model_type == "sparse"
        else {}
    )
    router_path = prefix_dir / "router_step_1000.pt"
    attempt_dir = attempt_dir or OUT / "attempts" / f"seed{seed}_{model_type}_{condition}" / f"attempt_{time.time_ns()}"
    had_attempt_directory = attempt_dir.exists()
    attempt_dir.mkdir(parents=True, exist_ok=True)
    ensure_native_output_directories(attempt_dir)
    checkpoint = attempt_dir / "interruption.pt"
    if (attempt_dir / "result.json").exists():
        raise FileExistsError(f"Attempt already has a completed result: {attempt_dir}")
    if had_attempt_directory and any(attempt_dir.iterdir()) and not checkpoint.is_file():
        raise RuntimeError(
            f"Existing attempt has outputs but no resumable interruption state: {attempt_dir}"
        )

    branch_start_payload = masked_branch_state(
        prefix_path, variant, masks, model_type, condition, router_path, train, validation
    )
    if checkpoint.exists():
        resume_payload = af_native.decode_numpy(torch_load(checkpoint))
        if resume_payload.get("identity") != branch_start_payload.get("identity"):
            raise RuntimeError("Existing interruption checkpoint has a different branch identity.")
        resume_step = int(resume_payload.get("step", -1))
        if not 1000 <= resume_step <= 1500:
            raise RuntimeError(f"Existing interruption step is outside the branch: {resume_step}.")
        if resume_payload.get("study_condition") != condition:
            raise RuntimeError("Existing interruption checkpoint has the wrong intervention.")
    else:
        resume_payload = branch_start_payload
        torch_save_atomic(checkpoint, af_native.encode_numpy(branch_start_payload))
    base_payload = branch_start_payload
    base_state = base_payload["model"]
    _PARAMETER_NAMES.clear()
    model = TinyMoELanguageModel(variant["config"]["model"])
    model.load_state_dict(base_state)
    _PARAMETER_NAMES.update({id(parameter): name for name, parameter in model.named_parameters()})
    if len(_PARAMETER_NAMES) != len(list(model.parameters())):
        raise RuntimeError("Unique parameter enumeration failed.")
    model = model.to(device)
    model.eval()
    with preserve_rng_and_modes(model):
        baseline = evaluate(model, validation, device)
    historical_baseline = historical_ce(seed, model_type, 1000)
    if abs(baseline["ce"] - historical_baseline) > CE_ATOL:
        raise RuntimeError(
            f"Branch-start validation CE differs from historical {model_type} step-1000 readout: "
            f"{baseline['ce']} vs {historical_baseline}."
        )
    del model
    json_save(
        attempt_dir / "branch_start.json",
        {
            "seed": seed,
            "model_type": model_type,
            "condition": condition,
            "state_sha256": state_dict_hash(base_state),
            "optimizer_state_entries": len(base_payload["optimizer"]["state"]),
            "scaler": base_payload["scaler"],
            "sampler_next_batch": base_payload["sampler_next_batch"],
            "mask_hash": base_payload["mask_hash"],
            "event": base_payload["event"],
            "baseline_ce": baseline["ce"],
            "historical_baseline_ce": historical_baseline,
            "baseline_ce_abs_difference": abs(baseline["ce"] - historical_baseline),
            "baseline_target_tokens": baseline["target_tokens"],
            "training_batch_hash": base_payload["study_identity"]["training_batch_hash"],
            "validation_batch_hash": base_payload["study_identity"]["validation_batch_hash"],
            "optimizer_reset": False,
            "trainability": base_payload["trainability"],
        },
    )
    baseline_model = TinyMoELanguageModel(variant["config"]["model"])
    baseline_model.load_state_dict(base_state)
    baseline_model = baseline_model.to(device)
    for name, parameter in baseline_model.named_parameters():
        parameter.requires_grad_(group_for_parameter(name) in TRAINABLE_GROUPS[condition])
    diagnostic_optimizer = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=float(variant["config"]["training"]["learning_rate"]),
        weight_decay=float(variant["config"]["training"]["weight_decay"]),
    )
    diagnostic_optimizer.load_state_dict(base_payload["optimizer"])
    diagnostic_scaler = create_grad_scaler(
        variant["config"]["training"]["precision"] == "fp16"
    )
    diagnostic_scaler.load_state_dict(base_payload["scaler"])
    batch_indices = probe_indices(seed, len(train))
    probe_batches = [train[index] for index in batch_indices]
    probe_manifest_path = OUT / "diagnostic_probe_manifest.json"
    if not probe_manifest_path.is_file():
        raise FileNotFoundError("The deterministic diagnostic probe manifest has not been sealed.")
    sealed_probe = next(
        (row for row in json_read(probe_manifest_path)["seeds"] if int(row["seed"]) == seed),
        None,
    )
    if sealed_probe is None or sealed_probe["batch_indices"] != batch_indices:
        raise RuntimeError("Runtime diagnostic batches differ from the sealed probe manifest.")
    cache_path = prefix_dir / "diagnostics" / f"{model_type}_start.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_runtime_path = attempt_dir / "diagnostic_runtime.json"
    diagnostic_runtime_seconds_total = (
        float(json_read(diagnostic_runtime_path)["total_seconds"])
        if diagnostic_runtime_path.is_file()
        else 0.0
    )
    cache_start_time = time.monotonic()
    with preserve_rng_and_modes(baseline_model):
        start_cache = create_start_cache(
            model=baseline_model,
            seed=seed,
            model_type=model_type,
            probe_batches=probe_batches,
            probe_indices_value=batch_indices,
            device=device,
            path=cache_path,
        )
    cache_generation_seconds = time.monotonic() - cache_start_time
    diagnostic_runtime_seconds_total += cache_generation_seconds
    json_save(
        diagnostic_runtime_path,
        {"total_seconds": diagnostic_runtime_seconds_total},
    )
    if start_cache["probe_identity"] != sealed_probe["probe_identity_sha256"]:
        raise RuntimeError("Runtime probe data identity differs from the sealed manifest.")
    if start_cache["positions"] != sealed_probe["fixed_token_position_indices"]:
        raise RuntimeError("Runtime fixed-token positions differ from the sealed manifest.")
    route_rows = []
    expert_rows = []
    shared_rows = []
    trajectory_rows = []
    diagnostic_start_time = time.monotonic()
    with preserve_rng_and_modes(baseline_model):
        initial_diagnostic = collect_diagnostics(
            model=baseline_model,
            optimizer=diagnostic_optimizer,
            scaler=diagnostic_scaler,
            cache=start_cache,
            probe_batches=probe_batches,
            seed=seed,
            model_type=model_type,
            condition=condition,
            offset=0,
            device=device,
            data_position=1000,
        )
    baseline_diagnostic_seconds = time.monotonic() - diagnostic_start_time
    diagnostic_runtime_seconds_total += baseline_diagnostic_seconds
    json_save(
        diagnostic_runtime_path,
        {"total_seconds": diagnostic_runtime_seconds_total},
    )
    trajectory_rows.append(
        {
            "seed": seed,
            "model_type": model_type,
            "condition": condition,
            "offset": 0,
            "global_step": 1000,
            **baseline,
            "applied_updates": 0,
            "amp_skips": 0,
            "source": "reconstructed complete branch start",
        }
    )
    frozen_parameters = {
        name: parameter.detach().clone()
        for name, parameter in baseline_model.named_parameters()
        if group_for_parameter(name) not in TRAINABLE_GROUPS[condition]
    }
    branch_start_parameters = {
        name: parameter.detach().clone()
        for name, parameter in baseline_model.named_parameters()
    }
    previous_parameters = {
        name: parameter.detach().clone()
        for name, parameter in baseline_model.named_parameters()
    }
    frozen_optimizer_start = snapshot_frozen_optimizer(
        baseline_model, diagnostic_optimizer, condition
    )
    clip_record: dict = {}
    update_path = {group: 0.0 for group in ("router", "experts", "shared")}
    update_log = attempt_dir / "study_optimizer_and_updates.jsonl"
    trajectory_log = attempt_dir / "study_trajectory_metrics.jsonl"
    route_log = attempt_dir / "study_routing_and_gate_diagnostics.jsonl"
    expert_log = attempt_dir / "study_expert_function_diagnostics.jsonl"
    shared_log = attempt_dir / "study_shared_representation_diagnostics.jsonl"
    if checkpoint.exists():
        checkpoint_step = int(resume_payload["step"])
        for path in (update_log, trajectory_log, route_log, expert_log, shared_log):
            _truncate_jsonl_after_step(path, checkpoint_step)
    route_rows = initial_diagnostic["routing_rows"]
    expert_rows = initial_diagnostic["expert_rows"]
    shared_rows = initial_diagnostic["shared_rows"]
    applied_updates = 0
    amp_skips = 0
    last_loss_scale = [float(base_payload["scaler"].get("scale", 1.0))]
    state_tracking_initialized = False
    start_time = time.monotonic()
    status_path = OUT / "status" / f"seed{seed}_{model_type}_{condition}.json"
    json_save(
        status_path,
        {
            "status": "RUNNING",
            "seed": seed,
            "model_type": model_type,
            "condition": condition,
            "attempt": str(attempt_dir),
            "global_step": int(resume_payload["step"]),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "heartbeat_unix": time.time(),
        },
    )

    def pre_update(current_model, step):
        nonlocal applied_updates, amp_skips, last_loss_scale, state_tracking_initialized
        if state_tracking_initialized:
            return
        optimizer = _ACTIVE_OPTIMIZER
        if optimizer is None:
            raise RuntimeError("Native AdamW optimizer was not exposed before the first update.")
        expected_step = int(resume_payload["step"]) + 1
        if step != expected_step:
            raise RuntimeError(
                f"Resume advanced to step {step}; expected {expected_step} from its checkpoint."
            )
        if state_dict_hash(current_model.state_dict()) != state_dict_hash(resume_payload["model"]):
            raise RuntimeError("Native model state differs from the saved interruption boundary.")
        if _structure_hash(optimizer.state_dict()) != _structure_hash(resume_payload["optimizer"]):
            raise RuntimeError("Native AdamW state differs from the saved interruption boundary.")
        if _ACTIVE_SCALER is None or _ACTIVE_SCALER.state_dict() != resume_payload["scaler"]:
            raise RuntimeError("Native GradScaler state differs from the saved interruption boundary.")
        if not _rng_matches(resume_payload, device):
            raise RuntimeError("Native RNG state differs from the saved interruption boundary.")
        if not verify_frozen_optimizer(
            current_model, optimizer, frozen_optimizer_start, condition
        ):
            raise RuntimeError("Frozen optimizer state changed before the first resumed update.")

        prior_rows = _read_jsonl(update_log)
        if prior_rows:
            by_group = {
                group: [
                    row for row in prior_rows
                    if row["group"] == group
                ]
                for group in ("router", "experts", "shared")
            }
            for group, rows in by_group.items():
                if rows:
                    update_path[group] = float(rows[-1]["path_length_l2_sum"])
            applied_updates = sum(
                bool(row["applied_update"])
                for row in prior_rows
                if row["group"] == "shared"
            )
            amp_skips = sum(
                bool(row["amp_skip"])
                for row in prior_rows
                if row["group"] == "shared"
            )
        with torch.no_grad():
            for name, parameter in current_model.named_parameters():
                previous_parameters[name].copy_(parameter.detach())
        last_loss_scale[0] = float(resume_payload["scaler"].get("scale", 1.0))
        state_tracking_initialized = True

    def clipping(groups, norm, coefficient):
        clip_record.clear()
        clip_record.update(groups=groups, global_norm=norm, coefficient=coefficient)

    def post_update(current_model, optimizer, scaler, step):
        nonlocal applied_updates, amp_skips, diagnostic_runtime_seconds_total
        if step == 0:
            # The callback runs after AdamW is constructed, so all original
            # optimizer IDs/groups remain intact when selected components freeze.
            ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            if ids != {id(parameter) for parameter in current_model.parameters()}:
                raise RuntimeError("Optimizer parameter membership changed at branch boundary.")
            return
        group_squares = {group: 0.0 for group in ("router", "experts", "shared")}
        with torch.no_grad():
            parameters = dict(current_model.named_parameters())
            for name, parameter in parameters.items():
                difference = parameter.detach() - previous_parameters[name]
                group_squares[group_for_parameter(name)] += float(
                    difference.float().square().sum(dtype=torch.float64)
                )
                previous_parameters[name].copy_(parameter.detach())
                if name in frozen_parameters and not torch.equal(
                    parameter.detach(), frozen_parameters[name]
                ):
                    raise RuntimeError(f"Frozen parameter changed: {name}.")
            for group, squared in group_squares.items():
                update_path[group] += math.sqrt(squared)
        current_scale = float(scaler.get_scale())
        skipped = current_scale < last_loss_scale[0]
        last_loss_scale[0] = current_scale
        applied = not skipped
        applied_updates += int(applied)
        amp_skips += int(skipped)
        frozen_state_equal = verify_frozen_optimizer(
            current_model, optimizer, frozen_optimizer_start, condition
        )
        if not frozen_state_equal:
            raise RuntimeError(f"Frozen AdamW state changed at step {step}.")
        rows = []
        for group in ("router", "experts", "shared"):
            rows.append(
                {
                    "seed": seed,
                    "model_type": model_type,
                    "condition": condition,
                    "global_step": step,
                    "group": group,
                    "one_attempt_parameter_change_l2": math.sqrt(group_squares[group]),
                    "path_length_l2_sum": update_path[group],
                    "applied_update": applied,
                    "amp_skip": not applied,
                    "preclip_gradient_l2": clip_record.get("groups", {}).get(group),
                    "global_preclip_gradient_l2": clip_record.get("global_norm"),
                    "clipping_coefficient": clip_record.get("coefficient"),
                    "loss_scale": float(scaler.get_scale()),
                    "frozen_optimizer_state_unchanged": frozen_state_equal,
                }
            )
        if step in (1025, 1100, 1250, 1500):
            elapsed = time.monotonic()
            metrics = evaluate(current_model, validation, device)
            diagnostic = collect_diagnostics(
                model=current_model,
                optimizer=optimizer,
                scaler=scaler,
                cache=start_cache,
                probe_batches=probe_batches,
                seed=seed,
                model_type=model_type,
                condition=condition,
                offset=step - 1000,
                device=device,
                data_position=step,
            )
            diagnostic_seconds = time.monotonic() - elapsed
            diagnostic_runtime_seconds_total += diagnostic_seconds
            json_save(
                diagnostic_runtime_path,
                {"total_seconds": diagnostic_runtime_seconds_total},
            )
            trajectory_row = {
                "seed": seed,
                "model_type": model_type,
                "condition": condition,
                "offset": step - 1000,
                "global_step": step,
                **metrics,
                "applied_updates": applied_updates,
                "amp_skips": amp_skips,
                "diagnostic_seconds": diagnostic_seconds,
                "source": "new exact continuation",
            }
            for row in diagnostic["routing_rows"]:
                recovery.append_jsonl(route_log, row)
            for row in diagnostic["expert_rows"]:
                recovery.append_jsonl(expert_log, row)
            for row in diagnostic["shared_rows"]:
                recovery.append_jsonl(shared_log, row)
            recovery.append_jsonl(trajectory_log, trajectory_row)
            net_squares = {group: 0.0 for group in ("router", "experts", "shared")}
            with torch.no_grad():
                for name, parameter in current_model.named_parameters():
                    delta = parameter.detach() - branch_start_parameters[name]
                    net_squares[group_for_parameter(name)] += float(
                        delta.float().square().sum(dtype=torch.float64)
                    )
            for row in rows:
                row["diagnostic_seconds_at_checkpoint"] = diagnostic_seconds
                row["net_displacement_from_branch_start_l2"] = math.sqrt(
                    net_squares[row["group"]]
                )
        for row in rows:
            recovery.append_jsonl(update_log, row)
        if step % 25 == 0 or step == 1500:
            json_save(
                status_path,
                {
                    "status": "RUNNING",
                    "seed": seed,
                    "model_type": model_type,
                    "condition": condition,
                    "attempt": str(attempt_dir),
                    "global_step": step,
                    "attempted_updates": step - 1000,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "heartbeat_unix": time.time(),
                },
            )

    # Used only when called from a production continuation. The native function
    # creates the optimizer internally, so pre_update resolves it via this slot.
    global _ACTIVE_OPTIMIZER, _ACTIVE_SCALER
    _ACTIVE_OPTIMIZER = None
    _ACTIVE_SCALER = None
    native_original = recovery._run_recovery_condition
    # Inject a callback at optimizer construction without rebuilding its groups.
    # The private function's state callback is invoked after construction but
    # does not expose the optimizer before its first update. The hook below is
    # installed by the local wrapper around optimizer creation.
    result = continue_with_optimizer_hook(
        native_original=native_original,
        variant=variant,
        model_state=base_state,
        router_checkpoint=router_path,
        masks=masks,
        train=train,
        validation=validation,
        device=device,
        seed=seed,
        output_dir=attempt_dir,
        checkpoint=checkpoint,
        condition=condition,
        pre_update=pre_update,
        post_update=post_update,
        clipping=clipping,
    )
    stats_path = attempt_dir / "gradient_stats" / "gradient_stats.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text().splitlines() if line.strip()]
    if len(stats) != 500:
        raise RuntimeError(f"Expected 500 attempted updates, found {len(stats)}.")
    stat_applied = sum(bool(row["optimizer_step_applied"]) for row in stats)
    all_update_rows = _read_jsonl(update_log)
    row_applied = sum(
        bool(row["applied_update"])
        for row in all_update_rows
        if row["group"] == "shared"
    )
    row_skips = sum(
        bool(row["amp_skip"])
        for row in all_update_rows
        if row["group"] == "shared"
    )
    if stat_applied != row_applied or 500 - stat_applied != row_skips:
        raise RuntimeError("Study update rows disagree with native attempted/applied accounting.")
    applied_updates = row_applied
    amp_skips = row_skips
    if applied_updates + amp_skips != 500:
        raise RuntimeError("Applied-update and AMP-skip accounting does not sum to 500.")
    if not (attempt_dir / "checkpoints/final_recovered.pt").is_file():
        raise RuntimeError("Native final checkpoint missing.")
    final_checkpoint = torch_load(checkpoint)
    if int(final_checkpoint["step"]) != 1500 or int(final_checkpoint["sampler_next_batch"]) != 1500:
        raise RuntimeError("Native final interruption state is not at global step 1500.")
    final_state = torch_load(attempt_dir / "checkpoints/final_recovered.pt")["model"]
    for name, start_parameter in base_state.items():
        if group_for_parameter(name) not in TRAINABLE_GROUPS[condition]:
            if not torch.equal(final_state[name], start_parameter):
                raise RuntimeError(f"Frozen parameter changed by endpoint: {name}.")
    parameter_names_value = list(dict(baseline_model.named_parameters()))
    initial_frozen_optimizer_hash = _frozen_optimizer_hash(
        base_payload["optimizer"], parameter_names_value, condition
    )
    final_frozen_optimizer_hash = _frozen_optimizer_hash(
        final_checkpoint["optimizer"], parameter_names_value, condition
    )
    if final_frozen_optimizer_hash != initial_frozen_optimizer_hash:
        raise RuntimeError("Frozen optimizer state differs from its branch-start state at endpoint.")
    if len(all_update_rows) != 1500:
        raise RuntimeError(f"Expected 1500 per-group update rows, found {len(all_update_rows)}.")
    if {
        int(row["global_step"])
        for row in all_update_rows
        if row["group"] == "shared"
    } != set(range(1001, 1501)):
        raise RuntimeError("Per-update accounting does not cover all 500 global steps.")
    if model_type == "sparse":
        for name, mask in masks.items():
            if torch.count_nonzero(final_state[name][~mask]).item():
                raise RuntimeError(f"Sparse mask regrew by endpoint: {name}.")
    all_trajectory_rows = trajectory_rows + _read_jsonl(trajectory_log)
    all_route_rows = route_rows + _read_jsonl(route_log)
    all_expert_rows = expert_rows + _read_jsonl(expert_log)
    all_shared_rows = shared_rows + _read_jsonl(shared_log)
    if sorted(int(row["global_step"]) for row in all_trajectory_rows) != list(GLOBAL_STEPS):
        raise RuntimeError("Validation trajectory is missing or duplicates a registered grid point.")
    if any(int(row["global_step"]) not in GLOBAL_STEPS for row in all_route_rows + all_expert_rows + all_shared_rows):
        raise RuntimeError("A diagnostic row is outside the registered measurement grid.")
    if condition == "full":
        expected_id = f"seed{seed}_{'dense' if model_type == 'unpruned' else 'abrupt'}"
        with (TRANSITION / "trajectory_metrics.csv").open(newline="", encoding="utf-8") as stream:
            history = {
                int(row["step"]): float(row["ce"])
                for row in csv.DictReader(stream)
                if row["id"] == expected_id
            }
        if 1500 not in history:
            raise RuntimeError(f"Historical full-control endpoint is missing for {expected_id}.")
        endpoint_ce = next(
            float(row["ce"]) for row in all_trajectory_rows if int(row["global_step"]) == 1500
        )
        if abs(endpoint_ce - history[1500]) > CE_ATOL:
            raise RuntimeError(
                f"Native full continuation differs from historical step-1500 CE for {expected_id}: "
                f"{endpoint_ce} vs {history[1500]}."
            )
    csv_save(attempt_dir / "optimizer_and_updates.csv", all_update_rows)
    csv_save(attempt_dir / "trajectory_metrics.csv", all_trajectory_rows)
    csv_save(attempt_dir / "routing_and_gate_diagnostics.csv", all_route_rows)
    csv_save(attempt_dir / "expert_function_diagnostics.csv", all_expert_rows)
    csv_save(attempt_dir / "shared_representation_diagnostics.csv", all_shared_rows)
    json_save(
        attempt_dir / "diagnostic_manifest.json",
        {
            "probe_indices": batch_indices,
            "probe_identity": start_cache["probe_identity"],
            "probe_cache": str(cache_path),
            "branch_start_state_hash": start_cache["start_state_hash"],
            "fixed_token_positions": start_cache["positions"],
            "fixed_token_position_count": len(start_cache["positions"]),
            "cache_generation_seconds_this_invocation": cache_generation_seconds,
            "probe_batches_may_overlap_training": True,
            "probe_is_independent_test_set": False,
            "diagnostic_preservation_verified_at_all_offsets": True,
            "actual_diagnostic_runtime_seconds": diagnostic_runtime_seconds_total,
        },
    )
    summary = {
        "seed": seed,
        "model_type": model_type,
        "condition": condition,
        "attempt": str(attempt_dir),
        "attempted_updates": 500,
        "applied_updates": applied_updates,
        "amp_skips": amp_skips,
        "native_result": result,
        "final_checkpoint": str(attempt_dir / "checkpoints/final_recovered.pt"),
        "final_checkpoint_sha256": sha256(attempt_dir / "checkpoints/final_recovered.pt"),
        "wall_seconds": time.monotonic() - start_time,
        "branch_start_validation": baseline,
        "frozen_parameters_bitwise_unchanged": True,
        "frozen_optimizer_state_unchanged": True,
        "mask_persisted": True if model_type == "sparse" else None,
        "mask_applicability": "IMP80 mask enforced" if model_type == "sparse" else "not applicable to unpruned branch",
        "optimizer_reset": False,
        "global_step": 1500,
    }
    if model_type == "sparse":
        summary["mask_persisted"] = True
    json_save(attempt_dir / "result.json", summary)
    json_save(
        status_path,
        {"status": "COMPLETE", "attempt": str(attempt_dir), "result": str(attempt_dir / "result.json")},
    )
    del baseline_model
    return summary


def snapshot_frozen_optimizer(model, optimizer, condition: str) -> dict:
    frozen = {}
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        name = named.get(id(parameter))
        if name is None or group_for_parameter(name) in TRAINABLE_GROUPS[condition]:
            continue
        frozen[name] = {
            key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in state.items()
        }
    return frozen


def verify_frozen_optimizer(model, optimizer, snapshot: dict, condition: str) -> bool:
    named_parameters = dict(model.named_parameters())
    for name, previous in snapshot.items():
        parameter = named_parameters[name]
        current = optimizer.state.get(parameter, {})
        if current.keys() != previous.keys():
            return False
        for key, value in previous.items():
            observed = current[key]
            if torch.is_tensor(value):
                if observed.device != value.device:
                    observed = observed.detach().cpu()
                    value = value.detach().cpu()
                if not torch.equal(observed, value):
                    return False
            elif observed != value:
                return False
    for name, parameter in named_parameters.items():
        if group_for_parameter(name) not in TRAINABLE_GROUPS[condition] and parameter.grad is not None:
            return False
    return True


def continue_with_optimizer_hook(
    *,
    native_original,
    variant,
    model_state,
    router_checkpoint,
    masks,
    train,
    validation,
    device,
    seed,
    output_dir,
    checkpoint,
    condition,
    pre_update,
    post_update,
    clipping,
):
    """Wrap the optimizer constructor to expose its persistent state to probes."""
    # The callback after optimizer construction can retain the actual object by
    # temporarily wrapping AdamW; it still delegates to the native constructor.
    original_adamw = torch.optim.AdamW

    def adamw_with_reference(*args, **kwargs):
        optimizer = original_adamw(*args, **kwargs)
        global _ACTIVE_OPTIMIZER
        _ACTIVE_OPTIMIZER = optimizer
        return optimizer

    original_assemble = native_original.__globals__["assemble_router_age_model"]

    def assemble_with_names(*args, **kwargs):
        model = original_assemble(*args, **kwargs)
        _PARAMETER_NAMES.clear()
        _PARAMETER_NAMES.update(
            {id(parameter): name for name, parameter in model.named_parameters()}
        )
        return model

    def native_evaluate(model, batches, eval_device, *args, **kwargs):
        return evaluate(model, batches, eval_device)["ce"]

    proxy_utils = TorchProxy(
        torch.nn.utils,
        {
            "clip_grad_norm_": lambda parameters, max_norm, *args, **kwargs: _clip_wrapper(
                parameters, max_norm, args, kwargs, clipping
            )
        },
    )
    proxy_nn = TorchProxy(torch.nn, {"utils": proxy_utils})
    proxy_optim = TorchProxy(torch.optim, {"AdamW": adamw_with_reference})
    proxy = TorchProxy(
        torch,
        {
            "nn": proxy_nn,
            "optim": proxy_optim,
            "save": _save_wrapper(output_dir, checkpoint, condition),
            "load": _load_wrapper,
            "assemble_router_age_model": assemble_with_names,
        },
    )
    globals_copy = dict(
        native_original.__globals__,
        torch=proxy,
        register_mask_gradient_hooks=register_mask_gradient_hooks,
        assemble_router_age_model=assemble_with_names,
        _evaluate_condition_loss=native_evaluate,
        RECOVERY_EVAL_INTERVAL=1_000_000,
    )
    function = types.FunctionType(
        native_original.__code__,
        globals_copy,
        native_original.__name__,
        native_original.__defaults__,
        native_original.__closure__,
    )
    function.__kwdefaults__ = native_original.__kwdefaults__

    # The step-0 native callback freezes parameters after AdamW group creation.
    def state_callback(model, optimizer, scaler, step):
        global _ACTIVE_SCALER
        if step == 0:
            _ACTIVE_SCALER = scaler
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(group_for_parameter(name) in TRAINABLE_GROUPS[condition])
        if step == 0:
            if condition in ("router_frozen", "router_experts_frozen"):
                if len(optimizer.param_groups) != 1:
                    raise RuntimeError("Unexpected native optimizer group structure.")
                # Keep native router-frozen checks while retaining the complete
                # branch optimizer group and its saved router moments.
                optimizer.param_groups[0]["params"] = list(model.parameters())
            ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if ids != {id(p) for p in model.parameters()}:
                raise RuntimeError("AdamW groups changed at branch start.")
        else:
            post_update(model, optimizer, scaler, step)

    calibration = recovery._calibration_batches(validation)
    selected = build_reference_selected(variant["config"], model_state, validation, device)
    # The native callback is step-local; install the optimizer reference before
    # evaluating its first parameter-update callback.
    def forward(model, tokens, step):
        if step == 1000:
            raise RuntimeError("Native resume did not advance beyond the saved prefix.")
        pre_update(model, step)
        return recovery._condition_forward(model, tokens, False, 1.0)

    return function(
        config=variant["config"],
        condition_name=f"seed{seed}_{condition}",
        pruned_base_state=model_state,
        router_checkpoint=str(router_checkpoint),
        router_age_percent=0,
        router_step=1000,
        masks=masks,
        expert_hash=state_dict_hash(
            {n: t for n, t in model_state.items() if parameter_group(n) == "expert"}
        ),
        shared_hash=state_dict_hash(
            {n: t for n, t in model_state.items() if parameter_group(n) == "shared"}
        ),
        mask_hash=recovery._mask_hash(masks),
        reference_selected=selected,
        calibration_batches=calibration,
        train_batches=train[:1500],
        validation_batches=validation,
        train_batch_hash=recovery._batch_sequence_hash(train[:1500]),
        validation_batch_hash=recovery._batch_sequence_hash(validation),
        device=device,
        recovery_steps=1500,
        dense_loss=1.0,
        output_dir=output_dir,
        confidence_control=False,
        target_confidence=None,
        seed=seed,
        sparsity=0.0 if not masks else 0.8,
        router_mode=(
            "frozen"
            if condition in ("router_frozen", "router_experts_frozen")
            else "trainable"
        ),
        state_snapshot_callback=state_callback,
        training_forward_callback=forward,
        interruption_checkpoint=checkpoint,
        interruption_interval=100,
    )


def _clip_wrapper(parameters, max_norm, args, kwargs, callback):
    parameters = list(parameters)
    gradients = {g: 0.0 for g in ("router", "experts", "shared")}
    for parameter in parameters:
        if parameter.grad is None:
            continue
        name = _PARAMETER_NAMES.get(id(parameter))
        if name is None:
            raise RuntimeError("Gradient clipping saw a parameter without an identity.")
        gradients[group_for_parameter(name)] += float(
            parameter.grad.detach().float().square().sum(dtype=torch.float64)
        )
    gradients = {name: math.sqrt(value) for name, value in gradients.items()}
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, *args, **kwargs)
    norm_value = float(norm.detach().cpu())
    coefficient = float(
        torch.clamp(torch.as_tensor(max_norm, device=norm.device) / (norm + 1e-6), max=1.0).cpu()
    )
    callback(gradients, norm_value, coefficient)
    return norm


def _save_wrapper(output_dir: Path, checkpoint: Path, condition: str):
    def safe_save(payload, path, *args, **kwargs):
        checkpoint_paths = {checkpoint, checkpoint.with_suffix(".tmp")}
        if Path(path) in checkpoint_paths and isinstance(payload, dict) and "optimizer" in payload:
            record = dict(payload)
            record["global_step"] = int(record["step"])
            record["sampler_next_batch"] = int(record["step"])
            record["study_condition"] = condition
            record["trainability"] = {
                group: group in TRAINABLE_GROUPS[condition]
                for group in ("router", "experts", "shared")
            }
            record["parameter_partition_hash"] = parameter_partition_hash()
            return torch_save_atomic(path, af_native.encode_numpy(record))
        return torch.save(af_native.encode_numpy(payload), path, *args, **kwargs)

    return safe_save


def _load_wrapper(path, *args, **kwargs):
    kwargs["weights_only"] = True
    return af_native.decode_numpy(torch.load(path, *args, **kwargs))
