from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from moe_lth.config import DEFAULT_CONFIG, save_config
from moe_lth.experiments.run_router_age_recovery import (
    _ensure_reference,
    run_router_age_recovery,
)
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import (
    assemble_router_age_model,
    build_fixed_pruned_base,
    calibrate_temperature,
    component_state_dict,
    forward_with_preserved_routing,
    set_router_temperature,
    state_dict_hash,
)
from moe_lth.training.checkpoint import load_checkpoint, save_checkpoint


def _tiny_config(tmp_path, *, output_name: str = "reference") -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    config.update({"seed": 13, "device": "cpu", "output_dir": str(tmp_path / output_name)})
    config["data"].update(
        {
            "path": None,
            "train_path": None,
            "validation_path": None,
            "seq_len": 8,
            "train_fraction": 0.9,
            "validation_blocks": 1,
        }
    )
    config["model"].update(
        {
            "vocab_size": 256,
            "max_seq_len": 8,
            "num_layers": 1,
            "num_heads": 2,
            "d_model": 16,
            "num_experts": 2,
            "expert_hidden_size": 32,
            "dropout": 0.0,
            "top_k": 1,
            "capacity_factor": 1.25,
        }
    )
    config["routing"].update({"mode": "learned", "aux_loss_weight": 0.01})
    config["training"].update(
        {
            "steps": 2,
            "batch_size": 2,
            "learning_rate": 3e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "eval_interval": 1,
            "log_interval": 1,
            "checkpoint_steps": [0, 1, 2],
            "save_optimizer": False,
            "record_train_routes": False,
            "precision": "fp32",
        }
    )
    return config


def test_preserved_routing_keeps_assignments_capacity_and_router_gradients():
    torch.manual_seed(5)
    config = deepcopy(DEFAULT_CONFIG["model"])
    config.update(
        {
            "vocab_size": 256,
            "max_seq_len": 8,
            "num_layers": 2,
            "num_heads": 2,
            "d_model": 16,
            "num_experts": 2,
            "expert_hidden_size": 32,
            "dropout": 0.0,
            "top_k": 1,
            "capacity_factor": 0.75,
        }
    )
    model = TinyMoELanguageModel(config)
    token_ids = torch.randint(0, 256, (4, 8))

    set_router_temperature(model, 1.0)
    with torch.no_grad():
        native = model(token_ids)
    controlled, integrity = forward_with_preserved_routing(model, token_ids, temperature=4.0)

    assert integrity["assignment_agreement_before_after"] == 1.0
    assert integrity["capacity_agreement_before_after"] == 1.0
    for native_trace, controlled_trace in zip(native.routes, controlled.routes):
        assert torch.equal(native_trace.selected_expert_indices, controlled_trace.selected_expert_indices)
        assert torch.equal(native_trace.accepted_mask, controlled_trace.accepted_mask)

    controlled.logits.square().mean().backward()
    router_gradients = [
        block.moe.router.projection.weight.grad
        for block in model.blocks
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in router_gradients)


def test_temperature_checkpoint_round_trip_and_legacy_loading(tmp_path):
    config = _tiny_config(tmp_path)["model"]
    model = TinyMoELanguageModel(config)
    set_router_temperature(model, 3.25)
    checkpoint = tmp_path / "temperature.pt"
    save_checkpoint(checkpoint, model, None, 0, None, {"model": config})

    restored = TinyMoELanguageModel(config)
    load_checkpoint(checkpoint, restored)
    assert all(block.moe.router.temperature.item() == pytest.approx(3.25) for block in restored.blocks)

    legacy_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    legacy_payload["model"] = {
        name: tensor
        for name, tensor in legacy_payload["model"].items()
        if not name.endswith(".moe.router.temperature")
    }
    legacy_checkpoint = tmp_path / "legacy.pt"
    torch.save(legacy_payload, legacy_checkpoint)
    legacy_restored = TinyMoELanguageModel(config)
    load_checkpoint(legacy_checkpoint, legacy_restored)
    assert all(block.moe.router.temperature.item() == pytest.approx(1.0) for block in legacy_restored.blocks)


def test_fixed_base_swaps_only_router(tmp_path):
    config = _tiny_config(tmp_path)["model"]
    final_model = TinyMoELanguageModel(config)
    early_model = TinyMoELanguageModel(config)
    final_checkpoint = tmp_path / "final.pt"
    early_checkpoint = tmp_path / "early.pt"
    save_checkpoint(final_checkpoint, final_model, None, 2, None, {"model": config})
    save_checkpoint(early_checkpoint, early_model, None, 0, None, {"model": config})

    masks = expert_local_magnitude_masks(final_model, 0.8)
    fixed = build_fixed_pruned_base(config, str(final_checkpoint), masks, torch.device("cpu"))
    assembled = assemble_router_age_model(
        config, fixed, str(early_checkpoint), masks, torch.device("cpu")
    )

    fixed_experts = {name: tensor for name, tensor in fixed.items() if ".moe.experts." in name}
    fixed_shared = {
        name: tensor
        for name, tensor in fixed.items()
        if ".moe.experts." not in name and ".moe.router." not in name
    }
    assert state_dict_hash(component_state_dict(assembled, "expert")) == state_dict_hash(fixed_experts)
    assert state_dict_hash(component_state_dict(assembled, "shared")) == state_dict_hash(fixed_shared)
    assert state_dict_hash(component_state_dict(assembled, "router")) == state_dict_hash(
        component_state_dict(early_model, "router")
    )


def test_ticket_uses_initial_expert_values_under_trained_mask(tmp_path):
    config = _tiny_config(tmp_path)["model"]
    initial_model = TinyMoELanguageModel(config)
    final_model = TinyMoELanguageModel(config)

    initial_path = tmp_path / "initial.pt"
    final_path = tmp_path / "final.pt"
    save_checkpoint(initial_path, initial_model, None, 0, None, {"model": config})

    with torch.no_grad():
        for block in final_model.blocks:
            for expert in block.moe.experts:
                for parameter in expert.parameters():
                    parameter.add_(7.0)
    save_checkpoint(final_path, final_model, None, 2, None, {"model": config})

    masks = expert_local_magnitude_masks(final_model, 0.8)
    ticket_state = build_fixed_pruned_base(config, str(initial_path), masks, torch.device("cpu"))

    reloaded_initial = TinyMoELanguageModel(config)
    load_checkpoint(initial_path, reloaded_initial, map_location="cpu")
    for name, mask in masks.items():
        expected = reloaded_initial.state_dict()[name]
        actual = ticket_state[name]
        assert torch.equal(actual[mask], expected[mask])
        assert torch.all(actual[~mask] == 0)


def test_partial_reference_and_nonempty_output_fail_loudly(tmp_path):
    config = _tiny_config(tmp_path)
    config_path = tmp_path / "partial.yaml"
    save_config(config, config_path)
    partial_dir = tmp_path / "reference"
    partial_dir.mkdir()
    (partial_dir / "do_not_overwrite.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing"):
        _ensure_reference(str(config_path))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing"):
        run_router_age_recovery([], str(occupied))


def test_tiny_end_to_end_router_age_run(tmp_path):
    config = _tiny_config(tmp_path)
    config_path = tmp_path / "router_age.yaml"
    save_config(config, config_path)
    output_dir = tmp_path / "recovery"

    result = run_router_age_recovery(
        [str(config_path)],
        str(output_dir),
        sparsity=0.8,
        recovery_steps=2,
        router_ages_percent=(0, 100),
        confidence_control_ages=(0, 100),
        confidence_control_seed_indices=(0,),
    )

    assert len(result["records"]) == 4
    assert (output_dir / "router_age_recovery_aggregate.csv").exists()
    assert (output_dir / "router_age_recovery_paired.csv").exists()
    assert (output_dir / "router_age_recovery_results.md").exists()
    assert (output_dir / "router_age_recovery_curves.svg").exists()

    native = [record for record in result["records"] if not record["confidence_control"]]
    assert len({record["expert_state_hash"] for record in native}) == 1
    assert len({record["shared_state_hash"] for record in native}) == 1
    assert len({record["mask_hash"] for record in native}) == 1
    assert len({record["training_batch_sequence_hash"] for record in native}) == 1
    assert len({record["validation_batch_sequence_hash"] for record in native}) == 1
    for record in result["records"]:
        metrics_path = (
            output_dir
            / "seed_13"
            / (
                f"age_{record['router_age_percent']:03d}pct_"
                f"{'confmatched' if record['confidence_control'] else 'native'}"
            )
            / "metrics.jsonl"
        )
        steps = [
            int(json.loads(line)["step"])
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
        assert steps == [0, 1, 2]
        if record["confidence_control"]:
            assert record["assignment_agreement_before_after_calibration"] == 1.0
            assert record["capacity_agreement_before_after_calibration"] == 1.0
            assert record["calibration_absolute_error"] <= 5e-4
