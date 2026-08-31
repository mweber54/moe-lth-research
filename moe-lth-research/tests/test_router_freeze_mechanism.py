from __future__ import annotations

from copy import deepcopy
import json

import torch
import torch.nn.functional as F

from moe_lth.config import DEFAULT_CONFIG
from moe_lth.experiments import run_router_age_recovery as recovery
from moe_lth.models import TinyMoELanguageModel
from moe_lth.pruning.magnitude_prune import expert_local_magnitude_masks
from moe_lth.pruning.router_age import (
    build_fixed_pruned_base,
    component_state_dict,
    load_model_from_checkpoint,
    parameter_group,
    selected_experts_per_batch,
    state_dict_hash,
)
from moe_lth.training.checkpoint import save_checkpoint


def test_frozen_router_remains_identical_while_ticket_experts_update(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config.update({"seed": 13, "device": "cpu"})
    config["data"].update({"path": None, "seq_len": 8, "validation_blocks": 1})
    config["model"].update({"vocab_size": 256, "max_seq_len": 8, "num_layers": 1, "num_heads": 2, "d_model": 16, "num_experts": 2, "expert_hidden_size": 32, "dropout": 0.0, "top_k": 1})
    config["training"].update({"steps": 2, "batch_size": 2, "learning_rate": 3e-4, "weight_decay": 0.0, "grad_clip": 1.0, "precision": "fp32"})
    initial, final = TinyMoELanguageModel(config["model"]), TinyMoELanguageModel(config["model"])
    initial_path, final_path = tmp_path / "step_0.pt", tmp_path / "step_2.pt"
    save_checkpoint(initial_path, initial, None, 0, None, config)
    save_checkpoint(final_path, final, None, 2, None, config)
    masks = expert_local_magnitude_masks(final, 0.85)
    ticket = build_fixed_pruned_base(config["model"], str(initial_path), masks, torch.device("cpu"))
    expert_hash = state_dict_hash({name: value for name, value in ticket.items() if parameter_group(name) == "expert"})
    shared_hash = state_dict_hash({name: value for name, value in ticket.items() if parameter_group(name) == "shared"})
    batches = [(torch.randint(0, 256, (2, 8)), torch.randint(0, 256, (2, 8))) for _ in range(2)]
    batch_hash = recovery._batch_sequence_hash(batches)
    reference = recovery.assemble_router_age_model(config["model"], ticket, str(final_path), masks, torch.device("cpu"))
    record = recovery._run_recovery_condition(
        config=config, condition_name="frozen_smoke", pruned_base_state=ticket, router_checkpoint=str(final_path), router_age_percent=100, router_step=2,
        masks=masks, expert_hash=expert_hash, shared_hash=shared_hash, mask_hash=recovery._mask_hash(masks),
        reference_selected=selected_experts_per_batch(reference, [batches[0][0]], torch.device("cpu")), calibration_batches=[batches[0][0]],
        train_batches=batches, validation_batches=batches[:1], train_batch_hash=batch_hash, validation_batch_hash=recovery._batch_sequence_hash(batches[:1]),
        device=torch.device("cpu"), recovery_steps=2, dense_loss=10.0, output_dir=tmp_path / "frozen", confidence_control=False,
        target_confidence=None, seed=13, sparsity=0.85, router_mode="frozen",
        diagnostic_steps=(0, 1, 2), reference_selected_by_age={100: selected_experts_per_batch(reference, [batches[0][0]], torch.device("cpu"))},
        save_assignment_snapshots=True,
    )
    assert record["router_mode"] == "frozen"
    assert record["router_hash_unchanged"] is True
    assert record["router_parameter_drift_final"] == 0.0
    assert record["router_trainable_parameter_count"] == 0
    final_state = torch.load(tmp_path / "frozen" / "checkpoints" / "final_recovered.pt", map_location="cpu", weights_only=False)["model"]
    assert state_dict_hash({name: value for name, value in final_state.items() if parameter_group(name) == "expert"}) != expert_hash
    assert state_dict_hash({name: value for name, value in final_state.items() if parameter_group(name) == "shared"}) != shared_hash
    assert torch.isfinite(torch.tensor(record["final_validation_loss"]))
    routing_rows = [
        json.loads(line)
        for line in (tmp_path / "frozen" / "routing_stats" / "routing_stats.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in routing_rows] == [0, 1, 2]
    assert all(row["router_parameter_drift_absolute"] == 0.0 for row in routing_rows)
    assert all(row["router_parameter_drift_normalized"] == 0.0 for row in routing_rows)
    assert all("agreement_with_R100_reference" in row for row in routing_rows)
    gradient_rows = [
        json.loads(line)
        for line in (tmp_path / "frozen" / "gradient_stats" / "gradient_stats.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert gradient_rows[0]["router_grad_norm"] == 0.0
    assert all(row["router_grad_norm"] == 0.0 for row in gradient_rows[1:])
    assert all(row["gradient_valid"] for row in gradient_rows[1:])
    assert len(list((tmp_path / "frozen" / "assignment_snapshots").glob("step_*.pt"))) == 3
