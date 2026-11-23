import pytest
import torch

from moe_lth.models import TinyMoELanguageModel
from scripts.imp_capacity_diagnostic_v1.run import (
    _assert_same_model_values,
    _capacity_policy_mapping,
    _contrast_tables,
    _evaluate_checkpoint,
    _validate_routing_rows,
)


def _small_config():
    return {
        "vocab_size": 32,
        "max_seq_len": 8,
        "num_layers": 2,
        "num_heads": 2,
        "d_model": 16,
        "num_experts": 4,
        "expert_hidden_size": 24,
        "dropout": 0.0,
        "top_k": 1,
        "capacity_factor": 0.5,
    }


def test_cpu_checkpoint_readout_checks_overflow_intervention_and_restoration():
    torch.manual_seed(19)
    config = _small_config()
    model = TinyMoELanguageModel(config)
    with torch.no_grad():
        for block in model.blocks:
            block.moe.router.projection.weight.zero_()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    tokens = torch.randint(0, config["vocab_size"], (2, 8))
    targets = torch.randint(0, config["vocab_size"], (2, 8))

    result, checks = _evaluate_checkpoint(
        state=state,
        masks={},
        config=config,
        batches=[(tokens, targets)],
        seed=7,
        model_type="sparse",
        offset=0,
        checkpoint_id="synthetic_overflow",
        checkpoint_hash="synthetic",
        validation_hash="synthetic",
        device=torch.device("cpu"),
    )

    assert set(result["policies"]) == {
        "NATIVE",
        "NO_DROP_ALL",
        "NO_DROP_L0",
        "NO_DROP_L1",
    }
    assert result["policies"]["NO_DROP_ALL"]["target_tokens"] == targets.numel()
    assert all(
        row["layer_rejected_assignments"] == 0
        for row in result["routing_rows"]
        if row["policy"] == "NO_DROP_ALL"
    )
    assert any(
        row["layer_rejected_assignments"] > 0
        for row in result["routing_rows"]
        if row["policy"] == "NATIVE"
    )
    assert checks["model_state_hash_preserved"]
    assert checks["rng_preserved"]
    assert checks["gradient_state_preserved"]
    assert checks["module_modes_preserved"]
    assert checks["first_intervened_layer_raw_scores_identical"]
    assert checks["layer_local_upstream_unchanged"]
    assert result["prediction_diagnostics"]["target_count"] == targets.numel()


def test_seed_contrast_report_preserves_algebraic_identities():
    readouts = []
    for seed in (7, 17, 29, 43, 61):
        for model_type in ("sparse", "unpruned"):
            for offset in (0, 500):
                native = 2.5 if model_type == "sparse" else 2.0
                native -= offset / 10000
                policies = {
                    "NATIVE": {
                        "task_ce": native,
                        "loss_sum": native,
                        "target_tokens": 1,
                        "batches": 1,
                        "auxiliary_loss_target_weighted_mean": 0.0,
                        "precision": "FP32_TF32off",
                    },
                    "NO_DROP_ALL": {
                        "task_ce": native - 0.1,
                        "loss_sum": native - 0.1,
                        "target_tokens": 1,
                        "batches": 1,
                        "auxiliary_loss_target_weighted_mean": 0.0,
                        "precision": "FP32_TF32off",
                    },
                }
                for layer in range(4):
                    policies[f"NO_DROP_L{layer}"] = {
                        "task_ce": native - 0.01 * (layer + 1),
                        "loss_sum": native - 0.01 * (layer + 1),
                        "target_tokens": 1,
                        "batches": 1,
                        "auxiliary_loss_target_weighted_mean": 0.0,
                        "precision": "FP32_TF32off",
                    }
                readouts.append(
                    {
                        "seed": seed,
                        "model_type": model_type,
                        "offset": offset,
                        "checkpoint_id": f"{seed}:{model_type}:{offset}",
                        "checkpoint_sha256": "synthetic",
                        "policies": policies,
                        "per_batch_rows": [],
                        "routing_rows": [],
                        "prediction_diagnostics": {
                            "prediction_top1_agreement": 1.0,
                            "mean_probability_distribution_js_divergence": 0.0,
                            "target_count": 1,
                            "native_rejected_subsets_by_batch": {},
                        },
                    }
                )

    _absolute, contrasts, layer_rows, _aggregates, _per_batch, _routing, _predictions = _contrast_tables(readouts)
    assert len(contrasts) == 10
    assert len(layer_rows) == 40
    for row in contrasts:
        assert abs(row["identity_residual_K"]) < 1e-12
    for seed in (7, 17, 29, 43, 61):
        start = next(row for row in contrasts if row["seed"] == seed and row["offset"] == 0)
        assert abs(start["identity_residual_J"]) < 1e-12


def test_model_state_comparison_treats_signed_zero_as_equal_values():
    left = {"weight": torch.tensor([-0.0, 1.0])}
    right = {"weight": torch.tensor([0.0, 1.0])}

    assert _assert_same_model_values(left, right, "test") == 1


def _routing_rows_for_mapping(layer_count, offset, *, forced_rejection=None):
    rows = []
    for policy, enabled_layers in _capacity_policy_mapping(layer_count, offset).items():
        for layer, capacity_enabled in enumerate(enabled_layers):
            per_expert = []
            for expert, proposed in enumerate((5, 5)):
                rejected = int(capacity_enabled)
                if forced_rejection == (policy, layer) and expert == 0:
                    rejected = 1
                accepted = proposed - rejected
                per_expert.append((proposed, accepted, rejected))
            layer_proposed = sum(row[0] for row in per_expert)
            layer_accepted = sum(row[1] for row in per_expert)
            layer_rejected = sum(row[2] for row in per_expert)
            for expert, (proposed, accepted, rejected) in enumerate(per_expert):
                rows.append(
                    {
                        "policy": policy,
                        "layer": layer,
                        "expert": expert,
                        "proposed_assignments": proposed,
                        "accepted_assignments": accepted,
                        "rejected_assignments": rejected,
                        "layer_proposed_assignments": layer_proposed,
                        "layer_accepted_assignments": layer_accepted,
                        "layer_rejected_assignments": layer_rejected,
                    }
                )
    return rows


def test_capacity_policy_mapping_drives_all_and_layer_local_flags():
    policies = _capacity_policy_mapping(layer_count=3, offset=0)

    assert policies["NATIVE"] == (True, True, True)
    assert policies["NO_DROP_ALL"] == (False, False, False)
    assert policies["NO_DROP_L1"] == (True, False, True)


def test_no_drop_all_fails_if_any_layer_rejects_assignments():
    rows = _routing_rows_for_mapping(
        layer_count=3,
        offset=0,
        forced_rejection=("NO_DROP_ALL", 2),
    )

    with pytest.raises(RuntimeError, match="Capacity-disabled layer rejected assignments"):
        _validate_routing_rows(rows, layer_count=3, offset=0)


def test_no_drop_l1_allows_rejection_in_untreated_layers():
    rows = _routing_rows_for_mapping(layer_count=3, offset=0)

    _validate_routing_rows(rows, layer_count=3, offset=0)


def test_no_drop_l1_fails_if_layer_one_rejects_assignments():
    rows = _routing_rows_for_mapping(
        layer_count=3,
        offset=0,
        forced_rejection=("NO_DROP_L1", 1),
    )

    with pytest.raises(RuntimeError, match="Capacity-disabled layer rejected assignments"):
        _validate_routing_rows(rows, layer_count=3, offset=0)


def test_routing_count_inconsistencies_and_unknown_policies_fail():
    rows = _routing_rows_for_mapping(layer_count=3, offset=0)
    rows[0]["accepted_assignments"] += 2

    with pytest.raises(RuntimeError, match="Accepted assignments exceed proposed"):
        _validate_routing_rows(rows, layer_count=3, offset=0)

    rows = _routing_rows_for_mapping(layer_count=3, offset=0)
    rows[0]["policy"] = "NO_DROP_L99"

    with pytest.raises(RuntimeError, match="Unknown capacity policy"):
        _validate_routing_rows(rows, layer_count=3, offset=0)


def test_invalid_routing_layer_indices_fail():
    rows = _routing_rows_for_mapping(layer_count=3, offset=0)
    rows[0]["layer"] = 3

    with pytest.raises(RuntimeError, match="Invalid routing layer index"):
        _validate_routing_rows(rows, layer_count=3, offset=0)
