import hashlib
from pathlib import Path

import numpy as np
import torch

from moe_lth.routing.deconfounded import deconfounded_identity_shuffle
from moe_lth.routing.interventions import RoutingController
from moe_lth.routing.rich_trace import RichRouteHistory
from moe_lth.routing.route_history import RouteHistory, load_validation_route_batches, save_validation_routes
from moe_lth.training.train import build_controller, build_validation_overrides


def test_random_every_step_is_balanced_and_reproducible():
    tokens = torch.arange(16).reshape(2, 8)
    controller = RoutingController("random_every_step", 1, 4, 7)
    first = controller.overrides(tokens, 3)[0]
    second = controller.overrides(tokens, 3)[0]
    later = controller.overrides(tokens, 4)[0]
    assert torch.equal(first, second)
    assert not torch.equal(first, later)
    assert torch.equal(torch.bincount(first.flatten(), minlength=4), torch.tensor([4, 4, 4, 4]))


def test_replay_swap_and_shuffle_preserve_expected_properties():
    tokens = torch.zeros((2, 4), dtype=torch.long)
    routes = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    history = RouteHistory()
    history.record(1, 0, routes)

    replay = RoutingController("replay", 1, 2, 7, history=history).overrides(tokens, 1)[0]
    swapped = RoutingController("swapped", 1, 2, 7, history=history, swap_pairs=[[0, 1]]).overrides(tokens, 1)[0]
    shuffled = RoutingController("shuffled_usage", 1, 2, 7, history=history).overrides(tokens, 1)[0]
    assert torch.equal(replay, routes)
    assert torch.equal(swapped, 1 - routes)
    assert torch.equal(torch.bincount(shuffled.flatten()), torch.bincount(routes.flatten()))
    assert not torch.equal(shuffled, routes)


def test_shuffled_usage_is_reproducible_and_changes_by_step():
    routes = torch.arange(32).remainder(4).reshape(4, 8)
    controller = RoutingController("shuffled_usage", 1, 4, 7)
    first = controller.transform_replayed(routes, 1, 0)
    repeated = controller.transform_replayed(routes, 1, 0)
    later = controller.transform_replayed(routes, 2, 0)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, later)
    assert torch.equal(torch.bincount(first.flatten()), torch.bincount(routes.flatten()))
    assert torch.equal(torch.bincount(later.flatten()), torch.bincount(routes.flatten()))


def test_layer_specific_swaps_only_affect_target_layer():
    routes = torch.tensor([[0, 1, 2, 3]])
    controller = RoutingController(
        "swapped",
        2,
        4,
        7,
        layer_swap_pairs={1: [[0, 3]]},
    )

    assert torch.equal(controller.transform_replayed(routes, 1, 0), routes)
    assert torch.equal(controller.transform_replayed(routes, 1, 1), torch.tensor([[3, 1, 2, 0]]))


def test_cyclic_swap_shift_wraps_expert_ids():
    routes = torch.tensor([[0, 1, 2, 3]])
    controller = RoutingController("swapped", 1, 4, 7, cyclic_shift=1)

    assert torch.equal(controller.transform_replayed(routes, 1, 0), torch.tensor([[1, 2, 3, 0]]))


def test_validation_route_snapshot_round_trip(tmp_path):
    routes = torch.tensor([[0, 1], [1, 0]])
    path = tmp_path / "routes.npz"
    save_validation_routes(path, 10, [[routes]])
    loaded = load_validation_route_batches(path, torch.device("cpu"))
    assert torch.equal(loaded[0][0], routes)


def test_graded_corruption_preserves_per_expert_counts_and_forwards_config():
    routes = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    controller = RoutingController("graded_corruption", 1, 4, 7, corruption_fraction=0.5)
    transformed = controller.transform_replayed(routes, 1, 0)
    assert torch.equal(torch.bincount(transformed.flatten(), minlength=4), torch.bincount(routes.flatten(), minlength=4))
    assert not torch.equal(transformed, routes)

    config = {
        "model": {"num_layers": 1, "num_experts": 4},
        "routing": {"mode": "graded_corruption", "corruption_fraction": 0.25},
    }
    built = build_controller(config)
    assert built.corruption_fraction == 0.25


def test_deconfounded_shuffle_is_nontrivial_and_preserves_invariants():
    original = torch.tensor(
        [[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]],
        dtype=torch.long,
    )
    gate = torch.tensor(
        [[0.9, 0.1, 0.7, 0.3], [0.6, 0.4, 0.8, 0.2], [0.5, 0.5, 0.1, 0.9], [0.8, 0.2, 0.6, 0.4]],
        dtype=torch.float32,
    )
    accepted = torch.tensor(
        [[True, True, True, True], [True, False, True, False], [False, True, True, False], [True, True, False, False]],
        dtype=torch.bool,
    )

    shuffled, preserved_gate, preserved_accepted = deconfounded_identity_shuffle(
        original,
        gate,
        accepted,
        step=5,
        layer_id=2,
        seed=7,
    )

    original_hash = hashlib.sha256(original.cpu().numpy().tobytes()).hexdigest()
    shuffled_hash = hashlib.sha256(shuffled.cpu().numpy().tobytes()).hexdigest()
    assert original_hash != shuffled_hash, "deconfounded shuffle degenerated to the identity mapping"
    assert not torch.equal(shuffled, original), "deconfounded shuffle must change the assignment tensor"
    assert torch.equal(torch.bincount(original.flatten(), minlength=2), torch.bincount(shuffled.flatten(), minlength=2))
    assert torch.equal(gate, preserved_gate)
    assert torch.equal(accepted, preserved_accepted)

    repeated, _, _ = deconfounded_identity_shuffle(original, gate, accepted, step=5, layer_id=2, seed=7)
    assert torch.equal(shuffled, repeated), "deconfounded shuffle must be deterministic"


def test_deconfounded_shuffle_handles_constant_route_traces():
    original = torch.full((2, 3), 2, dtype=torch.long)
    gate = torch.full((2, 3), 0.5, dtype=torch.float32)
    accepted = torch.ones((2, 3), dtype=torch.bool)

    shuffled, preserved_gate, preserved_accepted = deconfounded_identity_shuffle(
        original,
        gate,
        accepted,
        step=12,
        layer_id=3,
        seed=7,
    )

    assert torch.equal(shuffled, original)
    assert torch.equal(gate, preserved_gate)
    assert torch.equal(accepted, preserved_accepted)


def test_deconfounded_shuffle_tracks_gate_and_acceptance_with_the_permutation():
    original = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    gate = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
    accepted = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    shuffled, shuffled_gate, shuffled_accepted = deconfounded_identity_shuffle(
        original,
        gate,
        accepted,
        step=3,
        layer_id=1,
        seed=7,
    )

    assert not torch.equal(shuffled, original)
    assert torch.equal(torch.bincount(shuffled.flatten(), minlength=2), torch.bincount(original.flatten(), minlength=2))
    assert torch.equal(shuffled_gate, gate)
    assert torch.equal(shuffled_accepted, accepted)


def test_route_history_load_accepts_rich_trace_artifacts(tmp_path: Path):
    route_history = RichRouteHistory()
    route_history.metadata["num_experts"] = 2
    selected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    gate = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float16)
    accepted = np.array([[True, True], [False, True]], dtype=bool)
    route_history.traces[(1, 0)] = type("Trace", (), {
        "selected_expert_ids": selected,
        "gate_values": gate,
        "accepted_mask": accepted,
        "batch_indices": np.array([0, 0], dtype=np.int32),
        "seq_positions": np.array([0, 1], dtype=np.int16),
        "step": 1,
        "layer_id": 0,
    })()
    path = tmp_path / "rich_history.npz"
    route_history.save(path)

    loaded = RouteHistory.load(path)
    assert loaded.routes[(1, 0)].shape == (1, 2)
    assert np.array_equal(loaded.routes[(1, 0)], np.array([[0, 1]], dtype=np.int64))


def test_build_controller_uses_rich_history_for_deconfounded_shuffle(tmp_path: Path):
    route_history = RichRouteHistory()
    route_history.metadata["num_experts"] = 2
    selected = np.array([[0], [1]], dtype=np.uint8)
    gate = np.array([[0.7], [0.3]], dtype=np.float16)
    accepted = np.array([[True], [True]], dtype=bool)
    route_history.traces[(1, 0)] = type("Trace", (), {
        "selected_expert_ids": selected,
        "gate_values": gate,
        "accepted_mask": accepted,
        "batch_indices": np.array([0, 1], dtype=np.int32),
        "seq_positions": np.array([0, 0], dtype=np.int16),
        "step": 1,
        "layer_id": 0,
    })()
    path = tmp_path / "rich_history.npz"
    route_history.save(path)

    config = {
        "model": {"num_layers": 1, "num_experts": 2},
        "routing": {"mode": "deconfounded_shuffle", "replay_path": str(path)},
        "seed": 7,
    }
    controller = build_controller(config)

    assert hasattr(controller.history, "traces")
    assert controller.history.get(1, 0).selected_expert_ids.shape == (2, 1)
    override = controller._layer_override(torch.zeros((2, 1), dtype=torch.long), 1, 0)
    assert hasattr(override, "expert_ids")
    assert hasattr(override, "gate_values")


def test_build_validation_overrides_falls_back_to_nearest_saved_step(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    earlier = logs_dir / "validation_routes_step_25.npz"
    np.savez_compressed(earlier, checkpoint_25_batch_0_layer_0=np.array([[0, 1], [1, 0]], dtype=np.int64))

    config = {
        "model": {"num_layers": 1, "num_experts": 2},
        "routing": {"mode": "deconfounded_shuffle", "replay_path": str(logs_dir / "rich_train_route_history.npz")},
        "seed": 7,
    }
    controller = RoutingController("deconfounded_shuffle", 1, 2, 7)

    overrides = build_validation_overrides(config, 125, torch.device("cpu"), controller)
    assert overrides is not None
    assert len(overrides) == 1
    assert overrides[0][0].shape == (2, 2)
