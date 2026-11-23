import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/imp_postpruning_adaptation_v1"))
sys.path.insert(0, str(ROOT / "scripts/imp_adaptation_factorial_v1"))
sys.path.insert(0, str(ROOT / "src"))

import engine
import analyze
from diagnostics import _routes, collect_diagnostics, create_start_cache, probe_indices


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, values):
        return self.linear(values)


class _MoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 2
        self.experts = torch.nn.ModuleList([_Expert(), _Expert()])


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.moe = _MoE()


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.forward_calls = 0

    def forward(self, token_ids):
        self.forward_calls += 1
        features = torch.stack((token_ids.float(), token_ids.float() + 1), dim=-1)
        routes = []
        hidden = []
        for _block in self.blocks:
            hidden.append(features)
            ids = (token_ids.remainder(2)).unsqueeze(-1)
            routes.append(
                SimpleNamespace(
                    selected_expert_indices=ids,
                    selected_probabilities=torch.full_like(ids, 1.0, dtype=torch.float32),
                    accepted_mask=torch.ones_like(ids, dtype=torch.bool),
                )
            )
        return SimpleNamespace(routes=routes, pre_router_hidden_states=hidden)


class PostPruningDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.model = _ToyModel()
        self.model.train()
        self.batches = [
            (torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.zeros((2, 3), dtype=torch.long)),
            (torch.tensor([[6, 7, 8], [9, 10, 11]]), torch.zeros((2, 3), dtype=torch.long)),
        ]

    def test_route_collection_accumulates_batches_and_restores_mode(self):
        routes = _routes(self.model, self.batches, torch.device("cpu"))
        self.assertTrue(self.model.training)
        self.assertEqual(len(routes["ids"]), 2)
        self.assertEqual(tuple(routes["ids"][0].shape), (12, 1))
        self.assertEqual(tuple(routes["accepted"][1].shape), (12, 1))
        self.assertEqual(tuple(routes["gates"][0].shape), (12, 1))
        self.assertEqual(tuple(routes["inputs"][0].shape), (12, 2))
        self.assertTrue(torch.equal(routes["ids"][0][:6, 0], torch.tensor([0, 1, 0, 1, 0, 1])))

    def test_diagnostic_preserves_training_state_and_reports_grid_rows(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scaler = engine.create_grad_scaler(False)
        probe_batches = [batch for batch in self.batches]
        with tempfile.TemporaryDirectory(prefix="adaptation-v1-test-") as directory:
            cache = create_start_cache(
                model=self.model,
                seed=17,
                model_type="unpruned",
                probe_batches=probe_batches,
                probe_indices_value=[4, 9],
                device=torch.device("cpu"),
                path=Path(directory) / "cache.pt",
            )
            result = collect_diagnostics(
                model=self.model,
                optimizer=optimizer,
                scaler=scaler,
                cache=cache,
                probe_batches=probe_batches,
                seed=17,
                model_type="unpruned",
                condition="full",
                offset=0,
                device=torch.device("cpu"),
                data_position=1000,
            )
        self.assertTrue(self.model.training)
        self.assertTrue(result["diagnostic_rng_modes_gradients_buffers_parameters_optimizer_scaler_preserved"])
        self.assertEqual(len(result["routing_rows"]), 2 * 2 * 3)
        self.assertEqual(len(result["expert_rows"]), 2 * 2 * 2)
        self.assertEqual(len(result["shared_rows"]), 2 + 5)

    def test_probe_batch_selection_is_deterministic_and_unique(self):
        selected = probe_indices(17, 20)
        self.assertEqual(selected, probe_indices(17, 20))
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)

    def test_checkpoint_state_hash_is_device_independent(self):
        state = {"moment": torch.tensor([1.0, 2.0])}
        self.assertEqual(
            engine._structure_hash(state),
            engine._structure_hash({"moment": state["moment"].clone()}),
        )
        self.assertNotEqual(
            engine._structure_hash(state),
            engine._structure_hash({"moment": state["moment"] + 1}),
        )

    def test_prefix_verification_reference_is_dense_pre_event_ce(self):
        dense_ce = engine.historical_ce(17, "unpruned", 1000)
        sparse_ce = engine.historical_ce(17, "sparse", 1000)
        self.assertAlmostEqual(dense_ce, 2.0673522047200326, places=8)
        self.assertAlmostEqual(sparse_ce, 2.5169741710183007, places=8)
        self.assertGreater(sparse_ce - dense_ce, 0.4)

    def test_native_temporary_checkpoint_keeps_resume_metadata(self):
        with tempfile.TemporaryDirectory(prefix="adaptation-v1-checkpoint-test-") as directory:
            checkpoint = Path(directory) / "interruption.pt"
            temporary = checkpoint.with_suffix(".tmp")
            save = engine._save_wrapper(Path(directory), checkpoint, "router_frozen")
            save(
                {
                    "step": 1100,
                    "optimizer": {"state": {}, "param_groups": []},
                    "scaler": {},
                },
                temporary,
            )
            saved = engine._load_wrapper(temporary, map_location="cpu")
        self.assertEqual(saved["global_step"], 1100)
        self.assertEqual(saved["sampler_next_batch"], 1100)
        self.assertEqual(saved["study_condition"], "router_frozen")
        self.assertEqual(
            saved["trainability"],
            {"router": False, "experts": True, "shared": True},
        )

    def test_resume_discards_diagnostic_rows_after_checkpoint_boundary(self):
        with tempfile.TemporaryDirectory(prefix="adaptation-v1-resume-test-") as directory:
            path = Path(directory) / "study.jsonl"
            path.write_text(
                '{"global_step":1001}\n{"global_step":1100}\n{"global_step":1101}\n',
                encoding="utf-8",
            )
            engine._truncate_jsonl_after_step(path, 1100)
            rows = engine._read_jsonl(path)
        self.assertEqual([row["global_step"] for row in rows], [1001, 1100])

    def test_native_output_directories_are_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="adaptation-v1-output-test-") as directory:
            output = Path(directory) / "attempt"
            engine.ensure_native_output_directories(output)
            marker = output / "component_checkpoints" / "preserve.txt"
            marker.write_text("keep", encoding="utf-8")
            engine.ensure_native_output_directories(output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            for name in ("routing_stats", "gradient_stats", "checkpoints", "component_checkpoints"):
                self.assertTrue((output / name).is_dir())

    def test_cached_routes_allow_diagnostic_reconstruction_without_batch_forward(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scaler = engine.create_grad_scaler(False)
        with tempfile.TemporaryDirectory(prefix="adaptation-v1-cache-test-") as directory:
            cache = create_start_cache(
                model=self.model,
                seed=17,
                model_type="unpruned",
                probe_batches=self.batches,
                probe_indices_value=[4, 9],
                device=torch.device("cpu"),
                path=Path(directory) / "cache.pt",
            )
            calls_after_cache = self.model.forward_calls
            result = collect_diagnostics(
                model=self.model,
                optimizer=optimizer,
                scaler=scaler,
                cache=cache,
                probe_batches=[],
                seed=17,
                model_type="unpruned",
                condition="full",
                offset=0,
                device=torch.device("cpu"),
                data_position=1000,
                current_routes_override=cache["routes"],
            )
        self.assertEqual(self.model.forward_calls, calls_after_cache)
        self.assertTrue(result["diagnostic_rng_modes_gradients_buffers_parameters_optimizer_scaler_preserved"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for cross-device state comparison")
    def test_frozen_optimizer_state_comparison_is_device_independent(self):
        class _Router(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, device="cuda"))

        class _MoE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = _Router()

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.moe = _MoE()

        class _RouterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([_Block()])

        model = _RouterModel()
        parameter = model.blocks[0].moe.router.weight
        parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.state[parameter] = {
            "step": torch.tensor(3.0, device="cuda"),
            "exp_avg": torch.tensor([0.25], device="cuda"),
            "exp_avg_sq": torch.tensor([0.5], device="cuda"),
        }
        snapshot = {
            "blocks.0.moe.router.weight": {
                key: value.cpu().clone()
                for key, value in optimizer.state[parameter].items()
            }
        }
        self.assertTrue(
            engine.verify_frozen_optimizer(model, optimizer, snapshot, "router_frozen")
        )

    def test_trajectory_svg_accepts_flat_tuple_keyed_values(self):
        trajectories = {
            (seed, model_type, condition, step): 1.0 + step / 1000
            for seed in analyze.SEEDS
            for model_type in analyze.MODEL_TYPES
            for condition in analyze.CONDITIONS
            for step in analyze.GLOBAL_STEPS
        }

        svg = analyze._svg_trajectories(trajectories)

        self.assertTrue(svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"'))
        self.assertIn("Validation CE across the registered adaptation grid", svg)


if __name__ == "__main__":
    unittest.main()
