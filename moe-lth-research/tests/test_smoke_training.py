from pathlib import Path

import torch

from moe_lth.config import load_config
from moe_lth.data import build_dataloaders
from moe_lth.training.train import train_from_config
from moe_lth.utils import resolve_autocast_dtype, resolve_data_seed


def test_one_step_training_writes_checkpoint_and_routes(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["training"]["steps"] = 1
    config["training"]["checkpoint_steps"] = [0, 1]
    config["output_dir"] = str(tmp_path / "run")
    stale_log = tmp_path / "run" / "logs" / "train_metrics.jsonl"
    stale_log.parent.mkdir(parents=True)
    stale_log.write_text('{"stale": true}\n', encoding="utf-8")
    stale_checkpoint = tmp_path / "run" / "checkpoints" / "step_99.pt"
    stale_checkpoint.parent.mkdir(parents=True)
    stale_checkpoint.write_text("stale", encoding="utf-8")
    unrelated = tmp_path / "run" / "notes.txt"
    unrelated.write_text("keep me", encoding="utf-8")

    summary = train_from_config(config)

    assert summary["steps"] == 1
    assert (tmp_path / "run" / "checkpoints" / "step_1.pt").exists()
    assert (tmp_path / "run" / "logs" / "train_route_history.npz").exists()
    assert not stale_checkpoint.exists()
    assert '"stale"' not in stale_log.read_text(encoding="utf-8")
    assert unrelated.read_text(encoding="utf-8") == "keep me"


def test_data_order_seed_can_be_decoupled_from_model_seed():
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["seed"] = 17
    config["training"]["data_seed"] = 7
    first_loader, _ = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    config["seed"] = 29
    second_loader, _ = build_dataloaders(
        config["data"], int(config["training"]["batch_size"]), resolve_data_seed(config)
    )
    first_tokens, _ = next(iter(first_loader))
    second_tokens, _ = next(iter(second_loader))
    assert torch.equal(first_tokens, second_tokens)


def test_cpu_fp16_falls_back_to_fp32_autocast():
    assert resolve_autocast_dtype("fp16", torch.device("cpu")) is None
    assert resolve_autocast_dtype("fp16", torch.device("cuda")) == torch.float16
