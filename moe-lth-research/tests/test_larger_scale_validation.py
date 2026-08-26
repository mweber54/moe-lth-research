from moe_lth.config import load_config
from moe_lth.models import TinyMoELanguageModel


def test_p08_config_is_meaningfully_larger_and_reshuffled():
    config = load_config("configs/revision_larger_scale.yaml")
    model = TinyMoELanguageModel(config["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())

    assert parameters > 5 * 30_000_000
    assert config["model"]["num_layers"] == 8
    assert config["model"]["num_experts"] == 16
    assert config["model"]["top_k"] == 1
    assert config["data"]["reshuffle_each_epoch"] is True
    assert config["pruning"]["rewind_fractions"] == [0.0, 0.5, 0.8]
    assert config["training"]["checkpoint_steps"] == [0, 1250, 2000, 2500]
    assert config["training"]["save_optimizer"] is False
