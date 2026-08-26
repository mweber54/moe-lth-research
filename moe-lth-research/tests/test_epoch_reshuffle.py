import torch

from moe_lth.data import build_dataloaders


def test_build_dataloaders_can_refresh_epoch_order():
    data_config = {
        "path": None,
        "train_fraction": 0.9,
        "seq_len": 8,
        "batch_size": 2,
        "validation_blocks": 1,
        "tokenizer": "byte",
    }

    train_loader, _ = build_dataloaders(data_config, batch_size=2, seed=11, reshuffle_each_epoch=True)

    first_epoch = list(iter(train_loader))
    second_epoch = list(iter(train_loader))

    first_ids = first_epoch[0][0]
    second_ids = second_epoch[0][0]

    assert first_ids.shape == second_ids.shape
    assert not torch.equal(first_ids, second_ids)
