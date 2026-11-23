import json
import hashlib

import pytest
import torch

from scripts.imp_extended_recovery.pipeline import _is_complete, _write_absolute_jsonl


def test_absolute_curve_copy_adds_step_and_cross_entropy_names(tmp_path):
    source = tmp_path / "metrics.jsonl"
    destination = tmp_path / "metrics_absolute.jsonl"
    source.write_text('{"step": 0, "loss": 1.7}\n{"step": 2500, "loss": 1.2}\n')

    _write_absolute_jsonl(source, destination)

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["absolute_step"] for row in rows] == [2500, 5000]
    assert [row["validation_cross_entropy"] for row in rows] == [1.7, 1.2]
    assert '"absolute_step"' not in source.read_text()


def test_complete_extension_requires_matching_identity_and_step_checkpoint(tmp_path):
    identity = {"num_experts": 8, "seed": 7, "condition": "sparse", "sparsity": 0.8, "mask_hash": "mask"}
    output = tmp_path / "e8/seed_7/final_s0.80"
    (output / "checkpoints").mkdir(parents=True)
    checkpoint_path = output / "checkpoints/step_5000.pt"
    torch.save({"step": 5000, "mask_hash": "mask", "identity": identity}, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    (output / "metrics_absolute.jsonl").write_text('{"absolute_step": 5000}\n')
    (output / "extension_manifest.json").write_text(json.dumps({
        "identity": identity,
        "step_5000_checkpoint_sha256": checkpoint_hash,
    }))

    assert _is_complete(output, identity)
    with pytest.raises(RuntimeError, match="identity differs"):
        _is_complete(output, {**identity, "seed": 17})