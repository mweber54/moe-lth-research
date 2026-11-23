import json

from scripts.imp_extended_recovery.run import START_STEP, _annotate_jsonl


def test_annotate_jsonl_adds_absolute_steps(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 0, "loss": 2.0}\n{"step": 50, "loss": 1.5}\n')

    _annotate_jsonl(path, START_STEP)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["absolute_step"] for row in rows] == [2500, 2550]
    assert [row["loss"] for row in rows] == [2.0, 1.5]
    assert [row["validation_cross_entropy"] for row in rows] == [2.0, 1.5]