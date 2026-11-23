from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/imp_recovery_order_stability_v1/run.py"
SPEC = importlib.util.spec_from_file_location("imp_recovery_order_stability_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_order_generation_is_deterministic_and_segment_local():
    first = study._generate_orders()
    second = study._generate_orders()
    for order in study.ORDER_NAMES:
        for segment in ("segment_0", "segment_1"):
            assert np.array_equal(first[order][segment], second[order][segment])
            start = 0 if segment == "segment_0" else study.SEGMENT_SIZE
            expected = np.arange(start, start + study.SEGMENT_SIZE)
            assert np.array_equal(np.sort(first[order][segment]), expected)
            assert not np.array_equal(first[order][segment], expected)


def test_permutations_preserve_whole_batch_identity_and_multiplicity():
    batches = [
        (torch.tensor([[index, index + 1]]), torch.tensor([[index + 1, index + 2]]))
        for index in range(study.TOTAL_STEPS)
    ]
    hashes = [study._batch_content_hash(batch) for batch in batches]
    for order in study._generate_orders().values():
        for segment, start in (("segment_0", 0), ("segment_1", study.SEGMENT_SIZE)):
            indices = order[segment]
            assert sorted(hashes[int(index)] for index in indices) == sorted(
                hashes[start : start + study.SEGMENT_SIZE]
            )
            permuted = [batches[int(index)] for index in indices]
            assert all(
                torch.equal(actual[0], batches[int(index)][0])
                and torch.equal(actual[1], batches[int(index)][1])
                for actual, index in zip(permuted, indices)
            )


def test_permutation_generation_does_not_change_global_rng():
    import random

    random.seed(41)
    np.random.seed(42)
    torch.manual_seed(43)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    study._generate_orders()
    after_numpy = np.random.get_state()
    assert random.getstate() == before_python
    assert before_numpy[0] == after_numpy[0]
    assert np.array_equal(before_numpy[1], after_numpy[1])
    assert before_numpy[2:] == after_numpy[2:]
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_rank_helper_uses_average_ranks_for_ties():
    assert study._rank({7: 1.0, 17: 1.0, 29: 3.0}) == {7: 1.5, 17: 1.5, 29: 3.0}
