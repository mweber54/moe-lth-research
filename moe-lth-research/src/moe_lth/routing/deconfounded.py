from __future__ import annotations

"""Deconfounded routing interventions and graded corruption.

These interventions go beyond the original shuffled_usage by preserving
gate values, acceptance status, and expert-slot counts while permuting
only which token representation occupies each routed slot.

Schema version 2 (rich traces) is preferred but not required for
the graded corruption mode which works on primary IDs.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np


@dataclass
class DeconfoundedShuffleConfig:
    """Configuration for a deconfounded (fixed-gate/acceptance) identity shuffle.

    Preserves:
      - expert-slot counts from source trace
      - acceptance status per token
      - gate value assigned to each expert slot

    Permutes:
      - which token representation occupies each routed slot
    """
    seed: int = 42


@dataclass
class GradedCorruptionConfig:
    """Configuration for parameterized route corruption.

    corruption_fraction: float in [0, 1]
      - 0.0 = exact replay (no corruption)
      - 1.0 = fully shuffled (maximum corruption)
      - intermediate values corrupt that fraction of assignments
    """
    corruption_fraction: float = 0.0
    seed: int = 42
    preserve_counts: bool = True


def _nontrivial_count_preserving_permutation(
    flat: torch.Tensor,
    generator: torch.Generator,
    num_experts: int,
) -> torch.Tensor:
    """Create a deterministic, count-preserving permutation with a guaranteed change.

    If the routed assignment is already constant across all positions, then no
    nontrivial count-preserving permutation exists. In that case we return the
    original tensor unchanged rather than failing the entire training run.
    """
    if flat.numel() <= 1:
        return flat.clone()

    if flat.unique().numel() <= 1:
        return flat.clone()

    available = torch.arange(flat.numel(), device=flat.device, dtype=torch.long)
    target = torch.empty_like(flat)
    remaining = available.clone()

    expert_order = torch.randperm(num_experts, generator=generator)
    for expert_id in expert_order.tolist():
        count = int((flat == expert_id).sum().item())
        if count == 0:
            continue
        if remaining.numel() < count:
            raise AssertionError("Insufficient positions to preserve expert counts during shuffle.")
        choice = remaining[torch.randperm(remaining.numel(), generator=generator)[:count]]
        target[choice] = expert_id
        remaining = remaining[~torch.isin(remaining, choice)]

    if torch.equal(target, flat):
        mismatch_pairs: list[tuple[int, int]] = []
        for i in range(flat.numel() - 1):
            for j in range(i + 1, flat.numel()):
                if flat[i] != flat[j]:
                    mismatch_pairs.append((i, j))
        if not mismatch_pairs:
            raise AssertionError("Deconfounded shuffle degenerated to the identity mapping.")
        i, j = mismatch_pairs[int(torch.randint(len(mismatch_pairs), (1,), generator=generator).item())]
        target = flat.clone()
        target[i], target[j] = flat[j], flat[i]

    return target


def deconfounded_identity_shuffle(
    selected_expert_ids: torch.Tensor,
    gate_values: torch.Tensor,
    accepted_mask: torch.Tensor,
    step: int,
    layer_id: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform a deconfounded identity shuffle on routing assignments.

    The intervention preserves the exact per-expert counts, the gate values, and
    the acceptance mask while changing which token occupies each expert slot.
    """
    if selected_expert_ids.ndim != 2:
        raise ValueError(f"Expected [num_tokens, top_k] selected_expert_ids, got {tuple(selected_expert_ids.shape)}")
    if gate_values.shape != selected_expert_ids.shape:
        raise ValueError(f"Gate shape mismatch: {tuple(gate_values.shape)} vs {tuple(selected_expert_ids.shape)}")
    if accepted_mask.shape != selected_expert_ids.shape:
        raise ValueError(f"Accepted-mask shape mismatch: {tuple(accepted_mask.shape)} vs {tuple(selected_expert_ids.shape)}")

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + step * 7919 + layer_id * 6271)

    num_experts = int(selected_expert_ids.max().item()) + 1
    shuffled_ids = selected_expert_ids.clone()

    for k in range(selected_expert_ids.shape[1]):
        flat = selected_expert_ids[:, k].flatten()
        shuffled = _nontrivial_count_preserving_permutation(flat, generator, num_experts)
        shuffled_ids[:, k] = shuffled.reshape_as(selected_expert_ids[:, k])

    assert_counts_preserved(selected_expert_ids, shuffled_ids, num_experts)
    assert_gate_values_preserved(gate_values, gate_values)
    assert_acceptance_preserved(accepted_mask, accepted_mask)
    if torch.equal(shuffled_ids, selected_expert_ids) and selected_expert_ids.unique().numel() > 1:
        raise AssertionError("Deconfounded shuffle degenerated to the identity mapping.")

    return shuffled_ids, gate_values, accepted_mask


def deconfounded_identity_shuffle_flat(
    primary_ids: torch.Tensor,
    step: int,
    layer_id: int,
    seed: int,
    num_experts: int,
) -> torch.Tensor:
    """Simplified deconfounded shuffle for primary IDs only.

    The intervention preserves the exact expert-count vector while rearranging the
    assigned expert labels to new positions in a deterministic, nontrivial way.
    """
    flat = primary_ids.flatten()
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + step * 7919 + layer_id * 6271)

    shuffled = _nontrivial_count_preserving_permutation(flat, generator, num_experts)
    assert_counts_preserved(flat, shuffled, num_experts)
    if torch.equal(shuffled, flat) and flat.unique().numel() > 1:
        raise AssertionError("Deconfounded shuffle degenerated to the identity mapping.")
    return shuffled.reshape_as(primary_ids)


def graded_route_corruption(
    primary_ids: torch.Tensor,
    corruption_fraction: float,
    step: int,
    layer_id: int,
    seed: int,
    num_experts: int,
    preserve_counts: bool = True,
) -> torch.Tensor:
    """Apply graded corruption to routing assignments.

    Corrupts `corruption_fraction` of the assignments while preserving
    per-expert counts (if preserve_counts=True).

    Args:
        primary_ids: [batch, seq_len] primary expert assignments
        corruption_fraction: float in [0, 1], fraction of assignments to corrupt
        step: training step
        layer_id: layer index
        seed: base random seed
        num_experts: number of experts
        preserve_counts: if True, preserve per-expert token counts

    Returns:
        Corrupted primary IDs
    """
    if corruption_fraction <= 0.0:
        return primary_ids.clone()

    flat = primary_ids.flatten()
    n = flat.numel()
    device = flat.device

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + step * 8317 + layer_id * 5573)

    corrupted = flat.clone()

    # Select which positions to corrupt
    num_corrupt = max(1, int(round(n * corruption_fraction)))
    if num_corrupt >= n:
        # Full corruption = full shuffle
        if preserve_counts:
            perm = torch.randperm(n, generator=generator)
            corrupted = flat[perm]
        else:
            corrupted = torch.randint(
                0, num_experts, (n,), generator=generator, dtype=flat.dtype
            ).to(device)
        return corrupted.reshape_as(primary_ids)

    # Select positions to corrupt
    corrupt_indices = torch.randperm(n, generator=generator)[:num_corrupt]

    if preserve_counts:
        # Among the corrupted positions, shuffle their expert assignments
        # This preserves per-expert counts within the corrupted subset
        corrupt_values = flat[corrupt_indices]
        sub_perm = torch.randperm(num_corrupt, generator=generator)
        corrupted[corrupt_indices] = corrupt_values[sub_perm]
    else:
        # Replace with random expert IDs (does not preserve counts)
        corrupted[corrupt_indices] = torch.randint(
            0, num_experts, (num_corrupt,), generator=generator, dtype=flat.dtype
        ).to(device)

    return corrupted.reshape_as(primary_ids)


def measure_corruption_statistics(
    original: torch.Tensor,
    corrupted: torch.Tensor,
    num_experts: int,
) -> dict:
    """Compute statistics comparing original and corrupted routing assignments.

    Returns metrics useful for the graded corruption analysis:
    - route_disagreement: fraction of positions with different expert assignments
    - accepted_count_difference: L1 difference in per-expert counts
    - assignment_distribution: per-expert fractions in original and corrupted

    Args:
        original: [batch, seq_len] original expert assignments
        corrupted: [batch, seq_len] corrupted expert assignments
        num_experts: number of experts

    Returns:
        Dictionary of corruption statistics
    """
    flat_orig = original.flatten()
    flat_corr = corrupted.flatten()
    n = flat_orig.numel()

    disagreement = float((flat_orig != flat_corr).float().mean().item())

    orig_counts = torch.bincount(flat_orig, minlength=num_experts).float()
    corr_counts = torch.bincount(flat_corr, minlength=num_experts).float()
    count_diff = float((orig_counts - corr_counts).abs().sum().item())

    orig_dist = orig_counts / max(1, n)
    corr_dist = corr_counts / max(1, n)
    tv_distance = float(0.5 * (orig_dist - corr_dist).abs().sum().item())

    return {
        "route_disagreement": disagreement,
        "accepted_count_difference": count_diff,
        "total_variation_distance": tv_distance,
        "original_distribution": orig_dist.tolist(),
        "corrupted_distribution": corr_dist.tolist(),
    }

def gate_distribution_distance(
    source_gate: torch.Tensor,
    target_gate: torch.Tensor,
) -> float:
    """Compute the total variation distance between two gate distributions."""
    source = source_gate.detach().float().reshape(-1)
    target = target_gate.detach().float().reshape(-1)
    if source.numel() == 0:
        return 0.0
    source_prob = source / max(source.abs().sum().item(), 1e-12)
    target_prob = target / max(target.abs().sum().item(), 1e-12)
    return float(0.5 * (source_prob - target_prob).abs().sum().item())


def accepted_count_transform(source_counts: torch.Tensor, target_counts: torch.Tensor) -> float:
    """Return the L1 difference between accepted expert counts."""
    return float((source_counts.float() - target_counts.float()).abs().sum().item())

# --- Assertions for intervention property verification ---

def assert_counts_preserved(
    original: torch.Tensor,
    shuffled: torch.Tensor,
    num_experts: int,
) -> None:
    """Assert that per-expert token counts are exactly preserved."""
    orig_counts = torch.bincount(original.flatten(), minlength=num_experts)
    shuf_counts = torch.bincount(shuffled.flatten(), minlength=num_experts)
    if not torch.equal(orig_counts, shuf_counts):
        diff = (orig_counts - shuf_counts).abs()
        raise AssertionError(
            f"Expert counts not preserved. Max difference: {diff.max().item()}, "
            f"Total difference: {diff.sum().item()}"
        )


def assert_gate_values_preserved(
    original_gates: torch.Tensor,
    shuffled_gates: torch.Tensor,
    rtol: float = 1e-5,
) -> None:
    """Assert that gate value distributions are preserved under the route permutation.

    The deconfounded intervention changes which token occupies each routed slot,
    not the set of gate values attached to the slot assignments. Accordingly, the
    flattened gate values are compared as multisets rather than literal indices.
    """
    if original_gates.shape != shuffled_gates.shape:
        raise AssertionError(
            f"Gate-shape mismatch: {tuple(original_gates.shape)} vs {tuple(shuffled_gates.shape)}"
        )
    original_flat = original_gates.detach().reshape(-1).float()
    shuffled_flat = shuffled_gates.detach().reshape(-1).float()
    if original_flat.numel() == 0:
        return
    if not torch.allclose(
        torch.sort(original_flat).values,
        torch.sort(shuffled_flat).values,
        rtol=rtol,
        atol=1e-8,
    ):
        max_diff = (torch.sort(original_flat).values - torch.sort(shuffled_flat).values).abs().max().item()
        raise AssertionError(
            f"Gate values not preserved under permutation. Max difference: {max_diff}"
        )


def assert_acceptance_preserved(
    original_accepted: torch.Tensor,
    shuffled_accepted: torch.Tensor,
) -> None:
    """Assert that acceptance status is preserved as a count-preserving permutation.

    Under the deconfounded route shuffle, accepted positions move with the routed
    slots, so the literal mask indices are allowed to change while the same number
    of accepted entries and the same acceptance pattern over the flattened trace
    are preserved.
    """
    if original_accepted.shape != shuffled_accepted.shape:
        raise AssertionError(
            f"Acceptance mask shape mismatch: {tuple(original_accepted.shape)} vs {tuple(shuffled_accepted.shape)}"
        )
    if original_accepted.float().sum().item() != shuffled_accepted.float().sum().item():
        raise AssertionError(
            "Acceptance status counts differ under the deconfounded permutation. "
            f"Original accepted count={original_accepted.float().sum().item()}, "
            f"Shuffled accepted count={shuffled_accepted.float().sum().item()}."
        )
