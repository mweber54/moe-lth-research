from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64)
    second = second.astype(np.float64)
    first = first / max(first.sum(), 1.0)
    second = second / max(second.sum(), 1.0)
    middle = 0.5 * (first + second)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / np.maximum(right[mask], 1e-12))))

    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


def expert_token_histograms(
    token_batches: list[np.ndarray],
    route_batches: list[np.ndarray],
    num_experts: int,
    vocab_size: int,
) -> dict[int, np.ndarray]:
    histograms = defaultdict(lambda: np.zeros(vocab_size, dtype=np.int64))
    for tokens, routes in zip(token_batches, route_batches, strict=True):
        for expert_id in range(num_experts):
            selected_tokens = tokens[routes == expert_id]
            histograms[expert_id] += np.bincount(selected_tokens, minlength=vocab_size)
    return dict(histograms)


def pairwise_token_distribution_js(histograms: dict[int, np.ndarray]) -> list[dict]:
    results = []
    expert_ids = sorted(histograms)
    for index, first in enumerate(expert_ids):
        for second in expert_ids[index + 1 :]:
            results.append(
                {
                    "first_expert": first,
                    "second_expert": second,
                    "jensen_shannon": jensen_shannon(histograms[first], histograms[second]),
                }
            )
    return results


def context_embedding_cosine_similarity(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> np.ndarray:
    flat_hidden = hidden_states.detach().cpu().reshape(-1, hidden_states.shape[-1])
    flat_experts = expert_ids.detach().cpu().reshape(-1)
    means = []
    for expert_id in range(num_experts):
        selected = flat_hidden[flat_experts == expert_id]
        means.append(selected.mean(dim=0) if selected.numel() else torch.zeros(flat_hidden.shape[-1]))
    normalized = F.normalize(torch.stack(means), dim=-1)
    return (normalized @ normalized.T).numpy()
