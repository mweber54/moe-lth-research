from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


@torch.no_grad()
def router_vector_similarities(model: torch.nn.Module) -> list[dict]:
    results = []
    for layer_id, block in enumerate(model.blocks):
        vectors = F.normalize(block.moe.router.projection.weight.detach().cpu(), dim=-1)
        similarity = vectors @ vectors.T
        for first in range(similarity.shape[0]):
            for second in range(first + 1, similarity.shape[0]):
                results.append(
                    {
                        "layer_id": layer_id,
                        "first_expert": first,
                        "second_expert": second,
                        "cosine_similarity": float(similarity[first, second]),
                    }
                )
    return results


def hidden_state_separability(hidden_states: torch.Tensor, expert_ids: torch.Tensor, max_samples: int = 5000) -> dict:
    features = hidden_states.detach().cpu().reshape(-1, hidden_states.shape[-1]).numpy()
    labels = expert_ids.detach().cpu().reshape(-1).numpy()
    if features.shape[0] > max_samples:
        indices = np.linspace(0, features.shape[0] - 1, max_samples, dtype=int)
        features, labels = features[indices], labels[indices]
    if np.unique(labels).size < 2:
        return {"silhouette_score": 0.0, "linear_probe_accuracy": 1.0}
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.25, random_state=0, stratify=labels
    )
    probe = LogisticRegression(max_iter=200).fit(train_x, train_y)
    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "linear_probe_accuracy": float(probe.score(test_x, test_y)),
    }

