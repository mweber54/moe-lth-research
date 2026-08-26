from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


KEY_PATTERN = re.compile(r"checkpoint_(\d+)_batch_(\d+)_layer_(\d+)")


def load_validation_routes(path: str) -> dict[tuple[int, int, int], np.ndarray]:
    routes = {}
    with np.load(path) as archive:
        for name in archive.files:
            match = KEY_PATTERN.fullmatch(name)
            if match:
                routes[tuple(int(group) for group in match.groups())] = archive[name]
    return routes


def routing_agreement(first_path: str, second_path: str) -> dict:
    first = load_validation_routes(first_path)
    second = load_validation_routes(second_path)
    per_layer: dict[int, list[float]] = defaultdict(list)
    for (_, batch_id, layer_id), first_routes in first.items():
        candidates = [
            values
            for (_, other_batch_id, other_layer_id), values in second.items()
            if other_batch_id == batch_id and other_layer_id == layer_id
        ]
        if candidates and candidates[0].shape == first_routes.shape:
            per_layer[layer_id].append(float(np.mean(first_routes == candidates[0])))
    return {
        "per_layer": {
            str(layer_id): float(np.mean(agreements))
            for layer_id, agreements in sorted(per_layer.items())
        },
        "overall": float(np.mean([value for values in per_layer.values() for value in values]))
        if per_layer
        else 0.0,
    }

