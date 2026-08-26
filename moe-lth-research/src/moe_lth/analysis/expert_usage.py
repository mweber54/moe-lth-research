from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from moe_lth.utils import read_jsonl


def usage_summary(usage_log: str) -> dict:
    records = read_jsonl(usage_log)
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        grouped[(int(record["step"]), int(record["layer_id"]))].append(float(record["usage_fraction"]))

    summaries = []
    for (step, layer_id), usage in sorted(grouped.items()):
        values = np.asarray(usage, dtype=np.float64)
        positive = values[values > 0]
        entropy = float(-(positive * np.log(positive)).sum())
        summaries.append(
            {
                "step": step,
                "layer_id": layer_id,
                "entropy": entropy,
                "normalized_entropy": entropy / math.log(max(2, len(values))),
                "coefficient_of_variation": float(values.std() / max(values.mean(), 1e-12)),
                "max_to_min": float(values.max() / max(values.min(), 1e-12)),
                "dead_experts": int((values == 0).sum()),
            }
        )
    return {"layers_over_time": summaries}

