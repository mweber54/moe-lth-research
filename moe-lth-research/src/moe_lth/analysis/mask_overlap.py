from __future__ import annotations

from itertools import combinations

from moe_lth.pruning.masks import load_masks, mask_jaccard


def pairwise_mask_overlap(mask_paths: list[str]) -> list[dict]:
    results = []
    loaded = {path: load_masks(path) for path in mask_paths}
    for first, second in combinations(mask_paths, 2):
        results.append({"first": first, "second": second, "jaccard": mask_jaccard(loaded[first], loaded[second])})
    return results

