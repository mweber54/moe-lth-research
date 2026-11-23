import csv
from pathlib import Path

import numpy as np
import torch

root = Path("results/moe_group_gradient_consistency_v1")
rows = []

for seed in (7, 17, 29):
    for order in ("A", "B"):
        for step in (250, 500):
            path = root / f"gradients_seed{seed}_{order}_{step}.pt"
            d = torch.load(path, map_location="cpu", weights_only=False)

            occurrences = np.asarray(d["accepted_occurrences"])
            batches = np.asarray(d["contributing_batches"])

            eligible_total = 0
            agree_total = 0
            oppose_total = 0
            neutral_total = 0
            signed_x, signed_y = [], []

            for j in range(len(d["names"])):
                scores = []
                valid = []

                for domain in (0, 1):
                    signs = d["sign_sum"][domain][j].float()
                    resolved = d["resolved_batches"][domain][j].float()

                    score = signs / resolved.clamp_min(1)
                    eligible = resolved >= 8
                    if batches[domain, j] < 8 or occurrences[domain, j] < 32:
                        eligible = torch.zeros_like(eligible, dtype=torch.bool)

                    scores.append(score.numpy().ravel())
                    valid.append(eligible.numpy().ravel())

                mask = valid[0] & valid[1]
                x, y = scores[0][mask], scores[1][mask]

                signed_x.append(x)
                signed_y.append(y)

                product = x * y
                agree_total += int(np.count_nonzero(product > 0))
                oppose_total += int(np.count_nonzero(product < 0))
                neutral_total += int(np.count_nonzero(product == 0))
                eligible_total += len(x)

            x = np.concatenate(signed_x)
            y = np.concatenate(signed_y)

            corr = (
                float(np.corrcoef(x, y)[0, 1])
                if len(x) > 1 and x.std() > 0 and y.std() > 0
                else float("nan")
            )

            row = {
                "seed": seed,
                "order": order,
                "step": step,
                "eligible_weights": eligible_total,
                "same_direction_fraction": agree_total / eligible_total,
                "opposite_direction_fraction": oppose_total / eligible_total,
                "neutral_fraction": neutral_total / eligible_total,
                "signed_score_correlation": corr,
                "strong_same_fraction": float(
                    np.mean((x * y > 0) & (np.abs(x) >= .8) & (np.abs(y) >= .8))
                ),
                "strong_opposite_fraction": float(
                    np.mean((x * y < 0) & (np.abs(x) >= .8) & (np.abs(y) >= .8))
                ),
            }

            rows.append(row)
            print(
                f"seed{seed}_{order}_{step}: "
                f"same={row['same_direction_fraction']:.3f}, "
                f"opposite={row['opposite_direction_fraction']:.3f}, "
                f"corr={corr:.3f}, "
                f"strong_opposite={row['strong_opposite_fraction']:.3f}",
                flush=True,
            )

out = root / "domain_agreement.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nMean across 12 cells:")
for key in (
    "same_direction_fraction",
    "opposite_direction_fraction",
    "neutral_fraction",
    "signed_score_correlation",
    "strong_same_fraction",
    "strong_opposite_fraction",
):
    print(f"{key}: {np.mean([r[key] for r in rows]):.4f}")

print(f"\nSaved: {out}")
