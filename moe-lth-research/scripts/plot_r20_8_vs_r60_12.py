#!/usr/bin/env python3
"""Generate remaining available-data figures for the cross-architecture comparison."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "r20_8_vs_r60_12"
FIG = OUT / "figures"


def read(name):
    with open(OUT / name) as handle:
        return list(csv.DictReader(handle))


def mean_by(rows, expert_count, age, field):
    values = [float(row[field]) for row in rows if int(row["expert_count"]) == expert_count and int(row["router_age"]) == age and row.get(field) not in (None, "")]
    return sum(values) / len(values)


def main():
    FIG.mkdir(exist_ok=True)
    stability, distance, load, vectors = (read(name) for name in ("routing_stability.csv", "functional_router_distance.csv", "load_organization.csv", "developmental_vectors.csv"))
    for rows, name, field, title in (
        ([row for row in stability if row["relation"] == "to_R100"], "assignment_stability_by_age.png", "assignment_agreement", "Assignment agreement to R100"),
        ([row for row in distance if row["relation"] == "to_R100"], "js_to_maturity_by_age.png", "mean_js", "JS distance to R100"),
        (load, "load_cv_by_age.png", "utilization_cv", "Utilization CV"),
    ):
        plt.figure()
        for experts, ages in ((8, (0, 20, 100)), (12, (0, 10, 20, 40, 60, 80, 100))):
            plt.plot(ages, [mean_by(rows, experts, age, field) for age in ages], marker="o", label=f"{experts}E")
        plt.title(title); plt.xlabel("Router age (%)"); plt.legend(); plt.savefig(FIG / name, dpi=140, bbox_inches="tight"); plt.close()
    fields = ("normalized_entropy", "margin", "load_cv", "specialization", "assignment_agreement_to_R100", "functional_js_to_R100", "cumulative_exposure_per_expert")
    profiles = {"8E R20": next(row for row in vectors if int(row["expert_count"]) == 8 and int(row["router_age"]) == 20), "12E R40": next(row for row in vectors if int(row["expert_count"]) == 12 and int(row["router_age"]) == 40), "12E R60": next(row for row in vectors if int(row["expert_count"]) == 12 and int(row["router_age"]) == 60), "12E R80": next(row for row in vectors if int(row["expert_count"]) == 12 and int(row["router_age"]) == 80)}
    plt.figure()
    for name, row in profiles.items():
        values = [float(row[field]) for field in fields]
        plt.plot(range(len(fields)), values, marker="o", label=name)
    plt.xticks(range(len(fields)), ["H", "M", "U", "S", "D", "F", "X"]); plt.title("Developmental profiles (raw scales)"); plt.legend(); plt.savefig(FIG / "developmental_profile_comparison.png", dpi=140, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()