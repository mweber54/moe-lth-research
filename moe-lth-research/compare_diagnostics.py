#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from pathlib import Path

BASE = Path("results/imp_adaptation_factorial_v1/attempts")

def flatten_numeric(obj, prefix=""):
    """Flatten nested JSON into metric-name -> numeric-value pairs."""
    result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(value, name))

    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            name = f"{prefix}[{i}]"
            result.update(flatten_numeric(value, name))

    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(obj):
            result[prefix] = float(obj)

    return result


def load_steps(directory):
    """Load all step_*.json diagnostics."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    steps = {}

    for path in directory.glob("step_*.json"):
        match = re.fullmatch(r"step_(\d+)\.json", path.name)
        if not match:
            continue

        step = int(match.group(1))

        with path.open() as f:
            data = json.load(f)

        steps[step] = flatten_numeric(data)

    return steps


def compare(full_steps, shared_steps):
    rows = []

    all_steps = sorted(set(full_steps) | set(shared_steps))

    for step in all_steps:
        full = full_steps.get(step, {})
        shared = shared_steps.get(step, {})

        for metric in sorted(set(full) | set(shared)):
            a = full.get(metric)
            b = shared.get(metric)

            if a is None:
                status = "only_expert_shared"
            elif b is None:
                status = "only_full"
            else:
                status = "both"

            delta = b - a if status == "both" else None

            pct = None
            if delta is not None and a != 0:
                pct = 100 * delta / abs(a)

            rows.append({
                "step": step,
                "metric": metric,
                "full": a,
                "expert_shared": b,
                "delta_shared_minus_full": delta,
                "pct_change_from_full": pct,
                "status": status,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sparsity", default="imp80")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--output",
        default="diagnostics_full_vs_expert_shared.csv"
    )
    args = parser.parse_args()

    prefix = f"e32_seed{args.seed}_{args.sparsity}"

    full_dir = (
        BASE / f"{prefix}_full"
        / "attempt_0001" / "diagnostics"
    )

    shared_dir = (
        BASE / f"{prefix}_expert_shared"
        / "attempt_0001" / "diagnostics"
    )

    full_steps = load_steps(full_dir)
    shared_steps = load_steps(shared_dir)

    print("\nFULL:", full_dir)
    print("SHARED:", shared_dir)

    print("\nFull steps:", sorted(full_steps))
    print("Expert+shared steps:", sorted(shared_steps))

    rows = compare(full_steps, shared_steps)

    if not rows:
        raise SystemExit("No comparable numeric diagnostics found.")

    fields = [
        "step",
        "metric",
        "full",
        "expert_shared",
        "delta_shared_minus_full",
        "pct_change_from_full",
        "status",
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} comparisons to {args.output}")

    for step in sorted(set(r["step"] for r in rows)):
        matching = [
            r for r in rows
            if r["step"] == step
            and r["status"] == "both"
        ]

        missing = [
            r for r in rows
            if r["step"] == step
            and r["status"] != "both"
        ]

        matching.sort(
            key=lambda r: abs(r["pct_change_from_full"])
            if r["pct_change_from_full"] is not None
            else -1,
            reverse=True
        )

        print(f"\n{'=' * 90}")
        print(f"STEP {step}")
        print(f"Matching numeric metrics: {len(matching)}")
        print(f"Unmatched numeric metrics: {len(missing)}")
        print(f"{'=' * 90}")

        print(
            f"{'Metric':<48}"
            f"{'Full':>12}"
            f"{'Shared':>12}"
            f"{'Delta':>12}"
            f"{'Change %':>12}"
        )

        for r in matching[:args.top]:
            pct = r["pct_change_from_full"]
            pct_text = f"{pct:+.2f}%" if pct is not None else "N/A"

            print(
                f"{r['metric'][:47]:<48}"
                f"{r['full']:>12.5g}"
                f"{r['expert_shared']:>12.5g}"
                f"{r['delta_shared_minus_full']:>+12.5g}"
                f"{pct_text:>12}"
            )

        if missing:
            print("\nMetrics present in only one condition:")
            for r in missing[:10]:
                print(f"  {r['status']}: {r['metric']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
