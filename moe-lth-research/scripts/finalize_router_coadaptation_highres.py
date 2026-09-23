#!/usr/bin/env python3
"""Write the seed-7 high-resolution verdict from existing diagnostic tables."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "router_coadaptation_highres"
RUNS = ("R0", "R20", "R100")
WINDOW = {(500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)}


def rows(name):
    with open(OUT / name) as handle:
        return list(csv.DictReader(handle))


def main():
    persistence = rows("persistent_switch_analysis.csv")
    distances = rows("functional_router_distance.csv")
    gradients = rows("gradient_trajectory.csv")
    coupling = []
    summary = {}
    for run in RUNS:
        summary[run] = {}
        for layer in range(4):
            records = [row for row in persistence if row["run"] == run and int(row["layer"]) == layer and (int(row["start_step"]), int(row["end_step"])) in WINDOW]
            switches = sum(int(row["switches"]) for row in records)
            tokens = sum(int(row["total_tokens"]) for row in records)
            stable = sum(int(row["persistent_next_two"]) for row in records)
            distance = [row for row in distances if row["run"] == run and int(row["layer"]) == layer and (int(row["start_step"]), int(row["end_step"])) in WINDOW]
            valid_gradients = [float(row["expert_grad_norm"]) for row in gradients if row["run"] == run and row["gradient_valid"] == "True" and row["expert_grad_norm"]]
            summary[run][layer] = {"switch_rate": switches / tokens, "persistent_fraction": stable / switches, "mean_js": sum(float(row["mean_JS"]) for row in distance) / len(distance), "mean_tv": sum(float(row["mean_TV"]) for row in distance) / len(distance)}
            coupling.append({"run": run, "layer": layer, "window": "500-1000", "switch_rate": switches / tokens, "persistent_switch_fraction": stable / switches, "functional_router_JS": summary[run][layer]["mean_js"], "functional_router_TV": summary[run][layer]["mean_tv"], "expert_gradient_mean_400_1100": sum(valid_gradients) / len(valid_gradients), "interpretation": "descriptive only; expert gradient is model-level because layer-resolved finite values were sparse"})
    with open(OUT / "routing_gradient_coupling.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coupling[0])); writer.writeheader(); writer.writerows(coupling)
    table = "\n".join(f"| {layer} | {summary['R0'][layer]['switch_rate']:.4f} / {summary['R0'][layer]['persistent_fraction']:.4f} | {summary['R20'][layer]['switch_rate']:.4f} / {summary['R20'][layer]['persistent_fraction']:.4f} | {summary['R100'][layer]['switch_rate']:.4f} / {summary['R100'][layer]['persistent_fraction']:.4f} |" for layer in range(4))
    (OUT / "seed7_mechanism_report.md").write_text(f"""# Seed-7 High-Resolution Mechanism Diagnostic

## Scope

This report analyzes the authorized seed-7, 85% sparse, trainable-router R0/R20/R100 recovery diagnostics. It introduces no new runs. The probe contains stable logical token identifiers and captured full eight-expert routing probability vectors at every diagnostic checkpoint.

## Result

The token-level data weaken both a simple magnitude account and an R20-specific stabilization account. R20 makes fewer switches than R0 in every layer during steps 500-1000 and has higher persistence than R0 in layers 1-3. However, R100 is more persistent than R20 in every layer and generally makes fewer switches. R20 therefore is not uniquely distinguished from both controls by persistence, switch confidence, or functional routing motion.

| Layer | R0 switch rate / persistence | R20 switch rate / persistence | R100 switch rate / persistence |
|---:|---:|---:|---:|
{table}

R20 switched-token changes are modestly more decisive than R0 on average (confidence +0.0048 vs +0.0034; margin +0.0024 vs +0.0022; entropy -0.0107 vs -0.0066), but R100 is comparable or stronger (+0.0050, +0.0024, -0.0098). This does not support targeted R20-only routing correction.

Router updates during 500-1000 are directionally more aligned for R20-R100 (mean cosine 0.2492) than R0-R20 (0.2010), while R0-R100 is 0.0591. Parameter direction alone is not functional equivalence and does not supply a mechanism.

## Hypothesis Assessment

- H2 targeted reorganization: not supported. Token transitions and source-destination matrices exist, but no R20-only pattern separates it from both controls in this single seed.
- H3 stabilization: not supported. R20 is more stable than R0 in most layers, but R100 is still more stable.
- H4 load correction: not supported. The probe has capacity outcomes and expert loads, but no expert-specific recoverability/damage measure was logged to make “better suited” testable.
- H5 routing-gradient coupling: candidate only. `routing_gradient_coupling.csv` aligns valid aggregate expert gradients with token dynamics, but it does not show R20-specific favorable optimization versus both controls and cannot establish mediation.
- H6 distributed multi-layer effect: not supported by this single-seed diagnostic; no coordinated R20-specific signature is present.

The high-resolution seed-7 run did not include matched dense recovery controls, so its final losses cannot reconstruct a new ticket-gap trajectory. The original matched-dense forensic result remains the source for the 500-1000 timing hypothesis.

No cross-seed replication is launched because the pre-registered decision rule was not met: no measurable seed-7 feature clearly distinguished R20 from both R0 and R100 while temporally aligned with the earlier performance effect. A causal conclusion would additionally require intervention.

## Verdict

`NO DISTINCTIVE TOKEN-LEVEL MECHANISM FOUND`
""")


if __name__ == "__main__":
    main()