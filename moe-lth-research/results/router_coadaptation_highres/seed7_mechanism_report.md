# Seed-7 High-Resolution Mechanism Diagnostic

## Scope

This report analyzes the authorized seed-7, 85% sparse, trainable-router R0/R20/R100 recovery diagnostics. It introduces no new runs. The probe contains stable logical token identifiers and captured full eight-expert routing probability vectors at every diagnostic checkpoint.

## Result

The token-level data weaken both a simple magnitude account and an R20-specific stabilization account. R20 makes fewer switches than R0 in every layer during steps 500-1000 and has higher persistence than R0 in layers 1-3. However, R100 is more persistent than R20 in every layer and generally makes fewer switches. R20 therefore is not uniquely distinguished from both controls by persistence, switch confidence, or functional routing motion.

| Layer | R0 switch rate / persistence | R20 switch rate / persistence | R100 switch rate / persistence |
|---:|---:|---:|---:|
| 0 | 0.0776 / 0.6604 | 0.0420 / 0.6607 | 0.0256 / 0.7591 |
| 1 | 0.2244 / 0.4512 | 0.1733 / 0.5204 | 0.1763 / 0.5847 |
| 2 | 0.2641 / 0.3608 | 0.2090 / 0.4387 | 0.2067 / 0.5125 |
| 3 | 0.3215 / 0.3594 | 0.2893 / 0.3653 | 0.2652 / 0.4238 |

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
