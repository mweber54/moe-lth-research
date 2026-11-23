# Controlled token-to-routing-slot matching findings

Status: STOPPED / INCOMPLETE. Completed 6 of75 registered runs.

Stop record: {"reason": "Pilot identity replay changes learning regime beyond registered feasibility limits", "scientific_stop": true, "pilot": {"expand": false, "rows": [{"sparsity": 0.0, "native_ce": 1.3298474067150539, "identity_ce": 4.008660688566362, "shuffle_ce": 1.9421901199642269, "native_improvement_retained": 0.9843042129428489, "passed": false}, {"sparsity": 0.8, "native_ce": 1.351046686393117, "identity_ce": 6.14767225110317, "shuffle_ce": 1.8053145548694314, "native_improvement_retained": 0.9711028087216108, "passed": false}], "criteria": {"identity_native_improvement_retained_min": 0.9, "identity_final_ce_native_ratio_max": 1.5, "max_amp_skips_per5000": 500, "finite_endpoints": true, "penalty_sign_used": false}, "reason": "Identity learning-regime and numerical feasibility only; shuffled penalty sign is not a criterion"}}

Pilot feasibility: {"expand": false, "rows": [{"sparsity": 0.0, "native_ce": 1.3298474067150539, "identity_ce": 4.008660688566362, "shuffle_ce": 1.9421901199642269, "native_improvement_retained": 0.9843042129428489, "passed": false}, {"sparsity": 0.8, "native_ce": 1.351046686393117, "identity_ce": 6.14767225110317, "shuffle_ce": 1.8053145548694314, "native_improvement_retained": 0.9711028087216108, "passed": false}], "criteria": {"identity_native_improvement_retained_min": 0.9, "identity_final_ce_native_ratio_max": 1.5, "max_amp_skips_per5000": 500, "finite_endpoints": true, "penalty_sign_used": false}, "reason": "Identity learning-regime and numerical feasibility only; shuffled penalty sign is not a criterion"}

## Every source seed

| Seed | Sparsity | Step | Sparse penalty | Dense penalty | Differential | Status |
|---|---|---|---|---|---|---|
| 7 | 0.8 | 2500 |  |  |  | MISSING |
| 7 | 0.8 | 5000 |  |  |  | MISSING |
| 7 | 0.85 | 2500 |  |  |  | MISSING |
| 7 | 0.85 | 5000 |  |  |  | MISSING |
| 17 | 0.8 | 2500 |  |  |  | MISSING |
| 17 | 0.8 | 5000 |  |  |  | MISSING |
| 17 | 0.85 | 2500 |  |  |  | MISSING |
| 17 | 0.85 | 5000 |  |  |  | MISSING |
| 29 | 0.8 | 2500 |  |  |  | MISSING |
| 29 | 0.8 | 5000 |  |  |  | MISSING |
| 29 | 0.85 | 2500 |  |  |  | MISSING |
| 29 | 0.85 | 5000 |  |  |  | MISSING |
| 43 | 0.8 | 2500 |  |  |  | MISSING |
| 43 | 0.8 | 5000 |  |  |  | MISSING |
| 43 | 0.85 | 2500 |  |  |  | MISSING |
| 43 | 0.85 | 5000 |  |  |  | MISSING |
| 61 | 0.8 | 2500 |  |  |  | MISSING |
| 61 | 0.8 | 5000 |  |  |  | MISSING |
| 61 | 0.85 | 2500 |  |  |  | MISSING |
| 61 | 0.85 | 5000 |  |  |  | MISSING |

## Descriptive seed-level summaries

[]

No full-cohort conclusion is supported. Missing/failed draws are not averaged into complete-seed contrasts.

Ordinary native recovery and replay are separate learning regimes. The causal comparison is shuffled replay versus identity replay. Complete slots move expert identity, token-specific coefficient and drop identity together; assignment identity alone is not identified. This experiment does not establish IMP-encoded token specialization, the cause of historical sparse wins, novelty, generality, or a percentage of an earlier freezing penalty.

Raw per-condition endpoints, complete validation/training curves, named task gradients, component displacements, trace inventories, checksums, exposure counts and gate distributions remain in the isolated attempt directories. Missing/failed conditions are explicit in missing_failed.json.
