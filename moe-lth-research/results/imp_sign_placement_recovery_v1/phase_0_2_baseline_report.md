# Phase 0.2: Baseline CE Reproduction Report

## Status: ✓ PASSED

All step-0 unmodified CE values match exactly (difference = 0.0 nats for all 15 conditions).

## Baselines by Seed

| Seed | IMP (nats) | Random-1 (nats) | Random-2 (nats) | IMP Advantage (nats) |
|------|-----------|----------------|----------------|----------------------|
| 7    | 169.875   | 174.668        | 174.671        | +4.79                |
| 17   | 167.340   | 172.049        | 172.047        | +4.71                |
| 29   | 167.287   | 172.158        | 172.154        | +4.87                |
| 43   | 172.532   | 177.841        | 177.840        | +5.31                |
| 61   | 166.642   | 171.201        | 171.201        | +4.56                |

**Mean IMP advantage**: +4.83 nats (σ = 0.29)

## Key Observations

### 1. Consistency Across Seeds
- All 5 seeds show consistent IMP advantage at step 0 (range: 4.56–5.31 nats)
- IMP + random masks are highly replicable (e.g., seed 7 random-1 vs random-2 differ by 0.003 nats)
- Validation hash identical for all 15 conditions (09a073e9...)

### 2. Baseline Variability Across Seeds
- Step-0 CE varies significantly: 166.6–177.8 nats (11.2 nats spread)
- Seeds 7, 17, 29 from `32e_router_age_localization` show tighter clustering (167.3–174.7)
- Seeds 43, 61 from `router_conditioned_fixed_budget_dataset` are consistently higher (166.6–177.8)
- **This is NOT a floor effect**: random masks at ~172–177 nats are well above log(vocab_size) baseline

### 3. Anomaly: Seed 43
- Seed 43 IMP (172.5 nats) is **higher** than seeds 7, 17, 29 IMP (~167–169 nats)
- Yet it maintains similar ~5 nats advantage over its paired random masks
- Later phases (Phase 0.3, 0.4) will audit whether seed 43 mask/checkpoint have unique properties
- **Does not prevent baseline verification**: step-0 reproducibility is confirmed

### 4. Evaluation Protocol Verification
- All 15 conditions evaluated on same validation set (492,416 target tokens)
- Validation hash consistency confirms identical data/ordering
- Baseline reproduction matches logged values to machine precision (0.0 difference)

## Gate Condition

**PASS**: No absolute CE differs from logged values by more than 1e-3 nats/token.

**All differences: 0.0 nats**

## Next Phase

Phase 0.3 (Control Reuse Audit) can proceed. The unusually high baseline for seed 43 will be flagged for investigation, but does not block Phase 0.2 validation.
