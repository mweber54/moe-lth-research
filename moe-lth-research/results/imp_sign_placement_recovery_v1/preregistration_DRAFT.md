# PREREGISTRATION: Phase 2 Recovery Gap Analysis (Sign Preservation Hypothesis)

**Status**: DRAFT — Awaiting user approval before implementation

**Date**: 2025-01-XX (to be filled on approval)

**Analysis**: Sign-based recovery gap in IMP masks during 32-expert, 80% sparsity training

---

## Hypothesis

**Primary**: Preserved signs in IMP-selected coordinates allow faster recovery than random masks must achieve through full relearning.

**Supporting Evidence** (from Phases 0–1):
- Step-0 analysis: sign permutation damages IMP models by 4.8 nats; magnitude permutation by 0.5 nats
- Phase 1: 96–97% sign agreement on survivors between original init and trained parent
- Random masks show no such structure, yet damage is identical per-seed

**Prediction**: 
- IMP models recover to step-0 performance faster than random masks
- Gap widens during mid-training (steps 250–2000) as signs enable routing decisions
- Final performance (step 5000) similar or IMP-better, confirming signs as structural prior

---

## Experimental Design

### Checkpoints & Initialization
- **Source**: Step-0 original initializations (see Phase 0.1 provenance map)
  - Seeds 7, 17, 29: `32e_router_age_localization/references/seed_X/checkpoints/step_0.pt`
  - Seeds 43, 61: `router_conditioned_fixed_budget_dataset/sources/e32/seed_X/checkpoints/step_0.pt`
- **Masks**: IMP80 from round 7 (see Phase 0.1) + matched random masks from step-0 analysis
- **Setup**: Initialize with step-0 checkpoint, apply mask, train for 5000 steps

### Training Configuration
- **Batch stream**: Order A and Order B (both from `imp_recovery_order_stability_v1` setup)
- **Optimizer**: AdamW (lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
- **Learning rate schedule**: Constant (no warmup, no decay)
- **Segment boundary**: Steps 0–2500 (segment 0), reset AdamW/scaler, 2500–5000 (segment 1)
- **Mixed precision**: Bfloat16 (TF32 off)
- **Gradient clipping**: norm=1.0
- **Seeds**: 5 (7, 17, 29, 43, 61)

### Evaluation Protocol
- **Common eval set**: Same 31-batch stream as step-0 analysis (validation_hash = 09a073e9...)
- **Steps**: 0, 250, 500, 1000, 1500, 2000, 2500 (boundary), 3000, 4000, 5000
- **Metric**: Cross-entropy (nats, target-token weighting, FP32 inference)
- **Routing**: Expert assignment changes vs. unmasked dense model

### Replication
- **Reuse**: All 10 IMP80 runs (5 seeds × 2 orders) from `imp_recovery_order_stability_v1`
- **New runs**: Random masks (2 per seed × 2 orders = 4 new per seed, but available reuse from step-0 analysis endpoints)
- **Total**: 10 reused IMP80 + (4 new or reused random per seed × 5 seeds) = 10 + up to 20 runs

### Comparison Strategy
- **Primary contrast (R_a)**: Recovery ratio = I_a(step 2000) / I_a(step 250)
  - I_a = IMP advantage over random at step a
  - Negative ratio (smaller number at step 2000 vs 250) = gap closing → IMP recovers
- **Secondary**: Per-seed, per-order CE curves (step 0, 250, 2000, 5000)
- **Tertiary**: Routing stability (expert changes per token between step 0 and later steps)

---

## Stopping Conditions

1. **Loss divergence**: Any run has NaN or CE > 10× baseline → abort and report
2. **Seed mismatch**: Initialization hash differs from logged → abort and report seed + expected + observed
3. **Mask enforcement**: Surviving coordinate count differs from step-0 → audit and report
4. **Reuse incompatibility**: Run uses different batch stream/LR than prior recovery → skip reuse and note

---

## Expected Outcomes

### If hypothesis supported:
- IMP gap closes or reverses by step 2000 (R_a < 1 or near 1)
- Random masks maintain or increase gap over the same steps
- Effect consistent across seeds and orders

### If hypothesis unsupported:
- Both IMP and random recover identically (R_a ≈ 1 for both)
- Signs don't provide structural advantage during optimization
- Implications: mid-training gap must be explained by other factors (e.g., routing stability, expert conditioning)

### Secondary patterns to monitor:
- Order A vs. Order B differences (batch sensitivity)
- Seed 43 anomaly: does high baseline persist, or does it reach parity?
- Layer-wise routing changes: which experts dominate recovery?

---

## Outputs

1. **recovery_trajectories.csv**: seed, mask_type, order, step, CE, delta_CE, expert_changes
2. **recovery_gap_by_seed.csv**: seed, step, IMP_CE, random_CE, gap_nats, recovery_ratio
3. **plots/**:
   - ce_vs_step_per_seed.pdf (IMP + random-1 + random-2 + unmasked baseline)
   - gap_vs_step.pdf (recovery ratio with error bars)
   - routing_changes_per_seed.pdf
4. **preregistration_summary.md**: Actual protocol used, any deviations, results summary

---

## Approval Checklist

- [ ] User confirms training configuration (LR, schedule, clipping, precision)
- [ ] User confirms reuse/new-run split (which random masks to reuse vs. generate)
- [ ] User confirms evaluation steps (0, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000)
- [ ] User confirms that prior recovery infrastructure (`imp_recovery_order_stability_v1` compute stack) is OK for reuse
- [ ] User approves this preregistration document (I will commit it with this exact wording)

---

## Dependencies & Assumptions

1. **No changes to code**: Use existing inference/training harness from `imp_recovery_order_stability_v1`
2. **No tuning**: All hyperparameters fixed a-priori; no ablations unless results require it
3. **Live queue**: Runs will submit to the same queue as prior recovery; no interruption/requeue required
4. **Artifact immutability**: All step-0 checkpoints, masks, and prior recovery outputs remain read-only

---

## Deviation Tracking

If Phase 2 proceeds and deviations occur (e.g., different LR, different steps, different seeds), they will be documented here and flagged as **DEVIATION** in the final report with justification.

| Deviation | Reason | Impact |
|-----------|--------|--------|
| (None yet) | — | — |

---

## Sign-Off

**Prepared by**: AI assistant (Copilot)  
**Date prepared**: [Auto-filled on approval]  
**User approval date**: [To be filled]  
**Approved by**: [User name]

**Commit hash** (upon approval): [Auto-filled]
