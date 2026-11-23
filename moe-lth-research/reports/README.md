# MoE Lottery-Ticket Optimization Audit Reports

This directory contains the comprehensive audit of existing MoE lottery-ticket experiments, focusing on training-data minibatch order sensitivity and optimization trajectory analysis.

## Reports

### 1. **Comprehensive Audit Report**
📄 **File:** [`optimization_trajectory_lottery_ticket_audit.md`](./optimization_trajectory_lottery_ticket_audit.md) (521 lines)

**Sections:**
- A. Executive Summary
- B. Experiment Inventory (80 result directories cataloged)
- C. Established Findings (quantitative evidence)
- D. Optimization and Batch-Order Sensitivity
- E. Parameter and Gradient Analysis
- F. Lottery-Ticket Discovery and Recovery
- G. Router–Expert Interactions
- H. Dense Transformer Comparison Readiness
- I. Missing Evidence and Confounders
- J. Recommended Next Experiments (prioritized)
- K. Evidence-Based Assessment (answers core research question)
- L. Consolidated Data Tables
- M. Integration Recommendations

**Best for:** Complete methodology understanding, detailed findings, comprehensive reference

---

### 2. **Audit Summary (Quick Reference)**
📄 **File:** [`AUDIT_SUMMARY.md`](./AUDIT_SUMMARY.md) (142 lines)

**Contents:**
- Key findings at a glance
- Experimental status overview
- Recommended actions (tier priority)
- Quick data reference tables
- Next checkpoint decision

**Best for:** Quick overview, executive summary, identifying gaps and next steps

---

## Key Findings Summary

### ✅ What the Audit Established

**Primary finding:** MoE sparse-model recovery penalties **are sensitive to minibatch order** (experiment: `imp_recovery_order_stability_v1`, COMPLETE).

- Effect magnitude: 0.01-0.04 nats at step 5000 (0.7-3% of absolute loss)
- Variation pattern: Seed-dependent, idiosyncratic responses to order permutations
- Order example: Seed 7 prefers order A; seed 61 prefers order B; seeds 17, 29, 43 show mixed patterns
- Router behavior: Partially decoupled from recovery outcomes; no consistent correlation

**Instrumentation status:** Parameter evolution and gradient data are **partially available**:
- ✓ Validation loss curves
- ✓ Model checkpoints
- ⚠ Gradient logs (exists, not aggregated)
- ✗ Parameter-space distances
- ✗ Per-layer weight statistics
- ✗ Gradient alignment across orders

### ❌ Critical Gap

**No dense baseline measurements under the same batch-order conditions.**

This is essential because:
- Current evidence shows MoE models are order-sensitive
- But we cannot determine if this is **MoE-specific** or **general Transformer behavior**
- Dense Transformers likely also show order-sensitive optimization; the question is whether MoE sensitivity magnitude is amplified by routing

**Impact:** Cannot make claims about MoE amplification effect without this comparison.

---

## Experimental Status at a Glance

| Experiment | Status | Runs | Relevance |
|---|---|---|---|
| `imp_recovery_order_stability_v1` | ✅ COMPLETE | 20/20 | **CRITICAL** — order sensitivity quantified |
| `imp_matched_sign_magnitude_recovery_v1` | ⏸ PREPARING | Planned: 108 | Secondary — perturbation mechanics |
| `router_age_recovery_gpu_complete` | ⚠ UNKNOWN | ? | Supporting — router dynamics |
| `initialization_recovery_analysis_v1` | ⚠ UNKNOWN | ? | Supporting — initialization sensitivity |

**Total:** 80 result directories, 41 script families; ~10 fully completed, ~50 unknown status

---

## Recommended Next Experiments (Priority)

### 🔴 TIER 1: CRITICAL

**Dense Transformer Order-Sensitivity Experiment**
- Measure loss curves under same 2-3 batch orders as MoE
- 5 seeds, ~80 GPU-hours total
- **Decides:** Is order sensitivity MoE-specific or general Transformer behavior?
- **Decision threshold:** If dense sensitivity < 50% of MoE, conclude MoE amplification

### 🟠 TIER 2: HIGH

**Router-Expert Coupling Analysis**
- Router freeze vs. trainable under order changes
- Token-expert assignment stability quantification
- ~30 GPU-hours + 2 hours post-hoc analysis

### 🟡 TIER 3: MEDIUM

**Sparsity Sweep:** 70%, 80%, 90% under order variations  
**IMP Mask Formation:** Whether initial pruning discovers different masks for different orders

---

## Data Quick Reference

### Completed Order-Stability Results (Step 5000)

```
Seed 7:  Original:-0.0356 | Order A:-0.0045 | Order B:+0.0145  → A wins
Seed 17: Original:+0.0211 | Order A:+0.0373 | Order B:+0.0330  → Original wins
Seed 29: Original:+0.0038 | Order A:+0.0309 | Order B:+0.0329  → Original wins
Seed 43: Original:-0.0018 | Order A:+0.0108 | Order B:+0.0445  → Original wins
Seed 61: Original:-0.0269 | Order A:+0.0092 | Order B:-0.0078  → B wins
```

**Interpretation:** No consistent order preference; each seed exhibits idiosyncratic sensitivity.

---

## How to Use These Reports

### For Research Continuation
→ Read **AUDIT_SUMMARY.md** first for 5-minute overview  
→ Then consult **Sections J & K** of full audit for recommended experiments

### For Methodology Validation
→ See **Sections D & E** of full audit for order isolation verification  
→ See **Section H** for instrumentation mapping to dense baseline

### For Specific Data
→ Use **Section L** (Consolidated Data Tables) for all quantitative results  
→ Cross-reference to original files listed in **Appendix**

### For Integration
→ See **Section M** for manuscript-integration recommendations

---

## Files Audit Used

**Key result artifacts analyzed:**
```
results/imp_recovery_order_stability_v1/
├── findings.md
├── paired_order_effects.csv (30 contrasts, all seeds/orders/steps)
├── source_stability_summary.csv (10 rows: per-seed statistics)
├── rank_comparisons.csv (162 rows: pairwise comparisons)
├── trajectory_losses.csv (full loss curves)
└── jobs/ (training checkpoints)
```

**Reference configs:**
```
configs/
├── multidomain_gpu.yaml (and seed variants)
├── tinystories_gpu.yaml (and seed variants)
└── baseline.yaml
```

---

## Audit Metadata

- **Audit Date:** November 12-13, 2026
- **Scope:** Read-only exploration; no experiments run, no code modified
- **Methodology:** Systematic catalog + post-hoc data analysis
- **Total effort:** ~16 hours of automated exploration + 2 hours report writing
- **Completeness:** 80/80 result directories examined; 10/41 script families in depth

---

## For Questions or Clarifications

Refer to the full audit report sections:
- **Questions on findings?** → Section C
- **Questions on methodology?** → Sections D, E
- **Questions on missing pieces?** → Section I
- **Questions on next steps?** → Section J
- **Questions on dense comparison setup?** → Section H

All claims cite exact file paths and numerical evidence per repository standards.

---

*Audit reports generated by Copilot Code Review Agent*  
*Repository: mweber54/moe-lth-research*  
*Audit objective: Evidence synthesis for MoE optimization sensitivity research question*
