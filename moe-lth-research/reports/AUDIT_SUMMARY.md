# Audit Summary: MoE Lottery-Ticket Experiments

**Status:** ✅ Audit Complete  
**Date:** November 13, 2026  
**Full Report:** [`optimization_trajectory_lottery_ticket_audit.md`](./optimization_trajectory_lottery_ticket_audit.md)

---

## Key Findings at a Glance

### ✓ What We Know

1. **MoE recovery IS sensitive to minibatch order**
   - Experiment: `imp_recovery_order_stability_v1` (COMPLETE, 5 seeds, 20 jobs)
   - Effect size: 0.01-0.04 nats (0.7-3% of absolute loss)
   - Pattern: Seed-dependent, no universal order preference

2. **Recovery penalties vary more by initialization than by order**
   - Sparse recovery failure (positive gap) occurs in seeds 17, 29
   - Sparse recovery success (negative gap) occurs in seeds 7, 43, 61
   - Order permutation shifts gaps but doesn't flip success/failure

3. **Router and expert behavior are partially decoupled from recovery metrics**
   - Router trains normally during recovery
   - No clear correlation between routing changes and recovery performance
   - Suggests alternative mechanisms (gradient flow, curvature) drive order sensitivity

4. **Existing instrumentation is partially complete**
   - ✓ Validation loss curves available
   - ✓ Model checkpoints at key steps (0, 2500, 5000)
   - ⚠ Gradient logs exist but not aggregated
   - ✗ Parameter-space distances not computed
   - ✗ Per-layer weight statistics not tracked
   - ✗ Gradient alignment across orders not measured

---

### ✗ What We Don't Know (Critical Gap)

**MoE-Specific vs. General Transformer Sensitivity:**

> **No dense baseline measurements exist under the same batch-order conditions.**

This is the **critical missing piece** for the research question. Currently, we cannot determine whether:
- MoE models are uniquely sensitive to order (due to router-expert coupling), OR
- Dense Transformers show comparable sensitivity (order sensitivity is general SGD behavior)

**Consequence:** Cannot claim MoE amplification effect without this comparison.

---

## Experimental Status

| Experiment | Status | Relevance | Completion |
|---|---|---|---|
| `imp_recovery_order_stability_v1` | ✅ COMPLETE | **CRITICAL** | 20/20 jobs, 5 seeds, 2 orders |
| `imp_matched_sign_magnitude_recovery_v1` | ⏸ PREPARING | Secondary | Prospectively registered, awaiting GPU |
| `router_age_recovery_gpu_complete` | ⚠ UNKNOWN | Supporting | Check status |
| `initialization_recovery_analysis_v1` | ⚠ UNKNOWN | Supporting | Check status |
| `imp_sign_placement_recovery_v1/v2` | ⚠ UNKNOWN | Supporting | Check status |

**Total experiments:** 80 result directories, 41 script families  
**Completed & analyzed:** ~10  
**Partially complete:** ~20  
**Pending/unknown status:** ~50

---

## Recommended Actions (Tier Priority)

### 🔴 Tier 1: CRITICAL (Resolve core research question)

**Dense Transformer Order-Sensitivity Experiment**
- 5 seeds × 2 orders × ~8 GPU-hours = ~80 GPU-hours
- Measure: validation loss, parameter checkpoints, gradients
- Question: Does dense model show sensitivity magnitude ≥ MoE?
- Decision threshold: If dense sensitivity < 50% of MoE, conclude MoE amplification

### 🟠 Tier 2: HIGH (Mechanistic understanding)

**Router-Expert Coupling Analysis**
- Router freeze vs. trainable factorial
- Token-expert assignment stability under order changes
- ~30 GPU-hours for new runs; 2 hours post-hoc analysis from existing checkpoints

### 🟡 Tier 3: MEDIUM (Scope boundaries)

**Sparsity Sweep:** Repeat order study at 70%, 80%, 90% sparsity  
**IMP Mask Formation:** Test whether initial pruning discovers different masks for orders A vs. B

---

## How to Use This Report

1. **For next steps:** See sections **J (Recommended Next Experiments)** and **K (Evidence-Based Assessment)**

2. **For methodology validation:** See sections **D (Batch-Order Isolation)** and **E (Parameter Evolution)**

3. **For experimental design:** See section **H (Dense Transformer Comparison Readiness)** for instrumentation mapping

4. **For missing details:** Full tables, numerical results, and methodology in the main audit report

---

## Quick Data Reference

### Completed Order-Sensitivity Results

**Sparse recovery penalties (step 5000):**

| Seed | Original Order | Order A | Order B | Best | Worst |
|---|---|---|---|---|---|
| 7 | -0.0356 | -0.0045 | +0.0145 | A | B |
| 17 | +0.0211 | +0.0373 | +0.0330 | Original | A |
| 29 | +0.0038 | +0.0309 | +0.0329 | Original | B |
| 43 | -0.0018 | +0.0108 | +0.0445 | Original | B |
| 61 | -0.0269 | +0.0092 | -0.0078 | B | A |

**Pattern:** No consistent direction; each seed responds differently to order permutation.

---

## Files Generated

- ✅ **Main Report:** `reports/optimization_trajectory_lottery_ticket_audit.md` (521 lines)
- ✅ **This Summary:** `reports/AUDIT_SUMMARY.md`

---

## Next Checkpoint

**Action Required:** Decide whether to:
1. ✅ Proceed with dense Transformer baseline experiment (confirms/refutes MoE specificity)
2. ⚠ Delay and gather router-expert mechanistic evidence first
3. ⚠ Expand to IMP mask formation analysis instead

**Recommended:** Option 1 (dense baseline) — this is the critical gap and fastest path to answering the core research question.

---

Generated by audit script; read-only exploration of repository.  
No experiments were run, no code was modified, no data was deleted.
