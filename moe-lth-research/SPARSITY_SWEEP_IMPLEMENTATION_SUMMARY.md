# Sparsity Sweep Experiment Implementation Summary

**Date**: 2026-08-29  
**Commits**: 646e8a1 (implementation), 276f8cc (deployment guide)  
**Status**: Ready for VM deployment

## What Was Implemented

### 1. Sparsity Sweep Runner (`run_sparsity_sweep.py`)
A comprehensive experiment runner that extends the corrected router-age recovery protocol to test sparse lottery tickets across a range of expert sparsity levels.

**Key design principles:**
- **Pure rewind protocol**: E_sparse = m_s ⊙ E_0 (mask from E_T, values from E_0)
- **Isolated router effect**: Only router changes between R_0 and R_100; everything else fixed
- **Exact checkpoint mapping**: 2500-step budget with well-defined step-to-age mapping
- **Dense baseline reuse**: E_0 controls computed once per seed, reused across sparsities
- **Comprehensive auditing**: Hash tracking and integrity assertions throughout

### 2. Experimental Parameters
```
Sparsities:     60%, 70%, 90%, 95% (+ existing 80%)
Router ages:    R_0 (step 0) and R_100 (step 2500)
Seeds:          7, 17, 29
Conditions:     4 sparsities × 2 ages × 3 seeds × 2 types (sparse+dense) = 48 new runs
Recovery budget: 2500 steps (identical to 80% experiment)
```

### 3. Core Algorithm
For each sparsity level s:
1. Load final checkpoint, discover top-(1-s) weights per expert → mask m_s
2. Load initial checkpoint, build sparse base = m_s ⊙ E_0
3. For each seed and router age (R_0, R_100):
   - Run sparse recovery from m_s ⊙ E_0 with router swapped
   - Reuse dense E_0 control (only 6 unique dense runs needed)
   - Track loss trajectory, routing stats, gradient norms
   - Compute Δ_ticket(R, s) = L_sparse - L_dense

### 4. Protocol Compliance
Every condition validates:
- ✓ Rewind assertion: sparse weights match E_0, pruned = 0
- ✓ Mask source: m_s discovered from E_T, not E_0
- ✓ Shared state: identical hash across all conditions for same seed
- ✓ Router uniqueness: R_0 hash ≠ R_100 hash
- ✓ Data reproducibility: batch sequences hashed and validated
- ✓ Mask enforcement: pruned weights stay zero throughout recovery

Violations fail loudly with detailed error messages; no silent fallbacks.

### 5. Metrics Captured
**Per condition:**
- Initial/final/best validation loss
- Early AUC (first 50% of steps)
- Recovery fraction (distance to dense baseline)
- Time to reach 5% and 10% thresholds
- Routing: entropy, mean selected probability, expert utilization, margin
- Gradients: per-expert norms, by-component aggregates

**Derived (post-run):**
- Sparse-dense gap: Δ_ticket(R, s) = L_sparse(R, s) - L_dense(R, s)
- Router routing benefit: Δ_routing(s) = Δ_ticket(R_0, s) - Δ_ticket(R_100, s)
- Proportional gap reduction: [gap(R_0, s) - gap(R_100, s)] / gap(R_0, s)
- Sparsity frontier: max{s : Δ_ticket(R, s) ≤ ε}

### 6. Output Structure
```
results/router_conditioned_sparsity_sweep/
├── sparsity_0.60/
│   └── seed_{7,17,29}/
│       ├── age_000pct_sparse/   (E_sparse with R_0)
│       ├── age_000pct_dense/    (E_0 with R_0, reused)
│       ├── age_100pct_sparse/   (E_sparse with R_100)
│       ├── age_100pct_dense/    (E_0 with R_100, reused)
│       ├── pruning_metadata.json (mask stats: realized sparsity, hash)
│       └── lth_isolation_audit.json (protocol compliance)
├── sparsity_0.70/, 0.90/, 0.95/ (same structure)
├── sparsity_sweep_all_records.json  (48 full records with metadata)
├── sparsity_sweep_all_records.csv   (same, incrementally updated)
└── sparsity_sweep_paired.csv        (R_0 vs R_100 pairs: gap_reduction, proportional_reduction)
```

### 7. Tests
Created `tests/test_sparsity_sweep.py` with:
- `test_sparsity_sweep_tiny_protocol()`: Validates with 2-step budget
  - Checks sparse ticket correctness (E_0 under mask)
  - Validates dense control assembly
  - Confirms hash tracking and isolation
  - Verifies output structure

## Scientific Motivation

**Central Research Question**: *Does router maturity shift the sparsity frontier at which rewound expert subnetworks remain competitive?*

**Hypothesis**: 
- Young router (R_0) forces poorly-calibrated expert selection → sparse tickets suffer
- Mature router (R_100) learns specialized expert roles → sparse tickets recover better
- This effect should be strongest at medium-to-high sparsity (70-90%)

**Predictions**:
- At 60%: Both R_0 and R_100 may be dense-equivalent → minimal gap difference
- At 70-80%: R_100 benefit grows → Δ_routing(s) > 0
- At 90-95%: Extreme sparsity → gap may grow for both, but R_100 still better

## How to Deploy on VM

### Quick Start
```bash
cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research
git pull origin main  # Get commit 276f8cc
source .venv/bin/activate

# Full sparsity sweep (all 4 levels)
python -m moe_lth.experiments.run_sparsity_sweep \
  --configs \
    configs/router_age_reference_seed7.yaml \
    configs/router_age_reference_seed17.yaml \
    configs/router_age_reference_seed29.yaml \
  --output-dir results/router_conditioned_sparsity_sweep \
  --recovery-steps 2500
```

### Validate Protocol First (Optional)
```bash
python -m moe_lth.experiments.run_sparsity_sweep \
  --configs configs/smoke.yaml \
  --output-dir results/sparsity_sweep_test \
  --recovery-steps 2
```

### Expected Runtime
- 48 total conditions (~2-4 min each depending on GPU)
- Estimate: 2-3 hours wall-clock for full sweep
- Dense controls reused: saves ~8 condition runs (~20 min)

### Post-Run Analysis
See `SPARSITY_SWEEP_DEPLOYMENT.md` for:
- Combining with existing 80% results
- Python scripts to compute gaps and frontiers
- Matplotlib code for generating figures
- Interpretation guidelines

## Integration with Existing 80% Results

**Current state**: 
- 80% complete in `results/router_age_lth_80pct_dense_v2/`
- Full router ages: R_0, 10, 20, 40, 60, 80, 100 (all 7 points)
- Sparse and dense controls for all ages
- Confidence-matched variants for ages 0, 40, 80, 100

**New state after sweep**:
- 60%, 70%, 90%, 95% complete
- Only R_0 and R_100 endpoints (2 points per sparsity)
- Sparse and dense controls

**Combined analysis**:
- Sparsity range: 60% → 95% (5 levels)
- Router ages: R_0 and R_100 only (for fair comparison)
- Filter 80% results to just (R_0, R_100) for this analysis
- Create sparse-dense gap matrix: 5 sparsities × 2 router ages × 3 seeds

## Key Technical Notes

### Dense Baseline Reuse
The runner intelligently reuses dense controls:
```python
dense_cache_key = (seed, router_age_percent)
if dense_cache_key in sparsity_dense_baselines:
    # Reuse from cache with hash validation
    dense_record = cache[dense_cache_key].copy()
else:
    # Run new dense condition
    dense_record = _run_recovery_condition(...)
    cache[dense_cache_key] = dense_record
```

Reuse is validated by checking:
- Reference seed matches
- Router age and step match
- Expert hash (E_0 identical)
- Shared hash (shared parameters identical)
- Batch sequence hashes (data order identical)
- Recovery steps and optimizer config identical

### Exact Checkpoint Mapping
For 2500-step budget:
```python
EXACT_ROUTER_STEPS_BY_AGE = {0: 0, 100: 2500}

def _checkpoint_for_percent(run_dir, total_steps, percent):
    if total_steps >= 2500 and percent in EXACT_ROUTER_STEPS_BY_AGE:
        # Use exact step, fail if missing
        return available[EXACT_ROUTER_STEPS_BY_AGE[percent]]
    else:
        # Fall back to nearest checkpoint (for test budgets)
        return available[nearest_step]
```

This prevents accidental reuse of wrong router checkpoints while supporting small test budgets.

### Mask Enforcement
Pruned weights tracked throughout recovery:
```python
if (step + 1) % 500 == 0:
    for name, mask in masks.items():
        param = dict(model.named_parameters())[name]
        if not torch.all(param[~mask.bool()] == 0):
            raise RuntimeError(f"Mask enforcement violation in {name}")
```

Gradient hooks registered to prevent gradient flow to pruned locations:
```python
register_mask_gradient_hooks(model, masks)
```

## Expected Findings (Scientifically)

This experiment should answer:

1. **Does router maturity help sparse tickets?**
   - If Δ_routing(s) >> 0: YES, strong routing effect
   - If Δ_routing(s) ≈ 0: NO, routing age doesn't matter
   - If Δ_routing(s) < 0: REVERSE, young router better (unlikely but interesting)

2. **Is there a sparsity frontier?**
   - Compute s*(R_100) = max{s : Δ_ticket(R_100, s) ≤ 5%}
   - Compute s*(R_0) = max{s : Δ_ticket(R_0, s) ≤ 5%}
   - If s*(R_100) >> s*(R_0): router shifts frontier significantly

3. **Does the effect strengthen at extreme sparsity?**
   - Compare Δ_routing(70%) vs Δ_routing(90%) vs Δ_routing(95%)
   - If trend is increasing: routing provides disproportionate protection at high sparsity

## Next Steps (After Deployment)

1. **Run full sweep** on VM (commit 276f8cc)
2. **Collect results**: 48 conditions × ~3 min = ~2.5 hours
3. **Aggregate**: Combine all sparsities + existing 80%
4. **Generate figures**: 3 main plots as specified in deployment guide
5. **Interpret findings**: Write up results w.r.t. routing-conditioned lottery ticket hypothesis
6. **Archive**: Save all records, figures, and summary to results directory

## Files Changed
- ✓ `src/moe_lth/experiments/run_sparsity_sweep.py` (820 lines, new)
- ✓ `tests/test_sparsity_sweep.py` (130 lines, new)
- ✓ `SPARSITY_SWEEP_DEPLOYMENT.md` (comprehensive guide, new)

## Version
- **Git commits**: 646e8a1, 276f8cc on `main`
- **Base**: Extends corrected router_age_recovery (commit 55dd047)
- **Protocol**: True LTH rewind (E_0 under E_T-derived mask)
- **Calibration tolerance**: 1e-3 (0.1% of target confidence)
- **Recovery budget**: 2500 steps (matches 80% experiment)
