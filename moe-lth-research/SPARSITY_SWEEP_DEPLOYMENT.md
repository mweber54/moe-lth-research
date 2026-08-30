# Sparsity Sweep Deployment Guide

## Overview
The sparsity sweep experiment tests how router maturity (R₀ vs R₁₀₀) affects the sparse-vs-dense recovery gap across different expert sparsity levels (60%, 70%, 90%, 95%).

**Key research question**: Does router maturity provide increasing protection at higher sparsities? Is there a sparsity frontier where sparse expert rewinding remains competitive with dense baselines?

## Experimental Design

### Sparsity Levels
- **60%**: Test low-sparsity regime (light pruning)
- **70%**: Test moderate sparsity
- **90%**: Test high sparsity (extreme expert thinning)
- **95%**: Test ultra-high sparsity (near-removal regime)

Note: 80% already has complete results in `results/router_age_lth_80pct_dense_v2/`

### Router Ages (Endpoints Only)
- **R₀**: Router from step 0 (untrained)
- **R₁₀₀**: Router from step 2500 (fully trained)

This tests the extreme contrast without intermediate checkpoints for efficiency.

### Reference Seeds
- **Seed 7, 17, 29**: Same three as existing 80% experiment for consistency

### Conditions Per Sparsity
- 2 router ages × 3 seeds = 6 seed-age pairs
- Each pair: 1 sparse + 1 dense control
- Total per sparsity: 12 conditions (6 sparse + 6 dense)
- **Grand total**: 4 sparsities × 12 conditions = 48 new conditions

## Protocol (Lottery Ticket Hypothesis)

### Mask Discovery
For each sparsity level s ∈ {0.60, 0.70, 0.90, 0.95}:
1. Load fully trained expert weights **E_T**
2. Apply magnitude pruning: discover top-(1-s) weights per expert
3. Mask: **m_s = MagnitudeMask(E_T, s)**

### Rewind
1. Load initial expert weights: **E_0**
2. Build sparse ticket: **E_sparse,s = m_s ⊙ E_0**
   - Surviving locations: values from E_0
   - Pruned locations: exactly zero
   - **Never** use values from E_T

### Training
- Recover from fresh optimizer for 2500 steps
- Same data order as 80% experiment
- **Mask enforcement**: pruned weights must stay zero throughout

## VM Deployment Commands

### Prerequisites
On the VM, ensure you have:
```bash
cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research
source .venv/bin/activate  # or your venv path
git pull origin main  # Get commit 646e8a1 with sparsity sweep code
```

### Full Sparsity Sweep (All Four Levels)
```bash
python -m moe_lth.experiments.run_sparsity_sweep \
  --configs \
    configs/router_age_reference_seed7.yaml \
    configs/router_age_reference_seed17.yaml \
    configs/router_age_reference_seed29.yaml \
  --output-dir results/router_conditioned_sparsity_sweep \
  --recovery-steps 2500
```

**Expected duration**: 
- ~100-150 minutes total for all 48 conditions (4 sparsities × 12 conditions)
- ~2-3 minutes per condition (depending on GPU availability)
- Produces ~48 JSON records + aggregate CSVs

### Test Run (Sanity Check on 2 Steps)
Before full run, validate the protocol with a minimal budget:
```bash
python -m moe_lth.experiments.run_sparsity_sweep \
  --configs configs/smoke.yaml \
  --output-dir results/sparsity_sweep_test \
  --recovery-steps 2
```

This completes in ~10 seconds and validates:
- Config loading
- Mask discovery
- Dense/sparse condition assembly
- Rewind protocol correctness
- Output structure

## Dense Baseline Reuse
The code automatically reuses dense E₀ baselines across sparsities for efficiency:
- Dense R₀ computed once per seed
- Dense R₁₀₀ computed once per seed
- Reused for all four sparsity levels
- Hash validation ensures exact compatibility

**Result**: Only 6 dense conditions run (1 per seed-age pair), not 24.

## Output Structure
```
results/router_conditioned_sparsity_sweep/
├── sparsity_0.60/
│   ├── seed_7/
│   │   ├── age_000pct_sparse/
│   │   ├── age_000pct_dense/
│   │   ├── age_100pct_sparse/
│   │   ├── age_100pct_dense/
│   │   ├── pruning_metadata.json
│   │   └── lth_isolation_audit.json
│   ├── seed_17/
│   └── seed_29/
├── sparsity_0.70/
├── sparsity_0.90/
├── sparsity_0.95/
├── sparsity_sweep_all_records.json  (full records for all 48 conditions)
├── sparsity_sweep_all_records.csv   (incremental updates as conditions finish)
└── sparsity_sweep_paired.csv        (R_0 vs R_100 gaps per sparsity/seed)
```

## Key Metrics Per Condition

### Loss Metrics
- `initial_validation_loss`: At step 0
- `final_validation_loss`: At step 2500
- `best_validation_loss`: Minimum during recovery
- `early_auc`: Area under curve first 50% of steps
- `dense_reference_loss`: Dense baseline final loss for this router age

### Derived Metrics (Computed Offline)
- **Sparse-dense gap**: Δ_ticket(R, s) = L_sparse(R, s) - L_dense(R, s)
- **Router benefit**: Δ_routing(s) = Δ_ticket(R₀, s) - Δ_ticket(R₁₀₀, s)
  - Positive = mature routing reduces sparsity penalty
  - Zero = routing age doesn't matter for this sparsity
  - Negative = young routing is actually better (unexpected)

### Routing Metrics
- `mean_selected_probability_initial/final`: Average gating probability
- `routing_entropy_initial/final`: Entropy of expert selection distribution
- `top1_top2_margin`: Gap between top-1 and top-2 expert logits
- `expert_utilization_initial/final`: Fraction of experts receiving tokens

### Gradient Metrics
- `per_expert_gradient_norms`: Norm by expert (8 experts/layer × 4 layers)
- `expert_gradient_norms`: Aggregated by component
- `router_gradient_norms`: Router parameter gradient norms
- `shared_gradient_norms`: Shared parameter gradient norms

## Integrity Checks (Automatic)

The runner validates:
1. **Rewind correctness**: Every retained weight = E₀, every pruned weight = 0
2. **Mask source**: Mask derived from E_T, not E₀ (checked via hash)
3. **Shared state**: Identical across all conditions in a seed
4. **Router uniqueness**: R₀ hash ≠ R₁₀₀ hash within each sparsity
5. **Batch reproducibility**: Training and validation sequences hashed
6. **Component isolation**: Only router changes between R₀ and R₁₀₀

Fails loudly if any check violated; never silently falls back.

## Post-Experiment Analysis

### Combine with 80% Results
After the sweep completes, load existing results and merge:
```python
import pandas as pd
import json

# Load new results
new_records = json.load(open("results/router_conditioned_sparsity_sweep/sparsity_sweep_all_records.json"))

# Load existing 80% results
existing_80 = json.load(open("results/router_age_lth_80pct_dense_v2/router_age_recovery_all_records.json"))

# Filter to R_0 and R_100 only for 80%
existing_80_endpoints = [r for r in existing_80 if r["router_age_percent"] in [0, 100]]

# Combine
all_records = new_records + existing_80_endpoints
df = pd.DataFrame(all_records)
```

### Generate Comparison Table
```python
# Sparse-dense gap by sparsity and router age
pivot = df.groupby(["sparsity", "router_age_percent"]).agg({
    "final_validation_loss": ["mean", "std", "count"]
})
```

Expected shape:
```
                           sparse_final (mean)  dense_final (mean)  gap (mean)
sparsity  router_age
0.60      0                                                              
          100                                                            
0.70      0                                                              
          100                                                            
0.80      0                                                              
          100                                                            
0.90      0                                                              
          100                                                            
0.95      0                                                              
          100
```

### Create Figures
**Figure 1 – Sparse-dense gap vs sparsity**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
sparsities = [0.60, 0.70, 0.80, 0.90, 0.95]

for router_age in [0, 100]:
    gaps = []
    for s in sparsities:
        sub = df[(df["sparsity"] == s) & (df["router_age_percent"] == router_age)]
        gaps.append(sub["final_validation_loss"].mean() - sub[sub["sparse_condition"]]["final_validation_loss"].mean())
    
    ax.plot(sparsities, gaps, marker="o", label=f"R_{router_age}")

ax.set_xlabel("Expert Sparsity")
ax.set_ylabel("Sparse-Dense Gap (L_sparse - L_dense)")
ax.legend()
ax.grid()
plt.savefig("sparsity_ticket_gap.png", dpi=300, bbox_inches="tight")
```

**Figure 2 – Final loss vs sparsity**
- Four lines: sparse R₀, sparse R₁₀₀, dense R₀, dense R₁₀₀
- Shows overall loss trajectory and router effect

**Figure 3 – Router benefit (gap reduction)**
- Plot: Δ_routing(s) = gap(R₀, s) - gap(R₁₀₀, s)
- Shows magnitude of router age advantage per sparsity level

## Expected Findings

### Most Likely Outcome
- **Gap increases with sparsity**: Δ_ticket(R₀, s) grows as s increases
- **Router provides protection**: Δ_routing(s) > 0 (R₁₀₀ reduces gap)
- **Effect is sparsity-dependent**: Δ_routing(s) may grow, shrink, or plateau

### Interpretation
- **s* = max{s : Δ_ticket(R₁₀₀, s) ≤ ε}**: Sparsity frontier for mature routing
- Compare to R₀ frontier: does router maturity increase frontier?
- If R₁₀₀ frontier >> R₀ frontier: strong evidence of routing-conditioned tickets

## Troubleshooting

### Dense calibration error > 1e-3
- Expected for some dense conditions due to numerical precision limits
- Tolerance 1e-3 is strict (0.1% of target) but achievable
- If fails: expand calibration search (already set to max_temp=1M, 31 grid points, 4 rounds)

### Memory issues
- Each condition ≈ 1-2 GB during training
- Can run 1-2 conditions in parallel if VM has ≥ 8 GB VRAM
- Sequential execution (default) is safer

### Checkpoint not found errors
- Ensure reference configs (router_age_reference_seed*.yaml) exist
- Ensure they have checkpoint_steps: [0, 250, 500, 1000, 1500, 2000, 2500]
- Run reference training first if missing: `python -m moe_lth.training.train -c <config>`

## Timeline Estimate
- **Setup**: 5 minutes (git pull, verify reference checkpoints)
- **Full sweep**: 2-3 hours (48 conditions × 2.5-4 min each)
- **Post-analysis**: 30 minutes (aggregation, figures, summary)
- **Total**: ~3.5 hours wall-clock

## Success Criteria
✓ All 48 conditions complete without assertion failures  
✓ Sparse-dense gaps computed for all (sparsity, router_age, seed) pairs  
✓ Router benefit (gap reduction) computed and meaningful  
✓ Output CSVs contain valid numeric data (no NaN/Inf in final_loss)  
✓ Integrity audits pass (protocol compliance verified)  
✓ Figures generated and interpretable  

## Questions?
- Check `sparsity_sweep_all_records.json` for full condition metadata
- Check `<condition_dir>/metadata.json` for per-condition protocol details
- Check `<seed_dir>/lth_isolation_audit.json` for rewind correctness validation
