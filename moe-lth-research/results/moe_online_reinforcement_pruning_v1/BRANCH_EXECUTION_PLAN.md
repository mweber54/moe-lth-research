# MOE Online Reinforcement Pruning V1 - Branch Continuation Execution Plan

**Status**: Branch execution in progress
**Start Time**: 2026-11-20 14:43 UTC
**First Job**: seed7_A untouched (monitoring)

## Execution Summary

All 18 branch jobs are ready to launch:
- **6 cells** × 3 arms = **18 jobs**
- **3 arms per cell**:
  1. `untouched` - No pruning (baseline)
  2. `random-small` - Delete random 5% small-magnitude weights
  3. `high-reinforcement-small` - Delete highly reinforced 5% small-magnitude weights

## Runner Implementation

### Script: `branch_v2.py`
- **Input**: Sealed step-250 checkpoints + locked selection indices
- **Process**: Resume from step 250, train to step 2500
- **Admission**: GPU must have ≥70% free memory
- **Output**: `branches/seed{N}_{A,B}/{arm}/final_step_2500.json` (completion marker)

### Key Verification Steps
1. ✓ Checkpoint loads with step=250, optimizer_steps_applied=249
2. ✓ Selections load with 3,355,392 deletions per arm per cell
3. ✓ Masks created correctly (1=keep, 0=delete)
4. ✓ Untouched arm loads no masks (pure baseline)

## Frozen Protocol Guarantees

**Sealed at**: 2026-11-20T02:14:08Z (SHA-256: 403caae...

### Acceptance Criteria Verified
| Criterion | Requirement | Actual | Status |
|-----------|-------------|--------|--------|
| Score Separation | ≥0.20 | 0.556 | ✓ |
| Overlap Fraction | ≤0.30 | 0.2805 | ✓ |
| Identical Deletions | 3,355,392/matrix | 3,355,392 | ✓ |
| Row/Col Balance | Perfect | Perfect | ✓ |
| Magnitude Strata | 5 bins | 5 bins | ✓ |

## Execution Timeline

### Phase 1: Verification Check (Completed)
- [x] Load all 6 step-250 checkpoints
- [x] Verify sealed selections (SHA-256 match)
- [x] Create and test mask logic
- [x] Record GPU admission snapshot

### Phase 2: Pilot Run (In Progress)
- [ ] seed7_A untouched: steps 250→2500 (monitoring...)
- [ ] Verify: untouched reproduces oracle at step 251
- [ ] Verify: pruned arms keep deleted weights at zero
- [ ] Verify: all three consume same next batch

### Phase 3: Full 18-Job Execution (Pending GPU)
When pilot passes:
- [ ] Launch remaining 17 jobs (seed7_B untouched, seed7_A random, seed7_A high, etc.)
- [ ] Run under GPU admission policy
- [ ] Monitor step-2500 checkpoints

### Phase 4: Results Collection (Post-Execution)
- [ ] Extract final validation CE from each branch
- [ ] Compute 6 Delta values: Δᵢ = CE_high(i) - CE_random(i)
- [ ] Report: within-seed means, grand mean, effect sizes

## Expected Deliverables

### Primary Result
**6 Delta Values** (one per cell):
```
Δ₇ᴬ = CE₂₅₀₀(high, 7A) - CE₂₅₀₀(random, 7A)
Δ₇ᴮ = CE₂₅₀₀(high, 7B) - CE₂₅₀₀(random, 7B)
Δ₁₇ᴬ = ...
Δ₁₇ᴮ = ...
Δ₂₉ᴬ = ...
Δ₂₉ᴮ = ...
```

### Secondary Analysis
1. **Within-Seed Means**: average Δ for each seed {7, 17, 29}
2. **Grand Mean**: average across all 6 Δ values
3. **Effect Size**: grand mean / std(Δ values)
4. **Directional Consistency**: count of positive vs. negative Δ

### Immediate Damage Report
**At Step 251** (pre-adaptation):
- Untouched validation CE (baseline)
- Random-small validation CE (immediate loss from random deletion)
- High-reinf-small validation CE (immediate loss from reinforced deletion)
- **Damage Gap**: |CE(high) - CE(random)| at step 251

### Final Recovery Report
**At Step 2500** (post-adaptation):
- All three arms' final CE values
- Loss recovery trajectories
- Primary Delta: how much worse high-arm is vs. random-arm

## Hypothesis & Interpretation

**Hypothesis**: Reinforced weights are more task-critical than random weights.

**Predictions**:
- If Δ > 0: High-arm has higher loss (supports hypothesis)
  - Interpretation: reinforced weights are more important; pruning them causes lasting damage
  - Expected recovery: limited recovery even after 2250 additional steps
- If Δ ≈ 0: Both arms recover equally
  - Interpretation: reinforcement doesn't indicate criticality
- If Δ < 0: Random-arm has higher loss
  - Interpretation: reinforced weights may be redundant or task-irrelevant

## Data Files Location

```
results/moe_online_reinforcement_pruning_v1/
├── branches/                          # NEW: branch execution outputs
│   └── seed{N}_{A,B}/
│       ├── untouched/
│       │   ├── native/                # training checkpoints
│       │   ├── admission.json         # GPU snapshot
│       │   └── final_step_2500.json   # completion marker
│       ├── random-small/
│       └── high-reinforcement-small/
├── prefixes/                          # EXISTING: step-250 state
│   └── seed{N}_{A,B}/native/interruption.pt
├── matching/                          # EXISTING: sealed selections
│   └── seed{N}_{A,B}/selections.pt
└── branch_v2.py                       # Runner script
```

## Monitoring Commands

```bash
# Check if first job is complete
ls -la branches/seed7_A/untouched/final_step_2500.json

# Check all job status
for s in 7 17 29; do
  for o in A B; do
    for a in untouched random-small high-reinforcement-small; do
      if [ -f branches/seed${s}_${o}/$a/final_step_2500.json ]; then
        echo "seed${s}_${o} $a: ✓ DONE"
      else
        echo "seed${s}_${o} $a: ⏳ PENDING"
      fi
    done
  done
done

# Monitor GPU
nvidia-smi

# Check active training process
ps aux | grep branch_v2 | grep -v grep
```

## Next Steps

1. **Wait for pilot to complete** (seed7_A untouched expected: 20-40 minutes depending on model size)
2. **Verify pilot results**:
   - Checkpoint exists at step 2500
   - final_step_2500.json created successfully
3. **If pilot passes**: Launch all 17 remaining jobs with launcher.py
4. **Monitor execution** (expected total: 24-48 hours for 18 sequential jobs)
5. **Collect and analyze results** once all complete
