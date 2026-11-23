# Dataset A original-initialization IMP at 80% and 85%

Implementation is prepared; the 25 trajectories / 50 final recoveries have **not been launched**. Empty tables mean no scientific results, not failures. `preflight.json` records checkpoint availability and byte hashes. `screening.json` records completed-run counts only.

Canonical OMP uses **expert-local** magnitude pruning across each expert's weight matrices (not global pruning). `src/moe_lth/pruning/imp.py` preserves that scope, with integer keep counts computed independently from the original size of every expert. Consequently actual sparsity differs from nominal sparsity by unavoidable integer rounding, exactly as in OMP. Ties use stable ascending sorting and flattened canonical parameter order; this explicit deterministic policy is a controlled difference from unspecified `topk` tie ordering in OMP.

Nine discovery stages train masks at 0, 20, 36, 48.8, 59.04, 67.232, 73.7856, 79.02848, and 80 percent, each producing the next mask through 85 percent. Two separate final recovery stages evaluate saved masks 8 and 9. Total: 225 discovery stages + 50 final recoveries = 687,500 attempted steps. Discovery alone costs 562,500 attempted steps. Discovery compute must be reported separately from final recovery compute.

Each stage loads the entire original step-0 checkpoint, masks only expert weight matrices, and invokes the existing Dataset A `_run_recovery_condition` trainer. Expert and shared initialization hashes are checked; the router is loaded from the same step-0 checkpoint. Native routing and auxiliary loss remain gradient bearing. The trainer asserts router optimizer membership, fresh AdamW state, nonzero observed router gradients, and actual final parameter drift. Masks zero gradients and are reapplied after updates, including AMP skips. The canonical scheduler is **none**; adding a scheduler would change the comparison protocol. Both attempted and applied updates are recorded. Train and validation sequence hashes must match canonical records before any training.

## Commands

Run from the repository root with the verified `.venv-highres` torch 2.7.0 environment:

```bash
.venv/bin/python -m pytest tests/test_imp.py tests/test_pruning.py tests/test_router_age_recovery.py -q
.venv-highres/bin/python scripts/imp_80_85/run.py
.venv-highres/bin/python scripts/imp_80_85/benchmark.py --experts 8 --seed 7 --steps 20
```

After explicit authorization, launch all 25 trajectories:

```bash
.venv-highres/bin/python scripts/imp_80_85/run.py --execute
```

Launch an explicitly selected group, or resume all verified saved states:

```bash
.venv-highres/bin/python scripts/imp_80_85/run.py --execute --groups 8:7
.venv-highres/bin/python scripts/imp_80_85/run.py --execute --resume
```

Output is restricted to this directory or its subdirectories. An exclusive lock prevents competing writers. No canonical experiments are written. Original and matched dense checkpoint file hashes are checked; absent/mismatched files halt execution. No automatic initialization reconstruction or fallback is performed. Source-final checkpoints are used only for reference-routing diagnostics.

## Interruption and artifacts

Every 100 steps and at stage end, the opt-in existing trainer writes an atomic `interruption.pt` with model, AdamW, scaler, RNG states, batch position, curves, update counts, and log lengths. Resume checks configuration, component/mask/data identities, truncates uncommitted logs, restores all training state, and continues from the next saved batch. At most 99 attempts since the last snapshot are replayed from that exact snapshot. Optimizer continuity within a resumed stage is required; fresh optimizers apply between stages. The snapshot is removed after the durable `stage.json` commit to save disk. Completed checkpoints are verified before reuse. An interruption before the first durable snapshot fails loudly and preserves artifacts; it does not silently restart. Such a directory needs explicit manual archival before a fresh attempt.

Each trajectory preserves `mask_01.pt` through `mask_09.pt`, mask provenance JSON, all nine trained discovery checkpoints, and both independent final checkpoints. `stage.json` includes initial/final router hashes, actual drift, gradient observation, duration, attempted/applied counts, and batch hashes. Complete validation curves are `metrics.jsonl`; per-attempt training losses (including auxiliary loss), gradients, AMP flags, and scale values are `gradient_stats/gradient_stats.jsonl`. Router and shared component snapshots remain available for future geometry work. No geometry analysis is implemented.

`imp_round_metrics.csv`, `imp_mask_manifest.csv`, `final_recovery_metrics.csv`, `imp_vs_omp_comparison.csv`, and `checkpoint_manifest.csv` are reconstructed from committed stage metadata on startup and after each trajectory. Paired comparisons use the exact canonical architecture/seed/sparsity R0 rows, with positive improvement = OMP gap minus IMP gap. `screening.json` provides 2/5/10 percent validation screens and per-architecture/sparsity means and sample SDs. Curve paths support matched-step comparisons; no selected-seed-only summary is produced.

Thresholds are descriptive screens, not proof of winning tickets or statistical equivalence. Promising candidates still require untouched held-out test evaluation, matched-step dense/sparse trajectories, and seed variation/equivalence assessment. Expert-only pruning is an adaptation of original LTH. Test evaluation is deliberately not performed during discovery. All candidates and negative controls remain preserved.

## Resource planning

The short benchmark is separate under `benchmarks/` and excluded from all scientific tables. It measures an actual canonical-shape dense-mask stage using the same trainer, optimizer, precision, validation, and diagnostics. Its linear GPU-hour estimate includes fixed initial/final diagnostics, and is provisional on a shared busy GPU. A short E8 timing cannot establish scaling across all architectures. Storage estimates must include all 275 trained checkpoints, 225 masks, logs, and one temporary model + two AdamW moment tensors per active trajectory. Checkpoint sizes grow approximately with expert count. See `resource_estimate.json` for measured values and explicit limitations.

All 25 architecture/seed groups have verified existing original, source-final, and matched-dense checkpoints. The default `.venv` currently has torch 2.13.0; it is suitable for development tests but fails the full runner version guard. `.venv-highres` has the required torch 2.7.0. No missing checkpoint substitution is necessary.

Verification: 17 targeted tests pass with canonical torch 2.7.0, including a full tiny nine-stage discovery trajectory, both independent recoveries, and bit-identical mid-stage interruption/resume with dropout. The canonical E8/H100 20-step benchmark took 70.18 seconds (including diagnostics and resume snapshot I/O); linear E8-rate extrapolation is about 670 GPU-hours for 275 stages. An expert-count-proportional planning scenario is about 1,541 GPU-hours; neither scenario is a statistical bound or a reliable runtime forecast. The full experiment needs explicit launch authorization.

The canonical environment lacks pytest; tests were run using its torch while loading the installed development pytest package:

```bash
.venv-highres/bin/python - <<'PY'
import torch, sys, glob
sys.path.extend(glob.glob(".venv/lib/python*/site-packages"))
import pytest
raise SystemExit(pytest.main(["tests/test_imp.py", "tests/test_pruning.py", "tests/test_router_age_recovery.py", "tests/test_fixed_budget_dataset.py", "-q"]))
PY
```

Measured tensor storage across the actual five architecture sizes is 40.90 GiB for trained checkpoints plus 8.09 GiB for intermediate masks. Reserve at least 80 GiB for component snapshots, logs, serialization overhead, and temporary resume state. Existing baseline artifacts are referenced, not copied.
