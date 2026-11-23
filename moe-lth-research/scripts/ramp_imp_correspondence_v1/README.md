# RAMP–IMP correspondence v1

Pre-implementation audit: `reports/ramp_imp_audit.md`. Final report: `reports/ramp_imp_correspondence.md`.

This module performs CPU inference and retrospective analysis only. Historical checkpoints/masks/results are read-only inputs; no trainer is imported and no training launcher exists. Source is isolated from pre-existing repository edits.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/run.py --output results/ramp_imp_correspondence_v1
.venv/bin/python -m pytest tests/test_ramp_imp_correspondence.py tests/test_ramp_imp_pipeline.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/supplement.py --output results/ramp_imp_correspondence_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/validate_results.py --output results/ramp_imp_correspondence_v1
# Rebuild analysis from existing profile files:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/ramp_imp_correspondence_v1/run.py --output results/ramp_imp_correspondence_v1 --summarize-only
```

The fixed calibration manifest is committed before profiling: 64 sequences ×128 bytes, 32 per domain, calibration seed20261113, fixed batches of32; it removes exact sequences appearing anywhere in the full validation split. Larger calibration is allowed only in a new output directory; it must not be chosen from recovery outcomes. Source domains use exact original-source substring identity. Domain subsets share pooled dispatch competition.

The runner verifies all 30 historical endpoint identities, original initialization and binary mask hashes, final sparse checkpoint hashes, paired dense data hashes, and curve endpoints. It profiles reference steps0/250/1250/2500 and directly ancestral discovery round00 final. Resume requires matching registration and input/checkpoint hashes; complete result parquet files are rehashed. Registered source changes require a new output directory. Original masks are never rewritten. Undefined constant-score correlations remain NaN. No neuron-level p-values or fitted recovery models are produced.

Outputs:

- `registration.json`, `calibration_manifest.json`, `verified_recovery_pairs.csv`: reproducible identities.
- `e*_seed*/<profile>/profiles.parquet`: source-specific and pooled RAMP, activation and routing statistics.
- `e*_seed*/<profile>/neurons_s*.parquet`: pooled neuron–mask–checkpoint pairings, exact retention and initial/current surviving magnitudes/signs.
- `mask_run_results.csv`, `h1_seed_results.csv`, `h1_seed_summary.csv`, `h1_baseline_differences.csv`, `h2_descriptive_associations.csv`.
- `h1_groups.parquet`, `retention_deciles.parquet`, `checkpoint_rank_stability.csv`.
- `historical_mask_controls*.csv`: verified existing randomized supports; nested draws kept separate from source replications.
- `adaptation_context_separate_protocol.csv`: condition-matched 5000-step outcomes, not pooled with main H2.
- `figures/*.pdf`, `figures/*.png`, and per-figure source manifest.

Dependencies use the existing `.venv` (torch, numpy, scipy, sklearn, pandas, pyarrow, matplotlib). pandas is available through existing dataset dependencies; install nothing for this task. Calibration/profiling is FP32 CPU, with FP64 accumulation, rather than historical FP16 GPU training numerics. No exact historical dispatch equivalence is claimed.

`H3_design.md` prepares later controlled recovery. `allocate_weight_mask` is a tested non-training mask bridge; it is not applied to original artifacts or launched for recovery.
