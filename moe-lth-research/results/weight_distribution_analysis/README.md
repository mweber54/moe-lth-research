# Router and expert weight distributions

Run from repository root with `.venv-highres/bin/python -m moe_lth.analysis.weight_distributions` (set `PYTHONPATH=src` if not installed). Override `--cases`, `--root`, or `--output` as needed. Requires NumPy, SciPy, Matplotlib and PyTorch. Checkpoints are read-only; only one checkpoint is mapped at a time. Four CPU workers process extracted layer/matrix groups; use `--workers 1` for minimum memory. Scratch extracted matrices are deleted after each case. Statistics and distances use every weight, without sampling or rescaling.

## Checkpoints and verification

Each case uses its extension manifest and the matched dense manifest in the same architecture/seed directory. `provenance.json` records exact paths and verified SHA256s: dense source at step 2500, sparse beginning of extension at step 2500, sparse endpoint at 5000, and matched dense endpoint at 5000. The original IMP trajectory stage and mask_08 (80%) or mask_09 (85%) are checked against the source and mask hashes. The sparse beginning here is the recovered subnetwork before *extended continuation*, not immediately after pruning. Original recovery router provenance is also recorded. These are fresh AdamW/GradScaler continuations, not exact optimizer resumes. Training/validation sequence hashes and protocol identities must match. Success means sparse validation CE <= matched dense validation CE at 5000, with no tolerance or statistical-significance claim. Missing or incompatible inputs are reported, never replaced.

## Plots

Each layer/matrix directory contains numbered PNG/PDF sheets, with one row per expert (and a pooled row), or the layer router. The actual architecture uses fc1 = up projection followed by GELU and fc2 = down projection; there is no gated FFN or gate matrix. Biases and router temperature buffers are excluded. Experts are identified by layer and index, never matched by weight similarity.

Columns 1–3 show normalized signed histograms, signed empirical CDFs, and absolute empirical CDFs. Columns 4–6 repeat these with sparse exact zeros excluded (dense curves retain all entries). All four stages are overlaid, including initial/final sparse and final dense/sparse comparisons. Histogram area is one; sparse zero atoms are represented by a finite-width bin and must not be read as continuous density. CDF F(t) = count(w <= t)/N; absolute CDF uses |w|. No Gaussian assumption or KDE is used because sparse distributions contain point masses. CDF rendering retains up to 1500 ordered points plus zero and endpoints; computed statistics and distances are exact empirical calculations. Axes and histogram bins are shared across stages and experts within each layer/matrix type; raw weights are never standardized. Pooling is within layer and matrix type. Router distributions remain separate from expert distributions.

## Tables and interpretation

Each case exports CSV and JSON statistics, distances, growth, and heterogeneity. Standard deviation is population SD; absolute quantiles are 25/50/75/90/95/99 percentiles. L1 = sum |w| and L2 = sqrt(sum w²). All, surviving (mask true, including natural zeros), and nonzero (exact w != 0) subsets are distinct. Dense surviving subsets use the paired sparse mask coordinates; masked_fraction is zero for dense weights. natural_zero_count counts exact zeros at kept coordinates, not pruned zeros.

Distances cover every pair of stages for comparable layer/matrix/expert identities: Wasserstein-1 in weight units and KS = sup_t |F(t)-G(t)|, for signed and absolute weights. All-weight and fixed-mask-survivor comparisons are explicitly labeled. No KS p-values or independent-weight inference is claimed. Growth uses paired surviving coordinates: mean(|final|-|initial|), ratio of mean absolute magnitudes, fraction that grew, and change per 2500 attempted steps. Normalized parameter drift = ||final-initial||₂/||initial||₂; undefined zero-denominator quantities are null. Dense growth is also measured at the same sparse-mask coordinates. Router drift uses all router coordinates.

Heterogeneity summarizes population SD and range across experts of growth and drift, separately per layer and projection. Root router and expert comparison tables attach matched CE outcomes; `router_total_drift` combines all router layers and cross-checks normalized drift against the training manifests. `exploratory_findings.md` and `exploratory_summary` report configuration-level descriptive comparisons. The outcome plot shows layer router drift and per-layer expert growth heterogeneity against matched CE gaps. Router outcome sheets overlay initial and final router distributions across the selected cases, with shared raw weight limits. All comparisons are exploratory. Small selected configuration counts, dependent weights, different sparsities, and architecture/seed confounding prevent causal claims. Weight distribution similarity is distinct from geometric or subspace alignment and supplies no evidence of alignment.


Run results:

- `e32/seed_7/final_s0.80`: complete; sparse − dense CE = -0.035600
- `e32/seed_7/final_s0.85`: complete; sparse − dense CE = -0.021205
- `e32/seed_17/final_s0.80`: complete; sparse − dense CE = 0.021117
- `e16/seed_17/final_s0.80`: complete; sparse − dense CE = 0.001161

In `outcome_comparison`, expert-panel circles denote fc1 and triangles denote fc2; each point is a layer. `router_outcomes_sparse_initial` and `router_outcomes_sparse_final` overlay router distributions across configurations.
