# Focused E32 seed-7 weight distributions

Only `e32/seed_7/final_s0.80` and `e32/seed_7/final_s0.85` are included. Run:

```bash
PYTHONPATH=src .venv-highres/bin/python -m moe_lth.analysis.focused_weight_distributions
```

## Ten figures, each in 300-dpi PNG and vector PDF

- `router_stages_all`, `fc1_stages_all`, `fc2_stages_all`: one row per transformer layer; the left three columns show 80% sparsity, the right three show 85%. Columns are signed density histogram, signed empirical CDF, and absolute empirical CDF. Gray = dense source, orange = sparse initial, blue = sparse final, green = matched dense final.
- `fc1_stages_nonzero`, `fc2_stages_nonzero`: the same comparisons after removing exact zeros from sparse curves only; dense curves retain all weights.
- `router_final_all`, `fc1_final_all`, `fc2_final_all`: direct 80%- versus 85%-sparse final distributions, orange versus blue, with one layer per row.
- `fc1_final_nonzero`, `fc2_final_nonzero`: direct final comparisons after excluding exact zeros from both sparse models.

All 32 experts are pooled within each layer/projection. No per-expert pages, geometry figures, or other configurations are generated. fc1 is the GELU input/up projection and fc2 the output/down projection; no gate matrix exists. Routers remain separate and unpruned.

## Interpretation and controls

Weights are raw and never standardized. For each matrix type and layer, the same 100 histogram bins and weight limits cover every stage of both configurations, including the nonzero and direct-comparison versions. Histogram density integrates to one within each displayed distribution. Histogram y limits also match between the two sparsity columns. CDFs use F(t)=count(w<=t)/N, or |w| for magnitudes, with common probability limits. A sparse histogram's zero spike is a finite-bin representation of a point mass. No Gaussian model or smooth density fit is assumed. `plot_settings.json` records all limits and bin edges.

All weights contribute to counts, summaries, and distributional distances. For display only, CDFs retain up to 2500 ordered support locations plus zero and adjacent support points; probabilities are calculated from complete empirical counts. Removing exact zeros is not the same as selecting mask survivors: naturally zero surviving coordinates are excluded in nonzero figures, while fixed-mask survivor statistics include them.

## Provenance and statistics

The original analysis verifier and statistics implementation are reused unchanged: canonical dense matching, checkpoint SHA256s, seeds, architecture, source/target steps, original and continuation data hashes, continuation protocols, mask hashes, shapes, and masked-zero integrity are checked. Checkpoints are mapped sequentially on CPU; scratch extracted arrays are removed after success. The two cases must also share identical dense source and dense-final checkpoint hashes.

Each case directory contains verified `provenance.json` and the existing CSV/JSON `statistics`, `distances`, `growth`, and `heterogeneity` tables, including per-expert statistics but no per-expert figures. All-weight, mask-surviving, and nonzero statistics remain distinct. Signed/absolute Wasserstein and KS comparisons retain all entries. See the parent analysis README for full statistical definitions. Dense source and sparse initial refer to step 2500; final endpoints refer to step 5000. Sparse initial is the start of extended continuation, not immediately after pruning. Continuations use fresh AdamW/GradScaler state. Distributional similarity does not establish geometric or subspace alignment.
