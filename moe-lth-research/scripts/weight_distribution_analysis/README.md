# Weight distribution analysis

From the repository root:

```bash
PYTHONPATH=src .venv-highres/bin/python -m moe_lth.analysis.weight_distributions
```

The default run analyzes the four prioritized E32/E16 configurations. Use `--cases e32/seed_7/final_s0.80` to select a subset, or pass other existing extension cases. It performs no training and never writes to checkpoint directories. It needs NumPy, SciPy, Matplotlib, and PyTorch; the repository environment provides them.

Outputs go to `results/weight_distribution_analysis/`. The generated README describes matrix identity, plot interpretation, masking, statistics, limitations, and exact checkpoint provenance. PNG/PDF sheets provide individual expert rows and a pooled row for each layer/projection. All-weight and fixed-mask-survivor distances are separate. The sparse initial state is step 2500 before the extended continuation; it is not the immediate post-pruning state.

A missing or incompatible checkpoint produces an audit entry and a nonzero exit code. No alternative checkpoint is substituted. CPU extraction maps one checkpoint at a time and writes temporary NumPy arrays beneath the output case; those arrays are deleted on successful completion. Allow several minutes per configuration for exact pooled empirical distances and plot generation.

For the focused E32 seed-7 80%/85% comparison:

```bash
PYTHONPATH=src .venv-highres/bin/python -m moe_lth.analysis.focused_weight_distributions
```

This reuses checkpoint verification and all statistical calculations, but generates only ten layer-organized figure pairs under `results/weight_distribution_analysis/focused_winners/`: five four-stage overlays and five direct final comparisons. Expert weights are pooled across all 32 experts within each layer/projection. Exact-zero-excluded versions are included for fc1 and fc2. Shared bin edges and axes are recorded in `plot_settings.json`. No per-expert pages or geometry plots are produced in this mode.
