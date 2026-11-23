# Dataset A lower-sparsity endpoint analysis

Read `audit_report.md` before using `findings.md`. The original experiment artifacts are read-only; only this newly created analysis directory is written. No training, Dataset B restart, or training-code edit is performed.

From the repository root:

```bash
OMP_NUM_THREADS=2 .venv-highres/bin/python -u results/dataset_a_full_lower_sparsity/analysis_final/scripts/analyze.py
.venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/interpret.py
.venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/verify_random_controls.py
.venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/refine_figures.py
.venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/finalize_report.py
.venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/verify.py
```

`analyze.py` refuses existing generated files. For reproduction without overwriting this analysis, set `DATASET_A_ANALYSIS_OUT` to a fresh output directory for all six commands. This performs full-byte hashing (four independent CPU workers), CPU checkpoint inspection, independent mask reconstruction, metadata/endpoints/rewind matching, and statistical analysis. No model evaluation or optimizer update is executed. The first serial audit was stopped to improve I/O throughput; `analysis_parallel.log` records the completed audit.

- `canonical_lower_sparsity.csv`: 1,100 lower-sparsity sparse runs; `sparsity` is the target fraction, `actual_sparsity` the realized fraction of prunable expert weight elements. Retained whole-model counts are also recorded.
- `dense_references.csv`: 275 unique finalized table-backed dense endpoints; raw checkpoint availability is not raw verification.
- `canonical_combined.csv`: 2,750 unique learned-mask scientific conditions, including original 70–95% runs with finalized-table provenance.
- `setting_summary.csv`: five paired seed gaps, SD, t intervals, one-sided upper bounds, seed counts and Holm p values (220 settings).
- `ticket_candidate_counts.csv` and `replicated_candidate_counts.csv`: descriptive strict and exploratory sensitivity counts, with denominators.
- `paired_age_contrasts.csv`, `paired_sparsity_contrasts.csv`, `paired_65_vs_50.csv`: within-seed contrasts and stated multiplicity families.
- `highest_observed_sparsity.csv`: exact tested maxima, separately by architecture, router age and criterion; NA means no tested match.
- `random_controls.csv`: separately matched earlier 32E controls; limited age coverage.
- `figures/`: PDF/PNG; `plot_*_data.csv` and canonical tables preserve exact inputs.
- `verification_checks.csv`, `final_verification_checks.csv`, `input_hash_verification.csv`, `independent_verification.json`: audit evidence and independent output verification.

The unadjusted intervals assume independent source seeds and approximately normal paired gaps. With five seeds, precision is limited. Router ages and sparsities are repeated measurements, not independent replicates. Holm controls familywise error under arbitrary dependence; each family is stated in the reports and column names. Endpoint matches alone do not establish equivalence, mechanism, or initialization-only LTH.

To regenerate only figures from exported tables:

```bash
.venv-highres/bin/python results/dataset_a_full_lower_sparsity/analysis_final/scripts/plot.py
```

This writes a new `replotted/figures/` tree and refuses existing figures.
