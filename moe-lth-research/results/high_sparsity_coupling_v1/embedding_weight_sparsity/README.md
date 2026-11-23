# Reproduce this checkpoint-only analysis

From the repository root, using the existing environment:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-highres/bin/python results/high_sparsity_coupling_v1/embedding_weight_sparsity/analyze.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-highres/bin/python results/high_sparsity_coupling_v1/embedding_weight_sparsity/validate_and_enrich.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-highres/bin/python results/high_sparsity_coupling_v1/embedding_weight_sparsity/finalize.py
```

Run the scripts in this order; the second enriches the newly generated CSVs, and the third finalizes the figures, interpretation, validation and artifact manifest. All writes stay inside this directory. No training, CUDA execution, checkpoint mutation, pruning, recovery or modification of previous experiment outputs is performed. The file-hash cache reuses only same-path, same-size, same-mtime records from this analysis; the first execution recomputes all hashes. Remove only this directory's physical_file_hashes.json to force rehashing on another execution.

`analysis_config.json` was written before statistical computation. `sparsity=0` denotes the dense control and `.8`/`.85` denote expert pruning conditions, not measured embedding sparsity. `role=registered` records define the trajectories; other roles are checkpoint aliases and are not additional observations. Historical step zero measures the actual original saved embedding only, without rebuilding a masked model. Missing historical trained states are not reconstructed.

`checkpoint_provenance.csv` maps every record to a full file SHA256, original-dtype tensor hash and CDF. `cdf/<E_sha256>.npz` stores the complete empirical distribution: `absolute_weight`, `multiplicity`, `cumulative_count`, `cdf`, `entries`, and `exact_zero_count`. Hashes include original dtype and shape; conversion to float64 cannot recover precision absent from float32 weights. Saved head/embedding values are checked equal; runtime parameter identity is checked using the actual model class. Checkpoint serialization may clone the two state-dict entries.

`paired_threshold_curves.csv` contains all 61-threshold contrasts for both scales. `paired_row_statistics.csv` and `paired_column_statistics.csv` preserve ID-matched contrasts. Row/column statistics additionally report initialization changes. `endpoint_seed_summary.csv` uses five equally weighted seeds, with descriptive spread rather than entry-level confidence intervals. Both ticket-label tracks are annotations only.

See `findings.md`, `numerical_checks.json`, `independent_validation.json`, and `protected_output_validation.json` for results and checks. Figures A–F are available as PNG and PDF; all primary plots use rerun data, avoiding duplicate historical observations.
