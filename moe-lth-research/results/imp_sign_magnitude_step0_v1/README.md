This directory contains a checkpoint-only step-zero analysis. Existing checkpoints and prior results are read-only; no training is invoked.

`baseline_gate.py` reproduces the original fixed-validation CE using the original evaluator. Its 15 exact matches are in `baseline_reproduction.csv`; `baseline_gate.json` controls permission to run perturbations.

`protocol.md` defines provenance, transformations, RNG derivation, distances, routing readout, assertions, and interpretation limits. `analyze.py` runs requested doses and ablations. `matched_signs.py` adds sign permutations matched to A's distance, using weights alone. Results remain separate for random-mask realizations 1 and 2.

Resume the complete analysis from the repository root:

```bash
bash results/imp_sign_magnitude_step0_v1/finish.sh
```

A failed process stops the completion script. Completed rows are skipped on resume. Logs preserve interruptions. `completion.json` is produced only after all 3075 rows and invariant/identity checks pass. Until it exists with complete=true, outputs are incomplete and no final scientific summary should be inferred.

Final artifacts: `results.csv`, `condition_summary.csv`, `full_dose_per_seed.csv`, `distance_matched_comparisons.csv`, `distance_matched_per_seed.csv`, `findings.md`, `distance_IMP.png/pdf`, `distance_random.png/pdf`, `layers_IMP.png/pdf`, `layers_random.png/pdf`. Individual draws and RNG seeds are retained. `config_seed*.json` are original checkpoint configs. Run `report.py` after evaluation to regenerate tables and plots.
