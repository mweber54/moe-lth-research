# Router gate factorial mechanism screen

Authorized scope: E8, router ages 20/100, seeds 7/17/29, 85% learned expert-local masks, matched dense/sparse controls, 24 new recoveries. Original checkpoints/results are read-only inputs.

The predeclared estimands and stopping rule are in `protocol.json`. `validation_report.md`, `validation_details.json`, and `additional_validation.json` document passed pre-training tests. `execution_ready.json` freezes executable hashes; `execution_inputs.json` freezes original code, masks, checkpoints, configs and audit inputs. Workers refuse changed inputs and refuse to overwrite incomplete cells.

Arms: A native forward/native gate Jacobian (audited assignment replay); B donor forward/detached gate Jacobian (audited combined intervention); C donor forward/native gate Jacobian; D native forward/detached gate Jacobian. All retain donor assignments and capacity priorities, and native auxiliary probabilities with coefficient 0.1. The detach also affects downstream auxiliary paths through earlier gates; it is not a CE-exclusive graph cut.

Primary evaluation forces the same archived donor assignments and coefficients in every cell. Native and assignment-only evaluations are sensitivity analyses. Step 25 has no archived donor validation occurrence: a snapshot, native readout, and fixed-step 0 derivative diagnostic are saved, but step 25 is excluded from primary timing comparisons. No interpolated value is called observed.

Historical A checkpoints and A/B intermediate snapshots are unavailable. Their audited readout tables and training identities are retained and reused, without claiming a new checkpoint evaluation. B final checkpoints remain. Source weights and masks are retained and verified by their original hashes.

`recoveries/` contains legacy trainer outputs and per-cell completion records; `snapshots/` contains all nine new states with optimizer, scaler, mask and RNG; `measurements/` contains fixed-probe FP32 gradients, exact tied-embedding lookup/head decomposition, observed AdamW embedding updates, forward gate diagnostics, native/assignment readouts and mutation audits. Fixed-probe derivatives are diagnostics, not the gradients of the historical update batch. Update alignment names the source diagnostic step and includes momentum and weight decay in the actual observed displacement.

Execution first benchmarks one authorized cell, then schedules only remaining cells. Independent seed workers may share the H100, with per-condition and per-worker locks. Readouts, attempted-update budgets, precision and batches remain matched; AMP applied counts are reported separately. No fresh seeds or extra architecture/pruning searches are launched.

## Completed result

**NO-GO.** All 24 missing factorial recoveries completed; the complete 48-model factorial table reuses audited A/B conditions. Value-only replacement gives 98.3% mean per-seed attenuation under the donor readout, but its effect reverses in all three seeds under both native and assignment-only readouts. Jacobian-only effects are inconsistent across seeds. No fresh seeds were launched.

Start with [research verdict](research_verdict.md), [full report](report.md), [full factorial table](full_factorial_results.md), and [validation report](validation_report.md). All scientific/protocol inputs and original weights are unchanged. A concurrent final-newline-only change in the earlier narrative verdict was left untouched and documented in [the preservation exception](preservation_exception.md).

The frozen analyzer is `analyze.py`; `analyze_final.py` changes only the preservation audit/metadata for that proven formatting exception. Its numerical estimands and stopping checks are identical. Post-run analysis/diagnostic source scripts are retained under provenance; run from repository root with Python safe-path mode (`.venv-highres/bin/python -P ...`). Legacy recovery-fraction/threshold fields retain their old native-dense denominator; use the explicitly matched CE tables for this mechanism experiment.
