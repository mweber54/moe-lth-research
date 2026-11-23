# First-MoE global amplitude control

Run from the repository root on an admitted host:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u scripts/imp_first_moe_amplitude_v1/audit.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u scripts/imp_first_moe_amplitude_v1/run.py --evaluate
```

The audit freshly verifies exactly the prior interface checkpoint file/state/config/mask identities, all frozen router/expert groups including biases, changed shared groups, prior cache completeness/source seals, original full validation coordinates and the existing sealed four-batch training-only probe. Common-parent/transition provenance is inherited from the hash-verified prior audit; unrelated historical trajectory files are not rehashed or substituted. The full-state parent model config is read from `identity.config.model`, and endpoint config from `config.model`, through the previously validated loader. No historical artifact or registration is modified.

`specification_seal.json` freezes the code, fixed grid, source/probe identities and scientific definitions before calibration. `calibration_seal.json` then freezes all ten FP64 training-only RMS ratios and cache hashes before any new scaled validation losses. There is no CE calculation during calibration, no epsilon, no validation-based refit and no per-position/per-expert scaling. The previously used 16 diagnostic token positions do not restrict calibration: all 65,536 contextual positions in the four complete batches, including dropped zeros, are used. These training batches were previously used in training and diagnostics and are not held-out generalization data.

`engine.py` reuses the verified passive native donor capture and insertion boundary from the interface study. It forms the four global-scalar variants in a separate mapping, then calls the prior insertion with multiplier one. Only the first complete gated MoE contribution is replaced. New guards count exactly one insertion, require unchanged zero support and RMS linearity, and retain the previous residual/current-input replay/state/tied-weight/mask invariants. No historical engine or model code is edited.

The native mode is labeled NATIVE_DOWNSTREAM; RECIPIENT_BASELINE_REPLAY fixes only layers 1–3 to each recipient's baseline on the exact batch and recomputes experts on current hybrid inputs. A/D per-batch NLL is reused from exact verified interface caches. The predetermined seed17 pilot freshly re-evaluates all unscaled cells over the full validation set; every other pair verifies all unscaled cells on one representative complete batch. B/C are new scaled evaluations. All 160 logical cells remain explicit; reuse is not a new replication.

Each invocation has a unique attempt directory, and the study uses existing GPU admission, shared-home exclusive claims, atomic JSON/CSV and scientifically complete per-pair caches. Calibration caches are small sufficient-statistic records. Donor activations and logits are not persisted. Only small one-value-per-position vectors are temporarily pooled for exact cosine/norm-ratio quantiles. Calibration scaled RMS and perturbation measurements are derived from FP64 native sufficient moments and labeled as such; validation measures actually scaled FP32 tensors. Both nominal matched targets' actual validation mismatch is reported without changing the scale.

`report.py` exports seed-level effects before descriptive seed summaries, including the two arithmetic decompositions, same-seed sparse-minus-unpruned contrasts, native-minus-replay contrasts, paired batch NLL contrasts, interface distributions and five figures. It rechecks all contrast/aggregation identities independently from exported absolute losses. Missing values stay missing. The sample SD uses source seeds, never tokens/batches as independent replications. No percentage of recovery or isolated direction effect is reported.

Resume using the same evaluation command and unchanged seals. If a source or numerical check fails, preserve the failed attempt and its original caches; diagnose the exact defect rather than changing definitions or tuning scales. No training, mask discovery or later experiment is authorized by this workflow.
