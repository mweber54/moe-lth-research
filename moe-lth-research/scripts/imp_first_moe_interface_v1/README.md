# Checkpoint-only first-MoE interface experiment

The choice of layer 0 is post hoc, motivated by the completed expert-contribution ablation. The reciprocal substitution and fixed alpha grid are sealed before new outcomes. No training or permanent model changes occur.

From the repository root, run the read-only audit, then evaluation:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u scripts/imp_first_moe_interface_v1/audit.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u scripts/imp_first_moe_interface_v1/run.py --evaluate
```

`run.py --seal-only` seals without evaluation after an audit. The audit rehashes both IMP-transition input manifests and the post-pruning input manifest, verifies saved common parents and masks against transition snapshots, checks all ten frozen router/expert start/end pairs including biases, asserts shared parameters changed, verifies original validation identity, and verifies the previous ablation's source/provenance seal and completion. There is no training-prefix reconstruction: it reads the already saved verified parent and transition states. Historical alias inventory entries are not new replications. Exact paths are exported in `artifact_inventory.csv` and `checkpoint_and_validation_provenance.json`. The source audits distinguish seed 17's completed pilot from the four-seed production cohort summary.

Inspected sources include the compensation audit and completed tables; capacity findings/configuration/native dispatch and tests; expert-contribution findings, evaluator, isolated replay, tests and source seal; post-pruning protocol and partition; source mask-transition manifests/checkpoints; native transformer/MoE/router/expert/masking/data implementations. The existing replay preserves whole-batch accepted slots and native within-expert order. Here its flags are activated **only for layers 1–3** and only with the exact recipient checkpoint/batch payload. The donor tensor already contains layer-0 routing and weighting. Downstream expert outputs use current hidden inputs.

`engine.py` captures donor contributions and interface tensors in GPU memory per batch, with seed/type/checkpoint/token/target/batch identities. No logits or all-expert activation array is saved. Substitution hooks replace only the returned first-MoE contribution, assert the recipient residual and normalized input are untouched, and assert exact native residual addition of the linearly scaled donor tensor. The actual model forward is unchunked. Diagnostic selected-expert outputs include proposed-but-dropped tokens and may be chunked without changing previously captured routing. Same-expert and changed-expert populations are separate.

Synthetic tests precede admission. Full 31-batch preflight verifies native baselines and ZERO_L0 against the prior per-batch CE in both modes, including target counts and batch hashes. The first-batch full grid profile records runtime, memory and projected storage before cohort dispatch. Admission/locking use the existing shared-only worker conventions (at most 90% utilization, at least 24 GiB free GPU memory and 200 GiB free storage). The established Viper/Adder SSH transport can select an admitted host without interrupting unrelated processes.

Each atomic cache is one scientifically complete seed/model pair and carries both checkpoint/mask hashes, all validation batch hashes, seal hash and the entire recipient/donor/alpha/mode grid. Alpha-zero donor duplicates reuse validated outputs while retaining logical rows. Within-batch own native baseline outputs are reused after first-batch substitution equivalence checks. Replay baselines are re-evaluated. Incomplete pairs are recomputed; failed attempts and registrations remain intact. No cache can count toward completion without full coverage, fixed-donor perturbation equality across recipients, state preservation and prior-readout reproduction.

`analysis.py` exports every seed before aggregate summaries; recomputes V/Q/S/A/T and sparse/routing contrasts independently from exported CE; reproduces the prior diagonal ZERO_L0 contrast; streams sufficient statistics for pooled interface norms/cosines; renders six figures; and builds `summary.tar.gz`. No new conditions, alphas, alignment, classifier or causal percentage are selected from outcomes. Resume using the same evaluation command and unchanged seal.

Pre-seal engineering correction: the full-state parent stores model configuration at `identity.config.model`; endpoint snapshots use `config.model`. An early strict check incorrectly assumed the endpoint schema for both. Its source and failure are retained under `attempts/configuration_schema_assumption_v1/`. The corrected check uses each explicit schema, requires exact equality with the registered model configuration, verifies source-file hash and verifies the loaded model state hash. It does not accept missing metadata, relax tolerances or affect any ablation outcome/cache.

Final reporting amendment: after evaluation (including a resume), run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/imp_first_moe_interface_v1/amendments/finalize_metadata.py
```

Code review identified an aggregate field-name collision: grouping by `metric` and then storing the summary measure under the same name lost V/Q/T contrast labels in aggregate rows. The separately sealed metadata amendment preserves the original summary under `attempts/aggregate_label_metadata_v1/`, retains contrast and summary-measure labels in separate columns, and asserts every numerical summary value is unchanged. It independently rechecks the exported CE identities and refreshes the archive. No evaluation, condition, loss or seed-level estimate changes. The original evaluator seal and all caches remain intact. This helper lives outside the evaluator's sealed top-level source set so resumability remains compatible. Both engineering corrections are documented in findings/attempt records; no invariant tolerance was weakened.
