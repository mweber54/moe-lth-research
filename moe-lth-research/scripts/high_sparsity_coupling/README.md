# High sparsity router–expert coupling

Isolated exploratory analysis of E32 IMP .80/.85, five source seeds, ten sparse runs and five unique dense controls. All historical outputs are read only. The frozen registration and cohort live in `results/high_sparsity_coupling_v1/`.

Use the repository `.venv-highres/bin/python` (torch 2.7.0). Commands from repository root:

```bash
.venv-highres/bin/python scripts/high_sparsity_coupling/audit.py
.venv-highres/bin/python scripts/high_sparsity_coupling/validate.py --numerical-only
.venv-highres/bin/python scripts/high_sparsity_coupling/validate.py
.venv-highres/bin/python scripts/high_sparsity_coupling/analyze.py --seeds 7 --benchmark
.venv-highres/bin/python scripts/high_sparsity_coupling/cluster.py --verify
.venv-highres/bin/python scripts/high_sparsity_coupling/cluster.py --launch
.venv-highres/bin/python scripts/high_sparsity_coupling/report.py
```

Audit refuses an existing freeze. Do not delete or overwrite it to force a provenance check through. A genuinely changed registration requires another versioned output root. `cohort.json` enumerates exact original initialization, direct mask paths, source reference, boundary weights, and canonical matched dense pairing. Each file has SHA256; `freeze.json` detects cohort/protocol changes.

`cluster.py` launches one detached worker on Viper (seeds 7/17/29) and Adder (43/61). Host checks hash every registered input independently. Per-host locks and fixed disjoint seeds prevent duplicate dispatch. Per-condition locks also protect standalone runners. All incomplete attempts are retained; restarting verifies completed checkpoint hashes, then creates fresh attempts for unfinished conditions. No model-only snapshot is used to resume an optimizer. Workers emit persistent 15-second heartbeats; trainer callbacks record step progress. Failures are explicit JSON and logs. Stop/restart decisions never involve unrelated users' processes.

The native trainer performs both 2500-step segments. The second call deliberately constructs fresh AdamW/scaler and reseeds through native assembly. Segment 0 and segment 2500 preserve original batch ranges; instrumentation saves model-only diagnostic states at the registered times. Boundary optimizer/scaler states are separately saved before/after reset. Per-parameter actual changes are compared to clipped gradients at designated updates. These are AdamW updates, not raw SGD steps.

Analysis is offline on isolated model instances. Raw token/position/pair identities, activation scores, hidden boundaries, per-token CE and KL are bounded NPZ caches. `groups.csv` records per-expert Pearson/Spearman with explicit undefined reasons. `pools.csv` standardizes within each layer/expert/selection and records valid expert intersections for unmatched pairing controls. `within.csv` is a separate across-expert within-token statistic. Fixed masks select diagnostic pairs; they do not freeze a training router.

`weight_geometry.csv` reports independent ranks 8/16/32 with cutoff flags, full Gram scale and normalized metrics; `router_energy.csv` includes raw/centered q and unmatched pair distributions. `embedding*.csv` measures the tied byte embedding/LM-head matrix, not semantic word embeddings. Layerwise hidden comparisons are native-coordinate CKA, spectra, and drift. `function.csv` compares complete expert outputs on exactly the same inputs, with zero exclusions. Probe validation CE uses the full historical validation sequence; correlations use a fixed 2048-position subset while capacity routing uses the complete first validation batch.

`gradient_identity.csv` checks local single-token derivatives using a real accepted token and actual model CE upstream. Auxiliary diagnostics use original full-batch detached load, batch normalization and layer averaging for the direct local auxiliary term. They do not claim to isolate all downstream effects of auxiliary losses in later blocks. Task gradients of unselected router logits are expected for this repository's full-softmax gate. Mask projections are verified separately from signed router/expert direction and zero-gradient exclusions.

`report/findings.md` and `report/completion.json` explicitly identify incomplete runs. Historical validation tickets, rerun tickets and any independent test are kept distinct; no unused independent test has been established. Expert/token/layer/time observations are nested under five source seeds. No inferential token-level p-values are used. Generated plots retain individual source seeds.
