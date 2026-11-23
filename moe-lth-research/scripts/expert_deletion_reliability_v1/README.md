# Fixed-budget deletion-damage reliability

One nonmatching seed-17 32E IMP80 engineering case and its matched unpruned recovery,
four verified pre-reset checkpoints, original coordinate unions and output controls.
Exactly 32 new complete 128×128 training batches, grouped into eight four-batch
blocks. Discovery blocks 1–4 choose extrema; confirmation blocks 5–8 evaluate those
unchanged choices. This is a measurement study, with no training or pruning method.

The user authorized deterministic re-batching of intact canonical training sequences.
Sampling excludes prior A/B source windows, all input-plus-target span overlaps and
content-identical full windows. Historical shuffled minibatch membership is not
claimed. The plan, probes and analysis implementation were frozen before profiling
or new deletion measurements. Later reporting/verification scripts are separately
fingerprinted and do not change the frozen analysis.

Run from the repository root in the existing environments:

```bash
.venv/bin/python -m pytest -q tests/test_expert_deletion_reliability.py tests/test_expert_local_deletion_dose.py tests/test_expert_role_deletion.py
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/prepare.py
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/measure.py profile
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/accelerated.py validate
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/prefix_engine.py validate
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/prefix_engine.py measure
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/expert_deletion_reliability_v1/analyze.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/expert_deletion_reliability_v1/functional_uncertainty.py
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/report.py
.venv-highres/bin/python scripts/expert_deletion_reliability_v1/verify.py
```

Preparation/profile refuse to overwrite a completed freeze/profile. Measurement
resumes atomic per-checkpoint/batch caches using exact source, checkpoint, probe,
intervention, numerical-mode and frozen-implementation identities. It uses one GPU
worker, nonblocking host/study locks and persistent 15-second heartbeats. No unrelated
process is stopped. A profile is one of the 32 registered contexts and is reused.

Caches retain baseline losses/routes and compact FP64 paired sequence/subset sums
and counts, batch summaries and functional energies. They do not retain hidden
tensors or logits. Functional distortion pools numerator/denominator energies before
forming a ratio; the average batch ratio is a different, separately named quantity.

The analysis uses 2,000 fixed-seed paired complete-batch bootstrap resamples for
effects and all same-dose pair contrasts, both partitions and each block. It also
records block values, leave-one-block-out values and rank uncertainty/near ties.
Conditional bootstrap ratios recompute resampled sums/counts. No token bootstrap,
checkpoint/expert/dose pooling or confirmation-based reselection is used. Eight-group
singletons cannot identify same-dose rankings. Coverage-deficient cases remain in
the tables. Confirmation is a training-distribution diagnostic, not independent
generalization or source-seed replication. No follow-up experiment launches here.

Read `results/expert_deletion_reliability_v1/findings.md` and the per-condition A–E
verdicts in `condition_decisions.csv`, with the uncertainty and coverage tables.

The completed run uses the separately validated synchronization-only dispatch engine
for new caches. `accelerated.py validate` compares all 56 interventions on registered
batch 0 against the original native profile. It must pass bitwise for baseline,
CE/sequence statistics and functional energies. Integer assignment grouping changes;
router scores, capacity top-k, gate coefficients, expert calls and floating-point
operations do not. New caches bind `routing_engine_sha256`; the accelerated resume
path accepts both the preserved original-native caches and validated engine caches.
`host_compatibility.json`, `execution_transition.json` and `engineering_pause.json`
record the sole-worker host transfer and its numerical/provenance checks. Neither
change adds probe contexts or modifies the frozen scientific analysis.

The final worker additionally reuses only the unmodified upstream prefix for each
checkpoint/batch. `prefix_engine.py validate` must match all original profile effects
and energies bitwise. Complete-batch layer1 and all downstream routing are rerun.
Prefix working tensors stay ephemeral on GPU and are never saved. New caches bind
both integer-dispatch and prefix implementation hashes; validated older caches remain
readable. `prefix_validation.json` records the exact-equivalence proof.

`verdict.md` gives the curated A–E interpretation of the fixed tables. The functional
uncertainty supplement uses the same frozen paired-batch resampling vectors for
pooled energy ratios and within-condition functional/CE rank comparisons; it adds
no observations or selection rules. Historical primary statistical definitions
remain in the original freeze.
