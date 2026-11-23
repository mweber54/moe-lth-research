# Expert-local deletion dose and reliability check

This extends the existing Stage 1 seed-17 engineering pilot using read-only source
artifacts. It evaluates nested unions of original groups in **one expert fc1
matrix at a time**, and a separate expert-output-zero positive control. It never
calls training, recovery, optimizers, permanent pruning, clustering or predictors.

From the repository root, use the existing GPU environment:

```bash
.venv-highres/bin/python scripts/expert_local_deletion_dose_v1/run.py prepare
.venv-highres/bin/python scripts/expert_local_deletion_dose_v1/run.py measure
.venv-highres/bin/python scripts/expert_local_deletion_dose_v1/report.py
.venv-highres/bin/python scripts/expert_local_deletion_dose_v1/supplement.py
.venv-highres/bin/python scripts/expert_local_deletion_dose_v1/verify.py
```

Read `verdict.md` for the concise interpretation and `findings.md` for full
stratified tables. `supplement.py` adds a data-scaled whole-probe CE zoom and the
concise interpretation after reviewing the frozen results. It changes no
interventions, metrics, ranking strata, or reference thresholds; its source hash
is recorded in `presentation_supplement.json`.

Results and verified caches are written only to
`results/expert_local_deletion_dose_v1/`. Preparation refuses to overwrite a frozen
registration. Measurement reuses completed panels only if input/implementation
identities match. Workers hold nonblocking study/host locks and emit 15-second
heartbeats. Failures preserve explicit error records; interventions and hooks
restore through `finally` blocks.

Four fixed permutations are shared across trajectories/checkpoints. The manifest
contains unique physical interventions; `ordering_aliases.csv` preserves each
nested prefix without counting repeated unions as new observations. Eight groups
mean approximately 8% of the **sparse surviving fc1 matrix**, with a smaller active
fraction in the unpruned model. No Stage 1 global union cap applies. Positive
controls are not plotted as a weight dose.

`dose_common.py` captures native expert input/output rows and gates in actual
capacity-dispatch order. Cached single-group CE measurements are reused; their
expert-output metrics replay only the expert on identical accepted native rows.
Representative full-model checks verify both cached CE and replayed outputs.
Whole-model forward passes always use complete original A/B batches. The positive
control returns zero from one expert module; router selection, acceptance and gates
at that layer remain native and are explicitly checked.

`measurements.csv` reports whole, baseline-accepted and complement damage, exact
matrix fractions, functional RMS changes and union-minus-sum damage.
`paired_sequence_damage.csv` retains sequence identity, paired CE and coverage.
`ab_reliability.csv` compares unique interventions **within each dose and time**,
both across and within experts. Constant/small panels remain undefined. It also
provides an adequate-coverage view without dropping low-traffic primary rows.
Source seed is the replication unit; neither A/B nor groups, orderings, checkpoints
or sequences supply independent seed replications. No pooled cross-dose rank is
reported. The reference lines 0.001 and 0.0001 are not selection rules.

Run scientific-safeguard tests with the existing pytest environment:

```bash
.venv/bin/python -m pytest -q tests/test_expert_local_deletion_dose.py tests/test_expert_role_deletion.py
```

No cross-seed expansion or Stage 2 launcher is included.
