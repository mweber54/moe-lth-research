# Expert-role deletion-sensitivity Stage 1

This checkpoint-only diagnostic never calls a training runner, optimizer, recovery,
or mask search. The fixed engineering cohort is seed 17, 32E IMP80 and its matched
unpruned recovery at steps 25, 250, 1000, and 2500 (pre-reset). Historical inputs
remain read-only. Results are under `results/expert_role_deletion_v1/`.

The completed pilot is already cached. Read `findings.md` before considering any
expansion. No five-seed expansion or Stage 2 is launched by these entrypoints.

From repository root, using the existing GPU environment:

```bash
.venv-highres/bin/python scripts/expert_role_deletion_v1/study.py prepare
.venv-highres/bin/python scripts/expert_role_deletion_v1/study.py profile
.venv-highres/bin/python scripts/expert_role_deletion_v1/study.py measure
.venv-highres/bin/python scripts/expert_role_deletion_v1/report.py
.venv-highres/bin/python scripts/expert_role_deletion_v1/verify.py
```

Preparation/profile reject an already frozen study. Measurement verifies its
implementation/model/probe/group/checkpoint identities and reuses verified primary
caches. Every cache stores baseline contextual assignments, acceptance, gates,
router inputs and token CE, paired group token/sequence damages, and measurement
rows. A failed evaluation restores selected coordinates in `finally`. Measurement
holds nonblocking study/host advisory locks and emits a 15-second heartbeat.

The coordinate panel has eight uniform disjoint groups per expert, each containing
floor(1% of that tensor's actual IMP survivors). The same coordinates are deleted
in the unpruned model. Magnitude normalization divides group mean absolute weight
by the tensor mean absolute **nonzero** weight; recent movement is mean absolute
coordinate change since the previous selected checkpoint. No sensitivity score is
reused without matching provenance. All A decisions use past/current data only;
future B results cannot affect membership, dose, thresholds, or clustering.

Test the scientific safeguards with the existing test environment:

```bash
.venv/bin/python -m pytest -q tests/test_expert_role_deletion.py
```

The pilot treats whole-probe continuous signed task CE damage as primary. Threshold
ceilings, low traffic and conditional/union effects need to be read together. Seed,
not groups or tokens, is the source replication unit. A five-seed predictive study
would need a separately authorized expansion with seed-held-out preprocessing and
simple paired baseline/role-augmented models; this one-seed pilot fits no predictor.
