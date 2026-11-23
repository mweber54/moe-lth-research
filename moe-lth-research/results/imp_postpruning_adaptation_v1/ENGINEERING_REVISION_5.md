# Engineering revision 5 report

## Decision and scope

Revision 5 is an engineering-only continuation of the registered study. It
fixes a device mismatch in frozen optimizer-state validation and registers the
updated source hashes before resuming the seed-17 pilot. The router/expert
conditions, seeds, probes, endpoints, step window, analysis contrasts, and
acceptance criteria did not change. The probe-manifest SHA-256 remained
`506f897c83f58efbd5ca5b95ff95d1990ea836333a7dce7162f6add301d0fd6b`.

## Triggering failure

The seed-17 `unpruned/router_frozen` branch failed during its pre-update frozen
state check:

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument other in method wrapper_CUDA__equal)
```

The optimizer's saved state was on CUDA while the pre-update snapshot was on
CPU. `torch.equal` requires tensors on the same device. The preserved attempt
was at global step 1000 and sampler position 1000, with zero optimizer updates,
zero update rows, and no result checkpoint. No model update or training-batch
consumption occurred in this failed attempt.

## Changes

- [engine.py](../../scripts/imp_postpruning_adaptation_v1/engine.py):
  `verify_frozen_optimizer` now moves both tensors to CPU only when their
  devices differ, then still uses exact `torch.equal` comparison. Same-device
  tensors retain the existing direct comparison; non-tensor state retains
  ordinary equality. This changes validation compatibility, not optimizer
  behavior.
- [finalize_step1500.py](../../scripts/imp_postpruning_adaptation_v1/finalize_step1500.py):
  the checkpoint-only diagnostic path uses optimizer and scaler clones restored
  from the exact step-1000 state. This avoids relying on a read-only optimizer
  wrapper exposing native `.state` and keeps diagnostics from mutating the
  finalized checkpoint state. It performs no training updates or training
  batch consumption.
- [prepare.py](../../scripts/imp_postpruning_adaptation_v1/prepare.py):
  adds the revision-5 registration path. It verifies the failed attempt is
  still at the step-1000 boundary with no update rows, preserves the failed
  status and pilot report, verifies the prior revision-4 source hashes, and
  writes the revision record and replacement source-hash manifest.
- [test_imp_postpruning_adaptation_v1.py](../../tests/test_imp_postpruning_adaptation_v1.py):
  adds a CUDA regression test comparing CUDA optimizer-state tensors with a
  CPU frozen-state snapshot.
- [README.md](./README.md): documents the revision-5 registration and retry
  sequence.

## Source hashes

The revision-4 source-hash manifest SHA-256 was
`7f6fd1a02925e97c90191559ba89977dc2212aea3ba7883171bf324f0d614e04`. Revision
5 registered the following source changes:

| File | Revision 4 SHA-256 | Revision 5 SHA-256 |
|---|---|---|
| `scripts/imp_postpruning_adaptation_v1/engine.py` | `2ab3ce8985cde955c8b9d52160f2887eb13e024bdeb1bb5393aab280f755f3a2` | `b9faddc2cabfaae0e97d4d473bbc81c8eed2c9cb9d5948a491385040be8f2408` |
| `scripts/imp_postpruning_adaptation_v1/finalize_step1500.py` | `4c914aacf658589cb2719289ef316236eb7fe75cdb0f2aa01ceb7b1bf0dc63ce` | `10588caad0e0bd9918030b9d1b7de052ea37426bb62efe951032ecc8d5e1535f` |
| `scripts/imp_postpruning_adaptation_v1/prepare.py` | `b9e5566c49a35f12056d8c464b49a5fe1e8886399627e201f7f4e705b6220a41` | `2b437b834f9176456f550995db6dc5aceb98326e6c7128c78cc4928171251bbf` |

The revision-5 source-hash manifest SHA-256 is
`483b2ec65dea2c0d0d7b2d4fa00618341ac9bcebe4e07ab6bba01a3e16f90d8d`. The
revision record SHA-256 is
`c93077b40f69d9c4ecadfa8c642c75404fb8ac7a41e6bcb0a8378d380cdb1775`.

## Verification and pilot outcome

- Focused unittest suite: 10 tests passed, including the CUDA
  cross-device regression test.
- Python compilation and imports: passed.
- `git diff --check`: passed.
- Revision-5 registration preserved the failed attempt and recorded zero
  updates in it.
- The resumed seed-17 pilot passed all eight model-type × condition branches.
  Each branch recorded exactly 500 attempted and applied updates, zero AMP
  skips, and a verified step-1500 endpoint. Baseline CE was identical across
  the four conditions within each model type.
- `pilot_validation.json` reports `PASS`. This satisfies the registered
  engineering/resource gate; outcome direction was not used as a gate.

The cohort subsequently completed under the unchanged scientific registration:
40 branches, each with 500 attempted updates, were validated. Registered
analysis completed after the analysis-only renderer correction documented in
[ENGINEERING_REVISION_6.md](./ENGINEERING_REVISION_6.md). No training beyond
global step 1500, mask search, or other scientific-design change is included
in revision 5.
