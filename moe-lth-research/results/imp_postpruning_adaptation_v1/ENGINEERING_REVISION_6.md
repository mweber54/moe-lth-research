# Engineering revision 6 report

## Decision and scope

Revision 6 is an analysis-only engineering correction after the full registered
cohort completed. It fixes the trajectory SVG renderer's input-shape
assumption, registers the updated analysis and registration code, and reruns
the registered analysis. It did not run training, change model or optimizer
state, consume training batches, or change any scientific condition, probe,
endpoint, contrast, or analysis definition.

## Triggering failure

The first `analyze.py --analyze` invocation validated all registered branches,
then failed while generating the trajectory SVG:

```text
AttributeError: 'float' object has no attribute 'values'
```

`analyze()` constructs `trajectory` as a flat mapping from
`(seed, model_type, condition, global_step)` keys to CE floats.
`_svg_trajectories()` incorrectly treated those values as nested mappings while
computing the plot's y-axis range. The failure occurred after cohort completion
and before the final output report was written. It did not modify any branch
checkpoint or trigger training.

## Changes

- [analyze.py](../../scripts/imp_postpruning_adaptation_v1/analyze.py):
  compute the plot range directly from the flat mapping's CE values.
- [test_imp_postpruning_adaptation_v1.py](../../tests/test_imp_postpruning_adaptation_v1.py):
  add a regression test constructing the registered flat trajectory grid and
  asserting SVG generation.
- [prepare.py](../../scripts/imp_postpruning_adaptation_v1/prepare.py):
  add `--engineering-revision-6`. Before registering, it checks revision-5
  provenance, probe identity, pilot PASS, and all 40 branch statuses; it
  verifies no unapproved registered source changed and updates the source
  hashes. The missing revision-5 source-hash archive is copied from the active
  manifest only after its registered hash matches.
- [README.md](./README.md): document the required post-cohort registration and
  analysis sequence.

## Source hashes and registration

| File | Revision 5 SHA-256 | Revision 6 SHA-256 |
|---|---|---|
| `scripts/imp_postpruning_adaptation_v1/analyze.py` | `57a1ad05c4c2acc43a315c68fe205d0e220bc651e45713e7427c834685553a0f` | `10fbcc8b8ae756fca7dfccdbe8f08726fe592cbc81aefde30a5c95d874c7d567` |
| `scripts/imp_postpruning_adaptation_v1/prepare.py` | `2b437b834f9176456f550995db6dc5aceb98326e6c7128c78cc4928171251bbf` | `afa9b22015ec9f3662af56d9ee4dfc300174619b1e35bb8ffefb1315dd986e26` |

The revision-5 source-hash manifest SHA-256 is
`483b2ec65dea2c0d0d7b2d4fa00618341ac9bcebe4e07ab6bba01a3e16f90d8d`.
Revision 6 registered source-hash manifest
`43a0d3dc55c54fc697a742b3c1f879b6a77876331dcd0169f86387083410faf7` and
engineering revision record
`e8af63c309c0be186badac0c89175380d31a825f81ae8ba113d2f6a00195df32`.
The diagnostic probe-manifest hash remained
`506f897c83f58efbd5ca5b95ff95d1990ea836333a7dce7162f6add301d0fd6b`.

## Verification and outputs

- Focused unittest suite: 11 tests passed, including the SVG regression and
  CUDA frozen-state regression.
- Python compilation and whitespace checks passed.
- Revision-6 registration verified that all 40 registered branch statuses
  were `COMPLETE` and recorded zero training updates after cohort completion.
- Registered analysis completed with status `COMPLETE`: five seeds, 40
  branches, 200 trajectory points, and 60,000 per-group update rows. The
  analyzer verified each branch's final-checkpoint hash, 500-update accounting,
  validation grid, probe identity, diagnostic preservation, and required
  diagnostic row counts.
- The separate late-dense audit remained analysis-only: dense CE increased in
  seeds 7, 29, 43, and 61; 7/15 sparse-minus-dense comparisons were nonpositive
  at step 5000; sparse CE decreased from step 4500 to 5000 in 15/15
  comparisons. These descriptive observations are not causal conclusions.

SHA-256 values for the 10 registered analysis outputs and
`analysis_status.json`:

| Output | SHA-256 |
|---|---|
| `factorial_trajectory_metrics.csv` | `c393eb08c387941e6cbbb57ad5e2493b6a636186d9b0e9490f73bedb0f7c139b` |
| `seed_level_factorial_contrasts.csv` | `ba1382dc6dca8515a33ed2bc2ac08117661eacbbe1db61d289ed3a47be83cff1` |
| `seed_level_contrast_summary.csv` | `dfb716ca12eb15c394cf88147153f0fd66ee832cbe53a4afc3a5400c1db7c58d` |
| `trajectory_integrated_excess.csv` | `d1df2cbc26d9f583bb9f31058589504d0bda45fe39e8ed41559a2b60a6ccd356` |
| `trajectory_integrated_excess_summary.csv` | `51bc1a7f938519916b8dcc90f78278a26d7ed4e9650ea2265b0dbacce684c747` |
| `optimizer_compute_accounting.csv` | `214485d9588717c98f371f22d888fa816dd694b197cb21bd6b05a709dbdcca9b` |
| `trajectory_ce.svg` | `6df0bff0ed8a5d72992a2c7eb5390cdef71aa08e74aa055649259b1703c12841` |
| `endpoint_penalties.svg` | `557abb5c59896e2dd8dd4fb8373b0f21039807a8147c9a84e24e67ed29df323b` |
| `late_dense_audit.csv` | `487765c7e397e4767c6771b04f0211e7cf5a9ed611f6a31fb8cfc528de239974` |
| `late_sparse_comparisons.csv` | `9d5122e7d531a7065b2fe999e5b1cec81d8d247bc6740355ad968e9be3f89ec8` |
| `analysis_status.json` | `0a1431ed82dba14b0d877bf9bd3b451fd233675e1aca6c7ed29228e366686229` |

No follow-up training or mask search is authorized by this analysis correction.
