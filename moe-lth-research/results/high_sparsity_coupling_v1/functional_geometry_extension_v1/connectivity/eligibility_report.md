# Connectivity eligibility

**Verdict: unavailable without new training. No interpolation was evaluated and no endpoint was trained.**

Criteria (all required): same mask and parameterization; same original initialization where required; independently varied recovery randomness; verified comparable protocol; distinct trained solutions.

Inventory of trained endpoints sharing a cohort mask (found by mask-hash search over `results/`):

| Source | Relation | Distinct solution? |
|---|---|---|
| `results/imp_80_85/.../final_s0.80` and `.../round_08` | same mask_08, original init, R0, same data order | **No**: bit-identical weights in all five seeds (all identical: True; `endpoint_identity_checks.csv`) |
| `results/imp_extended_recovery` (historical 3000/4000/5000) vs `high_sparsity_coupling_v1/reruns` | historical vs rerun of the same trajectory | **No**: identical model states (see provenance_diagnosis.md); differences are TF32 only |
| `high_sparsity_coupling_v1/validation/*` | short instrumentation checks (enabled/disabled/reset) | No: not full-protocol recoveries |
| .85 mask (mask_09) | only one recovery per seed (`final_s0.85`); no round_09 recovery exists | No second endpoint |

Each (seed, sparsity) therefore has exactly one trained recovery trajectory; recovery randomness was never independently varied (training batches, routers and initialization are fixed per seed). Other masks in the repository (one-shot, router-age, other IMP sparsities) differ in mask and so are not same-mask endpoints.

Excluded by design: initialization-to-final interpolation (high initial loss makes barrier summaries uninformative), dense-to-sparse interpolation (leaves the sparse constraint set; not sparse-mode connectivity), and fitted nonlinear paths. A straight-line barrier would in any case not be evidence about all connecting paths.

To enable this analysis, at least two independent recoveries per (seed, mask) with varied data order/router randomness would be needed (new training, not authorized).
