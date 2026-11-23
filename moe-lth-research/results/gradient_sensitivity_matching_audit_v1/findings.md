# Gradient sensitivity matching feasibility audit

This array-only audit preserves the stopped intervention. The gradient pruning hypothesis remains untested; reproducible gradients do not establish pruning utility.

| Design and evaluated scope | Achieved candidate dose | Included tensors | Structural requirement | Solver certainty | Permitted claim |
|---|---|---|---|---|---|
| A: all1280, original deterministic greedy/cycle matcher | median 0.0015%, max 0.2182% | 0/1280 evaluated | exact row/column counts | heuristic; no full-graph maximum certificate | candidate feasibility only; no loss claim |
| B: fixed40, binary_alternative_edges | median 0.0065%, max 0.3769% | 1/40 evaluated | exact row/column counts | solver statuses/bounds per tensor; SMD/KS postchecked | candidate feasibility only; no loss claim |
| C: all1280, LP on original disjoint greedy pairs, then integer floor | median 0.0011%, max 0.4425% | 0/1280 evaluated | column TV<=0.05; exact rows | heuristic; no full-graph maximum certificate | candidate feasibility only; no loss claim |
| C: fixed40, binary_alternative_edges |median 0.0175%, max 0.5527% among returned candidates | 1/40 conditions; 21/40 incumbents | column TV<=0.05; exact rows | solver statuses/bounds per tensor; SMD/KS postchecked | candidate feasibility only; no loss claim |

The target is2621 pairs per tensor (0.999832%); minimum inclusion is656 (0.250244%), plus all feature/energy/contrast checks. Every source must retain >=80% tensors, >=26/32 in every layer/projection, and >=0.5% whole-expert dose per branch. Candidate edges are not achievable dose. All40 random tensors were retained regardless of original yield.

| Source seed | A included /256 | C LP/floor included /256 | C source coverage passed |
|---|---:|---:|---|
| 7 | 0 | 0 | False |
| 17 | 0 | 0 | False |
| 29 | 0 | 0 | False |
| 43 | 0 | 0 | False |
| 61 | 0 | 0 | False |

| Binary subset | Returned incumbents | Optimal encoded problems | Full-graph upper bound below656 | Restricted graphs | Scientific candidate passes |
|---|---:|---:|---:|---:|---:|
| B | 40/40 | 40 | 39 | 0 | 1 |
| C | 21/40 | 12 | 20 | 0 | 1 |

B incumbent improvement over A on returned cases: median 11.0 pairs; scientific subset passes 1/40. C binary subset passes 1/40. These are tensor diagnostics, not independent model replications.

Concrete defects versus design limitations: no divergence from the registered original matcher was found. Both early greedy partner consumption and greedy cycle decomposition can lose cardinality on synthetic examples. The actual optimization comparisons distinguish this limitation from absent overlap or a structural upper bound. Counts before row/column constraints, full-versus-greedy SCC diagnostics and complete-graph solver bounds are in the CSVs; A timeout alone or a failed postcheck is not an infeasibility certificate; a valid complete-graph upper bound below656 can still rule out the dose without resolving the exact maximum.

Feature-overlap populations are descriptive upper/lower A quartiles within each magnitude stratum, not new candidate eligibility restrictions. Row/column concentration or candidate-edge direction is not evidence that those neurons are important or that connectivity/routing causes sensitivity.

C changes only the column tolerance, keeping all original nonstructural thresholds, dose and coverage. TV=0.05 is a proposed design tolerance, not demonstrated harmless imbalance; occupancy, maximum absolute count differences and concentration are retained. A C comparison, if feasible, would test the compound selection rule with residual column-structure differences, not isolate sensitivity independently of connectivity.

Decision: B improves individual subset candidates, but the reviewed complete-graph bounds below rule out the original cohort coverage. D remains the decision for the intended intervention.

No gradient scores were recomputed; no model forwards/backwards, optimizer updates, pruning-mask applications or intervention/validation losses occurred. No conclusion that the gradient signal works or fails for pruning follows. No revised intervention is executed.

## Reviewed diagnosis

The original matching implementation reproduced its registration; no concrete implementation defect was found. The synthetic examples establish genuine greedy limitations, but those examples alone do not explain the actual cohort stop.

| Source seed | Complete-graph SCC bound below656 | Potentially includable tensors at most | Source coverage impossible under exact columns |
|---|---:|---:|---|
| 7 | 201/256 | 55 | True |
| 17 | 192/256 | 64 | True |
| 29 | 206/256 | 50 | True |
| 43 | 202/256 | 54 | True |
| 61 | 198/256 | 58 | True |

**The original exact-column design cannot satisfy the registered source-coverage gates in any of the five models, even if its greedy matcher is replaced.** This follows from complete admissible graphs and necessary SCC/coordinate bounds, before imposing aggregate SMD/KS or energy requirements. It is a combinatorial statement about these registered candidate populations and constraints, not evidence against the gradient pruning hypothesis.

| Source seed | Median admissible same-row edges | Median greedy disjoint pairs before columns | Median cycle pairs after exact columns | Quartile row TV median | Quartile column TV median |
|---|---:|---:|---:|---:|---:|
| 7 | 15410 | 11152 | 4.0 | 0.672 | 0.772 |
| 17 | 15576 | 11766 | 5.0 | 0.662 | 0.750 |
| 29 | 15622 | 11658 | 4.0 | 0.694 | 0.772 |
| 43 | 14560 | 10800 | 4.0 | 0.642 | 0.810 |
| 61 | 15062 | 11411 | 4.5 | 0.650 | 0.785 |

Thousands of actual disjoint row-matched pairs can exist before column balancing. This distinguishes raw/nonstructural overlap from the scarcity of column-balanced cycles. The complete-graph bound uses all admissible pair choices, so this conclusion is stronger than infeasibility on the original greedy-restricted graph. Aggregate balance can impose further limits; descriptive population overlap alone does not certify group feasibility.

Wall-killed solver jobs have unavailable peak RSS and no returned bound; these are not zero measurements. The8GiB address-space and45s external wall limits remained enforced. Time-limit returns with a valid complete-graph bound below656 can rule out the registered minimum even when the exact maximum is unresolved.

Design C is a different scientific question. Its all-tensor LP/floor generator is restricted to original greedy pairs and cannot certify full-graph infeasibility. The fixed-subset C binary solves use alternative admissible partners, but their bounds/timeouts apply only to that subset and stated graph. At fewer than20 pairs, TV<=0.05 itself forces exact columns because the smallest nonzero TV is1/k. This does not license treating four or five pairs as adequate dose.

**Decision D for the intended intervention:** stop the original matching design. Greedy limitations exist, but replacing the matcher cannot rescue its exact-column cohort under the certified coverage bounds above. The evaluated C candidates and bounded subset optimization do not establish a revised full-cohort comparison. Do not repeatedly relax settings or expand compute without a separately defined question. The0.05 tolerance is proposed, not demonstrated harmless; no model-loss or pruning utility claim is permitted.

### Per-tensor gains versus cohort feasibility

A returned candidate passing the per-tensor thresholds does not pass source coverage or the whole-expert dose gate. Every such candidate on the fixed subset is listed below; none is a pruning outcome.

| Design | Seed / parameter | Original pairs | Returned pairs | Column TV | Local checks |
|---|---|---:|---:|---:|---|
| B | 61 / blocks.1.moe.experts.10.fc1.weight | 566 | 988 | 0.000000 | passed locally; source gate not met |
| C | 61 / blocks.1.moe.experts.10.fc1.weight | 566 | 1449 | 0.049689 | passed locally; source gate not met |

Thus B/C may improve individual candidates while D remains the decision for the intended full-cohort intervention. These local gains do not override the exact-column source-coverage certificate, and C does not establish adequate full-cohort feasibility. No continued training or model-loss measurement is recommended or authorized from these examples.
