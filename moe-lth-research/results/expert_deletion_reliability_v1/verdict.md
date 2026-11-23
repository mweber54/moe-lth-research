# Verdict: mixed results requiring a narrower claim

The assay can distinguish some more-harmful from less-harmful deletions on new
contexts at equal dose, but it does **not** support a general CE-based group-selection
rule. Exactly 32 new complete batches were used; discovery and confirmation are
training-distribution diagnostic contexts in one nonmatching seed, not independent
source replications or a test set.

**A. Reproducibility.** Of 78 operationally covered expert/checkpoint/trajectory/dose
conditions with at least three unique unions, 26 discovery-selected high-minus-low
contrasts have confirmation 95% paired-batch intervals above zero; one is below zero;
51 remain unresolved. These counts do not represent independent replications, and
no multiplicity-adjusted global success is claimed. Confirming an extreme-pair
contrast does not validate the full ordering of all unions. Eight-group doses have
only one unique union per expert, so equal-dose ranking is unidentifiable there.

**B. Sampling precision.** Examples of reproduced contrasts, selected from discovery
only (nats/token; descriptive confirmation intervals):

| Trajectory | Step | Expert | Groups | Confirmation high-minus-low [95% interval] |
|---|---:|---:|---:|---|
| unpruned | 25 | 8 | 1 | 0.000122740 [0.000114864, 0.000130853] |
| unpruned | 25 | 8 | 2 | 0.000133853 [0.000125616, 0.000142220] |
| sparse | 2500 | 8 | 1 | 0.0000316773 [0.0000130712, 0.0000524610] |
| sparse | 2500 | 8 | 2 | 0.0000515048 [0.0000255534, 0.0000821173] |

All four confirmation block contrasts are positive in these examples. They are
illustrations, not a replacement for the complete per-condition tables. Some other
resolved contrasts are much smaller. The 0.0001/0.001 exploratory conventions do
not turn a small or unresolved contrast into harmlessness or permanent-pruning safety.

The apparent reversal at unpruned step2500, expert24, one group requires caution:
discovery contrast 2.86e-6 has interval [-2.00e-6,8.17e-6], while confirmation is
-5.93e-6 [-1.11e-5,-2.22e-7]. Discovery itself is imprecise. This is not sufficient
evidence of context-dependent importance from a negative rank correlation. Block
sizes, intervals, signs, coverage and leave-one-block-out values remain visible.

**C. Functional versus task ordering.** Functional distortion reproducibly orders
output changes, but it is not a generally reliable task-damage proxy. At unpruned
step25/expert8, the 1/2/4-group functional and CE orderings agree in both phases.
At sparse step2500/expert8, functional rankings reproduce perfectly at 2/4 groups,
but their confirmation association with CE is zero (four unions). This heterogeneity
and the small panels prevent a universal claim. Functional uncertainty uses the
same frozen 2,000 paired batch resamples, pooled energies before ratios, with
undefined/near-zero cases retained; see functional_task_rank_uncertainty.csv.

**D. Coverage.** Conditional ranking fails the prior 32-accepted-positions-per-block
rule for all four sparse experts at step25 and unpruned experts16/24 at step25.
Sparse expert0 has a minimum of26 and falls below32 in six blocks; the other listed
cases have zero-traffic blocks. Overall,46 of256 expert/checkpoint/trajectory/block
cells are below the operational minimum. Whole-probe estimates remain valid for
these sampled contexts, but low traffic does not demonstrate dispensability.
Concentration and contributing-batch counts are reported, not averaged away.

**E. Future use.** A later fixed cross-seed *measurement confirmation* is justified
only for explicitly named conditions with reproduced contrasts, such as the examples
above. Do not use this assay for CE group selection in unresolved or coverage-failing
conditions. It does not yet justify a generic future-persistence study, specialization
predictor, permanent deletion, recovery algorithm or Stage2. No expansion follows.

All 128 representative numerical repeats matched bitwise; zero dose, native baseline,
restoration, full model hashes, RNG, modes, gradients and IMP masks passed. Same-input
numerical agreement is distinct from cross-context reliability. All historical
artifacts were preserved. Full decisions for every expert/checkpoint/dose, all fixed
pairwise contrasts, batch-bootstrap intervals and block sensitivities accompany
findings.md. The final category is **mixed results requiring a narrower claim**.
