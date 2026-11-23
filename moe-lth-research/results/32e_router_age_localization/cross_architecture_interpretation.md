These results characterize router-age-conditioned sparse recoverability under a fixed global source-training and recovery budget.

Raw normalized router age is a fraction of 2500 source attempts, not expert developmental exposure. The experiment controls global source tokens, source attempted steps and recovery attempted steps. It does not equalize accepted tokens per expert, successful expert updates, or maturity. Per-expert source and recovery exposure decrease as expert count increases, so normalized router age should not be interpreted as equal developmental maturity across architectures.

Observed mean-winner sequence:
8E -> R20; basin [20, 100]
12E -> R60; basin [60, 80, 100]
16E -> R60; basin [60, 70, 80]
24E -> R50; basin [40, 50, 60, 80, 100]
32E -> R0; basin [0]

Monotonically nondecreasing observed ages: False; no scaling law follows. Feature distances in cross_architecture_developmental_distances.csv use the historical coordinate ranges; neither proximity nor a nearest age establishes one common developmental optimum. Existing normalization includes sampled exposure, which itself scales with expert count.

Seed-level disagreement is reported directly rather than absorbed into the architecture mean. 24E versus 32E winner spread: 80 versus 0; mean across-age seed SD: 0.023467 versus 0.015878. These are descriptive, with unequal adaptive grids and only three seeds. They do not establish a population trend in seed dependence. Mean curve ranges: 0.027658 versus 0.052564; relative-to-SD measures are saved, without a post-hoc significance/equivalence threshold.

No exposure matching, longer recovery, old-architecture retraining or mechanism experiments were performed. Future mechanistic target: none; recommendation only, not a launched experiment.
