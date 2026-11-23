# Final CPU-only audit decision

Do not launch connectivity recovery. No common frozen dose passes all gates in
all 18 configurations. GPU training remained paused throughout this audit.

The suspected sampler defect is disproved. In the row branch `x=cross_r` and
`y=cross_s`; in the column branch `x=cross_s` and `y=cross_r`. The historical
source rebuilds byte-for-byte to the archived `.so` (SHA-256
`7349b48f62db913cf420d4ea39d4987d3bc25420f1191ab4fa156491a6e5f40b`).
The original rebuild and separately named validated implementation also return
identical outputs across 19,999 deterministic seeds.

All 18 sealed v2 pairs pass every frozen gate. High/control counts match for
every row and column of every matrix, recomputed dose/energy/covariates pass,
and both arms reproduce archived step-0 hashes in 18/18 pairs. All 12 completed
v3 repair trajectories inherit the verified v2 draw-1 selections.

Connectivity results are identical for original and validated samplers: 0/18
pass at 0.05; exactly the six IMP configurations pass at every target from
0.075 through 0.34207. At 0.075 all 12 degree controls fail matrix coverage.
At 0.10 and above they fail coverage and target-dose error. No other frozen
gate fails.

Doubling proposals from 1M to 2M increases representative degree-control
maximum doses from 0.07783 to 0.11695 and 0.07541 to 0.11417, proving the dose
plateau was budget-limited. Coverage changes only 150 to 152 of 256 matrices in
one case and remains 147 of 256 in the other. This does not establish that the
uncovered matrices lack mathematically eligible rectangles; it establishes
failure of the current search/design under tested budgets.

The completed direct IMP-versus-degree-mask comparisons, observational sign
results, v2 interventions, and v3 sign-repair trajectories remain
interpretable. The heterogeneous v3 sign result still does not establish a
consistent early mechanism. No connectivity effect or intrinsic rectangle
capacity claim is supported. Any follow-up should be a separately
preregistered design, not a relaxed version of this frozen protocol.
