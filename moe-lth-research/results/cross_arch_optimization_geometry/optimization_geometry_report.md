# Cross-Architecture Sparse-Expert Optimization Geometry

## Finding

The historical results are protocol-compatible but insufficient for the pre-specified optimization-geometry test. The required gradient-vector and expert-update comparisons need matched sparse/dense recovery states at intermediate steps. Existing outputs retain only final recovery states and scalar gradient norms.

## Consequence

No conclusion is drawn about native versus forced-common-routing gradient alignment, update-direction alignment, trajectory divergence, expert functional alignment, routing-set overlap, gradient coherence, temporal ordering, or seed-specific optimization geometry. A final-only gradient probe would not answer whether alignment precedes the ticket-gap divergence.

## Minimum Diagnostic Rerun

Run exactly 18 sparse/dense paired recoveries at 85% with the existing scientific protocol: e8 R0/R20/R100 and e12 R20/R60/R100, each for seeds 7/17/29. Save full model or expert-only state snapshots at steps 0, 100, 250, 500, 750, 1000, 1500, 2000, and 2500. At each snapshot, run non-mutating fp32, AMP-unscaled diagnostic backwards on a persisted fixed batch under native routing and separately forced sparse-router assignments (and dense-router assignments if cost permits). Compare dense gradients and updates only on sparse mask-surviving coordinates. This is diagnostic instrumentation, not a new recovery condition.

## Verdict

INSUFFICIENT CHECKPOINTS — MINIMAL DIAGNOSTIC RERUN REQUIRED

**Most important implication for the paper:** The existing performance results show architecture-specific favorable router ages, but they cannot establish a common sparse-expert optimization geometry. Scalar gradient norms and final checkpoints are not substitutes for time-resolved gradient/update direction data.

**Most important next experiment:** Run the 18-condition, 85%-only paired sparse/dense diagnostic rerun described above, with intermediate state snapshots and forced-common-routing gradient vectors.

INSUFFICIENT CHECKPOINTS — MINIMAL DIAGNOSTIC RERUN REQUIRED
