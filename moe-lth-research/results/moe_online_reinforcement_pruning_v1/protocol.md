# Online reinforcement pruning v1 — frozen protocol

## Registration boundary

This protocol is written before generating or reading any pruned-continuation
outcome. The closed rectangle-connectivity experiment and its artifacts are
immutable and outside this study.

## Question and cohort

Primary hypothesis: at equal step-250 magnitude distribution and identical
deletion counts in every expert-matrix row and column, permanently deleting
highly reinforced small expert weights causes higher validation cross-entropy
at step 2500 than deleting matched random small weights.

Use the existing dense 32-expert model, optimizer, data, validation tokens, and
orders A/B for seeds 7, 17, and 29. Eligible tensors are every expert `fc1` and
`fc2` weight matrix. Routing and shared parameters remain trainable. Selection
must not use IMP masks, eventual-survivor labels, future updates, validation
outcomes, or branch outcomes.

## Reinforcement score

For optimizer attempts 1–250, accumulate each parameter's signed displacement
from AdamW's task-driven update after gradient unscaling and global clipping,
excluding decoupled weight decay. For scalar i,

`R_i = sign(w_i,0) * sum_t(u_i,t) / sum_t(abs(u_i,t))`.

Positive means cumulative task-driven displacement follows the starting sign.
Zero starting signs and zero update denominators are excluded from both arms
and counted. A hand-computed sign test is mandatory. The complete step-250
branch state includes model, optimizer moments/counters, scaler, scheduler (or
an explicit no-scheduler marker), next batch position, and Python/NumPy/Torch/
CUDA RNG. Restore must reproduce the next untouched update before branching.

## Dose and candidate pool

For each eligible matrix, deletion count is `floor(0.05*N + 0.5)`, where N is
the total scalar count. Candidates are eligible scored weights in the smallest
25% by absolute step-250 magnitude. The quartile size is `floor(N/4)` after a
stable sort by `(absolute magnitude, flat index)`; excluded entries are removed
from candidacy, not replaced from outside the frozen quartile. A cell fails if
the pool cannot supply the registered count.

The random arm is constructed first by uniform sampling without replacement
from each candidate pool. Its seed is the first 64 bits of SHA-256 over
`moe_online_reinforcement_pruning_v1|random|seed|order|tensor`. This draw is
never redrawn. Its row and column deletion totals are frozen targets.

The high arm maximizes total reinforcement over candidate edges subject to the
exact random row and column totals. Stable flat index is the final tie-break.
Magnitude balance may be improved only by deterministic degree-preserving
exchanges that increase or retain the frozen score objective subject to all
acceptance checks. There is no rectangle constraint.

Within each candidate pool, five equal-frequency magnitude bins use stable
`(absolute magnitude, flat index)` order and deterministic array splitting;
the largest and smallest bin sizes differ by at most one.

## Frozen acceptance requirements

- Identical deletion counts in every matrix, row, and column.
- Within every matrix, high/random deleted fractions in each of five magnitude
  bins differ by at most 0.02.
- Within every matrix, removed squared-weight energies differ by at most 5% of
  random energy; if random energy is zero, require equality.
- Cell-level mean reinforcement separation is at least 0.20.
- Cell-level shared-location fraction is at most 0.30.

Audit actual indices independently. Report overlap, score separation, current
magnitudes, removed energy, magnitude-bin fractions, and row/column energy
discrepancies by matrix and cell. Seal all six selections before continuation.
Any failed cell stops the study before branch training; random draws and gates
remain fixed.

## Branch plan and equivalence

From each identical step-250 state run untouched, random-small, and
high-reinforcement-small through step 2500. Pruned arms zero selected weights
immediately, clear their optimizer moments, mask gradients, and enforce zeros
after each update. No other parameter may change at deletion. Selected entries
must never regrow. Within each triplet subsequent batch IDs, learning-rate
schedule, optimizer/scaler starting state, and RNG start agree. Routing adapts
normally.

Evaluate the same validation tokens immediately before deletion, immediately
after deletion, and at steps 251, 275, 500, 1000, and 2500. Evaluation must
restore model modes and RNG and must not alter the training stream. Existing
untouched work is reusable only after exact state, stream, and protocol checks.

GPU admission uses the existing shared policy: do not kill other processes;
one worker per admitted device; require at least 24 GiB free GPU memory and no
more than 90% utilization; preserve locks, heartbeat, failures, and resumable
complete state. No GPU branch starts before all selection and restore gates pass.

## Primary contrast and reporting

For each seed/order cell,
`Delta = CE_2500(high-reinforcement-small) - CE_2500(random-small)`.
Positive means greater lasting damage from deleting reinforced weights. Report
all six deltas, A/B mean within seed, and the unweighted mean of three seed
means; seeds are independent replicates. Also report each pruned arm's excess
CE over untouched, paired gap trajectories, immediate-versus-final recovery,
directional consistency, effect size, and heterogeneity. A positive result
supports protecting some small weights only at this checkpoint and dose; it is
not evidence of an improved pruning algorithm without a later policy test.
