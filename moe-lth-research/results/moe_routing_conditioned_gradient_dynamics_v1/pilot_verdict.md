# Pilot verdict: routing-conditioned gradient dynamics v1

Cell: `seed7_A`, checkpoints 250→500, identical 90/90 canonical probe halves.

## Scientific answers

1. **Do initially conflicted connections exhibit temporal release?** 46,303 coordinates (0.0690% of qualified step-250 coordinates) meet the discovery release rule: high initial activity and cancellation, weak initial aggregate gradient, cancellation drop of at least 0.2, aggregate-gradient ratio at least 1.25, and retained activity. Mean independent-validation overlap across matrices is 58.6344%. The phenomenon is measurable but one-source and threshold-based. Weaker-activity, alignment, and dominance subtype indicators may overlap and are reported separately.

2. **Does release correspond to stronger subsequent aggregate signals?** By construction the operational release definition requires stronger aggregate magnitude, so this is an audited classification property, not independent evidence. Subsequent movement is reported relative to non-release controls in `cancellation_release_summary.csv`; it is not interpreted as importance.

3. **Are routing-composition changes associated with release?** Descriptive symmetric reweighting assigns 12.34% of the sum of aggregate-component magnitudes to acceptance/composition and 87.66% to conditional-gradient content; for absolute cancelled mass, the fractions are 26.25% routing and 73.75% content. Restricted to the operational release population, the fractions are 11.76% routing and 88.24% content for aggregate gradient, and 26.30% routing and 73.70% content for cancelled mass. These are not causal effects: context dependencies and accepted-token identity violate a composition-only interpretation.

4. **Can future learning-signal changes be predicted?**

- `log1p_future_aggregate`: conventional mean held-out metric 0.3067, domain/routing-augmented 0.3029, increment -0.0038.
- `log1p_future_activity`: conventional mean held-out metric 0.4008, domain/routing-augmented 0.4001, increment -0.0007.
- `log1p_future_movement`: conventional mean held-out metric 0.1056, domain/routing-augmented 0.1131, increment +0.0075.
- `release`: conventional mean held-out metric 0.9967, domain/routing-augmented 0.9953, increment -0.0013.

These grouped predictions are exploratory within one source. A positive increment would support feasibility only, not cross-seed generalization.

5. **Do conventional statistics already capture the structure?** Cancellation is mathematically constrained by aggregate and domain activity/moments. The baseline redundancy table retains the equal-mixture identity audit, and the prediction comparison tests incremental value rather than assuming independence.

6. **Can controlled deletion populations be constructed?** 0/256 matrices reached the 1% per-matrix pair target under matrix/sign/magnitude-bin/aggregate-bin matching. Only 12/256 achieved exact row/column degree balance with the simple feasibility matcher. The high-activity aligned control reached the same target in 0/256 matrices. Residuals and infeasible constraints are retained; no weights were deleted.

## Required causal intervention

A causal routing conclusion would require a frozen-checkpoint replay in which the routed assignment or acceptance composition is altered while preserving the same token content, checkpoint representation, gates/attention state where feasible, and loss terms. Its factorial controls must separately vary acceptance composition, gate amplitude/derivative, and accepted-token identity. The current descriptive reweighting cannot separate those mechanisms.

## Falsification conditions

- No reproducible temporal release remains under independent probes or nearby discovery-locked thresholds.
- Routing-composition reweighting reconstructs little of observed gradient or cancelled-mass evolution, while conditional content captures it.
- Domain/routing features provide no incremental grouped predictive performance beyond conventional magnitude, aggregate gradient, gradient energy, and movement.
- No matched small-magnitude populations remain after enforcing matrix, sign, magnitude, aggregate-gradient, and degree constraints.
- A frozen-router or accepted-token intervention eliminates the association in a design where the implementation is otherwise unchanged.

## Limitations

- One initialization/data-order source; matrices and coordinates are not independent trials.
- Two checkpoints only; no claim about temporal ordering between routing and gradient changes.
- Target-domain attribution does not equal the source domain of all contributing context routes.
- RAMP and activation measures are correlates, not importance.
- No masking, pruning, optimizer step, recovery, or router intervention occurred.
- No importance, dispensability, causal routing, or ICML-level novelty claim is made.

## Recommendation

Do not promote this mechanism on the current evidence. Release exists in a small population, but release-specific routing-composition contribution is about 11.76%, domain/routing features do not materially improve held-out aggregate, activity, or release prediction, and the registered matched populations are infeasible. Narrow the hypothesis to a conditional-gradient/content phenomenon or abandon the routing-conditioned dispensability interpretation unless a future causal replay materially changes these findings.
