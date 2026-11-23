# Exploratory temporal deletion damage findings

All measurements and both context populations were previously inspected. This is a post hoc checkpoint-only analysis of the seed17 NONMATCHING engineering case. No model execution, training, new probes or new model seeds occurred. Primary transitions are250→1000 and1000→2500;25→250 is secondary. Step2500 is pre-reset. Identical endpoint aliases count once.

Prediction scores always use whole-probe discovery damage. The same selected IDs are evaluated on whole and future-native baseline-accepted positions. Future accepted populations change over time; they are not the same routed population. Sparse and unpruned trajectories, experts, doses and transitions remain separate. Exact per-union magnitude is unavailable; expert load is constant across unions within a panel and is context only.

Tables contain every registered panel, including singleton trajectories and low-traffic cases. Dose1 has3 distinct unions, doses2/4 have4, doses0/8 and output-zero have1. Output-zero is a separate positive control. Singleton panels have no ranks, selected contrasts or policy comparisons. Overlapping unions are dependent. All95% intervals use2000 paired complete-batch resamples; they describe sampled contexts within one source seed, with no multiplicity correction.

## Current-damage temporal ordering

Counts below audit separate panels; they are not pooled correlations, independent successes or model-level replications. A positive fixed-selection contrast supports ordering; consistent signs across all four future blocks are a stricter descriptive check. Tied extrema are unidentified.

|Trajectory|Transition|Outcome|Covered panels|Positive fixed CI|Negative fixed CI|Positive CI and all4 blocks positive|
|---|---|---|---:|---:|---:|---:|
|sparse|1000_to_2500|baseline_accepted|12|2|3|1|
|sparse|1000_to_2500|whole|12|2|2|1|
|sparse|250_to_1000|baseline_accepted|12|1|3|0|
|sparse|250_to_1000|whole|12|1|3|0|
|sparse|25_to_250|baseline_accepted|0|0|0|0|
|sparse|25_to_250|whole|0|0|0|0|
|unpruned|1000_to_2500|baseline_accepted|12|4|0|3|
|unpruned|1000_to_2500|whole|12|3|1|2|
|unpruned|250_to_1000|baseline_accepted|12|4|2|4|
|unpruned|250_to_1000|whole|12|4|1|2|
|unpruned|25_to_250|baseline_accepted|6|1|0|1|
|unpruned|25_to_250|whole|6|1|0|1|

Every covered whole-outcome current-rule condition with a positive fixed-selection interval and all four future block contrasts positive is listed below. These are exploratory named candidates, not preselected confirmatory discoveries.

|Trajectory|Transition|Expert|Dose|Spearman|Future high−low (nats/token)|95% interval|Reselected95% interval|
|---|---|---:|---:|---:|---:|---|---|
|sparse|1000_to_2500|0|1|1.00|1.583e-05|6.366e-06 to 2.610e-05|-1.046e-05 to 2.515e-05|
|unpruned|25_to_250|0|4|0.40|5.272e-06|1.427e-06 to 9.248e-06|3.549e-07 to 9.210e-06|
|unpruned|250_to_1000|8|1|0.50|1.249e-06|7.722e-07 to 1.707e-06|-1.458e-06 to 1.848e-06|
|unpruned|1000_to_2500|8|1|0.50|8.660e-06|4.562e-06 to 1.257e-05|-2.832e-06 to 1.208e-05|
|unpruned|1000_to_2500|16|4|0.20|4.761e-06|2.383e-06 to 7.183e-06|-1.085e-05 to 6.986e-06|
|unpruned|250_to_1000|24|4|0.80|2.437e-06|4.119e-07 to 5.875e-06|-4.436e-06 to 5.687e-06|

All five primary named contrasts have discovery-reselection intervals crossing zero. Their fixed-ID future effects are measurable, but selection instability prevents calling these robust online temporal ranking rules at this budget. The secondary unpruned25→250 expert0 dose4 example survives reselection; it is explicitly outside the primary transitions.

## History and a concrete low-damage decision

Selection advantage is future panel mean minus selected future damage. The panel mean is uniform selection among these3/4 fixed overlapping unions, not random pruning throughout the model. Current-minus-rule damage is positive when history improves on current. Identical choices have exactly zero policy difference and are not independent corroboration. Future regret is an optimistic hindsight diagnostic.

|Trajectory|Transition|Rule|Covered panels|Same as current|Advantage CI>0|History improvement CI>0|History worse CI<0|
|---|---|---|---:|---:|---:|---:|---:|
|sparse|1000_to_2500|current|12|12|2|0|0|
|sparse|1000_to_2500|history_max|12|6|1|2|0|
|sparse|1000_to_2500|history_mean|12|7|2|1|0|
|sparse|250_to_1000|current|12|12|1|0|0|
|unpruned|1000_to_2500|current|12|12|3|0|0|
|unpruned|1000_to_2500|history_max|12|10|3|1|0|
|unpruned|1000_to_2500|history_mean|12|9|3|0|0|
|unpruned|250_to_1000|current|12|12|3|0|0|
|unpruned|250_to_1000|history_max|6|4|0|0|0|
|unpruned|250_to_1000|history_mean|6|4|0|0|0|
|unpruned|25_to_250|current|6|6|2|0|0|

All covered, identified history/current differences with fixed-selection intervals excluding zero (including losses) follow:

|Trajectory|Transition|Expert|Dose|Rule|Current−history future damage|Fixed95% interval|Reselected95% interval|
|---|---|---:|---:|---|---:|---|---|
|sparse|1000_to_2500|8|2|history_mean|2.760e-05|5.583e-06 to 5.347e-05|0.000e+00 to 4.806e-05|
|sparse|1000_to_2500|8|2|history_max|2.760e-05|5.583e-06 to 5.347e-05|-1.229e-05 to 5.454e-05|
|sparse|1000_to_2500|8|4|history_max|2.309e-05|7.817e-07 to 5.027e-05|-8.592e-06 to 6.381e-05|
|unpruned|1000_to_2500|8|4|history_max|5.840e-06|6.453e-07 to 1.055e-05|-9.654e-06 to 1.310e-05|

No primary current-rule panel has a positive selection-advantage interval after discovery reselection. The complete surviving set is secondary unpruned25→250 experts0 and8, both dose4. All four fixed-selection history improvements above touch or cross zero after discovery reselection. Thus history has not shown stable added value over current damage.

## Coverage, conditional effects and thresholds

Coverage qualification uses only the frozen≥32 accepted positions in each four-batch block, separately for early discovery and future confirmation, and previous discovery for history rules. All whole-probe outcomes remain in the tables despite failure. Native damage can evolve through weights, routing, gates, context populations and downstream computation; these data do not isolate a cause or specialization.

|Trajectory|Checkpoint|Partition|Expert|Minimum accepted/block|Contributing batches|Max batch fraction|HHI|
|---|---:|---|---:|---:|---:|---:|---:|
|sparse|25|discovery|0|28|16|0.0948|0.0675|
|sparse|25|discovery|8|0|0|nan|nan|
|sparse|25|discovery|16|0|0|nan|nan|
|sparse|25|discovery|24|0|0|nan|nan|
|unpruned|25|discovery|16|0|0|nan|nan|
|unpruned|25|discovery|24|0|0|nan|nan|

|Outcome|Tolerance|Threshold panels|Ceilings|Empty|Panels with point false-safe>0|
|---|---:|---:|---:|---:|---:|
|baseline_accepted|0.0001|160|144|16|58|
|baseline_accepted|0.001|160|154|6|13|
|whole|0.0001|160|144|16|2|
|whole|0.001|160|154|6|0|

Thresholds1e-4 and1e-3 are descriptive secondary conventions, not optimized cutoffs or safety criteria. Ceiling means lack of discrimination. Empty false-safe is undefined. Zero observed false-safe events do not prove safety. Coordinate fractions use actual set unions, with panel and active-matrix denominators; overlapping coordinates are counted once.

Conditional and whole damage values, intervals and threshold classifications are provided side by side in the tables and separate figures. Their different target denominators can produce much larger conditional damage despite small whole means. For example, unpruned step2500 expert0 union e0_n1_8b18d0c66a236ba2 has whole future damage−5.998e-7 but future-native accepted-position damage about8.1e-5 nats/token. This retrospective illustration is not a selected prediction rule or a causal explanation. Conditional bootstrap samples with no accepted targets are undefined and recorded; whole means are still reported.

## Decision and limits

1. Early whole damage has named temporal ordering candidates in 5 primary equal-dose panels under the positive-interval/all-block-sign descriptive check (listed above). Other panels remain unresolved or are reversal candidates; Spearman alone with3/4 unions is fragile.
2. History is not uniformly better: 4 covered identified history comparisons favor history and 0 favor current by fixed-selection intervals; identical selections give no added evidence. Reselected intervals and selection frequencies show sensitivity to discovery noise.
3. Current low-score selection beats fixed-panel uniform selection by a positive fixed-selection interval in 11 of 54 covered identified whole-outcome panels. This is a context-level description, not an independent-replication success rate.
4. Read effect sizes and both fixed/reselected intervals together. Many differences are small relative to1e-4/1e-3; the threshold tables disclose ceilings rather than tightening tolerances. Conditional effects can be concealed by whole-probe averaging.
5. Early low/absent traffic and tiny3/4-candidate overlapping panels limit interpretation; singletons provide no ordering information. Tokens, batches, groups, orderings and checkpoints are not independent model seeds.

Decision: practical temporal discrimination remains unresolved at this measurement budget. Improve selection stability before specifying a cross-seed confirmation of a usable temporal pruning rule. The fixed-ID primary signals are candidates to investigate, not sufficient support for a general rule; history adds no stable demonstrated value. No cross-seed or continuation experiment is launched.

This predicts later deletion damage along the original unmodified recovery trajectory. It does not establish that deleting early preserves subsequent learning, specialization as a cause, a winning-ticket-specific property, safe combinations of deletions or practical training savings.

Reproduce using prepare.py (fresh output only), analyze.py, report.py and verify.py in scripts/expert_deletion_temporal_prediction_v1/. The frozen config/protocol and estimand clarification predate temporal associations. Only coordinate tensors were loaded; checkpoints were byte-hashed without model deserialization. Original sealed artifacts remain unchanged.
