# Estimands and interpretation limits

Let L[a,s,r,k,e] be token-weighted validation CE after 2500 attempted updates, where a is factorial arm A/B/C/D, s is seed, r is router age20/100, k is sparse/dense, and e is readout. Primary e forces the identical archived donor assignments, capacity scores and gate coefficients. Each model is evaluated over the same complete validation batch sequence, without token resampling.

G[a,s,r,e] = L[a,s,r,sparse,e] - L[a,s,r,dense,e].
T[a,s,e] = G[a,s,20,e] - G[a,s,100,e].

The age contrast is signed; negative values mean the sparse-minus-dense penalty is smaller for R20. The screen is about changing this contrast, not proving a smaller absolute sparse CE or finding a winning sparse ticket.

The two isolated effects are T_C-T_A (donor values with native Jacobian) and T_D-T_A (Jacobian removal with native values). Combined effect is T_B-T_A. The interaction is T_B-T_C-T_D+T_A. Marginal factorial main effects average the two conditional effects: value = ((T_C-T_A)+(T_B-T_D))/2; detach = ((T_D-T_A)+(T_B-T_C))/2. When interaction is large, marginal and native-reference simple effects answer different questions and both must be retained.

Per-seed absolute attenuation is 1-|T_arm|/(|T_A|+1e-12). This can be negative or exceed simple signed-ratio interpretations when order reverses. Ratios with |T_A|<1e-5 are flagged. Report seed values, arithmetic mean, sample SD, and range. The attenuation of seed-mean contrasts is a different descriptive statistic and should not replace the mean of seed attenuations. The independent units are the three selected seeds, not tokens, experts, timesteps or validation batches. No confirmatory p-values or fresh-seed claims are warranted.

The stopping rule is frozen in protocol.json and the original decisive_experiment.md: finish exactly24 new recoveries; promotion requires an isolated effect exceeding .02 nats in the same direction in all three seeds, passed implementation controls, and an effect not solely explained by evaluation-policy reversal. No post-outcome success threshold or automatic additional seeds is permitted.

C supplies an explicitly specified surrogate backward Jacobian. It is not the ordinary finite-difference derivative of the donor-constant forward function. Equality is checked against the native gate path, not against the entire native parameter gradient: changed forward values change downstream derivatives. D removes the gate derivative while retaining native values. All arms use the same donor assignment and capacity archive. The native full-softmax auxiliary objective and coefficient .1 remain; cutting earlier gates also cuts their indirect contribution to later auxiliary losses. Thus this screen does not isolate a CE-exclusive gate-gradient intervention. A separate auxiliary-path control would be needed for that narrower interpretation.

Baseline A states and A/B intermediate states are unavailable. Their audited scalar readouts are reused with matching source/data/mask/precision/trainer identities and retained audit provenance. B final states remain. No new A/B evaluations, fabricated snapshots, or baseline recoveries are claimed. The C/R100/dense self-replay provides a full-trajectory identity test against a retained native donor control.

The same legacy AMP budget means2500 attempted updates in every new arm, not a forced equality of successful AdamW updates. Overflow skips are exposed separately rather than compensated with extra updates. The same validation token set was already used in the research investigation; this is an exploratory mechanism screen, not untouched-test generalization.
