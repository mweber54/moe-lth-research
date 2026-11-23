# IMP mask information pilot

Status: running. Completed recoveries: 16/16.

## Direct observations

Pilot checkpoint: imp_e16_seed17_final_s0.80. Estimated total GPU occupancy: 4.04 hours; contention can increase runtime.

Active condition: matrix, replicate 4.

Baseline reproduction gate passed: True; absolute CE discrepancy 0.000000.
IMP rerun final validation CE: 1.458223 nats/token.
global: 5 masks, mean CE 1.683511, sample SD 0.0127101584071728, random-minus-IMP +0.225288.
expert: 5 masks, mean CE 1.690385, sample SD 0.009439394479083263, random-minus-IMP +0.232162.
matrix: 5 masks, mean CE 1.678239, sample SD 0.008362915946536906, random-minus-IMP +0.220016.

## Statistical findings

The pilot has one independently trained source checkpoint. Mask realizations are nested within it. No seed-level confidence interval, statistical significance, or practical equivalence is established. The prespecified equivalence margin is +/-0.02 nats/token. Partial results are provisional.

## Mechanistic hypotheses and unresolved alternatives

Any observed contrast concerns connection placement beyond the specified sparsity allocation. It cannot establish indispensable individual connections, routed-token information, or disrupted early learning. Architecture, training seed, sparsity, router protocol and recovery duration remain possible moderators. Selection of a strong IMP endpoint limits generalization. Expand to independent seeds and nonwinning controls only after review; no automatic expansion.

See protocol.md, checkpoint_inventory.csv, audit_details.json and pilot_plan.json for provenance. Source IMP and dense endpoints are prior observations, not new pilot controls. The 5,000-step fresh-optimizer extensions remain separate.
