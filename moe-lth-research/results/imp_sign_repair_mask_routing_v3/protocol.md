# Sign repair, IMP connectivity, and routing

The sources are the sealed v2 draw sets, E32 checkpoints, probes, validation streams, and untouched IMP trajectories for seeds 7, 17, and 29 in orders A and B. Each seed is an independent replication. Report both orders within seed.

## Sign tracking and repair

Use absolute weight magnitude at most 1e-8 as the near-zero category. Use validation batches at zero-based positions 0, 7, 15, and 23 for the common early readout, preserving full batch shapes and target weighting. Full validation uses the archived v2 method in FP32 with TF32 disabled. Validate free replay at steps 0 and 250 against saved model hashes, verify step-0 full CE and count applied versus skipped AdamW steps through the replay window. Retained snapshots supply later sign observations. First return is known at attempted-step resolution through step 250. Afterwards, first observed return or continued agreement is measured only at retained checkpoints. Sign excursions between retained checkpoints are unobserved.

The optimizer projection uses draw 1 of every seed and order. For the first 250 attempted steps, after each applied AdamW update, a selected coordinate is set to exactly zero if its updated weight has the opposite sign from its post-flip starting weight. Exactly zero is on the boundary of the permitted half-line. Optimizer moments are preserved. At step 251 and later there is no projection. The held-out coordinates, including every other weight in the same matrix, are untouched by the projection. Event counts, projection distances and per-coordinate binding counts are retained.

The dense predictor uses the sign of the actual dense model weight at the 100th applied update. A fresh native replay must match the archived probe score and activity arrays exactly before its dense endpoint is accepted. The net adaptive displacement sign uses the original AdamW score multiplied by the initial sign. Decoupled decay is excluded from displacement, with its actual effect retained in the dense endpoint.

## IMP connectivity

Generate two independently seeded double-edge exchange masks per seed, with 24 proposals per original surviving edge. Verify every individual row and column survivor count and require at least 50% of IMP survivor edges to move for each full-model control. Reuse the v2 conditioned rectangles, acceptance gates, and original score definition. Search the common dose grid from 0.34207 down and reuse the sealed IMP comparison at 0.34207 only if regenerated IMP flip sets are exactly identical. For a lower common dose, rerun IMP at that dose. Apply the repair intervention to matched masks if the step-250 full-validation repair interaction is positive in every seed mean.

## Routing

The scaled condition multiplies each layer's native selected gate coefficients by 0.5 and retains the native selected IDs and unscaled native scores for capacity priority given that layer's inputs. Later layers can receive different representations and produce different endogenous assignments. The live task gradient with respect to the gate coefficient is scaled along with its forward value. The auxiliary routing path stays native. Interpret this as a combined forward and backward routing intervention. Both recovery regimes use the same initial checkpoint and data. Fresh native and scaled 100-update dense probes determine the flip sets before outcome evaluation. Require survivor top-quartile overlap at most 0.90 by Dice and selected high-set Dice overlap at most 0.60 for all six seed–order cells. If either check fails, retain diagnostics and withhold crossover recoveries.
