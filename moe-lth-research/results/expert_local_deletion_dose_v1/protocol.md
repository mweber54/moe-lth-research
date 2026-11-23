# Expert-local deletion dose and reliability check v1

This is a measurement study using the completed Stage 1 engineering pilot, not a
specialization study. Seed 17 remains a NONMATCHING 32E IMP80 engineering case.
Use only its verified sparse and matched unpruned recovery steps 25,250,1000,2500;
2500 is the pre-reset endpoint. Historical/rerun/reset endpoint aliases count once.
Layer index 1, experts 0,8,16,24, fc1.weight and the original eight disjoint groups
are fixed. No additional seeds, training, optimizer, permanent pruning, recovery,
cluster refit, specialization predictor, or Stage 2 is launched.

Four group permutations use fixed numpy seeds 170901,170902,170903,170904, identically
across experts, time and trajectories. Prefix unions have 0,1,2,4,8 groups. Deduplicate
identical (expert,coordinate-set) unions before evaluation and preserve every
ordering/dose alias separately. The empty union and all-eight union are each unique
per expert. No previous global union cap applies: each intervention affects exactly
ONE expert fc1 matrix. Coordinates match between sparse and unpruned models, not
relative density. Report both sparse mask-active matrix denominator and unpruned
full-matrix denominator; actual nonzero counts are separately recorded per snapshot.

Reuse the complete, original 2 x 128 x 128 A and B training-only batches. Both have
already been examined and are diagnostic populations, not independent generalization
data. Preserve native sequence structure, gates, top1 routing and capacity. Eval,
FP32, TF32 disabled, inference mode, signed actual-token-weighted task CE; auxiliary
loss stays separate. Use original baseline acceptance for conditional and complement
subsets; attention/later layers can spread effects beyond accepted positions.

Capture baseline expert inputs and outputs in the actual native dispatch order in
each complete batch. For cached single-group CE interventions replay only the expert
function on these *same native-dispatch rows*, not routing on isolated tokens. These
rows and gates cannot change under a layer-1 fc1 intervention. Verify against
representative full-model interventions at every expert/checkpoint/probe. Raw output
RMS difference = sqrt(mean((deleted_output-baseline_output)^2)) over accepted positions
and output channels. Relative difference divides by baseline RMS only if >1e-12;
otherwise undefined_near_zero_baseline. No accepted positions -> undefined, never zero.
Gate-weighted RMS = sqrt(mean((baseline_gate * output_difference)^2)), same denominator.
No change to capacity population or gate normalization. Coverage <32 accepted positions
is low_traffic, retained rather than removed. This minimum is descriptive, not a test.

A separate positive control returns zeros from the selected expert module using a
scoped forward hook, preserving native router/gate/capacity calculations at that
layer. It zeros the expert output contribution, including fc2 bias; it is NOT a
weight-deletion dose and is excluded from dose curves and weight-dose correlations.

For each union compute continuous whole/accepted/complement CE damage and per-sequence
paired damage/counts. Nonadditivity = union damage - sum of cached Stage 1 component
single-group damages, for each subset separately. No component subtraction is defined
for the output-zero positive control. Exact token-paired deltas and expert metrics are
cached with source, checkpoint, probe, implementation and intervention identities.

Reliability is A/B paired agreement and ranking. Main strata: each trajectory/time/
equal group dose, pooling unique expert/unions; also report within-expert equal-dose
rankings. No duplicate ordering aliases count as separate observations. A single union
per expert at eight groups has no within-expert rank reliability estimate. Report MAE,
RMSE, bias, ranges, sign agreement, Spearman and pairwise order concordance with ties
explicit; undefined constant/small panels remain undefined. Use a representative
repeatability floor (max measured repeat difference, at least 1e-12) only to flag ties
and numerically zero outcomes, not as a damage-selection threshold. Separately summarize
adequate-coverage paired panels; retain all low-traffic panels in primary outputs.
Do not pool doses to manufacture ranking reliability. No p-values or seed-level
confidence claims: one seed, nested experts/groups/orderings/checkpoints/sequences.
0.001 and 0.0001 nats/token are descriptive plot reference lines only. Do not choose
one to obtain a trend. Continuous outcomes and conditional/functional effects lead.

Validate zero dose, cached-baseline equivalence, representative single and 4-group
repeatability, positive-control repeatability, exact selected-coordinate restoration
after EVERY intervention, full model hash after each expert panel, unchanged RNG,
module modes/gradients and TF32 flags, sparse mask integrity, unchanged targeted-layer
routing/gates, and hook removal. Scoped finally blocks restore even on failures.
Workers use study/host advisory locks and 15-second heartbeats. Write only new results
under expert_local_deletion_dose_v1, preserve historical artifacts, and record actual
stage wall-time, CUDA-event forward time, forward counts, peak memory and storage.
