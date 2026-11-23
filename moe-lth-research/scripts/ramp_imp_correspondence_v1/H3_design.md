# H3 controlled recovery design — PREPARED, NOT LAUNCHED

This file does not authorize or execute training. Review completed retrospective results and validation before any later recovery launch. There is no training command in this module.

## Question and comparisons

Test whether routed neuron contribution improves weight-mask construction at fixed unstructured pruning budget, rather than whether deleting whole neurons helps immediate post-pruning loss. Include every source seed 7,17,29,43,61, E8/E16/E32, and .80/.85 sparsity; no seed is selected by recovery outcome. If resources require a smaller first cohort, register that choice and its non-outcome rationale before endpoints exist. Independent source initialization/discovery is the replication unit.

A: original immutable IMP support.
B: RAMP allocation of exact per-expert weight budget to neurons, followed by magnitude selection within each neuron's coordinates.
C1: matched magnitude allocation (same bridge, neuron score = mean absolute magnitude across fc1 row + fc2 column at the same scoring checkpoint).
C2: conventional expert-local weight magnitude ranking with exact same keep count (repository OMP scope).
D1: random weight support preserving each expert's exact keep count. D2: matrix-count-matched random support. D3: existing exact-degree-preserving control when feasible, reported as a distinct constraint.
E1: activation-only allocation using same bridge and checkpoint. E2: routing-weighted activation without down-norm. E3: exposure-only allocation. Since exposure is constant across neurons within one expert, this reduces to equal neuron allocation in the primary expert-local budget comparison; state this explicitly instead of pretending it provides meaningful intra-expert ranking.

`moe_lth.analysis.ramp.allocate_weight_mask` is the tested bridge. Each neuron owns one fc1 row and one fc2 column (disjoint parameter coordinates across neurons). Allocate the expert's *actual IMP integer keep count* with capped proportional largest-remainder allocation, deterministic neuron-index ties. Within each allocated neuron, select absolute largest weights across its up row and down column, canonical up-then-down coordinate ties. No gate coordinates, biases, router weights or shared parameters enter. Zero scores trigger uniform allocation. Exact total and dimensions are checked. Baseline allocations use exactly the same tie and within-neuron rule. Allocation and score files must be sealed with hashes before recovery outcomes exist. Do not require each neuron to retain both an incoming and outgoing connection unless all arms enforce that as a separately registered bridge.

The primary comparison holds total budget per expert, not matrix marginals or individual degrees. Different matrix allocations are a possible bridge mechanism and must be reported; matrix-count and degree-matched secondary comparisons address it separately. Record fc1/fc2 counts and zero-input/zero-output neurons. Structured neuron pruning would be a separate study with separate active parameter, FLOP, resident-memory and latency accounting; it cannot substitute for B.

## Checkpoint timing and controls

Primary mechanistic comparison: use the same predeclared unpruned discovery-final checkpoint for ranking B/C/E, then rewind the entire model to its exact original step-0 state for recovery. This is retrospective mask construction, not a during-training method. Secondary early-scoring comparison uses actual step250, then applies masks to the same step0 recovery start. All baseline ranking weights/activations come from that exact checkpoint; no final-state feature is permitted in this early comparison. Existing IMP uses its full discovery trajectory and has a larger search information/compute budget; explicitly report this asymmetry rather than claim equal discovery cost.

Hold full-model starting checkpoint, router initialization, expert biases, all shared parameters, training examples and batch order, training-token count, AdamW configuration, clipping, auxiliary routing loss, precision, learning rate and lack of scheduler, rewind, capacity/batch shape, dense comparison, and checkpoint selection fixed. Reuse canonical `build_dataloaders` and recovery utilities only after a separately authorized implementation/review. Preserve exact train/validation batch hashes. Use the canonical 2500-attempt budget with actual applied updates and AMP skips recorded; do not compensate skipped attempts. Fixed evaluation at canonical endpoints, not arm-specific best checkpoints. Record final and best validation CE and full curves, but primary endpoint remains prespecified final step.

Use a paired dense run with the same protocol and initialization. Verify hashes and curves before historical reuse; otherwise label missing references and request a later authorized matched design. Never silently train replacement references. Primary endpoint is relative sparse-minus-paired-dense CE. Report equal-seed paired differences B−A/B−C/B−D and uncertainty across independent seeds, stratified by architecture/sparsity/checkpoint timing; random draws are nested. Do not pool the existing 5000-step intervention cohorts with canonical 2500-step endpoints. No held-out loss may tune the mask allocation exponent, strong-retention threshold, calibration sample, scoring checkpoint or selected seeds.

## Launch gate and evidence threshold

Before a future authorized launch: all profiling/mapping/integration tests pass; all calibration and model/mask hashes sealed; exact parameter-count equality for every arm; no coordinate aliasing; state and batch hashes equal; checkpoint-timing leakage check; loss/gradient/AMP accounting and pause/resume consistent with the canonical trainer; compute/storage estimate based on a separately authorized benchmark. This task performs none of that training.

Immediate pruning performance is a diagnostic, not a lottery-ticket result. Recovery superiority requires consistent independent-seed evidence against matched alternatives; a formal winning-ticket claim additionally needs the repository's held-out/equivalence criteria. The present small-sample H2 correlations do not establish benefit. Existing degree controls motivate testing connection structure alongside neuron scores, not assuming RAMP will succeed.
