# Router–expert geometry v1: fixed registration

Registered before geometry outcomes: 32 experts; source seeds 7,17,29,43,61;
R0/R50/R100; learned magnitude and count-matched random masks at .50/.65;
15 matched unpruned recoveries, 60 sparse, 75 total. No source training or Dataset B
restart. Source and recovery each 2500 attempted steps. Measure 0,25,100,250,500,
1250,2500. Historical endpoints are explicitly historical, never trajectories.

## Actual model and coordinate system

Four blocks: pre_router_hidden = blocks.L.moe_norm(attention residual), dimension
256. Router and every expert receive identical vectors in this coordinate system.
blocks.L.moe.router.projection.weight [32,256], bias absent; logits / positive
stored temperature, softmax then top-1. Centered rows are primary: adding a common
row leaves softmax invariant (up to numerical rounding). Experts have
blocks.L.moe.experts.E.fc1.weight [1024,256], fc1.bias [1024], GELU, dropout 0,
fc2.weight [256,1024], fc2.bias [256]. No gated projection. Capacity is
ceil(1.25 * token_count / 32), confidence-ranked truncation. Accepted outputs
are multiplied by selected softmax probability, without gate renormalization.

Magnitude masks jointly rank absolute fc1/fc2 weights within each expert and retain
round(N*(1-sparsity)), at least one, using torch topk; biases are unpruned. Random
masks preserve exact per-tensor counts using dataset-b-random-v1 deterministic seed.
Existing serialized masks are reused and hash/count/generation checked. Experts
and all shared parameters rewind to source R0, then selected source-age router is
restored; router remains trainable. Native optimizer, LR .0003, weight decay .01,
clipping 1, fp16 AMP, fixed 128-example training batches and 32 validation batches
are preserved. Source checkpoints and canonical batch hashes constrain identity.

## Metrics and references

Right singular bases of fc1 via CPU FP64 Gram eigendecomposition (validated against direct SVD), returned in FP32; ranks 8,16(primary),32; centered/raw router
projection energy normalized by router norm; isotropic k/256; within-layer all
other-row shuffled pairing distribution; 32 fixed Gaussian directions per expert
(seed 918273+expert). These diagnostics are not replication seeds. Singular cutoff,
relative gap (<.001 flagged), retained energy, zero-norm and rank deficiency flags
are recorded. Undefined alignment is null. Rank-deficient/degenerate bases must be
excluded from headline inference. Subspace overlap = ||A.T B||F²/k, invariant to
basis rotations/signs. Compare source/current and recovery-initial/current bases;
current/current, initial-router/current-expert and current-router/initial-expert
separate descriptive movement. Do not interpret these as interventions.

Source-final trained, masked-trained (pruning only), unmasked initialization plus
selected router age, masked initialization plus selected age, and historical
recovered endpoints are separate states. Source-age geometry is additionally
reported from each actual source checkpoint; it is not recovery initialization.

Probe: first eight sequences from the existing fixed training loader, never
validation; all 1024 positions selected with generator seed 72419 in fixed random
order. Source-final native inputs cached per seed/layer are the frozen coordinate
reference. Native end-to-end uses the same token IDs and position order but current
representations. First 128 selected vectors feed every expert for counterfactual
preactivation, GELU mean/RMS, output RMS and source-response MSE/cosine. Positive
preactivation is not an activity definition for GELU. Store router probabilities,
entropy/log32, margins, assignments, confidence-accepted traffic and source/initial
agreement. Native source responses are evaluated on identical current input
vectors, so response comparisons do not confound different vectors. Winning-runner
row difference alignment is evaluated only for observed winners.

All diagnostics use eval, no_grad, FP32, restore each module's exact mode and
Python/NumPy/CPU/CUDA RNG; no optimizer, gradient, buffer or training-batch mutation.
Separate diagnostic time from total elapsed minus diagnostics. Checkpoint states
at each measurement are diagnostic model states, explicitly not resumable.

## Analysis and limitations

Independent n=5 source seeds. Primary families: learned-random pruning preservation;
paired recovery changes within seed/age/sparsity; paired age differences within seed;
within-condition geometry changes versus sparse-minus-matched-new-dense loss.
Report all seed effects, means and seed-bootstrap 95% intervals (fixed seed 8103).
Rank/layer sensitivity and correlations are exploratory. Do not pool architecture,
age, sparsity or time in mechanistic correlations. Explicitly assess improvement
without restoration, restoration without loss gain, similar geometry with different
loss, and inconsistent age effects. Observational association is not causation;
no extra intervention is authorized by this registration. Final report waits for
all 75 validated endpoints. Launch gates require initial analysis, focused tests,
GPU equivalence and benchmark, both-host input checks and measured memory margin.

## Verified historical availability
Both-host inventories verify 110 records per host, including all 15 canonical
historical dense recovery checkpoints. Fifteen additional table-backed reference
directories contain endpoint losses without a model file; those are not missing
canonical checkpoints. Partial validation attempts are preserved and excluded.
Existing states have frozen and native probes, with source-native assignment
agreement separated from same-input source-router agreement. Native validation is
cached by attempted step; step-25 validation is added uniformly to all conditions.

Native assignments and capacity acceptance are taken from the actual end-to-end
RouteTrace, reordered only for token identity. Source assignment agreement in native
mode compares identical token positions with source-native routing; the separately
reported source_same_input_assignment_agreement compares router decisions on
identical current vectors. Functional expert-output comparisons always use
identical vectors. Probe cache was generated on CPU in FP32; both-host diagnostics
use the same serialized cache.

Diagnostic frozen/native probes disable TF32 for full FP32 function comparisons;
exact caller matmul/cuDNN TF32 flags are restored. Native evaluation loss retains
the existing configure_device TF32 setting, including added step-25 evaluation.
Training precision and flags are unchanged.

Existing corpus provenance labels (TinyStories/WikiText103) are recovered only
after exact UTF-8 byte reconstruction using the repository sampling/interleaving
procedure. Separators are null. They annotate fixed probes without changing
selection. domain_routing.csv is descriptive; domains/tokens are not replication
units. Source file, metadata, training-data and fixed-probe hashes are recorded.
