# Fixed-budget deletion-damage reliability v1

Question: can deletions at a fixed checkpoint/expert/dose be ordered reproducibly
across additional training contexts? Measurement only, no specialization or pruning
algorithm. Seed17 remains a NONMATCHING 32E IMP80 engineering case. Retain verified
sparse/matched-unpruned steps25/250/1000/2500, pre-reset2500, layer1 experts0/8/16/24,
original fc1 groups, four orderings, 52 distinct weight unions and four separate
expert-output-zero controls. Coordinate-matched, not density-matched. No global cap.
Aliases with identical endpoint model hashes count once. No model/seed expansion.

The user explicitly authorized deterministic re-batching of INTACT canonical
128-token sequences. This preserves original source-window sequence boundaries,
complete128x128 native-shape batches, native collation/routing/capacity, but does NOT
claim to preserve historical shuffled minibatch membership. Use the canonical byte
TokenBlockDataset and data configuration. Seed171032 uniformly permutes source
sequence IDs, then greedily accepts the first4096 whose129-token input+target span
is disjoint from prior A/B spans and all previously accepted spans, also excluding
identical full-window contents at different source positions. No routing/loss is
consulted. Strict overlaps include the one-byte shifted-target boundary. If4096
unique valid windows are unavailable, stop and report the limitation; do not relax.
Pack in accepted order into exactly32 complete batches. Hash all sequences/batches.
Identical batches at all checkpoints/trajectories. Blocks1..8 contain four batches
each, in frozen order. Discovery blocks1..4; confirmation blocks5..8. These training
contexts may have been seen during training; confirmation is not independent test
or source-seed replication. No further contexts are evaluated. Profiling reuses
one of these32 batches and its measurements are cached for the main pass.

FP32, TF32-off, eval/inference, native top1 gates and capacity, task-only CE;
auxiliary loss separate. Compute paired token deleted-minus-baseline in FP64 before
FP64 aggregation; never clip negative damage. Conditional positions are baseline
accepted positions, fixed for all interventions. Effects may propagate outside
through attention/later layers. Store compact sequence/subset paired sums/counts,
batch sums and baseline routes/losses, not hidden tensors/logits. No old CE is reused
on these new contexts. Reuse only verified code, masks, coordinates and definitions.
Output RMS is on native-dispatch baseline accepted positions/channels. Save squared
difference/base energies and gate-weighted energies separately. Pool energies before
sqrt(numerator/denominator); do not replace by average batch ratios. Baseline RMS
<=1e-12 gives undefined relative distortion. No accepted output -> undefined.

Coverage is reported for EVERY expert/checkpoint/trajectory/batch/block. Threshold:
>=32 accepted positions PER FOUR-BATCH BLOCK for conditional ranking eligibility;
all eight blocks must qualify for the discovery/confirmation conditional ranking.
Still save raw low-coverage loss estimates and intervals. Report contributing batch
count, maximum batch fraction, HHI and effective contributing-batch count. Coverage
is an operational flag, not precision. Whole-probe results remain valid when traffic
is low, but zeros do not prove dispensability. Never replace experts or extend budget.

Ranks NEVER pool checkpoints, experts or doses. Each trajectory/time/expert/dose has
3 unique single-group unions,4 two-group,4 four-group,1 eight-group; zero and output
control are separate singleton cases. Spearman undefined if <3 unions, constant
range <=1e-12, or ineligible conditional coverage. Preserve all underlying values,
tie counts and distinct unions. Overlapping unions remain dependent observations.
Discovery selects maximal/minimal mean damage only. Tie break by lexicographic ID:
lowest first, highest last, so constant panels expose arbitrary tied choices. With
<2 unions selection is undefined. Evaluate the SAME pair on confirmation; report
confirmation/discovery contrast, each block contrast/sign/CI, and all fixed lexicographic
pairwise contrasts in both partitions and all eight blocks. Do not reselect with
confirmation. Report LOO block estimates for each phase and selected contrast.

Bootstrap complete batches ONLY, retaining paired all-intervention sums. Fixed
seed1732000, 2000 multinomial-resampled-batch count vectors per partition/block;
same count vectors for all interventions/subsets within that set. Discovery and
confirmation use independently seeded vectors. Conditional estimates resample sums
AND counts and divide AFTER summing; zero-denominator replicates undefined, report
frequency. Percentile2.5/97.5 intervals are descriptive sampled-context intervals in
one seed, not source-seed confidence or generalization. Bootstrap rank intervals
pair independent discovery/confirmation draws, preserving within-phase intervention
pairing; undefined replicates reported. Block CIs use four complete batches, not
individual tokens. LOO removes one whole four-batch block within each phase.

Continuous effects/intervals lead. Reference tolerances0.0001 and0.001: interval
entirely above -> exceeds exploratory convention; entirely below -> compatible
with below-convention damage on these contexts; crossing -> unresolved. No claim
of harmlessness, permanent-pruning safety or dispensability from a small effect.
Numerical repeatability is separate from context agreement. Negative ranks alone
are not context dependence. Inspect paired contrasts, intervals and coverage; avoid
semantic/domain labels or fitted specialization predictors. Functional distortion
rank agreement across phases and distortion-vs-CE correlations are also restricted
to the SAME expert/time/dose; singleton/constant panels undefined.

Condition verdicts retain expert identity: selected confirmation interval >0 and
positive discovery contrast -> reproduced directional contrast in these contexts;
interval <0 -> reversed directional contrast, candidate context dependence only;
interval crosses0 -> unresolved at this budget. Low coverage and <2/<3 union cases
explicit. This is descriptive, not a global PASS, and selected effects can be noisy.
Report A-E per fixed condition; final conclusion must preserve heterogeneous cases.

Profile sparse step1000 batch0 full panel before the full pass, after this freeze.
One GPU worker, locks,15-second heartbeat, atomic per-batch caches and completion.
Cache identity binds checkpoint, exact batch, coordinate/intervention seal, numerical
mode and implementation. Real empty deletion, repeat checks on batch0 and16 of every
checkpoint, native baseline check, scoped finally restoration and exact weight checks
per intervention; full hash/mask check per batch. All evaluation checks RNG/modes/
gradients. No backward, optimizer, training, recovery, mask search, permanent pruning,
new seeds, extra contexts or automatic follow-up. Engineering fixes must preserve
scientific definitions and retain old code/provenance.
