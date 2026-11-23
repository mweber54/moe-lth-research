# Expert role and persistent deletion sensitivity: Stage 1 engineering pilot

Seed17, 32E IMP80 is retained as a NONMATCHING recovery under the previous
5000-step criterion; this is an engineering case, not a winner. Four preferred
existing checkpoints 25,250,1000,2500 are available with no substitutions.
2500 is the pre-reset endpoint; the fresh optimizer/scaler segment is excluded.
Layer index1, experts0/8/16/24, fc1.weight only, eight disjoint uniformly sampled
groups per expert, floor(1% surviving mask coordinates). Fixed membership in both
trajectories: equal-coordinate deletion, not equal-relative-sparsity comparison.
Primary tolerances 0.001 nats/token, sensitivity 0.0001 and0.005, all exploratory.
Continuous signed task CE damage is primary; auxiliary loss is recorded separately.

H1 (more persistent low-damage groups) and H2 (roles add predictive information)
are separate and neither assumed true. Single A selection uses current damage;
persistent A selection uses previous/current only. Future outcomes always next B.
All groups remain in the prospective panel, including retrospectively rejected
ones. With four times only two two-history forecasts exist. No predictive model,
high-dimensional fit, or significance claim is supported by this one-seed pilot.
Prespecified descriptive comparisons: step; load/gates; magnitude/movement;
past damage; past damage plus separate persistence/differentiation. Eight groups
share an expert role measurement and are not independent role replications.

A/B contain complete deterministic native training batches with original sequence
order/capacity, disjoint raw byte windows including targets. Training may have seen
both. B is locked against candidate/dose/threshold/cluster decisions. Profile one
baseline and one first uniform-group deletion on A before fixing budget: if their
combined cost <=2 seconds choose two batches per probe, otherwise one. Upper bound
two batches each, fixed independently of effect direction. K16 clustering on the
earliest unpruned reference A only, feature standardization floor1e-6, seed170625,
n_init1, Lloyd max_iter100. Freeze centers and reference A/B labels; small clusters
(<32) retained and flagged. Expert populations <32 flagged unavailable for divergence.
Differentiation of routing over fixed context groups is Jensen-Shannon divergence
in nats against overall probe distribution; these are not semantic domains.

Proposed and accepted populations remain separate; capacity drops and gates are
baseline measurements. Adjacent contextual-set Jaccard empty/empty is undefined;
retention intersection/previous, inflow new/current. Shuffle references permute
whole per-token expert/acceptance/gate tuples within each complete batch, preserving
loads/capacity totals,64 deterministic replicates, independently by time.

Ablations start from baseline each time and restore in finally; baseline accepted
conditional positions are fixed before deletion. Attention/later layers spread
effects; conditional damage is not isolated expert loss. Token and sequence-paired
losses are stored but tokens are not independent source replications. Zero dose,
repeat and restoration controls required. Primary group dose never retuned.
Union diagnostics use persistent past-only A selection, canonical expert/group
order, at most floor(1% ALL eligible expert-weight coordinates across all MoE layers,
fc1+fc2); never split a group to fill the cap. Evaluate fixed union at decision and
next checkpoints on A/B; compare actual union damage with sum of singles. No
selection -> explicit undefined, no fake zero. No permanent deletion or recovery.

Checkpoint file and model hashes deduplicate historical/rerun/reset aliases. Raw
old metrics reused only if all provenance matches; none currently qualify. Cache
identity binds checkpoint/probe/group/mode/implementation. New results only.
Workers hold per-host and per-study advisory locks and emit15-second heartbeat.
No training, optimizer, mask search, five-seed expansion, or Stage2 is authorized.
Associations are observational. Low damage at this dose/probe does not prove
future-training dispensability or permanent-pruning recoverability. Final IMP mask
is a diagnostic cohort definition, unavailable to an eventual dense-training method.
