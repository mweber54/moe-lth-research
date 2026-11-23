# Known IMP80 mask transition protocol

Prospective controlled experiment motivated by previous exploratory findings; not project-level preregistration.

All five source seeds7,17,29,43,61 are retained, regardless of matching status.
Seed17 is the engineering pilot. Four conditions per seed: dense, fixed before
update1, abrupt immediately after attempted1000, gradual immediately after
attempted100,200,...1000. All sparse paths have the identical original final IMP80
mask for updates1001..5000. No new masks are discovered. The existing IMP mask is
privileged oracle information, not an online pruning algorithm. Its historical
discovery cost is excluded from diagnostic training but explicitly retained as a cost.

Gradual event assignments use SHA256('imp-mask-transition-v1:seed:parameter'),
first8 bytes big endian modulo(2^63-1), CPU torch.randperm on exactly final-discarded
coordinates. Event k deletes the prefix floor(k*N_discard/10); survivors never deleted.
One schedule per seed; no future statistics or outcome-dependent ordering.

Native32E model, complete initialization, batch128×128, deterministic batches0..4999,
AdamW lr3e-4 weight_decay.01, native default betas/eps, clip1, FP16 GradScaler,
auxiliary objective weight.1, full-softmax native selected gates and capacity1.25.
All parameters trainable. Masked gradients are zeroed before unscale/clipping;
weights enforced after every attempted step. New deletion zeros coordinate-shaped
Adam moments only; retained state and scalar counters persist. Events count AMP
skips and never reset RNG/scaler/optimizer. After2500 attempts, fresh optimizer,
scaler and RNG seeded with the source seed exactly as the established second call;
weights unchanged, second segment uses batches2500..4999.

Common task-only evaluation: FP32, TF32 off, eval/inference, ALL native validation
batches including final partial, actual-target weighting. Grid0,100..1000 by100,
1250,1500,1750,2000,2250,2500(pre-reset),2525,2600,2750,3000,3500,4000,4500,5000.
At1000 retain pre/post event; main trajectory is post. Reused endpoints must be
re-read under common evaluation. Unobserved reference grid points stay missing.
AUC uses common observed post1000 grid with≥3 points including1000 and5000;
otherwise unavailable. Trapezoidal AUC describes observed-grid coverage only.

Two complete training diagnostic batches: materialized batches0 and1, fixed before
new losses. Before/after every pruning event record task/aux, native proposals,
acceptance/drop rates, selected gates, per-layer expert loads, agreement/set overlap,
parameter norms. No diagnostic backward. Selected training updates on the grid and
first update in each segment record actual E/R/S update norms, native unscaled
gradient norms, skips/scaler and active counts. Tied parameters counted once.
Diagnostics preserve RNG/modes/gradients/model/buffers; do not touch optimizer,
scaler, sampler, masks or event counters. Cache only bounded compact route arrays.

Primary seed-level contrasts at5000: abrupt−fixed and gradual−fixed. Secondary:
gradual−abrupt, each−dense, pre-reset2500 and subsequent trajectory differences.
Report every seed before mean/median/range/sampleSD; n=5 source seeds, no token/
expert/checkpoint/event pseudoreplication, no winning-status filtering, no adaptive
duration or hyperparameter choice. No causal routing or isolated smoothness claim.
Equal attempted-token budgets have unequal cumulative active-coordinate budgets:
abrupt retains dense capacity longer than gradual; both exceed fixed early capacity.
Dense tensor zeros imply no measured FLOP/wall-clock savings.

Engineering before production: seed17 short native equivalence atol1e-6 rtol1e-5,
CE atol1e-6 (existing infrastructure), exact schedule/moment/event/resume/reset/RNG
checks. No tolerance relaxation after outcomes. Document engineering corrections
with versioned seals; no changes to timing/cohort/endpoints. Compatible dense/fixed
references can be reused with absent trajectories/diagnostics marked unavailable.
One detached worker per host, exclusive claims/host locks,≥24GiB and1.25×benchmark
peak admission, atomic status/heartbeats, full resumable states every100 attempts,
event state included, no duplicate resume events. Never overwrite an attempt or kill
unrelated jobs. Technical failures stop dispatch and preserve artifacts. Seed17 new
jobs run first; remaining seeds dispatch only after pilot numerical completion,
regardless of scientific effect direction. Only the declared20 endpoints authorized.
