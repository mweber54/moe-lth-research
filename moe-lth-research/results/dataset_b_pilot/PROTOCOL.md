# Dataset B: nominal exposure matching, staged 32E controlled follow-up

This design reuses Dataset A's five source seeds (7, 17, 29, 43, 61). It is a controlled follow-up, not independent confirmation. Only the 32E pilot is authorized for this queue. General budgets are configured for 8/12/16/24/32E as 2500/3750/5000/7500/10000 attempts; no all-architecture sweep is queued.

## Verified Dataset A semantics

Dataset A now has 25 sources, 275 distinct dense controls and 1650 sparse conditions. The original audit snapshot lacked the adder shard, but the current canonical table and checkpoint identities include it. Pilot reuse checks only the relevant 30 sparse and 15 dense conditions, their raw endpoints, metadata and checkpoint identities. The earlier unavailable-reference diagnosis is superseded by current artifacts.

Source training: 2500 attempts, batch 128, length 128, vocabulary 256, four layers, four heads, width 256, expert hidden size 1024, top-1 native learned routing, capacity factor 1.25, dropout zero, router auxiliary coefficient 0.1. AdamW LR 0.0003, weight decay 0.01, gradient clip 1.0, fp16 GradScaler. There is **no scheduler or warmup**. Attempted steps and successful updates are different quantities.

Learned masks jointly rank both eligible expert weight matrices **inside each expert** using that source's final weights; biases/shared/router weights are excluded. Keep count is `max(1, round(eligible_parameters * (1-sparsity)))`. The surviving expert values and shared parameters rewind to source R0. Only initial router parameters change across ages for a fixed source/seed/mask. All surviving experts, shared parameters, and routers train in recovery. Mask hooks zero pruned gradients and masks are reapplied after each update. New pilot checks also inspect pruned Adam moments at probe checkpoints.

The embedding and output head are tied and use the existing unscaled random initialization. Initial validation CE near 170 is not automatically an error: these logits strongly favor the current token while targets are shifted next-token IDs. `audits/initial_loss_verification.json` independently reconstructs a real 32E/R100/95% condition on CPU, compares manual negative log-softmax against token-mean cross entropy, and compares full validation to the stored initial endpoint. No valid initialization or loss scale is changed. The predeclared CPU-versus-historical-TF32 diagnostic tolerance is 0.02 nats/token (about 0.012% at loss 170); the independent CE reduction check uses 1e-4. Neither is a dense-equivalence or ticket threshold.

Validation is summed next-token cross entropy divided by the actual number of target tokens, in nats/token, excluding the auxiliary term. Recovery training adds 0.1 times auxiliary loss. Validation runs every 50 attempted steps; source validation every 250. Early AUC is an unnormalized trapezoidal validation-loss integral. Dataset A's 50%-horizon window is 0–1250. Dataset B explicitly fixes the comparable primary window to **0–1250** for both budgets, using the existing runner's integration code with `EARLY_AUC_WINDOW_FRACTION=1250/recovery_budget` in each isolated worker. A 10000-step run's first-half AUC is not silently compared against a 2500-step run's first-half AUC.

## Sampling and longer budgets

A private, seed-initialized DataLoader shuffles byte-token blocks once; `cycle(loader)` repeats that same first traversal. Longer training processes more token occurrences, not a larger unique corpus. Preprocessing, corpus files, batch size, sequence length and validation data stay fixed. Recovery materializes its specified stream before `seed_everything(seed)` resets Python/NumPy/Torch/CUDA RNG inside the native recovery runner. Every long recovery verifies its first 2500 batch-sequence hash against Dataset A and its validation sequence against the canonical hash.

Constant LR 0.0003 extends unchanged beyond step 2500. No schedule stretching is introduced. Historical checkpoints omit the full optimizer/scaler/RNG/data-stream state and are **not** valid continuation checkpoints. Long source and recovery runs start fresh from their specified initialization. Short and long jobs are not consolidated; prefix equivalence is not assumed from model checkpoints alone. Diagnostics are read-only, preserve Python/NumPy/Torch/CUDA RNG and restore model mode. Production replay is not claimed bitwise deterministic across hardware.

## Factorial pilot and sharing

| Arm | Source attempts | Recovery attempts |
|---|---:|---:|
| A | 2500 | 2500 |
| B | 10000 | 2500 |
| C | 2500 | 10000 |
| D | 10000 | 10000 |

All five seeds; R0/R50/R100; 75%/95%. There are 120 learned sparse cells and 60 dense cells. A/C share exact short-source artifacts and masks. B/D share exact new long-source artifacts and masks. Each source/seed/sparsity has one learned mask reused across ages. Dense controls are shared across sparsities and mask types only within the same source identity, seed, router checkpoint and recovery budget. R0 is not automatically deduplicated across source budgets. Short-source R50/R100 mean absolute steps 1250/2500; long-source R50/R100 mean 5000/10000. Long sources save 0/1250/2500/5000/10000, and record whether their R0 and step-2500 states actually equal short-source states.

Since masks depend on final-source weights, B versus A changes mask selection as well as source development and router absolute age. Shared/surviving expert values rewind to each source's own R0; equality across source budgets is measured rather than assumed. This is not a router-only causal intervention.

Arms A and D each add 20 random-mask recoveries: five seeds × R0/R100 × 75%/95%. Random masks preserve the exact eligible tensors and the learned mask's **per-tensor** retained counts. This conditions on the learned allocation; it tests weight locations, not allocation. A separate deterministic CPU Generator seeded by a SHA-256-derived source/seed/sparsity identifier creates one draw per source condition; it does not touch global initialization or recovery RNG. Each random mask is shared across its two router ages. One draw per source seed cannot identify within-source mask randomness separately.

Current validated reuse is 30 learned sparse A cells plus 15 dense A controls. New work: 20 A random recoveries; 45 C recoveries; five long sources; 45 B recoveries; 65 D recoveries. Total **180 new jobs**, 1,312,500 nominal attempts. No new A learned/dense runs are needed.

## Diagnostics, storage, and restart

Native compact per-layer/expert proposed/accepted counts are logged every attempt; dropped counts are their difference. Exact prefix reductions produce exposure at every saved source checkpoint and recovery steps 0/2500/final. Active attempted steps, active applied steps, load CV, applied updates and AMP skips remain separate. Missing checkpoint logs cause validation failure; whole-run exposure is never substituted. No token-level training histories are enabled. Unneeded source validation-route dumps and sampled text exports are suppressed without removing evaluation forwards.

Recovery probes use the first eight canonical validation batches at initialization, 2500 and the long endpoint. Entropy and probability margin are token/layer means, assignment agreement is against recovery initialization on the same probe, and router displacement is Euclidean parameter distance divided by initial Euclidean norm (also absolute distance). Existing per-layer load diagnostics and probe hashes are retained.

Final recovery models and source checkpoints are retained. No Dataset A artifact is overwritten. New attempts use numbered directories; incomplete attempts are preserved. `resume` means restart the durable queue: completed jobs are reused, live orphan workers are detected, and an interrupted trajectory starts a new fresh attempt. There is no unvalidated in-trajectory continuation. Scientific assertion failures stop for diagnosis. Dataset B mask files are never overwritten on a retry.

A host-local pilot flock prevents duplicate supervisors. The queue checks GPU 0 for **any external compute process** and waits if occupied; it never kills another job. Each new worker runs sequentially. A 30 GiB free-space reserve is checked before every job. Home-backed Dataset B outputs avoid the nearly full `/data` volume. Storage growth is estimated at roughly 75 GiB plus 30 GiB reserve; historical mtime-derived runtime is roughly 66–74 GPU-hours, median 68.3, excluding waiting and setup uncertainty. These are planning estimates, not completion promises.

## Prespecified analysis

Five source seeds are replication units. Save every paired value, mean, sample SD, SEM and sign count. For complete five-seed contrasts, reference Student-t intervals use df=4 and are explicitly small-sample, assumption-dependent summaries, not multiplicity-adjusted discoveries. Do not pool ages/sparsities/tokens/experts as independent samples. R50 is descriptive.

Primary contrasts, separately for sparse loss, dense loss and gap:

1. C − A (added recovery with short source).
2. B − A (added source at short recovery).
3. D − B (added recovery with long source).
4. (D − B) − (C − A).
5. R100 − R0 at each sparsity.
6. The preceding age effect at 95% minus that at 75%.
7. Random − learned sparse loss in A/D, and its R100-minus-R0 interaction.

Positive random-minus-learned advantage favors learned locations. A learned-mask main effect is not routing-specific compatibility; the age interaction is the relevant contrast. All primary contrasts were specified by the user before launch. Exploratory Dataset A findings do not change this grid or select seeds. Multiple comparisons and reused seeds limit confirmation claims.

Dense controls are matched by exact protocol/state identity, with 1e-10 tolerance only for serialized endpoint arithmetic. No scientific dense-equivalence margin is declared here, so no “lottery ticket” or “matches dense” conclusion follows from a small gap. Longer recovery helping sparse models does not alone establish tickets; residual gaps at 10000 do not establish irreducible capacity. Further attribution, frozen/shuffled routers or extra architectures are not launch dependencies or queued work.
