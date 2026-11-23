# Research audit — 2026-10-20 (America/Indianapolis)

## Scope and confidence

Repository-wide triage covers 45 result directories (including this new analysis), 173,219 file entries and 95 Markdown reports; see `repository_inventory.csv` and `report_triage.csv`. File counts include route archives, backup trees and partial attempts and are **not counts of independent models**. Detailed scientific revalidation targets the three strongest candidates, with independent geometry measurements of all 240 registered 32E rerun checkpoints. This is not a claim that every historical binary or every report assertion was independently reproduced. Legacy root summaries referring to unavailable older experiment directories are background, not primary evidence.

Everything newly written is under `results/research_investigation_v1/`. No training, model forward/backward on experimental checkpoints, pruning, recovery or production-code change was performed. Existing model tensors are read on CPU; the embedding cache is original float32 and statistics are float64. Full checkpoint SHA256 values are reused from the preceding verified audit; this investigation freshly verifies E content hashes rather than streaming another 145 GB. The per-checkpoint provenance distinguishes these verification levels.

## Exact architecture and optimization

| Fact | Code / artifact supporting it |
|---|---|
| Four pre-norm decoder blocks, causal multihead attention, four heads, width 256, expert hidden width 1024, dropout zero | `src/moe_lth/models/transformer.py:11`, `:31`, `:59`; `results/high_sparsity_coupling_v1/cohort.json` model config |
| Token embedding E is 256×256 and tied to the bias-free LM head; position embedding is separate, 128×256 | `src/moe_lth/models/transformer.py:70`, `:71`, `:76`, `:77`, `:103` |
| Token IDs are UTF-8 bytes, not semantic token classes | `src/moe_lth/data.py:29`; model vocabulary config; IDs 0–255 retained in new row tables |
| Router is a bias-free linear map; full softmax is computed **before** top-k selection and selected probabilities are not renormalized | `src/moe_lth/models/router.py:37`, `:57`; primary 32E cohort uses top-1 |
| GELU experts include biases and two dense linear weight matrices | `src/moe_lth/models/experts.py`; no SwiGLU gate branch in this architecture |
| Capacity is ceil(1.25×tokens/experts); highest selected-confidence assignments survive; accepted branch output is probability times expert output | `src/moe_lth/models/moe_layer.py:63`, `:80`, `:89` |
| Auxiliary objective is N×sum(detached proposed usage × mean native probability), averaged over layers | `src/moe_lth/models/moe_layer.py:96`; `transformer.py:106` |
| Training loss is next-byte CE +0.1×auxiliary loss; reported validation CE excludes auxiliary loss | `src/moe_lth/experiments/run_router_age_recovery.py:210`, `:652`, `:655` |
| AdamW, LR .0003, weight decay .01, clip norm 1, fresh optimizer, CUDA FP16 GradScaler; no scheduler/warmup | recovery implementation `:449`, `:461`, `:473`, `:656`, `:682`, `:686`, `:878`; cohort and per-run metadata |
| E remains trainable and is excluded from expert pruning | optimizer takes all requires-grad parameters at recovery `:460`; only optional router freezing at `:453`; expert mask eligibility below |

There are two distinct notions of sparsity: **conditional activation** (top-1 expert selection plus capacity dropping) and **coordinate pruning inside expert matrices**. Coordinate masks suppress expert weight gradients and keep pruned parameters zero; they do not structurally prune E, attention, biases or routers. Shared gradients, routing and activations can nevertheless change indirectly.

`src/moe_lth/pruning/masks.py:11` selects only expert `.weight` parameters; `:20` reapplies masks. `src/moe_lth/pruning/rewind.py:36` registers gradient-times-mask hooks. The native recovery invokes hooks at `:449` and reapplies masks at `:698` after optimizer steps. `src/moe_lth/pruning/imp.py:6` and `:20` implement nested expert-local magnitude pruning with stable ties and fixed integer budgets.

## Distinct experimental cohorts — do not pool them

**32E high-sparsity IMP:** ten .80/.85 sparse models and five dense controls, seeds 7,17,29,43,61. Mask discovery uses nine 2500-attempt recoveries from original initialization with nested masks; final .80/.85 recoveries rewind experts, shared parameters and R0 routers separately. This selection cost is not included in a final-recovery-only budget and cannot support a training-efficiency claim. See `scripts/imp_80_85/run.py:89`, `src/moe_lth/pruning/imp.py:6`, `results/high_sparsity_coupling_v1/protocol.md`.

Within seed, original E equality is freshly verified across all three conditions in `geometry_validation.json`. Data order and validation identities are explicitly checked by `scripts/high_sparsity_coupling/runner.py:55`. Batch size 128 × sequence length 128 gives 81,920,000 attempted target-token presentations over 5000 attempts. Optimizer/scaler reset after 2500 is common to sparse and dense and changes no model weights (`runner.py:57–61`). Applied updates are 4997–5000, not always 5000 (`report/update_budgets.csv`, `report/budget_log_audit.csv`). Native final local step 2500 in the second segment means global recovery step 5000.

**Gate replay:** 8E and 12E, .85 expert-local one-shot masks, seeds 7,17,29; ages 0/20/100 for 8E and 20/60/100 for 12E; 2500 recovery attempts. This is not the 32E IMP cohort. Expert/shared states rewind while router initialization changes. Initial shared/expert/mask identities are recorded in `results/gate_amplitude_replay_intervention/gate_replay_manifest.json` and each recipient `metadata.json`. The 36 Level-2 recipient endpoint summaries reproduce the corresponding common-gate outcomes to floating-point arithmetic precision in `gate_summary_recheck.csv`. The comparison is made within architecture/seed under the same donor readout, not by comparing raw losses across cohorts.

**Other controlled families:** `router_expert_geometry_v1/analysis_extended/v1` measures five reused seeds, three router ages and learned/random masks at .50/.65; `cross_arch_optimization_geometry/diagnostic_rerun` uses three reused seeds and 8E/12E at .85; lower-sparsity Dataset A contains 1100 new sparse rows and 275 dense controls. `dataset_b_pilot/PROTOCOL.md` explicitly identifies Dataset B as a nominal-exposure follow-up on the same source seeds/corpus, not independent confirmation. `HOME_BACKUP`, historical copies, same-seed reruns and reports do not add replication.

## Availability and provenance

32E rerun grid: 0,25,100,250,500,1000,1500,2000,2500,2525,2600,3000,3500,4000,4500,5000. All 240 registered snapshots exist. Historical full trained states exist at 2500,3000,4000,5000; original saved E provides initialization without reconstruction of missing trained states. There are 75 historical logical records and 165 unavailable historical slots. All 75 shared historical/rerun states are tensor-identical. The preceding embedding audit additionally retained 75 reset/native/interruption aliases, giving 390 provenance records but only 230 distinct E tensors. Independent unit is source seed, n=5, not tensor, token, coordinate, checkpoint or sparsity.

Authoritative paths: `results/high_sparsity_coupling_v1/cohort.json`, `freeze.json`, `checkpoint_coverage.csv`, `report/measurement_provenance.csv`, `report/provenance_diagnosis.md`, `embedding_weight_sparsity/checkpoint_provenance.csv`. Fresh E hashes and new cache files are in this directory. Original full model/component hashes and prior file-hash checks are retained without claiming a second fresh file-hash audit.

Gate replay retains 324 recipient snapshots; only nine common-readout baseline checkpoints support matched timing. Fifty-one observations in one arm cannot manufacture corresponding observations in another. Pattern decomposition's failed FP16 transformation builds are retained and superseded by corrected builds. Later status/results take precedence over stale preflight text saying execution has not started.

## Gradient evidence: what exists and what does not

`high_sparsity_coupling_v1/analysis/.../gradient_identity.csv` contains single-token local router/expert derivative diagnostics at 0/2500/5000, not historical full embedding gradients. `runner.py:16–28` saves scalar gradient/update alignment for **expert and router** parameters, excluding shared E. Cross-architecture diagnostic files include expert gradient/update cosines and scalar norms; none supplies a complete saved tied-E gradient trajectory. Adam moments in endpoint optimizer checkpoints are exponentially weighted history, not the current or individual historical gradient.

Deterministic materialized batches, weights, source config, masks and implementation fingerprints make **reconstructed diagnostic** E gradients feasible at available checkpoints. A fixed validation minibatch derivative is not a saved training derivative. Exact historical reproduction additionally needs the correct next training minibatch, AMP/autocast/loss-scale, clipping, accepted capacity assignments, RNG where relevant and numerical execution environment. Unobserved intermediate gradients cannot be recovered uniquely from checkpoint displacement or optimizer moments. No fresh expensive full-model backward was needed to rank candidates, and none was run here. Expert-related VJPs and tied input/output gradient decomposition remain proposed measurements, not discovered results.

## Validity problems and falsification checks

1. Previous geometry summaries mixed valid experts across cells: a 30/30 decreasing rank-16 overlap claim becomes **28/30** with common membership. The high-sparsity report likewise corrected different sparse/dense populations; the largest published correction was +.11552. Use endpoint four-cell intersections and separate all-time coverage.
2. Current-accepted pair populations are selected by the evolving router. Across-all-time valid membership retains only 11–14/128 units and can reverse accepted-pair conclusions. Fixed-source/all-expert effects are much better identified. Fixed token IDs do not freeze native hidden states.
3. Gate replay simultaneously substitutes coefficients and removes task derivatives through the gate. `src/moe_lth/experiments/gate_amplitude_replay.py:21–29` creates detached replay coefficients. `gradient_semantics_check.json` verifies this on a small fixture; its values are **not experimental gradient measurements**. Auxiliary gradients into routers/upstream states remain active. Therefore “gate values uniquely mediate the natural advantage” is unproved.
4. All three training arms must use a common readout for training-effect contrasts. Native versus donor readout changes signs in many cases. Large native-readout losses after detached training are a warning about distribution shift and portability, not evidence of better sparse models.
5. Shared-input single-token linear-gradient identities do not imply AdamW update alignment. Masks change x to Mx; moments, clipping, signs, auxiliary gradients and weight decay further change updates. The roughly square-root-density masked-angle effect is elementary linear algebra, not a new mechanism.
6. Centered `effective_rank` and uncentered `embedding_effective_rank` coexist in `scripts/high_sparsity_coupling/measure.py:24` and `:115`. New spectral tables state the squared-energy entropy definition and use uncentered E. Rank around 155 is not numerical rank 155 and must be compared with random initialization/null spectra.
7. Full-softmax-before-top1 sends task gradients to unselected logits through normalization, unlike masked-before-softmax formulations. Do not transplant selected-only gradient arguments across implementations.
8. Historical/rerun endpoint CE differences reproduce under TF32 off/on with identical weights; see prior `report/ce_execution_settings.csv`. These are numerical evaluation differences, not extra trained models or training effects.
9. Source-final dense, same-age recovery dense, same-step dense and fixed final dense are different references. Raw unnormalized distances, normalized Gram shape and top-k overlaps are different metrics and may disagree legitimately.
10. Validation influenced selection. No demonstrated unused independent test split is established. Older “best checkpoint” and age-grid searches are exploratory. A huge initial CE (~170) is consistent with the unscaled tied random embedding and shifted next-byte targets; early integrals and helper recovery fractions are not automatically meaningful efficiency measures.
11. Architecture comparisons at fixed token budget change per-expert exposure, parameter count and gate magnitude. The repository's 8→12→16/24/32 age-optimum story does not establish a scaling law. Lower-sparsity results find no all-five strict dense-match setting among 220 and no adjusted age-pair discovery; do not turn seed-conditioned minima into universal laws.

All candidate claims below are bounded empirical results. The new analysis does not establish causal E mediation, parameter dispensability, an independent-test gain, or scale-independent MoE behavior.

## Additional timing verification

The common-readout gate trajectories were independently reconstructed from saved sparse/dense checkpoint losses, including every level and all available points. The comparison figure uses only the intersection of observed native/assignment/gate checkpoints, not the denser gate-only curve. This verifies saved-loss arithmetic; it is not a fresh model evaluation and cannot turn diagnostic onset into mediation evidence. The 32E full-validation/geometry timeline additionally preserves the existing measured CE precision setting.
