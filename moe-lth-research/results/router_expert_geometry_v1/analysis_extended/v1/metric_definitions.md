# Measurement definitions and scope — extension v1

**Post hoc disclosure:** this extension was chosen after inspecting the original rank-16 findings. Rank 16 remains the original primary subspace metric. Ranks 8 and 32, corrected membership, full-spectrum Gram comparisons, GELU score–activation associations and loss relations are exploratory. This is the existing 32-expert, 50%/65% mask geometry experiment (75 runs), not a new 80% ticket training experiment.

## Paper provenance

The named PDF `routers_learn_expert_geometry(1).pdf` was not present in the accessible attachments. The matching public paper was read instead: Ahrac, Hochwald and Geva, *Routers Learn the Geometry of Their Experts: Geometric Coupling in Sparse Mixture-of-Experts*, arXiv:2605.12476v1, Section 4/Figure 2, https://arxiv.org/html/2605.12476v1#S4. Exact equivalence to the user's PDF bytes/version cannot be established.

The paper compares raw router scores with mean SiLU gate-neuron activations on routed token/expert pairs, normalizing each quantity separately per layer/expert before pooling. Its text does not specify the exact normalization formula or identify the correlation estimator beyond rho. Here population z-score normalization, Pearson correlation and Spearman sensitivity are explicit implementation choices. The paper's gated architecture, top-6 routing, loss-free balancing and evaluation corpus differ from this auxiliary-loss GELU model; this is an adapted measurement, not an exact replication.

## Deployed model

The current model module hashes match the archived launch seal (`deployed_model_code_audit.csv`). Each of four pre-norm blocks sends the same 256-dimensional `moe_norm` output to router and experts. `ExpertFFN.fc1.weight` has shape [1024,256], with a 1024-vector bias. Expert forward is `fc2(dropout(GELU(fc1(x))))`; dropout is zero in archived configs and diagnostics use eval. The GELU is the deployed `nn.GELU()` (exact/erf default), not SiLU and not a SwiGLU gate. `fc2` and its bias are outside the projection geometry statistic.

The router projection has shape [32,256] and no bias. Native logits are `projection(x) / temperature`, followed by full softmax and top-1 selection. We measure the raw pre-temperature linear score `projection(x)[e]`, as the paper defines r_e^T x. A positive scalar temperature changes neither ranking nor standardized correlation for a fixed group. Capacity is ceil(1.25*N*top_k/32), minimum one; selected softmax probabilities rank overflow assignments. Accepted expert outputs are multiplied by their selected softmax probability. This measurement uses activations before that multiplication.

## References and state identities

- **Source-final dense:** same-seed source step 2500, unmasked; its expert weights and native hidden inputs define source references.
- **Matched new dense recovery:** the unpruned v1 recovery at the same seed, router age and attempted recovery step. This is not source-final.
- **Recovery initialization:** surviving expert/shared values from source step 0, masked where applicable, with the selected source-age router (R0/R50/R100).
- **Current sparse recovery:** actual stored diagnostic model state at its attempted step.
- **Masked-trained:** source-final weights after the exact saved mask; pruning only, before initialization rewind.
- **Historical recovered endpoint:** actual original native endpoint; never interpreted as a seven-step v1 trajectory. Historical dense comparator is located through its historical origin metadata, not replaced with a v1 dense trajectory.

Source checkpoint, mask, probe and native endpoint SHA-256 identities are checked against prior recorded hashes. Intermediate diagnostic files lack an independent original SHA-256 attestation; new file identities are established and their finite tensors, seed/condition provenance, step/probe metadata, dimensions and masked zeros checked. Exact step-zero states are compared with constructed original initialization; step-2500 snapshots are tensor-compared to the hash-attested native endpoints. All masks are applied before Gram construction. Structural zeros are verified in both fc1 and fc2. Source masks themselves are verified against recorded file and state hashes.

The audit uses the existing probe's exact token IDs and random position ordering. All 1,024 positions are present once. Fixed-input capacity is computed after undoing that random position permutation, in the original 8×128 batch composition/order, then assignments are returned in probe order. Native acceptance uses actual end-to-end RouteTrace values. Probe checksums match metadata and existing probe records.

## SVD membership

The existing per-expert SVD tables are reused; a summary cannot reconstruct weights. Ranks 8/16/32 are reported separately. Invalid current matrices are excluded for zero norm, rank deficiency (cutoff <= max singular value × 1e-6), cutoff degeneracy (relative adjacent gap <1e-3), or missing overlap. The direct correction of the original claim retains its original current-matrix exclusion rules, but uses exactly the same `(layer,expert)` rows at step 0 and 2500. Source-reference flags are exported. `endpoint_reference_valid` and `trajectory_reference_valid` require valid current and source bases and drive the main figures and cross-metric comparisons. `endpoint_matched` and `trajectory_fixed` retain original-rule filters for the direct 28/30 correction; both filtering schemes reproduce 28/30.

Endpoint means use the valid two-endpoint intersection. Trajectories use one fixed valid intersection across all seven recorded steps. Learned/random pairs intersect across both mask types and all required steps. Per-layer and all-layer means are computed within seed. Joint endpoint tables also require the same experts to be eligible for SVD, Gram and activation statistics. Age contrasts match experts across ages. Individual experts are descriptive, not independent replicates. Rank-deficient and degenerate rows do not enter primary current-valid averages. Cutoff gaps, retained energy, exact exclusions and source-reference flags are retained in `svd_expert_raw.parquet` and associated coverage tables.

## Full-spectrum input Gram

For the actual masked fc1 matrix W [1024,256], compute G=W^T W in FP64 over all 256 input dimensions. It describes linear response energy ||Wx||²=x^T Gx.

- Shape distance: ||G/||G||F - Gref/||Gref||F||F.
- Frobenius cosine: <G,Gref>F/(||G||F ||Gref||F).
- These satisfy distance²=2−2cosine and are the same shape comparison, not independent evidence.
- Scale: trace(G)=||W||F²; trace ratio to each reference.
- Relative unnormalized distance: ||G−Gref||F/||Gref||F.
- Zero norms yield missing shape/cosine values, never zero replacements. A nonzero reference still permits the relative distance and zero trace ratio of a zero current matrix.

G excludes the bias, GELU, down projection and routing. Left-orthogonal transformations of W preserve G but need not preserve the nonlinear expert. Coordinatewise comparisons also do not remove upstream representation drift. Fixed/native hidden-vector relative L2 drift and cosine are exported separately.

## GELU analogue of score–activation coupling

For each layer/expert/token on the same hidden input x:

- Score s_e(x)=r_e^T x, signed, no softmax/routing weight.
- Activation a_e(x)=mean_j GELU((W_e x+b_e)_j), signed, no absolute value or norm.
- Each score and activation group is independently standardized using mean and population SD, within layer/expert/selection/mode/checkpoint.
- Minimum group size: 8. SD <=1e-12 is treated as zero variance. Nonfinite values are excluded/flagged. Undefined correlations remain missing.
- Pearson is primary. Spearman per expert and per-layer standardized pooled sensitivity are exported. Equal-expert Pearson means and token-count-weighted standardized pooled Pearson are distinct summaries. The latter is exactly the pooled z-score correlation because each eligible group has mean zero and variance one.

**Fixed mode:** identical cached source-final hidden vectors at every checkpoint. **Native mode:** same token sequences, using the current checkpoint's actual end-to-end hidden vectors. Proposed top-k pairs and actual capacity-accepted pairs are separate selections. The all-expert counterfactual uses the first 128 positions in the original fixed random probe order for every expert, regardless of selection; this subset is unchanged across checkpoints. Native outputs/traffic for capacity are computed on the complete original batch, not on the counterfactual subset.

Within-layer shuffled controls cyclically shift router expert rows by each nonzero offset 1–31 while retaining expert activations, selected pair identities and identical hidden inputs. Thus every unmatched score row appears once per expert; routes are not recomputed. Identity versus shuffled effects use the same expert intersection across all 32 pairings within each state. Shifts are control pairings, not independent training seeds. Counts and all null correlations are exported.

Within-token raw score/activation correlations across 32 experts on the 128-token all-expert subset are a separate extension, not the paper's routed-pair measurement. No within-expert normalization is imposed on this across-expert statistic. Raw token arrays (`scores`, `activations`, `selected`, `accepted`, token IDs and position order) are retained as compressed NPZ files. Per-expert raw means/SD/quantiles/negative counts appear in the raw tables alongside standardized correlations.

Endpoints and trajectories additionally hold valid expert membership fixed. Learned/random effect comparisons jointly match valid experts; differences use all five source seeds, not pooled token sample sizes. Routed coverage can be sparse and is not hidden by pooling. The original probe was not expanded, preserving comparability; excluded expert groups remain missing. No claim of comprehensive evaluation-set correlation is made.

## Uncertainty, caching and preservation

The replication unit is the source seed (7,17,29,43,61). Seed-level means and paired effects use 10,000 bootstrap resamples with RNG seed 8103, with all five values shown. Intervals are pointwise and exploratory, with small-n limitations. No p-values, confirmatory significance claims or multiplicity-adjusted discoveries are asserted. Across-seed correlation sensitivities remain stratified by mask type/sparsity/router age, n=5; experts, layers, tokens and steps are not independent seeds.

Inference uses no optimizer/gradients, protected eval/no_grad, strict FP32 activations with TF32 disabled, and restoration of caller modes and RNG/flags. Mathematical/native-forward tests and CPU/CUDA RNG conservation checks are recorded in `validation.json`. Bounded expert chunks avoid full-model activation stacks; checkpoint references/initialization are released after each seed. Caches key the input/reference/probe identity, config, model code and measurement code. Completion receipts include output hashes. Original output hashes and manifest identity are rechecked separately; all new artifacts are under this versioned output directory.
