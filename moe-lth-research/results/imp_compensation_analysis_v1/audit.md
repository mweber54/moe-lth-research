# IMP80 immediate damage and adaptation-gap analysis

## Status and scope

This is a post hoc, artifact-only analysis of the five registered source seeds. It did not launch training, optimizer updates, mask discovery, or a pruning-policy experiment. The independent replication unit is the source seed (n=5). The primary window is global step 1000 to 1500, with offsets 0, 25, 100, 250, and 500. Shared-only means the registered `router_experts_frozen` condition; router and expert parameters are frozen and shared parameters train.

All 40 adaptation branches and 20 transition jobs are reported against their registrations. The 200 common-validation grid rows have 492,416 target tokens over 31 batches per readout. Immediate pruning damage on validation and on original training-only event probes is reported separately.

## Common parent and immediate event damage

Common parent status is A for each seed: the step-1000 reconstructed dense model is bitwise equal to the historical dense checkpoint; optimizer/scaler and Python/NumPy/Torch CPU/CUDA RNG state are saved, sampler position is 1000, and branch starts share training and validation identities within each model type and across conditions. Sparse branches originate from that state, then apply the registered IMP80 mask and zero coordinate-shaped AdamW moments at masked coordinates. This is a treatment-inclusive comparison, not a weight-zeroing-only effect. The native sparse and unpruned start states differ as expected because of the event.

Validation `D = L_post - L_pre` is the immediate paired CE effect of the complete pruning event on identical 492,416 validation target tokens. Training-only event-probe damage is separately recorded and never substituted for validation D. See `immediate_damage.csv`.

| Seed | Validation D | Shared-only L_sparse(1500) | Shared-only L_unpruned(1500) | Sparse reduction | Unpruned reduction | C= D-G(500) |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.470440 | 2.057034 | 1.989057 | 0.501382 | 0.098920 | 0.402463 |
| 17 | 0.449622 | 2.028336 | 1.961325 | 0.488639 | 0.106027 | 0.382611 |
| 29 | 0.465983 | 2.049269 | 1.967500 | 0.486839 | 0.102626 | 0.384213 |
| 43 | 0.494836 | 2.060850 | 1.984956 | 0.535030 | 0.116088 | 0.418942 |
| 61 | 0.512726 | 2.043791 | 1.980973 | 0.577648 | 0.127740 | 0.449908 |

Positive C means the observed sparse-minus-unpruned CE gap narrowed relative to the measured immediate validation D; it is not uniquely attributable repair and is not proof that original function was restored. The equality `H=R_sparse-R_unpruned` was checked numerically at every seed, condition, and offset. Since `G(0)=D` within the registered CE tolerance, C and observed initial-gap closure H also agree within tolerance.

## Trajectories and permitted interventions

`absolute_losses.csv` keeps global step 1000's pre-pruning unpruned parent and post-pruning sparse state as distinct phases. `gap_closure_decomposition.csv` gives both absolute losses beside G, C, H, each model's reduction, the improvement-pattern label, and arithmetic residuals for all 100 seed × condition × offset observations. Integrals use only the common observed offsets and span; no extrapolation is used.

`condition_penalties.csv` reports P_R, P_E, and P_RE for each model type, seed, and grid point. Sparse-excess penalty identities and the H/freezing identity passed. `factorial_interactions.csv` uses I=L11-L10-L01+L00, where 1 means the router or expert group trains; shared parameters train in every cell. These identities are algebraic bookkeeping, not independent evidence.

The complete five-seed values precede means, medians, minima, maxima, and sample SDs in `seed_level_metrics.csv` and `aggregate_summaries.csv`. No binary recovered/harmless threshold, winner classifier, or component-percentage decomposition was used.

## Shared-only pathway diagnostics

Existing sealed-probe routing, gate, accepted-load, dropped-token, representation, expert-function, update, and freeze-verification diagnostics are consolidated in `shared_only_diagnostics.csv`. Frozen expert outputs on identical branch-start inputs are unchanged within 1e-7; that is a consistency check. Outputs on current native inputs vary with the changing inputs and do not imply expert parameter learning. Frozen expert and router parameters, including biases, were separately compared bitwise between shared-only start and endpoint checkpoints.

A bounded checkpoint-only evaluation on the sealed four-batch training probe compares shared-only start and endpoint final-normalized hidden states and logits. It reports target-token-weighted probe CE, activation/logit displacement and cosine, softmax Jensen-Shannon divergence, and top-1 agreement in `checkpoint_prediction_diagnostics.csv`. These training probes may overlap training and are not an independent test set. Shared parameter movement includes the tied embedding/output matrix, which has both input and output roles. These diagnostics are candidate-process measurements, not causal decomposition.

For all five seeds, the sparse shared-only CE reduction is larger than the unpruned shared-only reduction because the sparse state starts with the measured event gap. A larger reduction from a worse starting loss, by itself, does not establish a specialized compensatory mechanism. Frozen experts remain functionally in the pathway because shared updates can change their inputs, gates, assignments, and downstream use. Similar average CE does not imply restored predictions or expert behavior.

## Provenance and outputs

See `provenance_checks.json` for seed-level common-parent evidence, source registrations, event output hash checks, and cohort status; `measurement_coverage.csv` for every required observation; and `artifact_inventory.csv` for relevant original files and hashes. No historical report, checkpoint, manifest, registration, raw measurement, or failed attempt was overwritten.

The primary inputs were the abrupt IMP80 transition artifacts and completed post-pruning adaptation factorial artifacts. All new analysis products are isolated under `results/imp_compensation_analysis_v1/`.
