# RAMP–IMP pre-implementation audit

Audit date: 2026-11-13 (client local date). This audit precedes new implementation.

## Eligible historical cohort

`results/imp_80_85/final_recovery_metrics.csv` contains 30 committed endpoint rows: E8/E16/E32 × seeds 7,17,29,43,61 × sparsities .80/.85. `imp_mask_manifest.csv` supplies all nine nested masks per trajectory and the exact checkpoint used to discover each mask. Existing independent audit `results/imp_ticket_audit_v1/{provenance.csv,comparison.csv,validation.json,checkpoint_configs.json}` verified original initialization file hashes, binary mask tensor hashes, endpoint curves, and dense/sparse train/validation batch identities for these 30 rows. Recheck these identities in the new runner; do not infer eligibility solely from this audit.

All 15 initial states, 30 final sparse states, and 15 paired dense states exist. Initial checkpoint sizes are approximately 72/139/274 MB for E8/E16/E32. Dense reference trajectories preserve steps 0,250,500,750,1000,1250,1500,1750,2000,2250,2500 for all seeds; some additionally preserve 125-step intervals. Discovery round_00 is the unpruned trajectory directly ancestral to IMP; paired dense recovery is a distinct trajectory and must be identified separately. Profiling initial and reference dense trajectory states is early/retrospective analysis, not sparse recovered-model scoring.

Only 30/50 originally planned final recoveries are complete according to `screening.json`: E12/E24 are outside this eligible cohort. The IMP README still says none launched; committed stage metadata, tables, and hashes supersede that stale prose. No absent state will be reconstructed by training.

## Architecture and pruning coordinates

`src/moe_lth/models/experts.py`: fc1 [1024,256], bias, GELU, dropout, fc2 [256,1024], bias. No neuron-level gate or shared expert. Router projection is not an FFN gate. At evaluation dropout is disabled. Adapt Eq. (3) using GELU(fc1(x)); down column is fc2.weight[:,i]. Biases are not historically pruned and output bias is outside neuron contribution score.

`pruning/masks.py:expert_weight_parameters` and `pruning/imp.py:nested_expert_local_masks` prune only fc1.weight and fc2.weight, jointly within each expert. Masks are nested, integer keep budgets per expert, stable coordinate tie handling. fc1 row i and fc2 column i identify the same neuron. Never modify original masks.

`models/moe_layer.py` runs actual capacity-selected assignments with raw selected softmax coefficients, with no selected-top-k renormalization and no post-capacity renormalization. Trace.selected_expert_indices / selected_probabilities / accepted_mask identify executed edges, including secondary experts. Capacity is per forward batch; calibration batch partition must be fixed. Dropped assignments contribute zero. No rerouting. Existing hooks can observe actual fc2 inputs, but the trace arrives after expert execution; replaying the exact accepted activation on captured layer inputs in bounded chunks avoids retaining token×neuron tensors. Use native forward for trace and hidden inputs and replay deterministic eval experts only; validate against direct hooks.

## Data and instrumentation

Saved configs agree on byte tokenizer (256 fixed UTF-8 byte IDs), sequence length 128, TinyStories/WikiText mixed local train/validation files under `data/multidomain_balanced`. `metadata.json` supplies original domain source paths. `data.py:load_token_splits`, TokenBlockDataset, and build_dataloaders are canonical. Use deterministic training-split blocks for calibration, explicitly remove any sequence appearing in the complete validation split, and preserve source/block IDs, exact sequences, split/file/token hashes and batch partition. These are in-training calibration examples, not new held-out generalization evidence. Domain subsets need exact source membership, not inferred semantic labels.

Native routes and pre-router hidden states can be recomputed from saved checkpoints. Historical validation route snapshots/usage logs and routing/rich_trace.py expose assignment/acceptance instrumentation, but aggregate counts alone are insufficient for RAMP. Gradient logs are in discovery/recovery `gradient_stats/gradient_stats.jsonl`; they are aggregate gradients, not neuron activations. Existing expert deletion/functional geometry scripts instrument activations; reuse the native architecture and avoid modifying it (it already has local edits).

## Mechanistic cohorts and limitations

`results/imp_mask_information_v2` has completed E16 matrix-random and degree-preserving controls, 5 seeds; use matched within-seed contrasts, not these controls as independent tickets. `imp_adaptation_factorial_v1` contains completion, protocol, reuse checks and endpoint tables for router/expert adaptation. `imp_initial_value_placement_v1` contains sealed transformation identities, sign/value/placement variants and recovery curves. `imp_recovery_order_stability_v1` contains 20 complete order jobs and historical compatibility checks. `imp_sign_placement_recovery_v2` contains only phase-0 tables, not verified new complete recovery endpoints. `imp_matched_sign_magnitude_recovery_v1` is prospectively prepared; no missing endpoints can be invented. Inspect endpoint identities before optional joins. Comparisons with altered weights must profile the altered state for mechanistic claims; original-state profiles alone cannot identify sign-intervention effects.

Primary retrospective H1/H2 is feasible now on 15 trajectories / 30 masks. Five independent seeds per architecture/sparsity permit descriptive seed uncertainty, not a high-dimensional predictive model or causal claims. Initialization, early, intermediate, final dense states can be compared with eventual masks without future-state features in early scores. Later training phase H3 must remain an unexecuted design. No GPU training or downloads are necessary.

## Sources inspected

Repository README, pyproject, experiment registry `experiments/paper_experiments.yaml`, protocol and phase2 documentation, prior reports, relevant architecture/checkpoint/data/pruning/routing code, IMP orchestration and artifact audits, mask-information findings and mechanistic artifact inventories. No AGENTS.md found under repository parent. Existing user edits are preserved. Paper: https://arxiv.org/html/2611.13975v1, Section 3.2 Eq. (3), verified from primary source; GELU adaptation explicitly differs from its gated example.

## Audit refinement during implementation

The separately stored `results/imp_sign_recovery_stage1_v1` is complete (54 approved sign-only runs, sources 7/17/29, stopped with no continuation authorized). This differs from the incomplete `imp_sign_placement_recovery_v2` phase-0 tables. Its `per_seed_gate.csv` and `per_seed_order_summary.csv` support contextual sign-only evidence but not a RAMP-conditioned sign effect without profiling transformed states. Keep its protocol and three-source conditional uncertainty separate from the primary five-seed cohort.
