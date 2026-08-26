# Revision Implementation Status

This document tracks the repository against the revision plan in `iclr_revision_plan.md`.

Status key:
- DONE
- IMPLEMENTED — EXPERIMENT NOT RUN
- RUNNING
- BLOCKED
- NOT STARTED

## P0 — Must Fix Before Submission

### P0.1 — Add a Compute-Matched Rewind Experiment
- Status: DONE
- Implementation: `src/moe_lth/pruning/train_ticket_v2.py` includes `full_budget_early_state_retrain()` and `compute_matched_rewind()`, with explicit protocol naming and the compute-matched continuation semantics.
- Validation: `tests/test_rewind_protocols.py` asserts the protocol labels, rewind-step accounting, and masked continuation behavior.
- Outputs: results are written to `tables/ticket_result_fbr.json` and `tables/ticket_result_cm_<step>.json` under each run output directory.

### P0.2 — Deconfound the Shuffled-Usage Routing Intervention
- Status: BLOCKED — REFERENCE-GATE VALIDATION INCOMPLETE
- Implementation: `src/moe_lth/routing/deconfounded.py` contains deconfounded identity-shuffle logic and the graded corruption utilities; `src/moe_lth/routing/interventions.py` exposes `deconfounded_shuffle` and `graded_corruption` routing modes.
- Validation status: the logic is implemented, but the current quick seed-7 proxy gate still fails the historical-reference check. The proxy run in `scripts/seed7_wikitext_gate.py` is not the actual 2500-step CUDA/FP16 WikiText reference; it uses a 5-step CPU/FP32 configuration and prints `129.465` instead of the historical `~1.68`.
- Required gate: the learned baseline must reproduce the exact historical experiment from `configs/wikitext103_gpu.yaml`, and the deconfounded intervention must show route disagreement while preserving counts and acceptance/gate statistics.
- Outputs: interim artifacts are under `results/wikitext_reference_gate/...`, but broader revision sweeps remain blocked until both gate conditions pass.

### P0.3 — Add a Matched-Dense-Quality Routing Perturbation
- Status: DONE
- Implementation: `graded_route_corruption()` in `src/moe_lth/routing/deconfounded.py` supports deterministic corruption fractions and preserves per-expert counts where requested.
- Validation: the 0%, 10%, 25%, and 100% corruption conditions have three-seed support metrics recorded in `results/revision_progress/items_1_2_summary.md` and `items_1_2_summary.json`.
- Outputs: primary sweep artifacts are under `results/wikitext_reference_gate/graded_sweep/`.

### P0.4 — Properly Replicate Cross-Initialization Replay
- Status: DONE
- Implementation: `src/moe_lth/experiments/run_cross_init_replay.py` now supports a backward-compatible single-source mode and a combined multi-source matrix mode.
- Relevant scripts: `src/moe_lth/experiments/run_cross_init_replay.py` and `src/moe_lth/experiments/run_cross_init_rewind.py`.
- Validation: all 9 source/target cells completed at 50% and 80% sparsity.
- Outputs: `results/revision/p04_cross_init_3x3/cross_init_replay_matrix_summary.json` and `cross_init_replay_matrix_results.md`.
- Interpretation: off-diagonal replay adds approximately 0.0723 loss points versus 0.0054 on diagonal cells; this supports an initialization-dependent route-history effect but is not itself the P0.5 variance decomposition.

### P0.5 — Run a Routing × Initialization Factorial / Variance-Decomposition Experiment
- Status: DONE
- Implementation: `src/moe_lth/experiments/run_routing_init_factorial.py` computes route, init, and interaction variance from the cross-init matrix.
- Validation: the direct test in `tests/test_factorial_analysis.py` passes, and the real artifact was generated at `results/revision/p05_factorial_analysis/`.

### P0.6 — Strengthen or Weaken the "Expert-Specific" Claim
- Status: DONE
- Implementation: `src/moe_lth/experiments/run_expert_transfer_matrix.py` computes the full source-expert × target-expert substitution matrix, writes the raw JSON, and summarizes diagonal versus off-diagonal transfer penalties.
- Output: `results/revision/p06_expert_transfer_matrix/expert_transfer_matrix.json` and `expert_transfer_summary.json`.
- Interpretation: the first layer shows a strong diagonal advantage (mean penalty ~4.08), while later layers are nearly flat; this supports a qualified, layer-dependent expert-specificity claim rather than a blanket global one.

### P0.7 — Test the Repeated-Data-Order / Memorization Alternative
- Status: IMPLEMENTED — EXPERIMENT NOT RUN
- Implementation: `src/moe_lth/data.py` provides `EpochReshuffledDataLoader`, and `src/moe_lth/training/train.py` resets its iterator at epoch boundaries when `data.reshuffle_each_epoch` is enabled.
- Validation: `tests/test_epoch_reshuffle.py` confirms that successive epochs use different batch orders.
- Execution: the P0.8 configuration enables this control for the larger-scale run; a dedicated P0.7 comparison remains to be executed if needed.

### P0.8 — Add One Meaningfully Larger-Scale Validation
- Status: IMPLEMENTED — EXPERIMENT NOT RUN
- Implementation: `src/moe_lth/experiments/run_larger_scale_validation.py` runs the larger learned baseline, count-preserving `deconfounded_shuffle` replay, established magnitude/random/rewind controls, and mask-overlap summaries.
- Configuration: `configs/revision_larger_scale.yaml` uses an 8-layer, 16-expert, 512-wide top-1 model with epoch-wise data reshuffling, totaling approximately 277M parameters.
- Validation: `tests/test_larger_scale_validation.py` checks the scale, top-1 protocol, fresh-order setting, and 50%/80% rewind points.
- Execution: run `python -m moe_lth.experiments.run_larger_scale_validation --config configs/revision_larger_scale.yaml` on the configured CUDA environment. The config retains only rewind-required model checkpoints and disables optimizer serialization to control disk use. Outputs are written under `results/revision/p08_larger_scale/`.

## P1 — Strongly Recommended

### P1.1 — Increase Seeds Only for the Mechanistic Core
- Status: NOT STARTED
- Scope: only the mechanistic core should be expanded, not the full architecture sweep.

### P1.2 — Standardize Statistical Reporting
- Status: NOT STARTED
- Gap: there is no shared statistics utility yet; seed-level paired deltas and consistent sample-SD conventions are still to be implemented.

### P1.3 — Demote or Replace the Route-Agreement / Mask-Jaccard Correlation
- Status: NOT STARTED
- Implementation requirement: correlation with and without the replay point, plus explicit warnings about non-independence.

### P1.4 — Promote IMP Enough to Establish Protocol Dependence
- Status: NOT STARTED
- Relevant scripts: `src/moe_lth/experiments/run_imp_representative.py`.

### P1.5 — Clarify Top-1 vs. Top-2 Generalization
- Status: IMPLEMENTED — EXPERIMENT NOT RUN
- Partial implementation: the rich trace schema records top-k route information and the codebase is prepared to support richer replay semantics; full top-2 replay has not yet been executed.

### P1.6 — Update and Sharpen Related Work
- Status: NOT STARTED
- Need targeted discussion revisions in the manuscript text, not just code changes.

### P1.7 — Rewrite the Theoretical / Problem-Formulation Section
- Status: NOT STARTED
- Need manuscript-level reframing of the dependency framework and research questions.

## Manuscript-Supporting Code Changes

### P15 — Create a Canonical Experiment Registry
- Status: DONE
- File: `experiments/paper_experiments.yaml`
- This is already the manuscript-facing registry for experiments and is now treated as the single source of truth for the revision.

### P16 — Improve Reproducibility Metadata
- Status: IMPLEMENTED — EXPERIMENT NOT RUN
- Broader metadata hooks exist in the config and checkpoints, but not yet the full git/Python/PyTorch/CUDA provenance manifest expected for all runs.
- Relevant code: `src/moe_lth/training/checkpoint.py`, `src/moe_lth/config.py`.

### P17 — Refactor Figure Generation
- Status: IMPLEMENTED — EXPERIMENT NOT RUN
- Script: `scripts/generate_paper_figures.py` is the deterministic presentation layer over stored results.
- Remaining work: update it around the revision figure layout and the new comparison protocols.

### P18 — Refactor Tables
- Status: NOT STARTED
- Need manuscript-ready summaries for compute-matched rewind, deconfounded interventions, cross-init matrices, etc.

## Audit and Semantics Implemented in the Current Patch
- Rich route-trace schema versioning and metadata: `src/moe_lth/routing/rich_trace.py`
- Explicit compute-matched and full-budget protocol names: `src/moe_lth/pruning/train_ticket_v2.py`
- Training config support for rich traces and routing corruption fractions: `src/moe_lth/config.py`
- Replay validation for deconfounded and corruption modes: `src/moe_lth/training/train.py`
- Unit tests covering protocol semantics and trace integrity: `tests/test_rewind_protocols.py`, `tests/test_rich_trace.py`

## Current Verification
- Command run: `python -m pytest -q`
- Result: 23 passed in 10.35s
