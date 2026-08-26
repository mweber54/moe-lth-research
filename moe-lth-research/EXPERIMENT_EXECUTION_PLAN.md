# Experiment Execution Plan

This execution plan orders the revision experiments by scientific priority and by dependency chain. The sequence below is designed to avoid spending compute on large, weakly informed sweeps before the protocol-level claims are validated.

## Recommended Order

| Order | Experiment | Estimated run count | Expected compute cost | Dependencies | Output locations |
|---|---|---:|---|---|---|
| 1 | P0.1 compute-matched rewind | 2 dataset × 3 seeds × 2 sparsities × 4 rewind points × 4 conditions = ~192 short runs | Low to medium | Validated dense checkpoints and mask generation | `results/revision/p01_compute_matched_rewind/...` |
| 2 | P0.2 deconfounded shuffle | 2 datasets × 3 seeds × 2 sparsities × 3 rewind points × 1 intervention = ~36 runs | Medium | Rich trace and route archive | `results/revision/p02_deconfounded_shuffle/...` |
| 3 | P0.3 graded corruption | 1 dataset × 3 seeds × 6 corruption fractions × 2 sparsities = ~36 runs | Medium | P0.2 route controls and trace fidelity | `results/revision/p03_graded_corruption/...` |
| 4 | P0.4 cross-initialization replay | 3 sources × 5 targets × 2 sparsities × 2 rewind points = ~60 runs | Medium | P0.3 matched-quality controls | `results/revision/p04_cross_init_replicated/...` |
| 5 | P0.5 routing × initialization factorial | 3 route seeds × 5 init seeds × 2 sparsities × 2 rewind levels = ~60 runs | Medium | P0.4 and P0.3 | `results/revision/p05_factorial_analysis/...` |
| 6 | P0.6 full expert-transfer matrix | 1 dataset × layers × experts × 2 sparsities = moderate but dense matrix | Medium | Learned masks and a clean route archive | `results/revision/p06_expert_transfer_matrix/...` |
| 7 | P0.7 fresh epoch shuffle / non-repeated order | 1 dataset × 3 seeds × 2 protocols × 2 sparsities = ~12 runs | Medium | Data-loader semantics and route archive | `results/revision/p07_epoch_shuffle/...` |
| 8 | P0.8 larger-scale validation | 1 larger-scale run or 1 small cluster of runs | High | Prior protocol validation | `results/revision/p08_larger_scale/...` |
| 9 | P1.1 increased mechanistic-core seeds | ~8–10 seeds for a narrow set of conditions | Medium | Stable protocol-level result | `results/revision/p11_mechanistic_core_seeds/...` |
| 10 | P1.4 IMP comparison | 2 datasets × 2 methods × 2 rewind points × 2 sparsities = ~16 focused runs | Low to medium | Stable sparse-support result | `results/revision/p14_imp_comparison/...` |
| 11 | P1.5 top-2 route support | 1 targeted top-2 validation exercise | Low to medium | Rich trace schema upgrade | `results/revision/p15_top2_support/...` |

## Dependency Notes

- The protocol-level validations (P0.1–P0.3) are the gating experiments. If they do not hold under compute matching or gate-control, the manuscript should revise its causal language before commissioning large matrix sweeps.
- Cross-initialization and factorial work should be done only after the deconfounded shuffle and graded corruption utilities are exercised and their invariants are checked.
- The larger-scale validation is intentionally last because it is not a broad scaling study; it is a single external-validity probe.

## Output Hygiene

- All run outputs should be versioned under `results/revision/...`.
- Each result directory should contain the config, seed metadata, run manifest, trace hash, and output summaries.
- No historical result directories should be overwritten by default.

## Current Status

As of this repository state, the protocol-level implementation and validation are in place for the compute-matched rewind and the richer trace/schema work. The larger intervention sweeps and large-scale validation remain to be executed.

## Effective Stop Condition for the Current WikiText Gate

The repository is intentionally blocked from advancing to the graded corruption sweep until both of the following are proved on the exact historical reference path:

1. The learned baseline reproduces the historical WikiText reference run from `configs/wikitext103_gpu.yaml` and the saved historical result in `results/wikitext103_gpu_suite/normal/resolved_config.yaml`.
2. The deconfounded intervention is demonstrably nontrivial while preserving count, acceptance, and gate statistics.

Current evidence shows the quick seed-7 proxy is not yet a valid reference gate:

- `scripts/seed7_wikitext_gate.py` uses a CPU/FP32/5-step configuration with `validation_blocks = 1` and therefore does not match the paper’s 2500-step CUDA/FP16 reference path.
- The current proxy prints `LEARNED = 129.465`, `REPLAY = 129.465`, and `DECONFOUNDED_SHUFFLE = 129.465`, which is far from the historical `~1.68` reference result and indicates the gate is not reproducing the original experiment.
- The deconfounded-invariance check is necessary but not sufficient; it still has to show route disagreement and assignment churn while preserving counts/acceptance/gates.

Until both conditions hold, no broader sweep (graded corruption, rewind, cross-init, or scale) should proceed.
