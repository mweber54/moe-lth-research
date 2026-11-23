# Recovery batch-order stability findings

This is a fixed five-source exploratory follow-up. Source seeds, not repeated orders, are the replication units. A missing outcome is not imputed. No near-match tolerance is used.

Completed new jobs: 20/20; attempted updates: 100000/100000.

## Primary endpoint: step 5,000

| Seed | Original gap | Order A gap | Order B gap | Match count / evaluated orders |
|---:|---:|---:|---:|---:|
| 7 | -0.03560036722951243 | -0.004474860276375114 | 0.014501219388259212 | 2 / 3 |
| 17 | 0.021116930508198406 | 0.037308905878902454 | 0.03295786882147156 | 0 / 3 |
| 29 | 0.0037887048684128732 | 0.030914058491940066 | 0.0328846597535164 | 0 / 3 |
| 43 | -0.0017636812040022765 | 0.010820443827919357 | 0.04454267976313164 | 1 / 3 |
| 61 | -0.026912756390097048 | 0.009172814401180895 | -0.007826601836103064 | 2 / 3 |

## Per-source compatible outcomes

- Seed 7: original=-0.03560036722951243; A=-0.004474860276375114; B=0.014501219388259212.
- Seed 17: original=0.021116930508198406; A=0.037308905878902454; B=0.03295786882147156.
- Seed 29: original=0.0037887048684128732; A=0.030914058491940066; B=0.0328846597535164.
- Seed 43: original=-0.0017636812040022765; A=0.010820443827919357; B=0.04454267976313164.
- Seed 61: original=-0.026912756390097048; A=0.009172814401180895; B=-0.007826601836103064.

## Order effects and interpretation

Continuous gap ranges and sample SDs are in source_stability_summary.csv. Sparse and unpruned CE changes are reported separately in paired_order_effects.csv; `delta_gap = delta_sparse - delta_unpruned` is checked for every available contrast. These arithmetic contrasts do not identify a mechanism.

Pair rankings by sparse-minus-unpruned gap and by absolute sparse CE, including source values, pairwise agreement, and reversals, are in rank_comparisons.csv. Repeated orders are nested within each source initialization-mask pair.

The original-order references are compatible with the reconstructed stream, source initialization, IMP80 mask, matched unpruned source, and two-segment reset protocol. Original common-readout checkpoints exist at steps 2,500, 3,000, 4,000, and 5,000; other requested historical trajectory checkpoints are not available and are not invented.

No conclusions about all optimization randomness, mask-vs-initialization causality, or mechanism follow. This tests only temporal order within the two fixed segments.

Measured successful production GPU time: 14.815 hours. Engineering throughput extrapolation: 20.735612970828598 GPU-hours; kept separate from measured production cost.
