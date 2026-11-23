# Expert-Count Architecture Replication

## Study Status

The complete 72-run grid finished: 2 expert counts x 3 seeds x 3 sparsities x 3 router initializations, plus 18 matched dense controls. Historical outputs were not overwritten or mixed into the new statistics. All requested router checkpoint steps loaded exactly (0, 500, 2500), and all conditions report passed integrity checks.

## Architecture

Both architectures retain d_model=256, four MoE layers, expert hidden size=1024, top-k=1, capacity factor=1.25, learned linear routers, auxiliary loss weight=0.1, fp16, AdamW, learning rate 3e-4, weight decay 0.01, gradient clip 1.0, 2500 dense-reference steps, and 2500 recovery steps. Only the expert count changes: 8 versus 12. Sparsity is expert-local magnitude pruning at 75/85/95%; no experts are removed.

## Mean Ticket Gaps

| Experts | Sparsity | R0 | R20 | R100 | R0-R20 | R100-R20 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.75 | 0.1155 | 0.0429 | 0.0786 | 0.0726 | 0.0357 |
| 8 | 0.85 | 0.2597 | 0.1677 | 0.2040 | 0.0920 | 0.0363 |
| 8 | 0.95 | 0.4170 | 0.4148 | 0.4750 | 0.0021 | 0.0601 |
| 12 | 0.75 | 0.1042 | 0.1005 | 0.0916 | 0.0037 | -0.0089 |
| 12 | 0.85 | 0.2260 | 0.2266 | 0.1906 | -0.0006 | -0.0360 |
| 12 | 0.95 | 0.4099 | 0.4558 | 0.4415 | -0.0459 | -0.0143 |

## Seed-Level 85% Results

| Experts | Seed | R0 gap | R20 gap | R100 gap | R0-R20 | R100-R20 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7 | 0.2297 | 0.1563 | 0.2351 | 0.0734 | 0.0788 |
| 12 | 7 | 0.1736 | 0.2809 | 0.2332 | -0.1073 | -0.0477 |
| 8 | 17 | 0.2586 | 0.1792 | 0.1657 | 0.0794 | -0.0135 |
| 12 | 17 | 0.2740 | 0.2317 | 0.1775 | 0.0423 | -0.0542 |
| 8 | 29 | 0.2909 | 0.1677 | 0.2112 | 0.1232 | 0.0435 |
| 12 | 29 | 0.2305 | 0.1672 | 0.1610 | 0.0633 | -0.0061 |

## Answers

1. **8 experts:** yes, partial-to-strong replication at the primary 85% condition. R20 beats R0 in all three seeds and beats R100 in two of three; mean gap is 0.1677 versus 0.2597 (R0) and 0.2040 (R100).
2. **12 experts:** no R20 replication at 85%. R20 beats R0 in two seeds but loses to R100 in all three; mean gap is 0.2266 versus 0.2260 (R0) and 0.1906 (R100).
3. **85% strongest routing-sensitive regime:** only for e8. Its R20 improvement over R0 is 0.0920 at 85%, versus 0.0726 at 75% and 0.0021 at 95%. For e12, 85% has no R20 benefit and 75% is also nearly flat; the strongest clean intermediate effect is absent.
4. **95% capacity-limited:** yes. Mean gaps are approximately 0.415-0.475 for e8 and 0.410-0.456 for e12, substantially above the 75/85% gaps. The result is descriptive, not a claim that capacity is the sole cause.
5. **Best router state:** R20 remains best for e8 at 75% and 85%, but not meaningfully at 95%. For e12, R100 is best at 75% and 85%, while R0 is best at 95%; the apparent optimum shifts away from R20 and does not form a stable intermediate regime.
6. **Effect size with expert count:** the 85% R0-to-R20 mean improvement falls from 0.0920 with 8 experts to -0.0006 with 12 experts; the R100-to-R20 contrast falls from 0.0363 to -0.0360.
7. **Architecture-specificity:** the phenomenon is not established as architecture-independent. The e8 result replicates within the tested setup, but the 12-expert architecture fails the primary R20 comparison.

## Classifications

- e8: **STRONG REPLICATION** under the predefined rule, with seed-level R20-vs-R0 consistency and two-of-three R20-vs-R100 wins.
- e12: **PARTIAL REPLICATION** at most: R20 is not better than R100 at 85%, and its R0 comparison is mixed in effect size.

The 12-expert outcome does not justify adding router ages automatically. If refining the apparent intermediate regime is scientifically warranted later, it should be a separately approved small age refinement, not inferred from this result alone.

## Overall Verdict

PARTIAL CROSS-ARCHITECTURE REPLICATION
