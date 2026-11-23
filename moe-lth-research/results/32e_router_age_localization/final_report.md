32-EXPERT ROUTER-AGE TICKET LOCALIZATION

Verdict: 32E BOUNDARY OPTIMUM

Mean winner: R0
Mean ticket gap ± seed SD: 0.13484835 ± 0.01724690
Local uncertainty basin: [0]

Seed winners:
- seed 7: R0 (0.11564137); second R20 (0.15097047), difference 0.03532910
- seed 17: R0 (0.14900976); second R10 (0.15604793), difference 0.00703817
- seed 29: R0 (0.13989390); second R60 (0.16106228), difference 0.02116838

Candidate vs R0:
- mean paired difference: +0.00000000
- seed differences: {7: 0.0, 17: 0.0, 29: 0.0}

Candidate vs R100:
- mean paired difference: -0.04055391
- seed differences: {7: -0.0687171635083605, 17: -0.024436413730495515, 29: -0.028508151386321484}

Candidate beats both boundaries in all seeds?: False; seeds not favoring both: [7, 17, 29]
Pruning-shock winner: R100
Early-recovery winner: R0
Sparse-final-loss winner: R60
Matched-ticket-gap winner: R0

Late source-training improvement:
 seed  R80_R100_absolute_improvement  R80_R100_relative_improvement
    7                       0.112886                       0.070706
   17                       0.117103                       0.072561
   29                       0.118295                       0.073571

Per-expert exposure note:
 phase  reference_experts  relative_32E_exposure                                                      coverage
 dense                  8               0.239607                         complete recovery; common R0/R20/R100
 dense                 24               0.736239                         complete recovery; common R0/R20/R100
sparse                  8               0.241330                         complete recovery; common R0/R20/R100
sparse                 24               0.739742                         complete recovery; common R0/R20/R100
source                  8               0.239627 matched sampled batches only; complete 24E source unavailable
source                 24               0.738496 matched sampled batches only; complete 24E source unavailable

Developmental-state comparison: Descriptive only; full age means and historical scaled optimum distances are persisted. No common equal-maturity optimum established.

Cross-architecture sequence:
8E -> R20; basin [20, 100]
12E -> R60; basin [60, 80, 100]
16E -> R60; basin [60, 70, 80]
24E -> R50; basin [40, 50, 60, 80, 100]
32E -> R0; basin [0]

Does optimum move monotonically later with expert count?: False descriptively; no general law established.
Does a single coarse developmental state explain the optima?: Not established by historical descriptive distances.
Does seed dependence increase with expert count?: Compare winner spread and SD below; three seeds and unequal adaptive grids do not establish a trend.
Is the current fixed-budget phenomenon still present at 32E?: 32E BOUNDARY OPTIMUM; favorable interior evidence requires the paired-boundary comparisons above.
Recommended future mechanistic target, if any: none

Scientific interpretation:
These results characterize router-age-conditioned sparse recoverability under a fixed global source-training and recovery budget.

Raw normalized router age is a fraction of 2500 source attempts, not expert developmental exposure. The experiment controls global source tokens, source attempted steps and recovery attempted steps. It does not equalize accepted tokens per expert, successful expert updates, or maturity. Per-expert source and recovery exposure decrease as expert count increases, so normalized router age should not be interpreted as equal developmental maturity across architectures.

Observed mean-winner sequence:
8E -> R20; basin [20, 100]
12E -> R60; basin [60, 80, 100]
16E -> R60; basin [60, 70, 80]
24E -> R50; basin [40, 50, 60, 80, 100]
32E -> R0; basin [0]

Monotonically nondecreasing observed ages: False; no scaling law follows. Feature distances in cross_architecture_developmental_distances.csv use the historical coordinate ranges; neither proximity nor a nearest age establishes one common developmental optimum. Existing normalization includes sampled exposure, which itself scales with expert count.

Seed-level disagreement is reported directly rather than absorbed into the architecture mean. 24E versus 32E winner spread: 80 versus 0; mean across-age seed SD: 0.023467 versus 0.015878. These are descriptive, with unequal adaptive grids and only three seeds. They do not establish a population trend in seed dependence. Mean curve ranges: 0.027658 versus 0.052564; relative-to-SD measures are saved, without a post-hoc significance/equivalence threshold.

No exposure matching, longer recovery, old-architecture retraining or mechanism experiments were performed. Future mechanistic target: none; recommendation only, not a launched experiment.


Limitations: n=3 independent source seeds; adaptively sampled ages; no statistical significance or equivalence claim; R100 is not a convergence criterion. Historical source exposure is sampled, new source counters complete. Gate amplitudes, capacities, exposure and successful expert updates are not matched across architectures. Sparse loss alone never selects the winner. Immediate shock and early AUC are separate historical readouts. Seed SD uses ddof=1; load CV uses population SD. Maturity assignment/JS probes use first validation batch per layer; other historical probes use first eight batches. Exposure developmental metric remains sampled proposed counts, not new accepted counters.


Requested questions

1. **Does 32E exhibit an intermediate router-age ticket optimum?**

32E BOUNDARY OPTIMUM

2. **Which age minimizes mean ticket gap?**

R0

3. **What is mean ± seed SD?**

0.13484835 ± 0.01724690

4. **Does the candidate beat R0 on average?**

False

5. **Does it beat R100 on average?**

True

6. **Does it beat both in all three seeds?**

False

7. **Which seeds disagree?**

[7, 17, 29]

8. **What are the seed-specific winners?**

{7: 0, 17: 0, 29: 0}

9. **Is the optimum sharp or broad?**

Descriptive connected local basin [0]; no unsampled point precision.

10. **What is the local uncertainty basin?**

[0]

11. **Does immediate pruning shock identify the same age?**

R100; same=False

12. **Does early matched recovery identify the same age?**

R0; same=True

13. **Does absolute sparse loss identify the same age?**

R60; same=False

14. **Is the result driven by sparse loss or matched recoverability?**

Winner uses matched sparse-minus-dense loss. Comparing the readouts above does not identify a causal mechanism.

15. **How do developmental statistics evolve?**

Full historical metric age means and seed-age rows follow and are saved in developmental_age_means.csv. R100 agreement/JS are references, not proof of convergence.

16. **Is R100 still improving late in source training?**

[{'seed': 7, 'R80_R100_absolute_improvement': 0.11288560803351166, 'R80_R100_relative_improvement': 0.07070611163363197}, {'seed': 17, 'R80_R100_absolute_improvement': 0.11710297426125083, 'R80_R100_relative_improvement': 0.0725605018804095}, {'seed': 29, 'R80_R100_absolute_improvement': 0.11829486139910772, 'R80_R100_relative_improvement': 0.07357144602763785}]

17. **How does per-expert exposure compare with 24E?**

[{'phase': 'dense', 'reference_experts': 24, 'relative_32E_exposure': 0.7362387646885729, 'coverage': 'complete recovery; common R0/R20/R100'}, {'phase': 'sparse', 'reference_experts': 24, 'relative_32E_exposure': 0.7397418797447225, 'coverage': 'complete recovery; common R0/R20/R100'}, {'phase': 'source', 'reference_experts': 24, 'relative_32E_exposure': 0.7384963811551812, 'coverage': 'matched sampled batches only; complete 24E source unavailable'}]

18. **Does 32E show more seed dependence than 24E?**

Winner spread 0 versus 80; mean seed SD 0.015878 versus 0.023467. Mixed indicators are not forced into a trend claim.

19. **Is the mean curve flatter than 24E?**

Mean curve range 0.052564 versus 0.027658; range/mean-SD 3.310537 versus 1.178611. Unequal adaptive grids limit this comparison.

20. **Does 32E exhibit a boundary optimum?**

Aggregate winner at boundary: True; final hierarchy verdict 32E BOUNDARY OPTIMUM

21. **Is there evidence for monotonic shift with expert count?**

Observed mean age sequence nondecreasing=False; no scaling law.

22. **Is there a common coarse developmental-state optimum?**

Not established; historical-scaled optimum distances are descriptive and exposure differs.

23. **Does the fixed-budget phenomenon survive at 32E?**

32E BOUNDARY OPTIMUM; see paired boundary evidence rather than forcing an interior interpretation.

24. **Should 32E enter a future mechanism experiment?**

none; recommendation only; no mechanism run launched.

25. **What claim is justified without equal development?**

These results characterize router-age-conditioned sparse recoverability under a fixed global source-training and recovery budget.

Developmental age means:

```text
             entropy    margin   load_cv  token_specialization  assignment_agreement_to_r100  js_distance_to_r100  cumulative_expert_exposure
router_age                                                                                                                                   
0           3.308232  0.020204  0.596389              1.099011                      0.077718             0.171385                         0.0
10          3.387274  0.013432  0.421020              1.586122                      0.209096             0.121385                      5632.0
20          3.291042  0.025799  0.377803              1.761411                      0.240941             0.102223                     10752.0
40          3.074999  0.075577  0.370309              1.636389                      0.377563             0.063180                     20992.0
60          2.919892  0.124607  0.291032              1.534812                      0.526601             0.033091                     31232.0
80          2.815653  0.150097  0.266136              1.444641                      0.688985             0.010697                     41472.0
100         2.703497  0.175752  0.225854              1.406661                      1.000000             0.000000                     51712.0
```

Mean curve and seed-dependence comparison:

```text
 experts  mean_curve_range  mean_seed_sd  range_to_mean_sd
       8          0.092016      0.025778          3.569517
      12          0.053039      0.040815          1.299492
      16          0.063574      0.026006          2.444636
      24          0.027658      0.023467          1.178611
      32          0.052564      0.015878          3.310537
```

All 25 requested questions are addressed by the winner/boundary/shock/AUC fields, developmental tables, late source table, exposure comparisons, cross-architecture interpretation and limitations above. Raw values remain in CSV files.

32E BOUNDARY OPTIMUM
