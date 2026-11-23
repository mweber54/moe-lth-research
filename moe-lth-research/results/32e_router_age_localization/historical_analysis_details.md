# 32-expert router-age ticket localization

1. **Does 32E exhibit an intermediate router-age ticket optimum?**

32E BOUNDARY OPTIMUM

2. **What router age or basin minimizes mean ticket gap?**

Mean winner R0; local uncertainty basin [0], using the preregistered paired-SD descriptive rule.

3. **What is the mean ± seed SD?**

0.13484835 ± 0.01724690 (n=3 seeds).

4. **Does the optimum beat R0?**

Mean paired difference +0.00000000; negative favors candidate. Seed differences: {7: 0.0, 17: 0.0, 29: 0.0}.

5. **Does the optimum beat R100?**

Mean paired difference -0.04055391; seed differences: {7: -0.0687171635083605, 17: -0.024436413730495515, 29: -0.028508151386321484}.

6. **Does it beat both in all three seeds?**

False

7. **Which seeds disagree?**

[7, 17, 29]

8. **Is the optimum sharp or broad?**

Observed local basin [0]; inspect local_uncertainty.csv for raw differences and paired seed SD. No significance claim or unsampled-age precision.

9. **Does immediate pruning shock show the same optimum?**

Minimum mean shock R100; matches candidate: False. Shock is measured on matched rewound E0/shared, age-router initial states.

10. **Does early recovery show the same optimum?**

Minimum matched sparse−dense early AUC at R0; matches: True. Individual sparse/dense AUCs are retained.

11. **Is the result driven by sparse loss or matched recoverability?**

The selection uses matched ticket gap only. Sparse-loss-only winner is R60; matched winner is R0. These descriptive results do not identify a causal optimization mechanism.

12. **How do developmental statistics evolve?**

Age means are tabulated below. Entropy uses natural logs, margin uses probabilities, load CV uses population SD per layer, specialization uses historical token-ID/expert MI. Assignment/JS maturity uses the historical first-batch-per-layer implementation. Exposure is sampled logged assignments, not complete accepted updates.

13. **Which 32E checkpoint is closest to 8E R20?**

R60, by mean historical-range-normalized distance; individual seed distances retained.

14. **Which is closest to 12E R60?**

R80.

15. **Is the ticket optimum developmentally close to those optima?**

It coincides with the closest sampled checkpoint to 8E R20: False; to 12E R60: False. Distance is descriptive; nearest does not establish a common developmental mechanism.

16. **Does increasing expert count shift the optimum later?**

The observed mean sequence is 8E R20 → 12E R60 → 32E R0. A stable later interior optimum is supported descriptively: False. Three architectures do not establish a scaling law.

17. **What architecture-dependence claim is justified?**

The measured matched ticket-gap curves support 32e boundary optimum. Only these model sizes, training budget, data, sparsity and three seeds are covered; developmental similarity is secondary.

18. **Should 32E enter the next gate-amplitude causal experiment?**

No stable intermediate target recommended. Any future experiment requires a separate instruction.

Age summary (mean and sample SD across independent seeds):

```text
 experts  router_age  num_seeds  mean_pruning_shock  mean_sparse_final_loss  sd_sparse_final_loss  mean_dense_final_loss  sd_dense_final_loss  mean_ticket_gap  sd_ticket_gap
      32           0          3           -0.643177                1.624855              0.023580               1.490007             0.006561         0.134848       0.017247
      32          10          3           -1.572786                1.584689              0.021689               1.418410             0.028276         0.166279       0.015209
      32          20          3           -1.721172                1.576985              0.016164               1.402228             0.008263         0.174758       0.021467
      32          40          3           -1.873999                1.572013              0.024817               1.384601             0.004469         0.187412       0.020517
      32          60          3           -2.059924                1.562136              0.018158               1.385179             0.016435         0.176957       0.014215
      32          80          3           -2.238045                1.566035              0.026413               1.384978             0.016929         0.181057       0.014332
      32         100          3           -2.294472                1.567898              0.026329               1.392496             0.030421         0.175402       0.008156
```

Seed winners:

```text
 seed  best_router_age  best_ticket_gap  second_best_router_age  second_best_ticket_gap  difference
    7                0         0.115641                      20                0.150970    0.035329
   17                0         0.149010                      10                0.156048    0.007038
   29                0         0.139894                      60                0.161062    0.021168
```

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

All primary readouts are native, with matched router-age dense recoveries. No replay interventions were run. Historical 8E/12E artifacts were reused without retraining. Logged exposure and first-batch maturity limitations preserve comparability but constrain developmental interpretation. No token, layer, expert, checkpoint or recovery step is treated as an independent replicate.

32E BOUNDARY OPTIMUM

Recommended 32E target checkpoint for future mechanism experiments: No stable intermediate target recommended

Recommended control checkpoints: R0 and R100; neighboring basin checkpoints if a future protocol requires local controls.

Confidence based on seed consistency: Limited: seed disagreement or boundary optimum prevents a stable intermediate target.
