# 24-expert router-age ticket localization

1. **Does 24E exhibit an intermediate router-age ticket optimum?**

24E OPTIMUM SEED-CONDITIONED

2. **What router age or basin minimizes mean ticket gap?**

Mean winner R50; local uncertainty basin [40, 50, 60, 80, 100], using the preregistered paired-SD descriptive rule.

3. **What is the mean ± seed SD?**

0.15709681 ± 0.03248276 (n=3 seeds).

4. **Does the optimum beat R0?**

Mean paired difference -0.00244113; negative favors candidate. Seed differences: {7: 0.020830160368563916, 17: -0.0195993989171504, 29: -0.00855415270859658}.

5. **Does the optimum beat R100?**

Mean paired difference -0.01884543; seed differences: {7: -0.0037933718485740897, 17: -0.06010318415983629, 29: 0.007360277406450899}.

6. **Does it beat both in all three seeds?**

False

7. **Which seeds disagree?**

[7, 29]

8. **Is the optimum sharp or broad?**

Observed local basin [40, 50, 60, 80, 100]; inspect local_uncertainty.csv for raw differences and paired seed SD. No significance claim or unsampled-age precision.

9. **Does immediate pruning shock show the same optimum?**

Minimum mean shock R100; matches candidate: False. Shock is measured on matched rewound E0/shared, age-router initial states.

10. **Does early recovery show the same optimum?**

Minimum matched sparse−dense early AUC at R0; matches: False. Individual sparse/dense AUCs are retained.

11. **Is the result driven by sparse loss or matched recoverability?**

The selection uses matched ticket gap only. Sparse-loss-only winner is R60; matched winner is R50. These descriptive results do not identify a causal optimization mechanism.

12. **How do developmental statistics evolve?**

Age means are tabulated below. Entropy uses natural logs, margin uses probabilities, load CV uses population SD per layer, specialization uses historical token-ID/expert MI. Assignment/JS maturity uses the historical first-batch-per-layer implementation. Exposure is sampled logged assignments, not complete accepted updates.

13. **Which 24E checkpoint is closest to 8E R20?**

R40, by mean historical-range-normalized distance; individual seed distances retained.

14. **Which is closest to 12E R60?**

R80.

15. **Is the ticket optimum developmentally close to those optima?**

It coincides with the closest sampled checkpoint to 8E R20: False; to 12E R60: False. Distance is descriptive; nearest does not establish a common developmental mechanism.

16. **Does increasing expert count shift the optimum later?**

The observed mean sequence is 8E R20 → 12E R60 → 24E R50. A stable later interior optimum is supported descriptively: False. Three architectures do not establish a scaling law.

17. **What architecture-dependence claim is justified?**

The measured matched ticket-gap curves support 24e optimum seed-conditioned. Only these model sizes, training budget, data, sparsity and three seeds are covered; developmental similarity is secondary.

18. **Should 24E enter the next gate-amplitude causal experiment?**

No stable intermediate target recommended. Any future experiment requires a separate instruction.

Age summary (mean and sample SD across independent seeds):

```text
 experts  router_age  num_seeds  mean_pruning_shock  mean_sparse_final_loss  sd_sparse_final_loss  mean_dense_final_loss  sd_dense_final_loss  mean_ticket_gap  sd_ticket_gap
      24           0          3           -0.762413                1.636573              0.014039               1.477035             0.039795         0.159538       0.028153
      24          10          3           -1.761753                1.619680              0.021972               1.451531             0.037112         0.168149       0.019793
      24          20          3           -1.809436                1.604245              0.026168               1.447029             0.040188         0.157216       0.024386
      24          30          3           -1.830590                1.605008              0.009170               1.437010             0.039242         0.167997       0.031674
      24          40          3           -1.880983                1.580462              0.017043               1.420391             0.044274         0.160072       0.048763
      24          50          3           -1.964552                1.581680              0.016806               1.424583             0.034273         0.157097       0.032483
      24          60          3           -2.022736                1.572186              0.010980               1.399516             0.006193         0.172670       0.012172
      24          80          3           -2.164175                1.577699              0.004793               1.392944             0.007151         0.184755       0.008862
      24         100          3           -2.326984                1.574554              0.004508               1.398612             0.004437         0.175942       0.004917
```

Seed winners:

```text
 seed  best_router_age  best_ticket_gap  second_best_router_age  second_best_ticket_gap  difference
    7               20         0.134190                       0                0.146401    0.012211
   17               40         0.104773                      50                0.120755    0.015982
   29              100         0.175944                      80                0.176884    0.000940
```

Developmental age means:

```text
             entropy    margin   load_cv  token_specialization  assignment_agreement_to_r100  js_distance_to_r100  cumulative_expert_exposure
router_age                                                                                                                                   
0           3.024578  0.026258  0.530061              0.982468                      0.092316             0.177323                    0.000000
10          3.091661  0.020008  0.440413              1.543582                      0.230962             0.126764                 7509.333333
20          2.971158  0.036540  0.322665              1.719605                      0.275050             0.103844                14336.000000
30          2.841861  0.069628  0.334249              1.647296                      0.336304             0.081111                21162.666667
40          2.740783  0.105633  0.292829              1.553581                      0.415634             0.061953                27989.333333
50          2.668846  0.133193  0.287427              1.465569                      0.492778             0.047325                34816.000000
60          2.616417  0.150612  0.251819              1.423979                      0.571472             0.033939                41642.666667
80          2.511312  0.178708  0.231431              1.353130                      0.727931             0.010275                55296.000000
100         2.404041  0.207433  0.210485              1.323095                      1.000000             0.000000                68949.333333
```

All primary readouts are native, with matched router-age dense recoveries. No replay interventions were run. Historical 8E/12E artifacts were reused without retraining. Logged exposure and first-batch maturity limitations preserve comparability but constrain developmental interpretation. No token, layer, expert, checkpoint or recovery step is treated as an independent replicate.

24E OPTIMUM SEED-CONDITIONED

Recommended 24E target checkpoint for future mechanism experiments: No stable intermediate target recommended

Recommended control checkpoints: R0 and R100; neighboring basin checkpoints if a future protocol requires local controls.

Confidence based on seed consistency: Limited: seed disagreement or boundary optimum prevents a stable intermediate target.
