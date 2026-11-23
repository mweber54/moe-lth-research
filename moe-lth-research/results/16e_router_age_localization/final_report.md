# 16-expert router-age ticket localization

1. **Does 16E exhibit an intermediate router-age ticket optimum?**

16E OPTIMUM SEED-CONDITIONED

2. **What router age or basin minimizes mean ticket gap?**

Mean winner R60; local uncertainty basin [60, 70, 80], using the preregistered paired-SD descriptive rule.

3. **What is the mean ± seed SD?**

0.14844933 ± 0.04677394 (n=3 seeds).

4. **Does the optimum beat R0?**

Mean paired difference -0.05821921; negative favors candidate. Seed differences: {7: -0.006923337152539011, 17: -0.0744564804065247, 29: -0.0932778050318799}.

5. **Does the optimum beat R100?**

Mean paired difference -0.03394451; seed differences: {7: 0.024189327796279497, 17: -0.09397056055031769, 29: -0.032052288744584184}.

6. **Does it beat both in all three seeds?**

False

7. **Which seeds disagree?**

[7]

8. **Is the optimum sharp or broad?**

Observed local basin [60, 70, 80]; inspect local_uncertainty.csv for raw differences and paired seed SD. No significance claim or unsampled-age precision.

9. **Does immediate pruning shock show the same optimum?**

Minimum mean shock R100; matches candidate: False. Shock is measured on matched rewound E0/shared, age-router initial states.

10. **Does early recovery show the same optimum?**

Minimum matched sparse−dense early AUC at R0; matches: False. Individual sparse/dense AUCs are retained.

11. **Is the result driven by sparse loss or matched recoverability?**

The selection uses matched ticket gap only. Sparse-loss-only winner is R100; matched winner is R60. These descriptive results do not identify a causal optimization mechanism.

12. **How do developmental statistics evolve?**

Age means are tabulated below. Entropy uses natural logs, margin uses probabilities, load CV uses population SD per layer, specialization uses historical token-ID/expert MI. Assignment/JS maturity uses the historical first-batch-per-layer implementation. Exposure is sampled logged assignments, not complete accepted updates.

13. **Which 16E checkpoint is closest to 8E R20?**

R40, by mean historical-range-normalized distance; individual seed distances retained.

14. **Which is closest to 12E R60?**

R70.

15. **Is the ticket optimum developmentally close to those optima?**

It coincides with the closest sampled checkpoint to 8E R20: False; to 12E R60: False. Distance is descriptive; nearest does not establish a common developmental mechanism.

16. **Does increasing expert count shift the optimum later?**

The observed mean sequence is 8E R20 → 12E R60 → 16E R60. A stable later interior optimum is supported descriptively: False. Three architectures do not establish a scaling law.

17. **What architecture-dependence claim is justified?**

The measured matched ticket-gap curves support 16e optimum seed-conditioned. Only these model sizes, training budget, data, sparsity and three seeds are covered; developmental similarity is secondary.

18. **Should 16E enter the next gate-amplitude causal experiment?**

No stable intermediate target recommended. Any future experiment requires a separate instruction.

Age summary (mean and sample SD across independent seeds):

```text
 experts  router_age  num_seeds  mean_pruning_shock  mean_sparse_final_loss  sd_sparse_final_loss  mean_dense_final_loss  sd_dense_final_loss  mean_ticket_gap  sd_ticket_gap
      16           0          3           -0.788536                1.669072              0.016285               1.462403             0.029251         0.206669       0.020596
      16          20          3           -1.557931                1.627031              0.036660               1.415008             0.035277         0.212023       0.019923
      16          40          3           -1.597710                1.606352              0.026919               1.421446             0.037740         0.184906       0.022750
      16          50          3           -1.640865                1.599594              0.017763               1.422533             0.036131         0.177061       0.022025
      16          60          3           -1.695337                1.590626              0.027698               1.442177             0.058439         0.148449       0.046774
      16          70          3           -1.792226                1.584843              0.026096               1.416547             0.028118         0.168296       0.025738
      16          80          3           -1.864518                1.589151              0.028086               1.408993             0.020109         0.180158       0.030265
      16          90          3           -1.911142                1.592941              0.030695               1.410390             0.026080         0.182551       0.025578
      16         100          3           -1.932230                1.581196              0.040915               1.398802             0.021090         0.182394       0.020401
```

Seed winners:

```text
 seed  best_router_age  best_ticket_gap  second_best_router_age  second_best_ticket_gap  difference
    7              100         0.176710                      70                0.190970    0.014260
   17               60         0.111064                      50                0.154078    0.043014
   29               60         0.133385                      70                0.140319    0.006934
```

Developmental age means:

```text
             entropy    margin   load_cv  token_specialization  assignment_agreement_to_r100  js_distance_to_r100  cumulative_expert_exposure
router_age                                                                                                                                   
0           2.621556  0.039605  0.503783              0.873431                      0.126022             0.192765                         0.0
20          2.543708  0.064009  0.369681              1.573674                      0.337753             0.115000                     21504.0
40          2.337529  0.138537  0.261958              1.424748                      0.484573             0.074367                     41984.0
50          2.283677  0.159664  0.220624              1.341657                      0.558385             0.058872                     52224.0
60          2.224513  0.180598  0.221351              1.281105                      0.626531             0.040800                     62464.0
70          2.159467  0.202606  0.195589              1.260753                      0.707179             0.023710                     72704.0
80          2.095628  0.221866  0.191229              1.232284                      0.777593             0.011281                     82944.0
90          2.032171  0.240596  0.190382              1.220342                      0.847249             0.003697                     93184.0
100         1.972952  0.258952  0.188854              1.205281                      1.000000             0.000000                    103424.0
```

All primary readouts are native, with matched router-age dense recoveries. No replay interventions were run. Historical 8E/12E artifacts were reused without retraining. Logged exposure and first-batch maturity limitations preserve comparability but constrain developmental interpretation. No token, layer, expert, checkpoint or recovery step is treated as an independent replicate.

16E OPTIMUM SEED-CONDITIONED

Recommended 16E target checkpoint for future mechanism experiments: No stable intermediate target recommended

Recommended control checkpoints: R0 and R100; neighboring basin checkpoints if a future protocol requires local controls.

Confidence based on seed consistency: Limited: seed disagreement or boundary optimum prevents a stable intermediate target.
