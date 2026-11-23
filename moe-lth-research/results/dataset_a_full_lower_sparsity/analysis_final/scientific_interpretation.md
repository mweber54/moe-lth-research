# Numerical interpretation

These conclusions use all five seeds and the validated paired endpoint tables. Marginal confidence intervals average repeated ages/sparsities within seed first (five source-seed means), avoiding 1,100-run pseudoreplication.

50–65% expert sparsity sometimes matches or beats dense (212 strict endpoints) but does not do so across all five seeds at any of the 220 settings. Every architecture's average gap is positive. Strict matches are distributed across seed IDs as follows:

```
seed
7     30
17    59
29    79
43    35
61     9
```

The largest seed-ID contribution is 79/212 strict matches (37.3%), from seed 29; matches span 23/25 unique source trajectories. These counts quantify concentration across correlated runs. 13 settings have negative mean gaps; 5 remain negative after deleting any one seed, while 8 reverse sign after deleting at least one seed. The strongest selected mean settings are:

```
 num_experts  router_age  sparsity  mean_gap   sd_gap  upper95_one_sided  strict_passing_seeds  strict_p_one_sided  strict_p_holm  max_leave_one_seed_out_mean
          16          60      0.50 -0.015776 0.038369           0.020804                     2            0.204958            1.0                    -0.001352
          16          60      0.55 -0.010486 0.042717           0.030240                     2            0.306132            1.0                     0.006026
          12           0      0.50 -0.009289 0.032668           0.021857                     4            0.279737            1.0                    -0.001145
          24           0      0.50 -0.008938 0.015625           0.005959                     4            0.135011            1.0                    -0.003783
          32           0      0.50 -0.008083 0.029626           0.020162                     3            0.287379            1.0                     0.001785
```

Architecture and sparsity marginal summaries (pointwise t intervals, n=5 per group):

```
   grouping   group  n  mean_gap  ci95_lower  ci95_upper  passing_strict  denominator
num_experts    (8,)  5  0.017771    0.001416    0.034125              46          220
num_experts   (12,)  5  0.019302   -0.004067    0.042672              55          220
num_experts   (16,)  5  0.013849   -0.005541    0.033238              58          220
num_experts   (24,)  5  0.019899    0.004306    0.035492              43          220
num_experts   (32,)  5  0.033126    0.016548    0.049703              10          220
   sparsity  (0.5,)  5  0.010727   -0.002246    0.023701              88          275
   sparsity (0.55,)  5  0.015728    0.002561    0.028896              64          275
   sparsity  (0.6,)  5  0.022869    0.010612    0.035127              39          275
   sparsity (0.65,)  5  0.033832    0.019893    0.047772              21          275
```

Expert-count effects are not monotonic: 16E has the smallest average gap, 32E the largest. This is a comparison of the tested architectures under equal global attempted budgets, not equal per-expert exposure or a causal expert-count mechanism.

R100 versus R0, averaging the four sparsities within seed, decomposes as:

```
 num_experts  mean_R100_minus_R0_gap  ci95_lower  ci95_upper  mean_sparse_change  mean_dense_change  seeds_R100_better        p  p_holm_5
           8               -0.001807   -0.048634    0.045021           -0.066254          -0.064447                  2 0.919840  1.000000
          12                0.012648   -0.027369    0.052666           -0.070646          -0.083295                  2 0.429733  1.000000
          16               -0.026205   -0.060222    0.007812           -0.098744          -0.072539                  4 0.099215  0.496074
          24                0.004760   -0.021023    0.030543           -0.062516          -0.067276                  3 0.635253  1.000000
          32                0.012118   -0.011934    0.036170           -0.087574          -0.099692                  2 0.234425  0.937701
```

A negative gap change means R100 improves relative sparse performance; a negative sparse change alone means better absolute sparse CE. Relative changes must be read with dense changes. The separate five-architecture Holm family above applies only to these fixed marginal boundary comparisons; it does not validate selected interior ages.

Interior minima of the mean occur in 9/20 curves, but only 0 beat both endpoints after Holm over all 1,100 age comparisons. 0/20 mean-winning ages are selected by all five seeds. Mean curve shapes are {'nonmonotonic': 20}; nonmonotonic sample means alone are not reproducible optima. A curve can be statistically unresolved without being shown equivalent or truly flat. Seed rank trends and selected age-versus-boundary sparse/dense decompositions are in age_shape_summary.csv; all seed views are plotted. Lowest absolute sparse CE and lowest gap occur at different mean ages in 18/20 curves.

At 65% versus 50%, 55/55 architecture/age settings have worse mean gaps; 40/55 show this in all five seeds. 4/55 differences survive the separate two-sided Holm family. Full paired contrasts are in paired_65_vs_50.csv. Adjacent sparsity contrasts additionally connect all ten measured sparsities.

Original higher-sparsity mean-gap inventory:

```
              mean       min       max
sparsity                              
0.70      0.050441 -0.024879  0.126467
0.75      0.080511  0.004818  0.183692
0.80      0.130644  0.049310  0.258854
0.85      0.197310  0.104773  0.313751
0.90      0.284732  0.174640  0.396829
0.95      0.409374  0.265839  0.505268
```

No strict all-five-seed match exists in the lower grid. In the combined grid, strict all-five matches: 0/550. Highest observed all-five 5% candidate sparsities by architecture are descriptive maxima over measured cells:

```
              min   max  count
num_experts                   
8            0.50  0.65      9
12           0.55  0.60      9
16           0.60  0.65     10
24           0.50  0.65     10
32           0.50  0.60      5
```

The count column is the number of router ages with any qualifying tested sparsity. The min/max are over those ages, not an interpolated sparsity threshold. highest_observed_sparsity.csv gives the exact architecture/age criterion values, including individual seeds and 3/4/5 seed replication, and adjusted inference. A setting selected by its best observed age is subject to selection bias.

The three settings surviving the joint 880-test Holm family for the exploratory 5% tolerance are 8E/R70/50%, 8E/R70/55%, and 8E/R80/50%. Their mean gaps are −0.002257, −0.001465, and +0.005077 nats/token; adjusted p values are 0.040157, 0.030503, and 0.022485. These support mean performance within the stated exploratory tolerance, not strict superiority. The most negative selected mean, 16E/R60/50%, is −0.015776 with SD 0.038369 and one-sided upper95 +0.020804; only 2/5 seeds strictly match, so it is not a replicated strict ticket.

The smallest warranted follow-up is fixed-setting replication on fresh source seeds and an independent held-out corpus. Prioritize settings with all-five 5% matches and adjusted mean-below-5% evidence; strict negative means have weaker support. Test an interior candidate against both fixed endpoints only if its multiplicity-adjusted contrasts justify it. Do not expand to a full new grid or infer mechanisms from these endpoints.
