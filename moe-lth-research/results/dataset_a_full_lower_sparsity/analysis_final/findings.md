# Dataset A lower-sparsity findings

1. Validation **PASS**: 0 failed checks; exact 1,100/1,100 sparse coverage and 275 unique matched dense endpoints. Dense verification is table-backed; historical raw dense checkpoints are not claimed verified.
2. Strict endpoint matches: **212/1,100**.
3. Strict matches in all five seeds: **0/220 settings**.
4. Threshold sensitivity (1/2/5% worse perplexity, exploratory):

```
threshold  seed_matches  seed_denominator  all_five  at_least_four  at_least_three  setting_denominator  upper95_below  holm_220  holm_joint_880
   strict           212              1100         0              2              12                  220              0         0               0
 ppl_1pct           354              1100         3             12              50                  220              3         0               0
 ppl_2pct           528              1100        13             55             100                  220             14         0               0
 ppl_5pct           960              1100       122            184             215                  220            146         5               3
```

5. Unadjusted one-sided 95% upper-bound counts, Holm within each 220-setting threshold family and stricter joint 880-test family are above. These quantify mean-gap evidence under seed-level t assumptions; descriptive endpoint matches are not equivalence discoveries.
6. This measures **router-conditioned dense-matching sparse recovery**, or lottery-ticket candidates. Trained router checkpoints and source-final learned mask selection differ from initialization-only original LTH. No endpoint-only result establishes geometry, causal mechanism, equal expert exposure or theoretical novelty.

## Supported findings and practical limits

All 9,657 main audit checks passed, and all 40 random-control raw audits passed. Original higher-sparsity compatibility and the combined unique grid passed. The standalone output verifier separately confirms the endpoint arithmetic, counts and paired statistics.

Strict seed matches decline from 88/275 at 50% sparsity to 64/275 at 55%, 39/275 at 60%, and 21/275 at 65%. Mean 65%-minus-50% gaps are positive in all 55 architecture/age settings; 40/55 have this direction in every seed, and 4/55 paired differences survive Holm. This is evidence that increasing sparsity generally worsens the matched gap in the observed grid, without assuming monotonicity in every individual run.

No setting has a one-sided 95% upper bound below zero, and no strict mean-matching claim survives adjustment. The exploratory 5% tolerance has 122/220 settings passing descriptively in all five seeds; five mean-below-tolerance tests survive Holm over 220 settings and three survive joint Holm over all 880 setting/tolerance tests. Those three are **8E/R70/50%, 8E/R70/55%, and 8E/R80/50%**. The 1% and 2% tolerance tests yield no adjusted discoveries.

No age-pair contrast survives Holm over 1,100 tests. Lowest mean sparse CE and smallest mean gap occur at different ages in 18/20 architecture/sparsity curves. The observed mean interior minima are exploratory and not reproducible optima established by these five seeds. Failure to resolve an age contrast does not establish a flat population response.

The separate 32E random controls show learned-mask advantages of 0.0590–0.0927 nats/token, with all 8/8 paired settings surviving Holm. None of the 40 random runs strictly matches dense. This evidence is limited to 32E, R0/R100 and 50–65% sparsity.

Dense endpoints are verified against their finalized table and explicit reference provenance; this analysis does not claim new raw dense checkpoint verification. Original higher-sparsity rows retain their prior finalized-table provenance. Recovery attempts are fixed at 2,500, while observed applied updates range 2,496–2,499. Trained routers and source-final masks make these router-conditioned recovery candidates rather than initialization-only LTH.

## Observations

Architecture averages across the balanced observed grid (descriptive; 220 correlated runs per architecture):

```
                 mean       min       max
num_experts                              
8            0.017771 -0.061887  0.094229
12           0.019302 -0.041864  0.096811
16           0.013849 -0.076535  0.109001
24           0.019899 -0.029557  0.099228
32           0.033126 -0.047558  0.100792
Strict matches by architecture:
num_experts
8     46
12    55
16    58
24    43
32    10
```

Sparsity averages (descriptive):

```
              mean       min       max
sparsity                              
0.50      0.010727 -0.073470  0.071910
0.55      0.015728 -0.076535  0.109001
0.60      0.022869 -0.054972  0.099208
0.65      0.033832 -0.048926  0.100792
```

Mean-gap winning ages, absolute sparse-loss winning ages and five-seed winner stability:

```
 num_experts  sparsity  best_gap_age  best_sparse_age  mean_best_gap                                   seed_winner_ages  seeds_selecting_mean_winner leave_one_out_winner_ages
           8      0.50            30               70      -0.002995    {"7": 30, "17": 50, "29": 30, "43": 0, "61": 0}                            2      [70, 30, 70, 20, 30]
           8      0.55            70               70      -0.001465    {"7": 20, "17": 70, "29": 20, "43": 0, "61": 0}                            1      [70, 30, 70, 20, 70]
           8      0.60            70               60       0.013523   {"7": 20, "17": 60, "29": 10, "43": 0, "61": 90}                            0      [90, 30, 70, 20, 70]
           8      0.65            70               70       0.016260    {"7": 20, "17": 70, "29": 20, "43": 0, "61": 0}                            1      [70, 20, 70, 20, 70]
          12      0.50             0               80      -0.009289    {"7": 0, "17": 50, "29": 80, "43": 80, "61": 0}                            2           [0, 0, 0, 0, 0]
          12      0.55             0              100      -0.007712     {"7": 0, "17": 20, "29": 0, "43": 80, "61": 0}                            3           [0, 0, 0, 0, 0]
          12      0.60             0              100       0.004956     {"7": 0, "17": 20, "29": 0, "43": 80, "61": 0}                            3           [0, 0, 0, 0, 0]
          12      0.65             0              100       0.026530  {"7": 0, "17": 10, "29": 100, "43": 60, "61": 50}                            1          [60, 0, 0, 0, 0]
          16      0.50            60              100      -0.015776  {"7": 100, "17": 60, "29": 60, "43": 0, "61": 80}                            2      [60, 80, 60, 60, 60]
          16      0.55            60              100      -0.010486   {"7": 50, "17": 60, "29": 60, "43": 0, "61": 80}                            2      [60, 80, 60, 60, 60]
          16      0.60            60              100      -0.002767  {"7": 50, "17": 60, "29": 70, "43": 20, "61": 80}                            1      [60, 70, 60, 60, 60]
          16      0.65            60              100       0.002358  {"7": 50, "17": 60, "29": 60, "43": 80, "61": 70}                            2      [60, 70, 60, 60, 60]
          24      0.50             0               80      -0.008938   {"7": 0, "17": 50, "29": 0, "43": 100, "61": 80}                            2           [0, 0, 0, 0, 0]
          24      0.55             0               70      -0.000354 {"7": 0, "17": 40, "29": 100, "43": 100, "61": 10}                            1           [0, 0, 0, 0, 0]
          24      0.60             0               70       0.006788  {"7": 0, "17": 40, "29": 20, "43": 100, "61": 10}                            1         [40, 0, 0, 0, 20]
          24      0.65            20               90       0.024600  {"7": 0, "17": 20, "29": 10, "43": 100, "61": 90}                            1       [40, 0, 20, 20, 20]
          32      0.50             0              100      -0.008083    {"7": 30, "17": 0, "29": 100, "43": 0, "61": 0}                            3           [0, 0, 0, 0, 0]
          32      0.55             0              100       0.012009  {"7": 30, "17": 100, "29": 0, "43": 100, "61": 0}                            2         [0, 0, 0, 0, 100]
          32      0.60             0               60       0.013503    {"7": 0, "17": 100, "29": 60, "43": 0, "61": 0}                            3           [0, 0, 0, 0, 0]
          32      0.65             0               60       0.026852    {"7": 0, "17": 100, "29": 60, "43": 0, "61": 0}                            3           [0, 0, 0, 0, 0]
```

Absolute-loss and gap mean winners disagree in 18/20 architecture/sparsity curves. Interior mean-gap winners occur in 9/20 curves; selecting these ages on these same results is exploratory. All seed curves, leave-one-seed-out winning ages and candidate-versus-boundary comparisons are retained. An interior minimum of the mean alone does not establish a reproducible interior optimum.

Original 70–95% sweep has 7/1,650 strict seed matches. The compatible combined grid contains ten tested sparsities at each age; combined_setting_summary.csv and combined_gap_heatmaps show the connection. highest_observed_sparsity.csv reports maxima separately for every architecture/age, seed threshold, replicated count and inferential criterion. NA means no tested sparsity qualifies. No monotonicity assumption, interpolation or extrapolation is used.

## Statistical evidence and precision

Each setting uses five paired sparse-minus-dense gaps, sample SD (ddof=1), two-sided 95% t interval and one-sided 95% upper bound (df=4). Tests of mean gap below tolerance use p=t_CDF((mean−tolerance)/SE); rejection supports a mean below that tolerance under independent identically distributed source seeds and approximately normal paired gaps. Five seeds give low precision and weak diagnostics; seeds are not additional independent samples when reused across ages/sparsities. Dense endpoints are stored once and their repetition never increases n. Nonsignificance is not proof of equivalence.

paired_age_contrasts.csv contains all 55 within-seed age contrasts for each of 20 architecture/sparsity curves (1,100 tests), sparse and dense decomposition, five differences and two-sided intervals. Holm is applied over all 1,100 contrasts. 52 contrasts are unadjusted p<.05; 0 survive Holm. The sparse−dense decomposition distinguishes relative improvement from shared absolute improvement. paired_sparsity_contrasts.csv compares adjacent measured sparsities within seed, with a separate 495-test Holm family. These are exploratory post-sweep comparisons. Confidence intervals displayed are pointwise, not simultaneous; SD shading is dispersion, not a confidence band.

Mean-setting tests use Holm at alpha .05 for each threshold's 220 settings; joint Holm over all 880 tests is also provided. Combined-grid tests use separate 550-setting per-threshold and 2,200-test joint families. A strict seed-run count does not imply a supported negative population mean. Per-setting leave-one-seed-out mean ranges and selection stability expose single-seed influence. Inference concerns variation over these source seeds and this fixed validation corpus, not independent validation datasets or all possible architectures.

## Separate random-mask control evidence

40 earlier arm-A random conditions, limited to 32 experts, R0/R100 and 50–65% sparsity; validation pass count 40. Exact source/seed/router/data/budget/dense matching and per-tensor retained-count matching are checked. No other architecture or age is covered. Positive random−learned means learned masks do better:

```
 router_age  sparsity  n  mean_random_minus_learned  ci95_lower  ci95_upper        p  random_strict_matches   p_holm
        0.0      0.50  5                   0.068172    0.023240    0.113105 0.013559                      0 0.048790
        0.0      0.55  5                   0.060772    0.021946    0.099599 0.012198                      0 0.048790
        0.0      0.60  5                   0.071429    0.019250    0.123609 0.019092                      0 0.048790
        0.0      0.65  5                   0.080283    0.027547    0.133018 0.013405                      0 0.048790
      100.0      0.50  5                   0.059018    0.031845    0.086192 0.003812                      0 0.027665
      100.0      0.55  5                   0.067621    0.036620    0.098622 0.003752                      0 0.027665
      100.0      0.60  5                   0.082174    0.036623    0.127724 0.007444                      0 0.037221
      100.0      0.65  5                   0.092659    0.051112    0.134206 0.003458                      0 0.027665
```

Separate Holm family of eight paired tests. These controls cannot support full-grid random-mask generalization.

## Interpretation and smallest follow-ups

Endpoint evidence can support router-conditioned recovery where the descriptive and uncertainty criteria agree. Router-dependent differences may reflect optimization, routing or capacity effects, but endpoint CE cannot identify those mechanisms. A selected best checkpoint carries selection bias; confirm a small fixed set of strongest candidate settings against fixed boundary ages on fresh source seeds and an independent held-out evaluation corpus before making discoveries. If an interior optimum survives paired and multiplicity checks, test only that fixed age and its two neighboring measured ages plus boundaries; otherwise prioritize candidate replication. A learned-vs-random follow-up needs only the selected architecture/age/sparsity and exactly count-matched masks, rather than a new full sweep. No follow-up was launched; Dataset B remains deferred.

## Reproduction and artifacts

Run scripts/analyze.py from repository root as documented in its docstring. It refuses existing generated outputs; use a fresh analysis output tree for a rerun. Inputs are read-only and no training entrypoint is invoked. Exact plot inputs are plot_curve_data.csv, plot_heatmap_data.csv, plot_seed_data.csv and canonical_lower_sparsity.csv. Figures are PDF and PNG in figures/. Input hashes and per-condition checks are retained. See audit_report.md for protocol and provenance limitations.


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
