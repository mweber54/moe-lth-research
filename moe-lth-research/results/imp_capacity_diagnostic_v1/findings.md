# IMP capacity diagnostic findings

## Scope and status

Checkpoint-only FP32 evaluation of the existing abrupt IMP80 shared-only trajectory. No training, optimizer updates, mask discovery, or new recovery branch was run. The same canonical validation set was examined repeatedly; it is not an independent test set. Source seed is the replication unit (n=5).

Positive `B` means disabling capacity rejection reduced that model's CE. Positive `K` means it reduced the sparse–unpruned gap. `J` is the difference in observed gap closure between native and no-drop evaluation, not a uniquely attributable mediation effect.

## Seed-level primary contrasts

| Seed | G native (0) | G no-drop (0) | B sparse (0) | B unpruned (0) | K (0) | G native (500) | G no-drop (500) | B sparse (500) | B unpruned (500) | K (500) | H native | H no-drop | J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.47043961 | 0.47839475 | 0.00622735 | 0.01418248 | -0.00795513 | 0.06797715 | 0.07632550 | 0.00555434 | 0.01390269 | -0.00834834 | 0.40246246 | 0.40206925 | 0.00039321 |
| 17 | 0.44962197 | 0.45263108 | 0.00976027 | 0.01276939 | -0.00300911 | 0.06701065 | 0.06581252 | 0.00960478 | 0.00840665 | 0.00119813 | 0.38261132 | 0.38681856 | -0.00420725 |
| 29 | 0.46598266 | 0.46718638 | 0.01102004 | 0.01222375 | -0.00120372 | 0.08176933 | 0.08601668 | 0.00828315 | 0.01253050 | -0.00424735 | 0.38421333 | 0.38116970 | 0.00304363 |
| 43 | 0.49483603 | 0.49692292 | 0.01380165 | 0.01588854 | -0.00208688 | 0.07589435 | 0.07750017 | 0.01219583 | 0.01380165 | -0.00160582 | 0.41894169 | 0.41942275 | -0.00048106 |
| 61 | 0.51272550 | 0.53698629 | 0.01263115 | 0.03689194 | -0.02426079 | 0.06281783 | 0.07379052 | 0.01760424 | 0.02857693 | -0.01097269 | 0.44990768 | 0.46319577 | -0.01328810 |

## Direct answers

A. Immediate gap: K(0) is positive in 0/5 seeds; K(0) values by seed are {"7": -0.00795513495021094, "17": -0.0030091143436514756, "29": -0.0012037152105661697, "43": -0.002086882390150979, "61": -0.02426079064114406}. `G_native(0)` versus `G_no_drop_all(0)` is shown seed by seed above.
B. Immediate model benefits: `B_sparse(0)` is positive in 5/5 seeds and `B_unpruned(0)` is positive in 5/5. Values are retained signed; no-drop may worsen CE.
C. Endpoint effect: K(500) is positive in 1/5 seeds; signed values are {"7": -0.008348342290485977, "17": 0.0011981314978934954, "29": -0.004247348675853857, "43": -0.0016058207561444604, "61": -0.010972694188029708}. `B_sparse(500)` positive in 5/5 and `B_unpruned(500)` positive in 5/5.
D. Layer-local K values are listed for every seed in `layer_local_contrasts.csv`. The largest by mean absolute start effect is L0 (mean |K|=0.00968503); this is descriptive, not a significance ranking. Per-layer signed values and sign counts (positive/negative out of five) are: L0 start={"7": -0.008733026832640878, "17": -0.005812979886050673, "29": -0.003571020984332929, "43": -0.005418166610705377, "61": -0.024889935028200583} (0/5), end={"7": -0.006889296278608237, "17": 0.0015678097217786124, "29": -0.0038652971683195148, "43": -0.000979676699709664, "61": -0.010617444002169352} (1/4); L1 start={"7": 2.215137981398385e-05, "17": 0.0031605865338635297, "29": 0.0026757286126009028, "43": 0.0010399437519343913, "61": -0.0013745732017409473} (4/1), end={"7": -0.0012056108813851552, "17": -2.336128356739131e-05, "29": -2.4922469823218663e-05, "43": -0.00031670660796767613, "61": -0.0002043554773072387} (0/5); L2 start={"7": 0.00023708379151576864, "17": 5.6727931112554586e-05, "29": -0.0007221242888122426, "43": 0.0004916546000628408, "61": 0.001911704417927762} (4/1), end={"7": -0.00019788888800298388, "17": -0.00018298887400169939, "29": -0.0002376985856720193, "43": -0.0003331862983524392, "61": 3.9114785519878126e-05} (1/4); L3 start={"7": 0.00014897384640155664, "17": -0.00028000928744864595, "29": 0.0007711007485156074, "43": 0.0019407892440459484, "61": -3.4538928646021816e-06} (3/2), end={"7": -0.00014799585535008042, "17": -0.00032525025058904866, "29": 1.3569874789620684e-05, "43": -8.914518006064576e-05, "61": -0.0001262980089242749} (1/4).
E. The no-drop gap is positive in 5/5 seeds at offset 0 and 5/5 at offset 500. `G_no_drop_all(0)` is mean 0.48642428, median 0.47839475, min 0.45263108, max 0.53698629, sample SD 0.03257027; at 500 it is mean 0.07588908, median 0.07632550, min 0.06581252, max 0.08601668, sample SD 0.00726800. Signed results are not clamped.
F. NO_DROP_ALL worsens sparse CE at offset 0 in 0/5 seeds and unpruned CE in 0/5; at offset 500 the counts are 0/5 and 0/5, respectively.
G. Capacity-policy sensitivity of observed gap closure J is {"7": 0.0003932073402750369, "17": -0.004207245841544971, "29": 0.0030436334652876873, "43": -0.0004810616340065188, "61": -0.01328809645311435}; mean -0.00290791, median -0.00048106, min -0.01328810, max 0.00304363, sample SD 0.00635697. Positive J means observed native gap closure is smaller when both checkpoints are evaluated without dropping.

## Routing and prediction diagnostics

`routing_capacity_metrics.csv` records batch/layer/expert counts. Layer-level load CV, capacity, maximum load, gate summaries, agreement, overlap, and coverage are repeated across expert rows with an explicit row-scope label; they are not independent expert replications. `prediction_diagnostics.csv` reports paired top-1 agreement, mean probability JS divergence, and native-rejected-position summaries. Conditional rejected-position subsets overlap across layers and are not additive.

## Interpretation boundary

A favorable result supports a capacity-policy contribution to checkpoint performance under this evaluation intervention. It does not establish why IMP discovers tickets, why sparse beats dense, why the observed behavior occurs, or that capacity-aware pruning improves training. Drop rates alone do not quantify performance impact. K and J are not mediation fractions and do not describe training without capacity limits.

## Display the share bundle

```sh
cat results/imp_capacity_diagnostic_v1/summary_bundle/absolute_losses.csv
cat results/imp_capacity_diagnostic_v1/summary_bundle/seed_level_capacity_contrasts.csv
cat results/imp_capacity_diagnostic_v1/summary_bundle/findings.md
```
