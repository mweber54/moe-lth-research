# Exploratory expert exposure and weight development

This retrospective analysis uses all five compatible original, unpruned 32E source trajectories. Cohort/grid/features were frozen after the coverage audit and before outcome associations. No training, recovery, IMP, pruning sweep, or new model forward/backward was run. Historical artifacts remain untouched.

## Individual-seed rank results

Primary joint fc1+fc2 ranks, step 250 → step 2500. Low means bottom50%; late-high means top 20%%. These are exact coordinate counts, not sampled coordinates.

| Source seed | Rank correlation | Early-low → late-top fraction | Later top 20% omitted by hypothetical early80% pruning | Absolute magnitude change among late risers |
|---|---:|---:|---:|---:|
| 7 | 0.709583 | 0.03521568 | 0.298242 | 0.04215862 |
| 17 | 0.730577 | 0.03168496 | 0.283807 | 0.04213994 |
| 29 | 0.713274 | 0.03495532 | 0.297884 | 0.04286747 |
| 43 | 0.688531 | 0.03838485 | 0.319188 | 0.04432538 |
| 61 | 0.731938 | 0.03164518 | 0.282640 | 0.04182161 |

Top15 sensitivity, within-tensor ranks, every expert/layer/interval, transition matrices, persistence and continuous rank changes are in the CSVs. Initialization step0 is explicitly flagged and excluded from the primary early-low interpretation. Additional early checkpoints for only three seeds are not substituted into five-seed comparisons.

## Stabilization across observed intervals

fc1:
- 250–500: seed-mean rank correlation range 0.972860–0.977473; top 20% turnover range 0.078537–0.087482.
- 500–1000: seed-mean rank correlation range 0.943770–0.951972; top 20% turnover range 0.121836–0.137501.
- 1000–1750: seed-mean rank correlation range 0.898555–0.918758; top 20% turnover range 0.175211–0.203256.
- 1750–2500: seed-mean rank correlation range 0.903699–0.918518; top 20% turnover range 0.171599–0.188797.
fc2:
- 250–500: seed-mean rank correlation range 0.935725–0.942688; top 20% turnover range 0.136558–0.146335.
- 500–1000: seed-mean rank correlation range 0.885390–0.902939; top 20% turnover range 0.184697–0.206632.
- 1000–1750: seed-mean rank correlation range 0.816695–0.845287; top 20% turnover range 0.243455–0.266870.
- 1750–2500: seed-mean rank correlation range 0.829264–0.856377; top 20% turnover range 0.221180–0.238581.
joint:
- 250–500: seed-mean rank correlation range 0.963054–0.967852; top 20% turnover range 0.062023–0.069118.
- 500–1000: seed-mean rank correlation range 0.928734–0.939302; top 20% turnover range 0.098568–0.111197.
- 1000–1750: seed-mean rank correlation range 0.878246–0.899442; top 20% turnover range 0.144275–0.167301.
- 1750–2500: seed-mean rank correlation range 0.882661–0.901394; top 20% turnover range 0.148876–0.165979.

Unequal interval lengths and update counts are explicit in expert_interval_metrics.csv; raw stability should not be compared as an equal-horizon learning rate. There is no unique stabilization time inferred between checkpoints. The earliest common nonzero checkpoint is250; behavior before that is limited.

## Exposure association and prediction

Quantity counts cover every attempted step and distinguish proposed, accepted, dropped, applied-update accepted exposure and skipped-update exposure. The probe feature is proposed-membership Jaccard turnover on saved validation routing. Accepted training-token composition, contextual hidden inputs, gate-weighted exposure, and genuine domain distributions are unavailable. Biased first-two context snippets are not treated as composition measurements.

Descriptive correlations compare experts within seed/layer/interval/ranking. Source seed is the replication unit. Same-interval correlations describe co-development, while predictive features stop at the outcome start. Every source seed is held out with all of its experts and intervals. One Ridge(alpha=1) model, train-only preprocessing, fixed features, and adjacent outcomes were used; no model/hyperparameter search.

| Held-out seed | Outcome mean | Baseline MSE | Exposure MSE | Relative MSE reduction |
|---|---:|---:|---:|---:|
| 7 | 0.00946604 | 7.60835444e-05 | 7.35443989e-05 | 3.337% |
| 17 | 0.00802294 | 5.73929049e-05 | 5.54327261e-05 | 3.415% |
| 29 | 0.00926367 | 7.27183298e-05 | 7.07480974e-05 | 2.709% |
| 43 | 0.01025199 | 8.7890903e-05 | 8.76776098e-05 | 0.243% |
| 61 | 0.00821725 | 6.87377199e-05 | 6.85118444e-05 | 0.329% |

Exposure improves held-out MSE in 5/5 source seeds. Mean relative MSE reduction: 2.007%; range 0.243%–3.415%. No source-seed significance or causal claim is made. The changing outcome base rate and seed-mean null MSE are reported explicitly.

Per-seed mean stratified Spearman associations with subsequent rank rise:

- prior_log_applied_rate: seed7=-0.0639, seed17=-0.1868, seed29=-0.1228, seed43=-0.0884, seed61=-0.1786
- prior_log_rate_change: seed7=0.0657, seed17=0.0021, seed29=0.0073, seed43=0.0073, seed61=0.1331
- prior_probe_turnover: seed7=-0.0913, seed17=0.0547, seed29=0.0928, seed43=-0.0187, seed61=0.0740

## Alternative explanations and limits

Initialization_scale_summary.csv records observed initialization magnitudes by seed/layer/projection. The model uses PyTorch Linear initialization: fc1 has fan-in 256 and fc2 fan-in 1024, so their initial uniform bounds differ by a factor of 2. Joint ranking therefore starts with different projection scales; within-tensor results help distinguish this from development. Initial mean absolute magnitude/initialization scale, current mean/low-half magnitude, prior magnitude-growth rate, prior late-rising fraction, prior top-set turnover, layer/tensor identity, global progress, interval duration, and prior applied-update count enter the baseline. fc1/fc2 and layers are reported separately; low/high prior utilization is compared within strata in utilization_strata.csv. Experts share exposure across coordinates: millions of weights are never treated as independent exposure replications. Expert IDs are not matched across seeds.

Both projections are pooled for the actual expert-local pruning budget. realized_survivor_fractions.csv shows the realized allocation by projection; nominal 80/85% expert sparsity is not imposed independently on fc1/fc2. Repository torch.topk has unspecified boundary tie membership; exact counts and all tied-boundary membership discrepancies are audited against the deterministic ordinal convention.

- Joint retained0.15, fc1: realized tensor survivor fraction range 0.201267–0.299999.
- Joint retained0.15, fc2: realized tensor survivor fraction range 0.000000–0.098732.
- Joint retained0.20, fc1: realized tensor survivor fraction range 0.263096–0.400002.
- Joint retained0.20, fc2: realized tensor survivor fraction range 0.000000–0.136906.

The primary source trajectory has no interior optimizer reset. Recovery resets at absolute step 2500 and IMP discovery rounds are inventoried as distinct excluded protocols. Weight decay was AdamW 0.01; absent optimizer states prevent isolating its contribution from gradients. Applied-update/AMP differences are retained. Fixed-probe changes may reflect shared-network learning as well as router development.

coordinate_examples.csv rereads exact named coordinates and illustrates rank rise with absolute growth versus rank rise without growth. These are illustrative outcome-selected examples, not independent evidence. Rank rise can result from other weights shrinking. Magnitude, late growth, and hypothetical mask membership do not establish functional importance. No final IMP mask is used as useful-weight ground truth.

An expert-level signal could identify unstable experts without identifying which coordinates to prune. Associations cannot establish the direction of causation. The eventual criterion is improved task loss/recovery after actual pruning, which this retrospective analysis does not measure.

## Required verdict

See the reviewed verdict below. All findings concern observed checkpoints and quantities; the missing historical input-composition measurements preclude a direct composition mechanism test.

**Reviewed verdict: Available data is insufficient.**

This verdict concerns evidence for a routing-aware pruning experiment, not coverage for the descriptive analysis. The five compatible source trajectories support exact rank description and a small held-out expert-level predictive comparison. They do not establish a useful coordinate-selection rule or a practically meaningful loss/recovery benefit. The exposure extension improves MSE in all five seeds, but two improvements are below 0.4%, and their importance for pruning is unknown. No prospective expansion is recommended or launched from this first pass.

1. **Do early magnitude ranks stabilize, and when?** No clear stabilization checkpoint is established. Ranks retain substantial structure but remain mobile through step 2500. In the two equal 750-attempt late intervals, joint top 20% turnover averages about15.48% and15.46%, respectively. Higher correlations over shorter horizons are not proof that learning has stabilized.
2. **How often do early-low weights become large?** For step 250 bottom-half to step 2500 top 20% joint ranks, the per-seed expert-mean fraction is 3.16%–3.84%. Within-tensor and top 15% sensitivity results are reported separately. Absolute growth accompanies the primary late risers; rank rise alone is not treated as growth or usefulness.
3. **Exposure quantity, composition, both, or neither?** Same-interval changes in accepted rate have positive mean stratified associations in all five seeds (roughly 0.15–0.25 Spearman); drop fraction also has positive same-interval associations. These are descriptive and can reflect confounding. Past accepted rate has negative mean associations with subsequent rise, while rate-change and fixed-probe associations are weak/mixed. Historical accepted contextual composition is unrecorded, so a composition mechanism is unanswerable; proposed fixed-probe membership is only a limited surrogate.
4. **Does exposure add information beyond current magnitude and prior growth?** A small observed increment: relative held-out MSE reduction averages 2.01%, range 0.24%–3.42%, with 5/5 positive signs. This is one fixed expert-level model and does not identify which individual coordinates to remove. It does not establish a robust functional pruning advantage.
5. **Which conclusions are coverage-limited?** Before the earliest common nonzero checkpoint 250, between sparse checkpoints, and beyond step 2500, five-seed dynamics are unresolved. Accepted token-ID/context distributions, hidden inputs, gate-weighted exposure, domain labels, and optimizer states are absent. Weight-decay mechanisms and causal direction cannot be isolated. Probe identity is supported by common data/configuration and batch shapes; the archived assignment files do not contain token hashes.
6. **Enough evidence for a prospective pruning experiment?** Not from this bounded pass alone. The result is a modest ranking-instability forecast without a validated pruning decision or task-level endpoint. No prospective comparison is implemented or launched. The eventual standard remains matched-sparsity, matched-budget loss/recovery improvement after actual pruning.
