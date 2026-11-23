# 12-Expert Router-Development Review

## Executive Summary

Protocol validity is supported by the recorded metadata: 12 experts, top-1 routing, four MoE layers, `d_model=256`, expert hidden size 1024, 2500-step recovery, trainable routers, E_T magnitude masks, E_0 rewind, and matched dense controls. The retry contains all 126 sparse conditions and all 21 dense controls. Of the 147 matched rows, 36 were reused from the compatible e12 replication and 111 were newly run; no failed or missing condition is recorded.

Mean ticket-gap optima are R100 at 60% (0.0064), R60 at 70% (0.0331), R60 at 75% (0.0675), R60 at 85% (0.1847), R60 at 90% (0.2856), and R0 at 95% (0.4099). The strongest age effect is at 85%: R60 is 0.0414 below R0 and 0.0059 below R100. Thus R100 was not genuinely optimal at 85%, although the R60 versus R100 difference is small relative to seed variation. A shifted intermediate optimum exists in the mean, especially from 70% through 90%, but seed consistency is weak to moderate.

The available probe measurements show a smooth maturation trajectory: entropy falls, confidence/margin rise, utilization entropy falls slightly, utilization CV rises, and normalized token-ID specialization rises. Assignment stability, JS movement, adjacent JS, and cumulative routed-token exposure were not stored in the retry artifacts, so developmental-state matching and exposure explanations cannot be tested. The evidence therefore supports a shifted intermediate optimum in this 12-expert sweep, not a common cross-architecture developmental state or a causal mechanism.

**Most important next experiment:** rerun the same fixed-probe 8- and 12-expert comparison while recording assignment agreement, JS/TV movement, and cumulative routed-token exposure at every checkpoint, with enough seeds to distinguish the R60 effect from seed noise.

## 1. Reconstruction and Completeness

The intended grid is 6 sparsities x 7 ages x 3 seeds = 126 sparse cells and 7 ages x 3 seeds = 21 dense controls. Observed:

| Item | Expected | Observed |
|---|---:|---:|
| Sparse conditions | 126 | 126 |
| Dense controls | 21 | 21 |
| Matched rows | 147 | 147 |
| Seeds | 7, 17, 29 | all three |
| Sparsities | 60, 70, 75, 85, 90, 95% | all six |
| Ages | R0, R10, R20, R40, R60, R80, R100 | all seven |
| Failed/missing rows | 0 | 0 recorded |

`matched_runs.csv` reports 111 newly run rows and 36 rows reused from `expert_count_replication/e12`. The dense table similarly contains 12 new and 9 reused rows. Reuse is explicit, not inferred from filenames.

## 2. Protocol Audit

The preflight metadata in `preflight_audit.md` specifies 12 experts, top-k=1, four MoE layers, `d_model=256`, expert hidden size 1024, capacity factor 1.25, learned linear routers, and auxiliary loss 0.1. `checkpoint_manifest.csv` contains 21 entries: each seed has R0/R10/R20/R40/R60/R80/R100 at steps 0/250/500/1000/1500/2000/2500. Checkpoint hashes are distinct across requested ages and seeds.

The matched records report `recovery_steps=2500`, `router_mode=trainable`, fresh AdamW, and the same recovery/evaluation protocol. Shared-state, training-batch, validation-batch, and probe hashes are repeated across the matched conditions. Sparse records use `pruning_method=expert_local_magnitude`, a mask hash, and the reference path; the preflight states masks were discovered from E_T and surviving values rewound from E_0. The same seed+sparsity mask is used across ages. Recorded integrity checks pass. These records support, but do not independently prove every tensor-level invariant; no raw mask-tensor audit was performed here.

## 3. Ticket-Gap Results

Values are means across seeds; SD is the sample SD across the three seed-level gaps.

### Mean gap

| Sparsity | R0 | R10 | R20 | R40 | R60 | R80 | R100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60% | .0074 | .0158 | .0202 | .0204 | .0066 | .0178 | **.0064** |
| 70% | .0600 | .0509 | .0656 | .0486 | **.0331** | .0507 | .0403 |
| 75% | .1042 | .0981 | .1005 | .0716 | **.0675** | .0926 | .0916 |
| 85% | .2260 | .2377 | .2266 | .2017 | **.1847** | .2075 | .1906 |
| 90% | .2973 | .3387 | .3306 | .3135 | **.2856** | .3056 | .2917 |
| 95% | **.4099** | .4601 | .4558 | .4367 | .4272 | .4536 | .4415 |

### SD

| Sparsity | R0 | R10 | R20 | R40 | R60 | R80 | R100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60% | .0202 | .0268 | .0333 | .0032 | .0254 | .0292 | .0153 |
| 70% | .0444 | .0391 | .0274 | .0204 | .0326 | .0246 | .0170 |
| 75% | .0532 | .0527 | .0433 | .0174 | .0372 | .0259 | .0278 |
| 85% | .0504 | .0470 | .0570 | .0260 | .0245 | .0430 | .0378 |
| 90% | .0654 | .0215 | .0422 | .0198 | .0222 | .0132 | .0061 |
| 95% | .0604 | .0262 | .0163 | .0219 | .0165 | .0286 | .0263 |

### Seed-level gaps

Columns are R0, R10, R20, R40, R60, R80, R100.

| Sparsity | Seed | Gaps |
|---|---:|---|
| 60% | 7 | .0073, .0392, .0500, .0231, .0352, .0503, .0239 |
| 60% | 17 | .0276, -.0134, -.0158, .0168, -.0134, .0096, -.0010 |
| 60% | 29 | -.0128, .0216, .0264, .0212, -.0019, -.0064, -.0038 |
| 70% | 7 | .0245, .0876, .0967, .0722, .0707, .0722, .0592 |
| 70% | 17 | .1098, .0098, .0453, .0371, .0144, .0561, .0264 |
| 70% | 29 | .0456, .0551, .0547, .0364, .0142, .0239, .0351 |
| 75% | 7 | .0639, .1425, .1482, .0914, .1105, .1072, .1105 |
| 75% | 17 | .1645, .0398, .0638, .0590, .0452, .1079, .1047 |
| 75% | 29 | .0841, .1121, .0895, .0644, .0469, .0626, .0597 |
| 85% | 7 | .1736, .2917, .2809, .2186, .2077, .2313, .2332 |
| 85% | 17 | .2740, .2149, .2317, .2148, .1874, .2333, .1775 |
| 85% | 29 | .2305, .2064, .1672, .1718, .1589, .1579, .1610 |
| 90% | 7 | .2272, .3560, .3698, .3081, .3070, .3086, .2944 |
| 90% | 17 | .3566, .3147, .3361, .3354, .2873, .3170, .2959 |
| 90% | 29 | .3082, .3454, .2860, .2970, .2626, .2912, .2847 |
| 95% | 7 | .3436, .4701, .4478, .4147, .4416, .4602, .4451 |
| 95% | 17 | .4618, .4304, .4745, .4585, .4307, .4784, .4659 |
| 95% | 29 | .4244, .4798, .4451, .4371, .4092, .4222, .4137 |

Per-seed best ages are: 60% R0/R20/R0 for seeds 7/17/29; 70% R0/R10/R60; 75% R0/R10/R60; 85% R0/R100/R80; 90% R0/R60/R60; and 95% R0/R10/R60. R60 is therefore consistent for seeds 17 and 29 at 90%, but not at 85%; seed 7 favors R0 at every sparsity. The mean R60 result is not a three-seed consensus.

## 4. Response Shape and Architecture Comparison

The age responses are irregular at 60% and 95%, have an intermediate minimum at 70%, 75%, 85%, and 90%, and are not monotonic. They do not show a clean U-shape at every sparsity. At 85%, R60 (.1847) beats R100 (.1906), so the earlier three-point conclusion that R100 was best was a sampling artifact, but the margin is modest.

The compatible 8/12 comparison is:

| Experts | Sparsity | Best sampled age | Gap |
|---:|---:|---|---:|
| 8 | 75% | R20 | .0429 |
| 8 | 85% | R20 | .1673 |
| 8 | 95% | R20/R0 nearly tied | .4148 |
| 12 | 75% | R60 | .0675 |
| 12 | 85% | R60 | .1847 |
| 12 | 95% | R0 | .4099 |

Chronological optima shift later in the 12-expert sweep at 75% and 85%. Equivalent normalized developmental measurements for the 8-expert R20 state were not available, so the stronger claim that 12-expert R60 matches the functional state of 8-expert R20 is unsupported.

## 5. Router Development

The four developmental CSVs contain the same 84 rows (four layers x seven ages x three seeds) and the same metric columns. Across layers and seeds, the available trajectory is:

| Age | H/log(12) | Top-1 conf. | Margin | Util. entropy | Util. CV | Norm. token-ID MI |
|---|---:|---:|---:|---:|---:|---:|
| R0 | .9410 | .1892 | .0509 | 2.3981 | .4233 | .3136 |
| R10 | .8566 | .3018 | .1615 | 2.2989 | .6363 | .5505 |
| R20 | .8444 | .3113 | .1700 | 2.3068 | .6274 | .5596 |
| R40 | .8125 | .3390 | .1984 | 2.2961 | .6365 | .5739 |
| R60 | .7739 | .3698 | .2310 | 2.2904 | .6390 | .5832 |
| R80 | .7451 | .3900 | .2484 | 2.2881 | .6443 | .5881 |
| R100 | .7247 | .4051 | .2617 | 2.2846 | .6485 | .5897 |

The largest changes occur from R0 to R10: entropy drops sharply, confidence and margin rise sharply, specialization rises, and utilization becomes less uniform. From R10 onward entropy continues falling and confidence/margin continue rising, while specialization increases more slowly. No inactive experts are recorded. Higher utilization CV and lower utilization entropy indicate increasingly organized, less uniform load rather than collapse.

The probe-facing measurements are suitable for descriptive comparison: the tables contain common seed and age coverage, and the matched records contain a common `probe_sequence_hash`. Raw semantic token categories were not recorded, so token-ID MI is the only specialization measure and should not be treated as semantic specialization.

No assignment-agreement trajectory, adjacent-checkpoint JS/TV table, distance-to-R100 table, or cumulative routed-token-per-expert history exists in this result directory. Consequently, it is impossible to determine whether R60 is partially stabilized, whether functional movement changes regime there, or whether routed-token exposure explains the chronological shift. These are missing measurements, not negative findings.

## 6. Gap Versus Developmental State

The age curves show that favorable recovery at 70–90% occurs while entropy is intermediate-to-low, margin and confidence are intermediate-to-high, and specialization is already near its mature plateau. This is compatible with an intermediate developmental state, but it does not identify which statistic matters. The same statistics continue changing after R60, while the gap often worsens again; correlations are therefore descriptive and cannot establish mechanism.

The requested separate analyses against normalized entropy, margin, utilization CV, utilization entropy, specialization, assignment stability, JS-to-R100, and adjacent JS cannot all be performed. The first five are available in `router_development_vectors.csv`; the latter three are absent. No post-hoc scalar developmental score is constructed.

## 7. Hypothesis Assessment

* **H1, fixed R20:** disfavored. R20 is not optimal in the dense sweep, and R60 wins the mean at four sparsities.
* **H2, shifted intermediate optimum:** supported at the mean level, especially 70–90%, with the important qualification that seed 7 repeatedly favors R0.
* **H3, common functional developmental state:** untested/unsupported because equivalent 8-expert probe metrics and stability/movement measures are absent.
* **H4, mature-router advantage:** not supported at 70–90%; R100 is not best at 85%.
* **H5, sparsity-dependent optimum:** supported descriptively. The mean best age is R100, R60, R60, R60, R60, R0 across the six sparsities, and seed optima vary substantially.
* **H6, capacity-limited regime:** supported descriptively. At 95%, gaps remain large (.410-.460), and router age does not rescue the ticket.
* **H7, no simple metric:** remains plausible. Available metrics evolve smoothly and no recorded analysis establishes a consistent predictor, especially across seeds.

## 8. Limits and Figure Status

No analysis figures were present in the retry directory. The complete numerical matrices and seed-level table above preserve the disagreement that a smoothed plot could hide. The requested stability/exposure plots cannot be regenerated from absent measurements; no existing summary was silently substituted for them.

The strongest justified conclusion is that the 12-expert architecture contains a sparsity-dependent, seed-sensitive intermediate recovery regime whose mean optimum is commonly R60. It is not justified to call R20 intrinsically special, to claim a universal developmental state across architectures, to claim causality for entropy/specialization/load, or to claim that 12-expert routers develop slower solely because there are more experts.

**Primary verdict: SHIFTED INTERMEDIATE OPTIMUM SUPPORTED**