# Expert-contribution ablation

This prospective diagnostic is motivated by previously examined results; it is not project-level preregistration. The validation set has been repeatedly examined and is not an independent test. Five source seeds are replication units. Complete contribution removal creates potentially out-of-distribution internal states. Lower damage does not prove dispensability; higher damage does not prove semantic specialization. Improvement under ZERO_ALL does not establish that training without experts would achieve it: experts were present throughout historical training. Shared-only recovery does not allocate causal percentages to embeddings, attention, normalization, routing or output head. The tied embedding/output matrix has both input and prediction roles. Contribution RMS alone is not functional importance. No safety of further weight pruning, mechanism of ticket formation, or explanation of four final sparse/dense matches is established. Native-minus-replay includes gates and capacity response, not isolated assignment effects. Single-layer effects must not be summed as independent contributions.

Individual seeds precede cohort summaries below. Positive delta_A means increased harm.


## NATIVE_RESPONSE

Seed 7: sparse delta_A=0.192252, R_baseline=0.501382, R_ZERO_ALL=0.309130; unpruned delta_A=0.083442, R_baseline=0.098920, R_ZERO_ALL=0.015478; X=0.108810.

Seed 17: sparse delta_A=0.345432, R_baseline=0.488639, R_ZERO_ALL=0.143207; unpruned delta_A=0.168392, R_baseline=0.106027, R_ZERO_ALL=-0.062365; X=0.177040.

Seed 29: sparse delta_A=0.205276, R_baseline=0.486839, R_ZERO_ALL=0.281563; unpruned delta_A=0.137393, R_baseline=0.102626, R_ZERO_ALL=-0.034767; X=0.067883.

Seed 43: sparse delta_A=0.429450, R_baseline=0.535030, R_ZERO_ALL=0.105580; unpruned delta_A=0.134636, R_baseline=0.116089, R_ZERO_ALL=-0.018548; X=0.294814.

Seed 61: sparse delta_A=0.279695, R_baseline=0.577647, R_ZERO_ALL=0.297953; unpruned delta_A=0.116145, R_baseline=0.127740, R_ZERO_ALL=0.011595; X=0.163550.

sparse ZERO_L0: delta_A mean 0.261863, median 0.279105, min 0.175739, max 0.355503, sample SD 0.072697.

sparse ZERO_L1: delta_A mean 0.003770, median -0.004005, min -0.004359, max 0.031023, sample SD 0.015355.

sparse ZERO_L2: delta_A mean -0.002106, median -0.001838, min -0.003763, max -0.001134, sample SD 0.000985.

sparse ZERO_L3: delta_A mean -0.001177, median -0.001396, min -0.002218, max -0.000336, sample SD 0.000767.

sparse ZERO_ALL: delta_A mean 0.290421, median 0.279695, min 0.192252, max 0.429450, sample SD 0.099193.

unpruned ZERO_L0: delta_A mean 0.124813, median 0.113401, min 0.084111, max 0.158568, sample SD 0.031944.

unpruned ZERO_L1: delta_A mean 0.005206, median 0.001668, min 0.000416, max 0.020049, sample SD 0.008358.

unpruned ZERO_L2: delta_A mean -0.000169, median 0.000205, min -0.002246, max 0.000729, sample SD 0.001184.

unpruned ZERO_L3: delta_A mean -0.000422, median -0.000232, min -0.001410, max -0.000080, sample SD 0.000557.

unpruned ZERO_ALL: delta_A mean 0.128002, median 0.134636, min 0.083442, max 0.168392, sample SD 0.031184.


## BASELINE_REPLAY

Seed 7: sparse delta_A=0.192252, R_baseline=0.501382, R_ZERO_ALL=0.309130; unpruned delta_A=0.083442, R_baseline=0.098920, R_ZERO_ALL=0.015478; X=0.108810.

Seed 17: sparse delta_A=0.345432, R_baseline=0.488639, R_ZERO_ALL=0.143207; unpruned delta_A=0.168392, R_baseline=0.106027, R_ZERO_ALL=-0.062365; X=0.177040.

Seed 29: sparse delta_A=0.205276, R_baseline=0.486839, R_ZERO_ALL=0.281563; unpruned delta_A=0.137393, R_baseline=0.102626, R_ZERO_ALL=-0.034767; X=0.067883.

Seed 43: sparse delta_A=0.429450, R_baseline=0.535030, R_ZERO_ALL=0.105580; unpruned delta_A=0.134636, R_baseline=0.116089, R_ZERO_ALL=-0.018548; X=0.294814.

Seed 61: sparse delta_A=0.279695, R_baseline=0.577647, R_ZERO_ALL=0.297953; unpruned delta_A=0.116145, R_baseline=0.127740, R_ZERO_ALL=0.011595; X=0.163550.

sparse ZERO_L0: delta_A mean 0.276519, median 0.283393, min 0.181697, max 0.389068, sample SD 0.086613.

sparse ZERO_L1: delta_A mean 0.003831, median -0.003885, min -0.004384, max 0.031191, sample SD 0.015411.

sparse ZERO_L2: delta_A mean -0.002101, median -0.001790, min -0.003801, max -0.001146, sample SD 0.001003.

sparse ZERO_L3: delta_A mean -0.001177, median -0.001396, min -0.002218, max -0.000336, sample SD 0.000767.

sparse ZERO_ALL: delta_A mean 0.290421, median 0.279695, min 0.192252, max 0.429450, sample SD 0.099193.

unpruned ZERO_L0: delta_A mean 0.117381, median 0.112975, min 0.081310, max 0.162072, sample SD 0.029394.

unpruned ZERO_L1: delta_A mean 0.004526, median 0.001544, min -0.000047, max 0.016704, sample SD 0.006928.

unpruned ZERO_L2: delta_A mean -0.000229, median 0.000058, min -0.002273, max 0.000888, sample SD 0.001196.

unpruned ZERO_L3: delta_A mean -0.000422, median -0.000232, min -0.001410, max -0.000080, sample SD 0.000557.

unpruned ZERO_ALL: delta_A mean 0.128002, median 0.134636, min 0.083442, max 0.168392, sample SD 0.031184.


Net harmful expert contributions (negative A) occur in 0 logical cells; inspect signed seed-level table.

Nonadditivity seed 7 sparse NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.007873.

Nonadditivity seed 7 sparse NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.027358.

Nonadditivity seed 7 sparse BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.007814.

Nonadditivity seed 7 sparse BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.021304.

Nonadditivity seed 7 unpruned NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.045902.

Nonadditivity seed 7 unpruned NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.043675.

Nonadditivity seed 7 unpruned BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.059429.

Nonadditivity seed 7 unpruned BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.060221.

Nonadditivity seed 17 sparse NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.021341.

Nonadditivity seed 17 sparse NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.078115.

Nonadditivity seed 17 sparse BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.015972.

Nonadditivity seed 17 sparse BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.041258.

Nonadditivity seed 17 unpruned NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.076062.

Nonadditivity seed 17 unpruned NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.085345.

Nonadditivity seed 17 unpruned BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.044889.

Nonadditivity seed 17 unpruned BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.051278.

Nonadditivity seed 29 sparse NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.009921.

Nonadditivity seed 29 sparse NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.018346.

Nonadditivity seed 29 sparse BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.008517.

Nonadditivity seed 29 sparse BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.018848.

Nonadditivity seed 29 unpruned NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.011756.

Nonadditivity seed 29 unpruned NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.032506.

Nonadditivity seed 29 unpruned BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.033283.

Nonadditivity seed 29 unpruned BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.043874.

Nonadditivity seed 43 sparse NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.042938.

Nonadditivity seed 43 sparse NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.091842.

Nonadditivity seed 43 sparse BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.030971.

Nonadditivity seed 43 sparse BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.046182.

Nonadditivity seed 43 unpruned NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.120303.

Nonadditivity seed 43 unpruned NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.079364.

Nonadditivity seed 43 unpruned BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.065604.

Nonadditivity seed 43 unpruned BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.071082.

Nonadditivity seed 61 sparse NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.008227.

Nonadditivity seed 61 sparse NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.014991.

Nonadditivity seed 61 sparse BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.004504.

Nonadditivity seed 61 sparse BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.006928.

Nonadditivity seed 61 unpruned NATIVE_RESPONSE offset 0: joint minus sum single-layer damage=0.041325.

Nonadditivity seed 61 unpruned NATIVE_RESPONSE offset 500: joint minus sum single-layer damage=0.047322.

Nonadditivity seed 61 unpruned BASELINE_REPLAY offset 0: joint minus sum single-layer damage=0.036627.

Nonadditivity seed 61 unpruned BASELINE_REPLAY offset 500: joint minus sum single-layer damage=0.047107.


Routing response contrasts for A and delta_A are in routing_mode_contrasts.csv. Absolute CE scales are in absolute_losses.csv and Figure 1; no percentages of causal recovery are assigned.

## Answers to the research questions


### NATIVE_RESPONSE

sparse: ZERO_ALL removal becomes more harmful in 5/5 seeds and less harmful in 0/5. Individual delta_A in seed order 7,17,29,43,61: 0.192252, 0.345432, 0.205276, 0.429450, 0.279695. mean 0.290421, median 0.279695, min 0.192252, max 0.429450, sample SD 0.099193.

Absolute CE improvement visible under ZERO_ALL for sparse: 0.309130, 0.143207, 0.281563, 0.105580, 0.297953; mean 0.227487, median 0.281563, min 0.105580, max 0.309130, sample SD 0.095551. Experts were present during historical training, so this is not a training-without-experts counterfactual.

sparse ZERO_L0: sensitivity increases in 5/5 and decreases in 0/5; seed values 0.175739, 0.296121, 0.202848, 0.355503, 0.279105; mean 0.261863, median 0.279105, min 0.175739, max 0.355503, sample SD 0.072697.

sparse ZERO_L1: sensitivity increases in 2/5 and decreases in 3/5; seed values 0.000261, -0.004005, -0.004359, 0.031023, -0.004071; mean 0.003770, median -0.004005, min -0.004359, max 0.031023, sample SD 0.015355.

sparse ZERO_L2: sensitivity increases in 0/5 and decreases in 5/5; seed values -0.001838, -0.002029, -0.001134, -0.003763, -0.001767; mean -0.002106, median -0.001838, min -0.003763, max -0.001134, sample SD 0.000985.

sparse ZERO_L3: sensitivity increases in 0/5 and decreases in 5/5; seed values -0.001396, -0.001429, -0.000505, -0.002218, -0.000336; mean -0.001177, median -0.001396, min -0.002218, max -0.000336, sample SD 0.000767.

unpruned: ZERO_ALL removal becomes more harmful in 5/5 seeds and less harmful in 0/5. Individual delta_A in seed order 7,17,29,43,61: 0.083442, 0.168392, 0.137393, 0.134636, 0.116145. mean 0.128002, median 0.134636, min 0.083442, max 0.168392, sample SD 0.031184.

Absolute CE improvement visible under ZERO_ALL for unpruned: 0.015478, -0.062365, -0.034767, -0.018548, 0.011595; mean -0.017721, median -0.018548, min -0.062365, max 0.015478, sample SD 0.032580. Experts were present during historical training, so this is not a training-without-experts counterfactual.

unpruned ZERO_L0: sensitivity increases in 5/5 and decreases in 0/5; seed values 0.084111, 0.158568, 0.113401, 0.156207, 0.111781; mean 0.124813, median 0.113401, min 0.084111, max 0.158568, sample SD 0.031944.

unpruned ZERO_L1: sensitivity increases in 5/5 and decreases in 0/5; seed values 0.001668, 0.000416, 0.003039, 0.020049, 0.000857; mean 0.005206, median 0.001668, min 0.000416, max 0.020049, sample SD 0.008358.

unpruned ZERO_L2: sensitivity increases in 4/5 and decreases in 1/5; seed values 0.000122, 0.000205, 0.000346, 0.000729, -0.002246; mean -0.000169, median 0.000205, min -0.002246, max 0.000729, sample SD 0.001184.

unpruned ZERO_L3: sensitivity increases in 0/5 and decreases in 5/5; seed values -0.000232, -0.000080, -0.000143, -0.001410, -0.000244; mean -0.000422, median -0.000232, min -0.001410, max -0.000080, sample SD 0.000557.

Sparse-specific change X: 0.108810, 0.177040, 0.067883, 0.294814, 0.163550; mean 0.162419, median 0.163550, min 0.067883, max 0.294814, sample SD 0.085957.


### BASELINE_REPLAY

sparse: ZERO_ALL removal becomes more harmful in 5/5 seeds and less harmful in 0/5. Individual delta_A in seed order 7,17,29,43,61: 0.192252, 0.345432, 0.205276, 0.429450, 0.279695. mean 0.290421, median 0.279695, min 0.192252, max 0.429450, sample SD 0.099193.

Absolute CE improvement visible under ZERO_ALL for sparse: 0.309130, 0.143207, 0.281563, 0.105580, 0.297953; mean 0.227487, median 0.281563, min 0.105580, max 0.309130, sample SD 0.095551. Experts were present during historical training, so this is not a training-without-experts counterfactual.

sparse ZERO_L0: sensitivity increases in 5/5 and decreases in 0/5; seed values 0.181697, 0.327460, 0.200979, 0.389068, 0.283393; mean 0.276519, median 0.283393, min 0.181697, max 0.389068, sample SD 0.086613.

sparse ZERO_L1: sensitivity increases in 2/5 and decreases in 3/5; seed values 0.000251, -0.003885, -0.004384, 0.031191, -0.004020; mean 0.003831, median -0.003885, min -0.004384, max 0.031191, sample SD 0.015411.

sparse ZERO_L2: sensitivity increases in 0/5 and decreases in 5/5; seed values -0.001790, -0.001999, -0.001146, -0.003801, -0.001766; mean -0.002101, median -0.001790, min -0.003801, max -0.001146, sample SD 0.001003.

sparse ZERO_L3: sensitivity increases in 0/5 and decreases in 5/5; seed values -0.001396, -0.001429, -0.000505, -0.002218, -0.000336; mean -0.001177, median -0.001396, min -0.002218, max -0.000336, sample SD 0.000767.

unpruned: ZERO_ALL removal becomes more harmful in 5/5 seeds and less harmful in 0/5. Individual delta_A in seed order 7,17,29,43,61: 0.083442, 0.168392, 0.137393, 0.134636, 0.116145. mean 0.128002, median 0.134636, min 0.083442, max 0.168392, sample SD 0.031184.

Absolute CE improvement visible under ZERO_ALL for unpruned: 0.015478, -0.062365, -0.034767, -0.018548, 0.011595; mean -0.017721, median -0.018548, min -0.062365, max 0.015478, sample SD 0.032580. Experts were present during historical training, so this is not a training-without-experts counterfactual.

unpruned ZERO_L0: sensitivity increases in 5/5 and decreases in 0/5; seed values 0.081310, 0.162072, 0.123313, 0.112975, 0.107232; mean 0.117381, median 0.112975, min 0.081310, max 0.162072, sample SD 0.029394.

unpruned ZERO_L1: sensitivity increases in 4/5 and decreases in 1/5; seed values 0.001544, -0.000047, 0.003475, 0.016704, 0.000951; mean 0.004526, median 0.001544, min -0.000047, max 0.016704, sample SD 0.006928.

unpruned ZERO_L2: sensitivity increases in 4/5 and decreases in 1/5; seed values 0.000027, 0.000058, 0.000156, 0.000888, -0.002273; mean -0.000229, median 0.000058, min -0.002273, max 0.000888, sample SD 0.001196.

unpruned ZERO_L3: sensitivity increases in 0/5 and decreases in 5/5; seed values -0.000232, -0.000080, -0.000143, -0.001410, -0.000244; mean -0.000422, median -0.000232, min -0.001410, max -0.000080, sample SD 0.000557.

Sparse-specific change X: 0.108810, 0.177040, 0.067883, 0.294814, 0.163550; mean 0.162419, median 0.163550, min 0.067883, max 0.294814, sample SD 0.085957.


### Routing response and harmful contributions

sparse ZERO_L0: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values -0.005958, -0.031339, 0.001869, -0.033565, -0.004288. This contrast includes changes in assignments, gates and capacity.

sparse ZERO_L1: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.000010, -0.000120, 0.000025, -0.000167, -0.000051. This contrast includes changes in assignments, gates and capacity.

sparse ZERO_L2: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values -0.000048, -0.000029, 0.000012, 0.000038, -0.000001. This contrast includes changes in assignments, gates and capacity.

sparse ZERO_L3: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.000000, 0.000000, 0.000000, 0.000000, 0.000000. This contrast includes changes in assignments, gates and capacity.

sparse ZERO_ALL: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.000000, 0.000000, 0.000000, 0.000000, 0.000000. This contrast includes changes in assignments, gates and capacity.

unpruned ZERO_L0: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.002801, -0.003505, -0.009912, 0.043232, 0.004549. This contrast includes changes in assignments, gates and capacity.

unpruned ZERO_L1: native and replay delta_A signs agree in 4/5 seeds; native-minus-replay seed values 0.000123, 0.000462, -0.000436, 0.003345, -0.000094. This contrast includes changes in assignments, gates and capacity.

unpruned ZERO_L2: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.000095, 0.000148, 0.000190, -0.000159, 0.000028. This contrast includes changes in assignments, gates and capacity.

unpruned ZERO_L3: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.000000, 0.000000, 0.000000, 0.000000, 0.000000. This contrast includes changes in assignments, gates and capacity.

unpruned ZERO_ALL: native and replay delta_A signs agree in 5/5 seeds; native-minus-replay seed values 0.000000, 0.000000, 0.000000, 0.000000, 0.000000. This contrast includes changes in assignments, gates and capacity.

Negative A (removal improves CE) occurs in 0 intervention cells. Exact seed, model, offset, mode, intervention and signed damage follow.

Joint minus summed single-layer damage spans 0.004504 to 0.120303 CE across checkpoints and modes. Joint and single-layer interventions are distinct and must not be interpreted as independent component contributions.
