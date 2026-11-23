# First-MoE amplitude control

The first MoE layer and this follow-up question were selected after previous diagnostics. The new amplitude intervention is sealed before its outcomes; this is not project-level preregistration. Calibration uses an already sealed four-complete-batch training probe previously used in training and diagnostics, not held-out generalization data. Validation was repeatedly examined and is not an independent test set. Source seed (n=5) is the independent replication unit; tokens, batches, recipient contexts, modes and variants are nested measurements. One training-only scalar per seed/model type changes only global contribution amplitude. Residual differences can include directions, token-specific amplitude allocation, zero support, assignments and gates; they are not an isolated direction effect. Global norms/cosines describe interventions, not why they help. Recipient context includes its own residual and subsequent computation, not downstream weights alone. Native/replay differences include downstream routing responses and do not isolate a routing mechanism. Sparse-minus-unpruned effects compare each trajectory's own intervention and are not equal absolute-amplitude perturbations. No equivalence claim, percentage of recovery explained, expert weight learning, training counterfactual, standalone sparse recoverability, online pruning criterion or generalization beyond this architecture/mask family/window is established. No follow-up is launched.

Positive B_content means a recovered-contribution advantage at the nominally matched amplitude. Negative effects are retained. Individual seed values appear first.


## NATIVE_DOWNSTREAM


### sparse / start recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.120067 | 0.116704 | 0.003363 | 0.001505 | 0.118562 | 0.001858 |
| 17 | 0.147120 | 0.133489 | 0.013631 | 0.008499 | 0.138621 | 0.005132 |
| 29 | 0.116657 | 0.101509 | 0.015148 | 0.012800 | 0.103857 | 0.002348 |
| 43 | 0.153530 | 0.139512 | 0.014017 | 0.010984 | 0.142546 | 0.003033 |
| 61 | 0.101892 | 0.093678 | 0.008214 | 0.005713 | 0.096179 | 0.002501 |
B_original: n=5/5, mean 0.127853, median 0.120067, range [0.101892, 0.153530], sample SD 0.021740.

B_scale_start: n=5/5, mean 0.116979, median 0.116704, range [0.093678, 0.139512], sample SD 0.019765.

B_content_end: n=5/5, mean 0.010875, median 0.013631, range [0.003363, 0.015148], sample SD 0.004981.

B_content_start: n=5/5, mean 0.007900, median 0.008499, range [0.001505, 0.012800], sample SD 0.004460.

B_scale_end: n=5/5, mean 0.119953, median 0.118562, range [0.096179, 0.142546], sample SD 0.020525.

I: n=5/5, mean 0.002974, median 0.002501, range [0.001858, 0.005132], sample SD 0.001277.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### sparse / end recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.110932 | 0.050088 | 0.060844 | 0.042962 | 0.067970 | 0.017882 |
| 17 | 0.169083 | 0.080714 | 0.088369 | 0.056046 | 0.113037 | 0.032323 |
| 29 | 0.108569 | 0.044932 | 0.063637 | 0.045512 | 0.063057 | 0.018125 |
| 43 | 0.184668 | 0.076040 | 0.108628 | 0.070376 | 0.114292 | 0.038252 |
| 61 | 0.136021 | 0.054930 | 0.081091 | 0.045316 | 0.090706 | 0.035776 |
B_original: n=5/5, mean 0.141855, median 0.136021, range [0.108569, 0.184668], sample SD 0.034178.

B_scale_start: n=5/5, mean 0.061341, median 0.054930, range [0.044932, 0.080714], sample SD 0.016034.

B_content_end: n=5/5, mean 0.080514, median 0.081091, range [0.060844, 0.108628], sample SD 0.019520.

B_content_start: n=5/5, mean 0.052042, median 0.045512, range [0.042962, 0.070376], sample SD 0.011429.

B_scale_end: n=5/5, mean 0.089812, median 0.090706, range [0.063057, 0.114292], sample SD 0.024147.

I: n=5/5, mean 0.028472, median 0.032323, range [0.017882, 0.038252], sample SD 0.009786.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### unpruned / start recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.031172 | -0.000305 | 0.031477 | 0.031443 | -0.000271 | 0.000034 |
| 17 | 0.040545 | 0.000414 | 0.040131 | 0.040153 | 0.000392 | -0.000022 |
| 29 | 0.037935 | 0.000204 | 0.037731 | 0.037266 | 0.000669 | 0.000464 |
| 43 | 0.035926 | -0.000758 | 0.036684 | 0.036954 | -0.001029 | -0.000271 |
| 61 | 0.045135 | -0.000127 | 0.045262 | 0.045316 | -0.000181 | -0.000054 |
B_original: n=5/5, mean 0.038143, median 0.037935, range [0.031172, 0.045135], sample SD 0.005201.

B_scale_start: n=5/5, mean -0.000115, median -0.000127, range [-0.000758, 0.000414], sample SD 0.000456.

B_content_end: n=5/5, mean 0.038257, median 0.037731, range [0.031477, 0.045262], sample SD 0.005032.

B_content_start: n=5/5, mean 0.038227, median 0.037266, range [0.031443, 0.045316], sample SD 0.005063.

B_scale_end: n=5/5, mean -0.000084, median -0.000181, range [-0.001029, 0.000669], sample SD 0.000658.

I: n=5/5, mean 0.000030, median -0.000022, range [-0.000271, 0.000464], sample SD 0.000269.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### unpruned / end recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.062725 | 0.000200 | 0.062525 | 0.062174 | 0.000551 | 0.000351 |
| 17 | 0.076124 | -0.000757 | 0.076881 | 0.075870 | 0.000255 | 0.001012 |
| 29 | 0.065851 | -0.000284 | 0.066134 | 0.065215 | 0.000636 | 0.000919 |
| 43 | 0.072997 | 0.000043 | 0.072954 | 0.072597 | 0.000400 | 0.000357 |
| 61 | 0.082161 | -0.000010 | 0.082171 | 0.081986 | 0.000175 | 0.000185 |
B_original: n=5/5, mean 0.071972, median 0.072997, range [0.062725, 0.082161], sample SD 0.007828.

B_scale_start: n=5/5, mean -0.000162, median -0.000010, range [-0.000757, 0.000200], sample SD 0.000376.

B_content_end: n=5/5, mean 0.072133, median 0.072954, range [0.062525, 0.082171], sample SD 0.007942.

B_content_start: n=5/5, mean 0.071568, median 0.072597, range [0.062174, 0.081986], sample SD 0.008011.

B_scale_end: n=5/5, mean 0.000403, median 0.000400, range [0.000175, 0.000636], sample SD 0.000194.

I: n=5/5, mean 0.000565, median 0.000357, range [0.000185, 0.001012], sample SD 0.000374.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### Paired sparse-minus-unpruned residual advantages

start recipient B_content_end, seed order 7,17,29,43,61: -0.028115, -0.026501, -0.022583, -0.022667, -0.037048; positive in 0/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.

start recipient B_content_start, seed order 7,17,29,43,61: -0.029938, -0.031654, -0.024466, -0.025970, -0.039603; positive in 0/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.

end recipient B_content_end, seed order 7,17,29,43,61: -0.001681, 0.011488, -0.002497, 0.035674, -0.001080; positive in 2/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.

end recipient B_content_start, seed order 7,17,29,43,61: -0.019212, -0.019823, -0.019703, -0.002221, -0.036670; positive in 0/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.


## RECIPIENT_BASELINE_REPLAY


### sparse / start recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.119101 | 0.116394 | 0.002707 | 0.001175 | 0.117926 | 0.001532 |
| 17 | 0.146216 | 0.133917 | 0.012299 | 0.007377 | 0.138838 | 0.004921 |
| 29 | 0.114364 | 0.100583 | 0.013781 | 0.011865 | 0.102499 | 0.001916 |
| 43 | 0.149991 | 0.139481 | 0.010510 | 0.008121 | 0.141870 | 0.002389 |
| 61 | 0.101364 | 0.093555 | 0.007809 | 0.005430 | 0.095935 | 0.002379 |
B_original: n=5/5, mean 0.126207, median 0.119101, range [0.101364, 0.149991], sample SD 0.021059.

B_scale_start: n=5/5, mean 0.116786, median 0.116394, range [0.093555, 0.139481], sample SD 0.020068.

B_content_end: n=5/5, mean 0.009421, median 0.010510, range [0.002707, 0.013781], sample SD 0.004363.

B_content_start: n=5/5, mean 0.006794, median 0.007377, range [0.001175, 0.011865], sample SD 0.003913.

B_scale_end: n=5/5, mean 0.119413, median 0.117926, range [0.095935, 0.141870], sample SD 0.020744.

I: n=5/5, mean 0.002627, median 0.002379, range [0.001532, 0.004921], sample SD 0.001331.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### sparse / end recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.111481 | 0.050836 | 0.060645 | 0.042719 | 0.068762 | 0.017926 |
| 17 | 0.172453 | 0.082022 | 0.090431 | 0.058551 | 0.113902 | 0.031880 |
| 29 | 0.109691 | 0.045311 | 0.064380 | 0.046555 | 0.063136 | 0.017825 |
| 43 | 0.200191 | 0.084361 | 0.115830 | 0.084505 | 0.115686 | 0.031325 |
| 61 | 0.137071 | 0.055387 | 0.081684 | 0.045713 | 0.091359 | 0.035972 |
B_original: n=5/5, mean 0.146177, median 0.137071, range [0.109691, 0.200191], sample SD 0.039452.

B_scale_start: n=5/5, mean 0.063583, median 0.055387, range [0.045311, 0.084361], sample SD 0.018271.

B_content_end: n=5/5, mean 0.082594, median 0.081684, range [0.060645, 0.115830], sample SD 0.022251.

B_content_start: n=5/5, mean 0.055609, median 0.046555, range [0.042719, 0.084505], sample SD 0.017246.

B_scale_end: n=5/5, mean 0.090569, median 0.091359, range [0.063136, 0.115686], sample SD 0.024515.

I: n=5/5, mean 0.026985, median 0.031325, range [0.017825, 0.035972], sample SD 0.008508.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### unpruned / start recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.030157 | -0.000271 | 0.030428 | 0.030427 | -0.000270 | 0.000001 |
| 17 | 0.039050 | 0.000337 | 0.038713 | 0.038702 | 0.000348 | 0.000011 |
| 29 | 0.034177 | 0.000293 | 0.033884 | 0.033503 | 0.000674 | 0.000381 |
| 43 | 0.027115 | -0.000773 | 0.027888 | 0.028255 | -0.001139 | -0.000367 |
| 61 | 0.042802 | -0.000088 | 0.042890 | 0.042942 | -0.000140 | -0.000052 |
B_original: n=5/5, mean 0.034660, median 0.034177, range [0.027115, 0.042802], sample SD 0.006384.

B_scale_start: n=5/5, mean -0.000100, median -0.000088, range [-0.000773, 0.000337], sample SD 0.000455.

B_content_end: n=5/5, mean 0.034761, median 0.033884, range [0.027888, 0.042890], sample SD 0.006093.

B_content_start: n=5/5, mean 0.034766, median 0.033503, range [0.028255, 0.042942], sample SD 0.006024.

B_scale_end: n=5/5, mean -0.000105, median -0.000140, range [-0.001139, 0.000674], sample SD 0.000691.

I: n=5/5, mean -0.000005, median 0.000001, range [-0.000367, 0.000381], sample SD 0.000266.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### unpruned / end recipient

| Seed | Original benefit | Scale start | Content end-target | Content start-target | Scale end | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.062768 | 0.000112 | 0.062656 | 0.062212 | 0.000556 | 0.000444 |
| 17 | 0.077865 | -0.000860 | 0.078725 | 0.077731 | 0.000134 | 0.000994 |
| 29 | 0.067562 | -0.000334 | 0.067896 | 0.066925 | 0.000637 | 0.000971 |
| 43 | 0.076626 | 0.000093 | 0.076532 | 0.076341 | 0.000284 | 0.000191 |
| 61 | 0.082407 | 0.000002 | 0.082404 | 0.082232 | 0.000174 | 0.000172 |
B_original: n=5/5, mean 0.073446, median 0.076626, range [0.062768, 0.082407], sample SD 0.008040.

B_scale_start: n=5/5, mean -0.000197, median 0.000002, range [-0.000860, 0.000112], sample SD 0.000412.

B_content_end: n=5/5, mean 0.073643, median 0.076532, range [0.062656, 0.082404], sample SD 0.008134.

B_content_start: n=5/5, mean 0.073088, median 0.076341, range [0.062212, 0.082232], sample SD 0.008247.

B_scale_end: n=5/5, mean 0.000357, median 0.000284, range [0.000134, 0.000637], sample SD 0.000227.

I: n=5/5, mean 0.000554, median 0.000444, range [0.000172, 0.000994], sample SD 0.000405.

B_content_end positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

B_content_start positive in 5/5 seeds; scaling alone leaves the signed residual differences above.

Recovered contribution retains an advantage at both nominal amplitude targets in 5/5 seeds. The two target-specific contrasts and I report context/target interaction; no equivalence or causal percentage is inferred.


### Paired sparse-minus-unpruned residual advantages

start recipient B_content_end, seed order 7,17,29,43,61: -0.027721, -0.026414, -0.020103, -0.017378, -0.035081; positive in 0/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.

start recipient B_content_start, seed order 7,17,29,43,61: -0.029252, -0.031325, -0.021637, -0.020133, -0.037513; positive in 0/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.

end recipient B_content_end, seed order 7,17,29,43,61: -0.002011, 0.011706, -0.003516, 0.039297, -0.000720; positive in 2/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.

end recipient B_content_start, seed order 7,17,29,43,61: -0.019494, -0.019180, -0.020370, 0.008163, -0.036520; positive in 1/5 seeds. These are trajectory-specific, not equal absolute-amplitude perturbations.


## Calibration and validation mismatch

Seed 7 sparse: training-only r=1.680710; a0=0.602628, a1=1.012842; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 7 unpruned: training-only r=1.027221; a0=2.805219, a1=2.881579; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 17 sparse: training-only r=1.769952; a0=0.680539, a1=1.204522; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 17 unpruned: training-only r=1.026321; a0=3.130257, a1=3.212648; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 29 sparse: training-only r=1.612848; a0=0.699301, a1=1.127867; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 29 unpruned: training-only r=1.024188; a0=3.299780, a1=3.379595; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 43 sparse: training-only r=1.777639; a0=0.627545, a1=1.115549; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 43 unpruned: training-only r=1.024927; a0=2.962749, a1=3.036600; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 61 sparse: training-only r=1.599601; a0=0.775924, a1=1.241168; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 61 unpruned: training-only r=1.011830; a0=3.553532, a1=3.595570; AVAILABLE. Four complete, previously examined training batches; no validation CE in this estimate.

Seed 7 sparse target end: actual validation scaled/reference RMS ratio 0.999089; relative mismatch -0.000911; no scale adjustment.

Seed 7 sparse target start: actual validation scaled/reference RMS ratio 1.000912; relative mismatch 0.000912; no scale adjustment.

Seed 7 unpruned target end: actual validation scaled/reference RMS ratio 1.000530; relative mismatch 0.000530; no scale adjustment.

Seed 7 unpruned target start: actual validation scaled/reference RMS ratio 0.999470; relative mismatch -0.000530; no scale adjustment.

Seed 17 sparse target end: actual validation scaled/reference RMS ratio 1.000932; relative mismatch 0.000932; no scale adjustment.

Seed 17 sparse target start: actual validation scaled/reference RMS ratio 0.999069; relative mismatch -0.000931; no scale adjustment.

Seed 17 unpruned target end: actual validation scaled/reference RMS ratio 1.000818; relative mismatch 0.000818; no scale adjustment.

Seed 17 unpruned target start: actual validation scaled/reference RMS ratio 0.999182; relative mismatch -0.000818; no scale adjustment.

Seed 29 sparse target end: actual validation scaled/reference RMS ratio 0.999173; relative mismatch -0.000827; no scale adjustment.

Seed 29 sparse target start: actual validation scaled/reference RMS ratio 1.000828; relative mismatch 0.000828; no scale adjustment.

Seed 29 unpruned target end: actual validation scaled/reference RMS ratio 1.000269; relative mismatch 0.000269; no scale adjustment.

Seed 29 unpruned target start: actual validation scaled/reference RMS ratio 0.999731; relative mismatch -0.000269; no scale adjustment.

Seed 43 sparse target end: actual validation scaled/reference RMS ratio 1.001407; relative mismatch 0.001407; no scale adjustment.

Seed 43 sparse target start: actual validation scaled/reference RMS ratio 0.998595; relative mismatch -0.001405; no scale adjustment.

Seed 43 unpruned target end: actual validation scaled/reference RMS ratio 1.000138; relative mismatch 0.000138; no scale adjustment.

Seed 43 unpruned target start: actual validation scaled/reference RMS ratio 0.999862; relative mismatch -0.000138; no scale adjustment.

Seed 61 sparse target end: actual validation scaled/reference RMS ratio 0.999221; relative mismatch -0.000779; no scale adjustment.

Seed 61 sparse target start: actual validation scaled/reference RMS ratio 1.000779; relative mismatch 0.000779; no scale adjustment.

Seed 61 unpruned target end: actual validation scaled/reference RMS ratio 1.001871; relative mismatch 0.001871; no scale adjustment.

Seed 61 unpruned target start: actual validation scaled/reference RMS ratio 0.998133; relative mismatch -0.001867; no scale adjustment.

Largest absolute relative validation RMS mismatch across the nominal pairs is 0.001871. The exact mismatch is shown for every seed/target. No prespecified equivalence margin was introduced: interpret the contrasts as nominal training-calibrated matching, with this remaining global-amplitude limitation, not exact validation matching.


## Verdict and interpretation

sparse, NATIVE_DOWNSTREAM: positive recovered-contribution residual advantage at both targets in 10/10 seed-by-recipient combinations. Where both remain positive, global amplitude alone is insufficient under these training-calibrated interventions; residual differences include more than direction. Otherwise the per-seed target/context interaction remains the conclusion.

unpruned, NATIVE_DOWNSTREAM: positive recovered-contribution residual advantage at both targets in 10/10 seed-by-recipient combinations. Where both remain positive, global amplitude alone is insufficient under these training-calibrated interventions; residual differences include more than direction. Otherwise the per-seed target/context interaction remains the conclusion.

sparse, RECIPIENT_BASELINE_REPLAY: positive recovered-contribution residual advantage at both targets in 10/10 seed-by-recipient combinations. Where both remain positive, global amplitude alone is insufficient under these training-calibrated interventions; residual differences include more than direction. Otherwise the per-seed target/context interaction remains the conclusion.

unpruned, RECIPIENT_BASELINE_REPLAY: positive recovered-contribution residual advantage at both targets in 10/10 seed-by-recipient combinations. Where both remain positive, global amplitude alone is insufficient under these training-calibrated interventions; residual differences include more than direction. Otherwise the per-seed target/context interaction remains the conclusion.

Native versus replay contrast differences are fully reported in routing_mode_contrasts.csv and Figure 5. They assess sensitivity to downstream routing response, not an isolated routing mechanism. Calibration and validation direction/support/gate diagnostics are descriptive. No subsequent experiment or training is launched.
