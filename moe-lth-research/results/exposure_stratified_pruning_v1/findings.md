# Exposure-stratified pruning experiment v1

Status: STOPPED_BALANCE_GATE.

Historical exposure is cumulative accepted tokens through1000, within layer. It measures quantity, not relevant input coverage. Gate-weighted exposure and historical diversity are unavailable. Calibration sensitivity arrays are reused from the verified native training-only task-gradient measurement; no gradient scoring was repeated.

| Seed | Layer0 H/L exposure | Layer1 | Layer2 | Layer3 | Matching pairs passed /64 |
|---|---:|---:|---:|---:|---:|
| 7 | 1.717 | 1.140 | 1.140 | 1.161 | 53/64 |
| 17 | 1.740 | 1.151 | 1.169 | 1.184 | 45/64 |
| 29 | 1.587 | 1.141 | 1.158 | 1.105 | 45/64 |
| 43 | 1.696 | 1.141 | 1.136 | 1.181 | 50/64 |
| 61 | 1.682 | 1.138 | 1.134 | 1.137 | 44/64 |

The frozen approximate matching design did not establish adequate five-seed coverage. It was stopped before any pruning loss, mask application to a model, engineering update or production continuation. Candidate coordinate files are retained for audit and are not approved intervention masks. No tolerance or dose was relaxed.

83/320 paired tensor selections failed one or more fixed gates. Full counts, magnitude/score distributions, energy balance and residual history/structure are in the CSVs.

This failure applies to the fixed exposure-rank expert pairing and single-pass nearest16 generator with its joint calipers; it is not a proof that all cross-expert approximate designs lack overlap. The intended pruning comparison remains unanswered. Exposure variation exists, but that alone does not validate exposure-aware protection or show that waiting makes a weight useful. Baseline expert function and post-pruning trajectories were not measured because the pre-loss selection gate failed. No method development is justified by this feasibility result.

Failed-gate counts (a pair can fail multiple): A_ks=4, A_smd=1, dose=74, normalized_magnitude_ks=20, normalized_magnitude_smd=10, raw_magnitude_ks=4, raw_magnitude_smd=1.

35/320 fixed expert pairs have RMS differences incompatible with the simultaneous raw/normalized magnitude calipers: feasibility requires absolute log RMS ratio <=0.105. This rules out pairs only for those fixed expert pairings and calipers. Other incomplete yields are single-pass heuristic results, not global infeasibility certificates. These constraints were not relaxed. Normalized-magnitude calipers and balance were additional registered requirements, so this failure does not establish that raw-magnitude/sensitivity matching alone is infeasible. For the singleton candidate, SMD=1e99 is a zero-variance unequal-mean failure sentinel, not an interpretable standardized effect; that pair already fails dose.
