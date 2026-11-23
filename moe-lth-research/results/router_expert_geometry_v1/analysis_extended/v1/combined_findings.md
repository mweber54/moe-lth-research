# Geometry and score–activation extension v1

**Post hoc exploratory analysis.** These additional metrics and aggregation corrections were selected after inspecting original rank-16 results. Existing analysis outputs, checkpoints, and Dataset B were preserved. No training or optimizer steps were run. Rank 16 remains the original primary SVD metric; ranks 8/32, Gram metrics, and GELU associations are complementary exploratory measurements.

## Input and reference audit

Measured 525/525 recovery states (75 runs × seven steps) and 175 existing source/pruning/initialization/historical states. State identities, probe hashes, checkpoint origins, mask-zero verification, and sample coverage are exported. Mid-trajectory diagnostic snapshots did not carry an independent prior file hash; this pass establishes SHA-256 identity, checks their step/probe metadata, and verifies finite tensors and mask zeros. Step zero is checked against the exact constructed initialization; step 2500 is tensor-compared against the original hash-attested native endpoint.

Source-final dense, current matched dense recovery at the same router age/step, and sparse initialization are separate Gram references. Historical dense comparators remain historical. Source/pruning-only states have no invented recovery comparator. Summary CSVs are used for recorded SVD metrics only; new Gram and token statistics are computed from actual weights and probes. See `metric_definitions.md`, `state_coverage_inventory.csv`, `coverage/`, and per-state `completed.json`.

## Corrected SVD findings

The original rank-16 learned-mask claim changes from 30/30 to **28/30** when both endpoints use identical valid (layer, expert) membership. Seed 17/R0/50% and seed 43/R0/50% instead increase slightly. The original step-specific mean changed which experts were included; therefore its universal sign claim is invalid.

| Rank | Mask type | Decreasing source overlap | Increasing source overlap |
| --- | --- | ---: | ---: |
| 8 | dense | 0/15 | 15/15 |
| 8 | learned | 29/30 | 1/30 |
| 8 | random | 0/30 | 30/30 |
| 16 | dense | 0/15 | 15/15 |
| 16 | learned | 28/30 | 2/30 |
| 16 | random | 0/30 | 30/30 |
| 32 | dense | 0/15 | 15/15 |
| 32 | learned | 29/30 | 1/30 |
| 32 | random | 0/30 | 30/30 |

Endpoint matching, fixed trajectory intersections, and learned/random joint intersections are reported separately. The main figures, direction counts and cross-metric cases require valid current and source-reference bases; the direct original-rule correction is preserved separately. Both reproduce 28/30. Cutoff gaps and retained energy are preserved in the expert raw tables. An endpoint decrease is not a monotonic trajectory claim.

## Full-spectrum Gram findings

- learned: 22/30 runs move farther from source-final Gram shape; 8/30 move closer. This is a separate measurement from top-rank subspace overlap.
- random: 17/30 runs move farther from source-final Gram shape; 13/30 move closer. This is a separate measurement from top-rank subspace overlap.
- dense: 3/15 runs move farther from source-final Gram shape; 12/15 move closer. This is a separate measurement from top-rank subspace overlap.

All seed values for source-final Gram shape distance changes (negative = approach):

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | -0.66091 | -0.67097 | -0.68908 | -0.74177 | -0.66822 | -0.68619 | [-0.71506, -0.66584] |
| dense | 0.0 | 50 | -0.00640 | -0.00055 | -0.01111 | -0.04071 | 0.01205 | -0.00935 | [-0.02584, 0.00449] |
| dense | 0.0 | 100 | -0.00567 | 0.00491 | -0.00924 | -0.03095 | 0.01929 | -0.00433 | [-0.01874, 0.00931] |
| learned | 0.5 | 0 | 0.02192 | -0.01807 | 0.00867 | -0.05025 | 0.04866 | 0.00219 | [-0.02938, 0.02997] |
| learned | 0.5 | 50 | 0.02683 | -0.00042 | 0.00181 | -0.03487 | 0.00881 | 0.00043 | [-0.01924, 0.01682] |
| learned | 0.5 | 100 | 0.00661 | 0.00389 | 0.00720 | -0.04585 | -0.00186 | -0.00600 | [-0.02644, 0.00619] |
| learned | 0.65 | 0 | 0.06644 | 0.09976 | 0.08310 | 0.04149 | 0.09694 | 0.07755 | [0.05813, 0.09530] |
| learned | 0.65 | 50 | 0.05104 | 0.02653 | 0.03671 | -0.00202 | 0.04338 | 0.03113 | [0.01347, 0.04511] |
| learned | 0.65 | 100 | 0.04734 | 0.03308 | 0.03863 | -0.01133 | 0.03268 | 0.02808 | [0.00747, 0.04212] |
| random | 0.5 | 0 | 0.01968 | 0.04742 | -0.00169 | -0.00628 | 0.04023 | 0.01987 | [0.00075, 0.03899] |
| random | 0.5 | 50 | -0.01183 | -0.00488 | -0.02689 | -0.05445 | 0.00595 | -0.01842 | [-0.03792, -0.00194] |
| random | 0.5 | 100 | 0.00092 | 0.01906 | -0.02999 | -0.04425 | 0.00391 | -0.01007 | [-0.03177, 0.00937] |
| random | 0.65 | 0 | 0.10285 | 0.10093 | 0.03560 | 0.06641 | 0.09359 | 0.07988 | [0.05483, 0.10023] |
| random | 0.65 | 50 | -0.00306 | 0.04699 | -0.00952 | -0.02719 | 0.04597 | 0.01064 | [-0.01530, 0.03657] |
| random | 0.65 | 100 | 0.00942 | 0.05011 | -0.01060 | -0.03116 | 0.02608 | 0.00877 | [-0.01493, 0.03317] |

Matched dense reference and scale are separately reported in `gram_endpoint_seed_summary.csv`. The normalized Frobenius cosine and shape distance satisfy d²=2−2cosine and are not independent evidence. Trace and trace ratio measure projection energy; relative unnormalized distance combines shape and scale. Coordinatewise comparisons do not remove upstream representation drift.

## GELU score–activation association

The statistic uses the raw linear router score and mean signed post-GELU fc1 activation before route-weight multiplication. Fixed source hidden inputs and native current representations are separate modes. Proposed assignments, capacity-accepted assignments, and the fixed 128-token all-expert counterfactual subset are distinct. Each layer/expert/selection group is z-scored separately before pooling. Pearson is primary; Spearman is sensitivity. Per-expert associations, equal-expert summaries, and token-weighted standardized pooled Pearson are kept distinct.

### fixed, proposed

- learned: association increases at the endpoint in 18/30 evaluable runs; final association exceeds within-layer shuffled pairings in 25/30. Run counts are descriptive; source seed is the replication unit.
- random: association increases at the endpoint in 18/30 evaluable runs; final association exceeds within-layer shuffled pairings in 8/30. Run counts are descriptive; source seed is the replication unit.
- dense: association increases at the endpoint in 0/15 evaluable runs; final association exceeds within-layer shuffled pairings in 14/15. Run counts are descriptive; source seed is the replication unit.

Equal-expert endpoint Pearson changes with all five seed values:

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | -0.21044 | -0.09139 | -0.16575 | -0.11404 | -0.15614 | -0.14755 | [-0.18222, -0.11079] |
| dense | 0.0 | 50 | -0.20228 | -0.16894 | -0.13890 | -0.05556 | -0.12745 | -0.13863 | [-0.17693, -0.09261] |
| dense | 0.0 | 100 | -0.23785 | -0.13695 | -0.18649 | -0.08266 | -0.10258 | -0.14931 | [-0.20053, -0.10148] |
| learned | 0.5 | 0 | 0.02151 | -0.00997 | 0.01397 | -0.01372 | 0.06170 | 0.01470 | [-0.00668, 0.03858] |
| learned | 0.5 | 50 | 0.09842 | -0.09624 | 0.06334 | -0.02048 | 0.01775 | 0.01256 | [-0.04917, 0.06825] |
| learned | 0.5 | 100 | -0.03281 | -0.00449 | -0.01417 | -0.05796 | -0.04861 | -0.03161 | [-0.04919, -0.01403] |
| learned | 0.65 | 0 | 0.12287 | 0.10287 | 0.12361 | 0.06523 | 0.15381 | 0.11368 | [0.08781, 0.13744] |
| learned | 0.65 | 50 | 0.13346 | -0.03649 | 0.07969 | 0.08557 | 0.01709 | 0.05586 | [0.00637, 0.10355] |
| learned | 0.65 | 100 | 0.10270 | -0.01836 | 0.05215 | 0.02818 | -0.05212 | 0.02251 | [-0.02452, 0.06838] |
| random | 0.5 | 0 | -0.08934 | 0.05643 | 0.03473 | -0.07664 | -0.01548 | -0.01806 | [-0.06949, 0.03337] |
| random | 0.5 | 50 | 0.02711 | 0.02741 | 0.02331 | -0.01006 | 0.12603 | 0.03876 | [0.00487, 0.08570] |
| random | 0.5 | 100 | -0.01371 | -0.01281 | -0.00562 | -0.06056 | 0.02380 | -0.01378 | [-0.04002, 0.00897] |
| random | 0.65 | 0 | 0.10388 | 0.04444 | 0.18342 | 0.19457 | 0.03227 | 0.11172 | [0.05146, 0.17197] |
| random | 0.65 | 50 | 0.01142 | -0.03933 | 0.04731 | 0.02807 | 0.11178 | 0.03185 | [-0.00888, 0.07829] |
| random | 0.65 | 100 | -0.00645 | -0.11716 | 0.02884 | -0.02272 | 0.05230 | -0.01304 | [-0.06907, 0.03261] |

### fixed, accepted

- learned: association increases at the endpoint in 18/30 evaluable runs; final association exceeds within-layer shuffled pairings in 17/30. Run counts are descriptive; source seed is the replication unit.
- random: association increases at the endpoint in 20/30 evaluable runs; final association exceeds within-layer shuffled pairings in 7/30. Run counts are descriptive; source seed is the replication unit.
- dense: association increases at the endpoint in 0/15 evaluable runs; final association exceeds within-layer shuffled pairings in 5/15. Run counts are descriptive; source seed is the replication unit.

Equal-expert endpoint Pearson changes with all five seed values:

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | -0.21674 | -0.10393 | -0.14958 | -0.08972 | -0.15592 | -0.14318 | [-0.18201, -0.10580] |
| dense | 0.0 | 50 | -0.18963 | -0.17336 | -0.14277 | -0.02743 | -0.11056 | -0.12875 | [-0.17375, -0.07357] |
| dense | 0.0 | 100 | -0.24535 | -0.13981 | -0.17957 | -0.07920 | -0.07932 | -0.14465 | [-0.19899, -0.09137] |
| learned | 0.5 | 0 | 0.04446 | -0.01500 | -0.00485 | 0.00690 | 0.02261 | 0.01082 | [-0.00656, 0.03023] |
| learned | 0.5 | 50 | 0.08276 | -0.08768 | 0.04913 | -0.01143 | 0.01040 | 0.00864 | [-0.04507, 0.05719] |
| learned | 0.5 | 100 | -0.05741 | -0.01678 | -0.02568 | -0.06011 | -0.05607 | -0.04321 | [-0.05822, -0.02669] |
| learned | 0.65 | 0 | 0.13334 | 0.09627 | 0.11920 | 0.07779 | 0.14915 | 0.11515 | [0.09260, 0.13684] |
| learned | 0.65 | 50 | 0.13566 | -0.01845 | 0.05965 | 0.08595 | 0.01404 | 0.05537 | [0.01017, 0.10140] |
| learned | 0.65 | 100 | 0.09359 | -0.00245 | 0.05272 | 0.01528 | -0.05855 | 0.02012 | [-0.02508, 0.06229] |
| random | 0.5 | 0 | -0.09116 | 0.04995 | 0.04249 | -0.07345 | -0.01039 | -0.01651 | [-0.06792, 0.03490] |
| random | 0.5 | 50 | 0.04093 | 0.00880 | 0.03811 | 0.01179 | 0.13528 | 0.04698 | [0.01582, 0.09115] |
| random | 0.5 | 100 | -0.01619 | -0.01776 | -0.01434 | -0.07182 | 0.01661 | -0.02070 | [-0.04951, 0.00317] |
| random | 0.65 | 0 | 0.11087 | 0.04930 | 0.20679 | 0.18167 | 0.03861 | 0.11745 | [0.05734, 0.17756] |
| random | 0.65 | 50 | 0.01713 | -0.03560 | 0.07798 | 0.03541 | 0.12503 | 0.04399 | [-0.00234, 0.09290] |
| random | 0.65 | 100 | 0.01003 | -0.11828 | 0.05135 | -0.04404 | 0.05438 | -0.00931 | [-0.06951, 0.04430] |

### fixed, all_expert

- learned: association increases at the endpoint in 30/30 evaluable runs; final association exceeds within-layer shuffled pairings in 30/30. Run counts are descriptive; source seed is the replication unit.
- random: association increases at the endpoint in 30/30 evaluable runs; final association exceeds within-layer shuffled pairings in 30/30. Run counts are descriptive; source seed is the replication unit.
- dense: association increases at the endpoint in 3/15 evaluable runs; final association exceeds within-layer shuffled pairings in 3/15. Run counts are descriptive; source seed is the replication unit.

Equal-expert endpoint Pearson changes with all five seed values:

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | -0.00636 | 0.04058 | -0.05666 | 0.05367 | 0.00926 | 0.00810 | [-0.02648, 0.03955] |
| dense | 0.0 | 50 | -0.13420 | -0.07002 | -0.23803 | -0.06628 | -0.15279 | -0.13226 | [-0.18688, -0.08136] |
| dense | 0.0 | 100 | -0.21046 | -0.03926 | -0.19446 | -0.14450 | -0.15004 | -0.14775 | [-0.19407, -0.09135] |
| learned | 0.5 | 0 | 0.17238 | 0.14161 | 0.19412 | 0.14366 | 0.19271 | 0.16890 | [0.14858, 0.18921] |
| learned | 0.5 | 50 | 0.18907 | 0.11322 | 0.12894 | 0.13658 | 0.08231 | 0.13002 | [0.10096, 0.16340] |
| learned | 0.5 | 100 | 0.14180 | 0.10634 | 0.13209 | 0.04553 | 0.06829 | 0.09881 | [0.06679, 0.13082] |
| learned | 0.65 | 0 | 0.27933 | 0.26546 | 0.31068 | 0.21205 | 0.30288 | 0.27408 | [0.24090, 0.30129] |
| learned | 0.65 | 50 | 0.29093 | 0.18222 | 0.29031 | 0.21962 | 0.20506 | 0.23763 | [0.19884, 0.27642] |
| learned | 0.65 | 100 | 0.31229 | 0.17800 | 0.29774 | 0.18416 | 0.16027 | 0.22649 | [0.17214, 0.28253] |
| random | 0.5 | 0 | 0.31519 | 0.27457 | 0.30033 | 0.26377 | 0.27966 | 0.28671 | [0.27127, 0.30409] |
| random | 0.5 | 50 | 0.25694 | 0.38057 | 0.20506 | 0.22919 | 0.27023 | 0.26840 | [0.22509, 0.32557] |
| random | 0.5 | 100 | 0.22325 | 0.31733 | 0.20226 | 0.22283 | 0.19558 | 0.23225 | [0.20245, 0.27542] |
| random | 0.65 | 0 | 0.39955 | 0.36714 | 0.39455 | 0.39502 | 0.33371 | 0.37799 | [0.35356, 0.39674] |
| random | 0.65 | 50 | 0.44565 | 0.38996 | 0.43570 | 0.45405 | 0.41688 | 0.42845 | [0.40816, 0.44702] |
| random | 0.65 | 100 | 0.40214 | 0.38120 | 0.43123 | 0.41811 | 0.37793 | 0.40212 | [0.38408, 0.42016] |

### native, proposed

- learned: association increases at the endpoint in 0/30 evaluable runs; final association exceeds within-layer shuffled pairings in 22/30. Run counts are descriptive; source seed is the replication unit.
- random: association increases at the endpoint in 24/30 evaluable runs; final association exceeds within-layer shuffled pairings in 13/30. Run counts are descriptive; source seed is the replication unit.
- dense: association increases at the endpoint in 15/15 evaluable runs; final association exceeds within-layer shuffled pairings in 15/15. Run counts are descriptive; source seed is the replication unit.

Equal-expert endpoint Pearson changes with all five seed values:

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | 0.06242 | 0.04962 | 0.08217 | 0.13694 | 0.08705 | 0.08364 | [0.05986, 0.11206] |
| dense | 0.0 | 50 | 0.04369 | 0.03070 | 0.04072 | 0.10773 | 0.06579 | 0.05773 | [0.03731, 0.08393] |
| dense | 0.0 | 100 | 0.04194 | 0.07074 | 0.06262 | 0.09247 | 0.14432 | 0.08242 | [0.05597, 0.11489] |
| learned | 0.5 | 0 | -0.17703 | -0.16968 | -0.24109 | -0.14303 | -0.19737 | -0.18564 | [-0.21486, -0.15923] |
| learned | 0.5 | 50 | -0.13388 | -0.19214 | -0.12618 | -0.05910 | -0.19432 | -0.14112 | [-0.18136, -0.09934] |
| learned | 0.5 | 100 | -0.11163 | -0.16471 | -0.06985 | -0.04987 | -0.07472 | -0.09416 | [-0.13117, -0.06283] |
| learned | 0.65 | 0 | -0.10246 | -0.09979 | -0.21432 | -0.16283 | -0.14798 | -0.14548 | [-0.18165, -0.11050] |
| learned | 0.65 | 50 | -0.17470 | -0.23260 | -0.14149 | -0.12389 | -0.22709 | -0.17995 | [-0.21881, -0.14109] |
| learned | 0.65 | 100 | -0.10777 | -0.21964 | -0.11282 | -0.05358 | -0.13776 | -0.12631 | [-0.17691, -0.08226] |
| random | 0.5 | 0 | 0.08723 | 0.01234 | -0.05800 | -0.02594 | -0.06052 | -0.00898 | [-0.05259, 0.04270] |
| random | 0.5 | 50 | 0.10225 | 0.00809 | 0.07890 | 0.00770 | 0.08940 | 0.05727 | [0.02209, 0.09244] |
| random | 0.5 | 100 | 0.11495 | 0.13901 | 0.05257 | 0.06026 | 0.08736 | 0.09083 | [0.06261, 0.11906] |
| random | 0.65 | 0 | 0.00529 | -0.08532 | 0.00798 | 0.00243 | 0.03418 | -0.00709 | [-0.04908, 0.02205] |
| random | 0.65 | 50 | 0.03234 | -0.00571 | -0.05774 | 0.07595 | 0.04606 | 0.01818 | [-0.02652, 0.05527] |
| random | 0.65 | 100 | 0.03034 | 0.01787 | 0.02727 | 0.08468 | 0.04100 | 0.04023 | [0.02413, 0.06259] |

### native, accepted

- learned: association increases at the endpoint in 3/30 evaluable runs; final association exceeds within-layer shuffled pairings in 23/30. Run counts are descriptive; source seed is the replication unit.
- random: association increases at the endpoint in 22/30 evaluable runs; final association exceeds within-layer shuffled pairings in 14/30. Run counts are descriptive; source seed is the replication unit.
- dense: association increases at the endpoint in 15/15 evaluable runs; final association exceeds within-layer shuffled pairings in 15/15. Run counts are descriptive; source seed is the replication unit.

Equal-expert endpoint Pearson changes with all five seed values:

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | 0.07247 | 0.05644 | 0.08121 | 0.14171 | 0.09940 | 0.09024 | [0.06780, 0.11751] |
| dense | 0.0 | 50 | 0.03570 | 0.03934 | 0.05059 | 0.10798 | 0.06621 | 0.05996 | [0.04014, 0.08517] |
| dense | 0.0 | 100 | 0.03317 | 0.07304 | 0.08272 | 0.09759 | 0.15239 | 0.08778 | [0.05597, 0.12452] |
| learned | 0.5 | 0 | -0.13426 | -0.16127 | -0.22520 | -0.11468 | -0.17929 | -0.16294 | [-0.19783, -0.13183] |
| learned | 0.5 | 50 | -0.08202 | -0.11964 | -0.06482 | 0.02099 | -0.13145 | -0.07539 | [-0.11684, -0.02660] |
| learned | 0.5 | 100 | -0.06896 | -0.10257 | -0.02335 | 0.04021 | -0.00830 | -0.03259 | [-0.07568, 0.00868] |
| learned | 0.65 | 0 | -0.07343 | -0.08867 | -0.20349 | -0.14772 | -0.13460 | -0.12958 | [-0.16937, -0.09133] |
| learned | 0.65 | 50 | -0.11132 | -0.15960 | -0.06725 | -0.03277 | -0.16867 | -0.10792 | [-0.15357, -0.06227] |
| learned | 0.65 | 100 | -0.07081 | -0.17124 | -0.05369 | 0.02862 | -0.06738 | -0.06690 | [-0.12696, -0.00978] |
| random | 0.5 | 0 | 0.09263 | 0.01168 | -0.06244 | -0.02307 | -0.06408 | -0.00906 | [-0.05523, 0.04543] |
| random | 0.5 | 50 | 0.11567 | 0.01473 | 0.08428 | 0.02724 | 0.08993 | 0.06637 | [0.03227, 0.09910] |
| random | 0.5 | 100 | 0.11668 | 0.13984 | 0.07805 | 0.07685 | 0.07902 | 0.09809 | [0.07776, 0.12262] |
| random | 0.65 | 0 | 0.01816 | -0.08270 | 0.01229 | -0.00017 | 0.03050 | -0.00438 | [-0.04470, 0.02192] |
| random | 0.65 | 50 | 0.03606 | -0.02679 | -0.05717 | 0.07894 | 0.02279 | 0.01077 | [-0.03245, 0.05056] |
| random | 0.65 | 100 | 0.04277 | -0.00162 | 0.04602 | 0.08219 | 0.03111 | 0.04010 | [0.01679, 0.06409] |

### native, all_expert

- learned: association increases at the endpoint in 18/30 evaluable runs; final association exceeds within-layer shuffled pairings in 30/30. Run counts are descriptive; source seed is the replication unit.
- random: association increases at the endpoint in 30/30 evaluable runs; final association exceeds within-layer shuffled pairings in 30/30. Run counts are descriptive; source seed is the replication unit.
- dense: association increases at the endpoint in 3/15 evaluable runs; final association exceeds within-layer shuffled pairings in 3/15. Run counts are descriptive; source seed is the replication unit.

Equal-expert endpoint Pearson changes with all five seed values:

| mask_type | sparsity | router_age | Seed 7 | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Mean | 95% seed bootstrap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense | 0.0 | 0 | -0.04655 | 0.02704 | -0.06229 | 0.03767 | 0.00188 | -0.00845 | [-0.04316, 0.02626] |
| dense | 0.0 | 50 | -0.14734 | -0.10362 | -0.14719 | -0.05681 | -0.15116 | -0.12122 | [-0.14884, -0.08489] |
| dense | 0.0 | 100 | -0.17023 | -0.03903 | -0.10914 | -0.14867 | -0.11827 | -0.11707 | [-0.15121, -0.07498] |
| learned | 0.5 | 0 | 0.00628 | -0.02298 | -0.03297 | -0.01600 | 0.00478 | -0.01218 | [-0.02558, 0.00123] |
| learned | 0.5 | 50 | 0.03008 | -0.10371 | -0.06103 | -0.00562 | -0.06325 | -0.04071 | [-0.07899, 0.00152] |
| learned | 0.5 | 100 | -0.01445 | -0.05348 | -0.06387 | -0.06343 | -0.07676 | -0.05440 | [-0.06944, -0.03405] |
| learned | 0.65 | 0 | 0.12103 | 0.12908 | 0.09742 | 0.07594 | 0.11765 | 0.10822 | [0.08925, 0.12358] |
| learned | 0.65 | 50 | 0.13542 | 0.01639 | 0.08529 | 0.07275 | 0.05191 | 0.07235 | [0.03893, 0.10869] |
| learned | 0.65 | 100 | 0.20294 | 0.02810 | 0.07785 | 0.06763 | 0.03675 | 0.08265 | [0.03946, 0.14469] |
| random | 0.5 | 0 | 0.33303 | 0.30995 | 0.31295 | 0.29819 | 0.34197 | 0.31922 | [0.30585, 0.33321] |
| random | 0.5 | 50 | 0.29349 | 0.37967 | 0.24819 | 0.29080 | 0.30347 | 0.30312 | [0.26830, 0.34466] |
| random | 0.5 | 100 | 0.30591 | 0.35051 | 0.26143 | 0.27623 | 0.25127 | 0.28907 | [0.26033, 0.32174] |
| random | 0.65 | 0 | 0.42133 | 0.41419 | 0.39251 | 0.41905 | 0.41800 | 0.41301 | [0.40261, 0.41975] |
| random | 0.65 | 50 | 0.47195 | 0.41301 | 0.40707 | 0.47014 | 0.48849 | 0.45013 | [0.42123, 0.47821] |
| random | 0.65 | 100 | 0.45022 | 0.44116 | 0.42554 | 0.46203 | 0.49725 | 0.45524 | [0.43596, 0.47662] |

## Joint observations and discordant cases

- learned: leading-rank overlap decreases in 28/30; both overlap decreases and full Gram distance increases in 22/30. Native proposed-pair coefficients increase despite decreasing overlap in 0/30, and despite increasing full Gram distance in 0/30. Positive final association above shuffled controls coexists with full Gram departure in 15/30. A higher coefficient can remain negative; endpoint increases alone are not evidence of positive coupling.
- random: leading-rank overlap decreases in 0/30; both overlap decreases and full Gram distance increases in 0/30. Native proposed-pair coefficients increase despite decreasing overlap in 0/30, and despite increasing full Gram distance in 14/30. Positive final association above shuffled controls coexists with full Gram departure in 6/30. A higher coefficient can remain negative; endpoint increases alone are not evidence of positive coupling.
- dense: leading-rank overlap decreases in 0/15; both overlap decreases and full Gram distance increases in 0/15. Native proposed-pair coefficients increase despite decreasing overlap in 0/15, and despite increasing full Gram distance in 3/15. Positive final association above shuffled controls coexists with full Gram departure in 3/15. A higher coefficient can remain negative; endpoint increases alone are not evidence of positive coupling.

Every concordant and discordant case is retained in `combined_seed_cases.csv`; loss and sparse-minus-matched-dense gap accompany each metric. `joint_expert_endpoint_effects.parquet` and `joint_matched_expert_seed_effects.csv` additionally enforce identical expert eligibility across SVD, Gram and activation measurements. Learned/random and router-age joint comparisons match experts across conditions before seed bootstrap uncertainty. See `existing_reference_findings.md` for separate source and pruning-only findings. The following illustrative rows are selected descriptively, not for a hypothesis test:

| Condition | Δ rank-16 overlap | Δ source Gram distance | Δ native proposed Pearson | Final dense gap |
| --- | ---: | ---: | ---: | ---: |
| G_e32_seed43_R100_learned_s0.50 | -0.00646 | -0.04585 | -0.04987 | 0.00195 |
| G_e32_seed43_R50_learned_s0.50 | -0.01084 | -0.03487 | -0.05910 | 0.01169 |
| G_e32_seed17_R50_learned_s0.50 | -0.04400 | -0.00042 | -0.19214 | 0.01636 |
| G_e32_seed17_R100_dense | 0.08980 | 0.00491 | 0.07074 | 0.00000 |
| G_e32_seed61_R50_dense | 0.08532 | 0.01205 | 0.06579 | 0.00000 |
| G_e32_seed61_R100_dense | 0.07481 | 0.01929 | 0.14432 | 0.00000 |
| G_e32_seed43_R0_learned_s0.50 | 0.00526 | -0.05025 | -0.14303 | -0.02032 |
| G_e32_seed17_R0_learned_s0.50 | 0.00163 | -0.01807 | -0.16968 | -0.01540 |
| G_e32_seed7_R100_dense | 0.08814 | -0.00567 | 0.04194 | 0.00000 |
| G_e32_seed61_R0_learned_s0.50 | -0.07339 | 0.04866 | -0.19737 | -0.04756 |

## Coverage and uncertainty

There are 80294 excluded expert/group/time/mode/selection observations, with reasons and counts in `activation_coverage.csv`. Groups need at least eight samples and nonzero variance. Repeated tokens, experts, layers, and checkpoints are not independent training replicates. The original probe was not enlarged; sample-poor routed experts remain explicitly missing. This probe is descriptive and may underpower routed correlations.

All 89,600 expert/state Gram calculations used actual weights. Of these, 86656 also match the independently recorded original SVD energy within relative tolerance 2e-6. Older SVD tables omit that field for 2944 rows (23 states); their actual Gram trace is available, but that extra archival energy check cannot be made. Exact seed/condition rows are marked in `svd_gram_weight_identity_consistency.csv`. This is a missing legacy field, not a missing weight checkpoint.

All SVD and Gram endpoint summary strata retain five seeds. Some activation layer/mode/selection strata retain zero to four seeds after fixed expert matching, and joint three-metric strata can retain only two seeds. Empty strata have undefined means/intervals, not zero effects. These coverage-limited results are not five-seed evidence; all exceptions and available seed values are in `summary_coverage_exceptions.csv`. Two-seed intervals are particularly unstable and support no strong inference.

Paired learned/random comparisons match valid experts across both mask types and required steps before computing effects. Uncertainty uses 10,000 paired seed bootstraps at fixed seed 8103, n≤5, and all five individual values are provided. These are pointwise exploratory intervals, not simultaneous confidence bands. No formal p-values or multiplicity-adjusted significance claims are made; multiplicity is not corrected, so results must not be interpreted as confirmatory. Within-condition loss associations use five seed-level values and remain unstable descriptive sensitivities.

Equal Gram matrices do not imply identical nonlinear experts: left-orthogonal input-projection transformations preserve G but need not preserve GELU. Gram excludes bias, fc2, nonlinear computation, and routing. Higher score–activation association is not proof of source restoration or a causal mechanism. Representation drift is measured separately, and conclusions vary with reference, probe mode, rank, and layer.

## Conclusion

**(c) Metric-dependent behavior** is the appropriate overall conclusion: rank-16 overlap decreases in 28/30 learned runs, while full-spectrum source Gram distance increases in 22/30. Source approach occurs in some states/metrics; it is not universal restoration. Native proposed coefficients increase while full Gram shape departs in 0/30 learned runs, but only 15/30 simultaneously show positive final proposed association above shuffled controls and full Gram departure. Selected, accepted and all-expert results must be distinguished. Where favorable losses coexist with positive above-control association and departure, they support limited descriptive evidence for (b), not a causal mechanism. Read the per-layer, per-seed, mask-control, probe-mode and reference-choice results before narrowing this conclusion.

## Reproduction

Run from repository root:

```bash
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/validate.py
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/svd.py
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/run.py --seed 7 --benchmark
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/run.py
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/analyze.py
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/existing_summary.py
OMP_NUM_THREADS=2 .venv-highres/bin/python scripts/router_expert_geometry_extended/audit.py
```

Inference-only per-seed workers may use `--seed 7`, `17`, `29`, `43`, or `61`; completion caches are keyed by checkpoint/reference/probe/config and measurement-code hashes. Original artifacts are read-only. All outputs live in `analysis_extended/v1/`. Figures are PNG and PDF. Raw token pairs retain sample identity in `cache/**/token_pairs.npz`.
