# Tied byte-token embedding coordinate sparsity

Primary analysis: 240 rerun checkpoint records (15 conditions ×16 steps); 75 historical records are separate and have identical E at all common states. Additional reset/native/interruption aliases are documented in checkpoint_provenance.csv and excluded from comparisons. Historical initialization uses the actual saved original E, with no missing trained-state reconstruction. All 165 absent historical slots are listed in coverage.csv.

E is token_embedding.weight, shape 256×256 (65,536 entries), tied to lm_head.weight. Position embeddings are excluded. All values were saved in torch.float32 and converted to CPU float64 solely for statistics. Per-checkpoint file hashes were freshly recomputed; source files and prior experiment outputs were not modified.

1. **Exact zeros:** 0 across all primary checkpoint records; all tensors are finite. No E acquires exact zeros. Repeated records are not independent observations.

2. **Scale and shape:**

- Dense: final RMS mean 0.982112992; mean relative change from initialization -1.771104% (seed range -1.812197% to -1.754023%).
- 80% sparse: final RMS mean 0.980925646; mean relative change from initialization -1.889760% (seed range -1.951820% to -1.833305%).
- 85% sparse: final RMS mean 0.980954065; mean relative change from initialization -1.886964% (seed range -1.920282% to -1.851398%).
- tau=.1, current_rms, dense: final fraction mean 0.079998779, initialization change mean -0.000039673, seed range -0.001449585 to +0.001281738.
- tau=.1, current_rms, 80%: final fraction mean 0.080334473, initialization change mean +0.000296021, seed range -0.001586914 to +0.001098633.
- tau=.1, current_rms, 85%: final fraction mean 0.080368042, initialization change mean +0.000329590, seed range -0.001037598 to +0.000747681.
- tau=.1, original_initialization_rms, dense: final fraction mean 0.081475830, initialization change mean +0.001437378, seed range +0.000000000 to +0.002624512.
- tau=.1, original_initialization_rms, 80%: final fraction mean 0.081799316, initialization change mean +0.001760864, seed range -0.000061035 to +0.002548218.
- tau=.1, original_initialization_rms, 85%: final fraction mean 0.081860352, initialization change mean +0.001821899, seed range +0.000549316 to +0.002365112.

The initialization scale retains the effect of global magnitude shrinkage; the current scale removes that overall scale and measures residual distribution-shape changes. Both effects can coexist. The complete 61-point curves and full empirical CDFs are exported; conclusions are not restricted to tau=.1.

3. **Matched sparse–dense comparisons:** Initial E tensors are verified identical across all three conditions within each seed.

- 80%: final RMS difference mean -0.001187347, range -0.001929280 to -0.000755035; negative in 5/5 seeds.
  tau=.1 (current_rms) difference mean +0.000335693, range -0.000183105 to +0.000885010; positive in 3/5 seeds.
  tau=.1 (original_initialization_rms) difference mean +0.000323486, range -0.000076294 to +0.000900269; positive in 3/5 seeds.
- 85%: final RMS difference mean -0.001158928, range -0.001612653 to -0.000952631; negative in 5/5 seeds.
  tau=.1 (current_rms) difference mean +0.000369263, range -0.000534058 to +0.001037598; positive in 4/5 seeds.
  tau=.1 (original_initialization_rms) difference mean +0.000384521, range -0.000259399 to +0.000930786; positive in 4/5 seeds.

4. **Rows and coordinates:** Row and column CSVs preserve IDs and include all requested exact/near-zero fractions under both global scales. A small row has both an input-embedding and tied output-prediction role; its norm does not establish whether the token is used.

- Dense rows: final L2 range 13.491421–17.795073; relative change median -1.663221%, 5th–95th percentiles -2.655895% to -1.050585% (descriptive range over IDs and seeds, not uncertainty). No all-zero rows; 0 whole rows have RMS <=.1×initial global RMS.
- 80% rows: final L2 range 13.408525–17.787847; relative change median -1.783263%, 5th–95th percentiles -2.781579% to -1.223541% (descriptive range over IDs and seeds, not uncertainty). No all-zero rows; 0 whole rows have RMS <=.1×initial global RMS.
- 85% rows: final L2 range 13.432472–17.779055; relative change median -1.792424%, 5th–95th percentiles -2.775013% to -1.221439% (descriptive range over IDs and seeds, not uncertainty). No all-zero rows; 0 whole rows have RMS <=.1×initial global RMS.
- Dense columns: final L2 range 13.528342–18.718893; relative change median -1.773125%, 5th–95th percentiles -2.376304% to -1.152085% (descriptive range over IDs and seeds, not uncertainty). No all-zero columns; 0 whole columns have RMS <=.1×initial global RMS.
- 80% columns: final L2 range 13.522175–18.764997; relative change median -1.895832%, 5th–95th percentiles -2.522698% to -1.253641% (descriptive range over IDs and seeds, not uncertainty). No all-zero columns; 0 whole columns have RMS <=.1×initial global RMS.
- 85% columns: final L2 range 13.511193–18.718493; relative change median -1.891553%, 5th–95th percentiles -2.521224% to -1.241972% (descriptive range over IDs and seeds, not uncertainty). No all-zero columns; 0 whole columns have RMS <=.1×initial global RMS.

5. **Ticket labels (descriptive only):** Four of ten sparse models have strict sparse<=dense validation labels in each track; these are seed 7 at .80/.85, seed 43 at .80, and seed 61 at .80. Historical/rerun weights duplicate and do not provide replication. Thresholds were fixed before computation and not chosen to separate labels.

- Seed 7, 80%, winner: RMS sparse−dense -0.000755035.
- Seed 7, 85%, winner: RMS sparse−dense -0.000952631.
- Seed 17, 80%, nonwinner: RMS sparse−dense -0.001185832.
- Seed 17, 85%, nonwinner: RMS sparse−dense -0.001236910.
- Seed 29, 80%, nonwinner: RMS sparse−dense -0.001276334.
- Seed 29, 85%, nonwinner: RMS sparse−dense -0.001021841.
- Seed 43, 80%, winner: RMS sparse−dense -0.000790253.
- Seed 43, 85%, nonwinner: RMS sparse−dense -0.000970603.
- Seed 61, 80%, winner: RMS sparse−dense -0.001929280.
- Seed 61, 85%, nonwinner: RMS sparse−dense -0.001612653.

The same direction of a small RMS change appears among winners and nonwinners; it does not separate the labels. Per-seed exceptions and the dependent second sparsity prevent interpreting pooled winner averages as independent evidence. See endpoint_ticket_annotations.csv for all metrics and labels.

## Interpretation and limitations

Small coordinates are near-zero entries, not evidence of inactive, useless, or safely removable parameters. The row/column results must not be interpreted as token usage. Coordinate sparsity is distinct from low-rank structure and earlier CKA measurements. No causal, dispensability, or winning-ticket mechanism claim follows from this checkpoint-only comparison. Seeds (n=5) are the independent units; no entry-, token-, coordinate-, or checkpoint-level confidence intervals are used. Equal-seed averages and seed ranges above are descriptive. Historical missing states remain unavailable. All plots use rerun trajectories as primary; the identical historical endpoints are retained in tables, not double-counted.

## Outputs

analysis_config.json fixes the tau grid, scales, quantiles, algorithms and input hashes before summary computation. checkpoint_provenance.csv includes physical file SHA256, E hash, saved dtype/shape, track/seed/condition/step and alias roles. checkpoint_summary.csv and threshold_curves.csv include changes from initialization. row_statistics.csv and column_statistics.csv retain IDs. sparse_dense_paired.csv contains scalar contrasts; paired_threshold_curves.csv contains every threshold contrast, and paired_row_statistics.csv / paired_column_statistics.csv contain ID-matched norm and near-zero contrasts. cdf/*.npz exports every unique empirical CDF without downsampling (including exact zeros). figures/ contains A–F in PNG and PDF. numerical_checks.json records focused checks and coverage. All files are confined to this directory.

## Row/coordinate concentration and label detail

Storage aliases: 135/390 physical provenance records retain shared E/head storage in their saved state dict. Every record has identical E/head values; the instantiated model ties the parameter regardless of whether a checkpoint cloned its state-dict tensors. E is counted once.

For rows, the fraction with a negative norm change ranges from 0.992 to 1.000 across all 15 conditions. The top 26/256 IDs account for 0.150–0.166 of summed absolute norm change. Thus the change is spread across many IDs rather than confined to a few zero or globally weak rows/columns. Specific extreme IDs and ranges are retained in row_concentration.csv; these are descriptive magnitude changes, not token usage or coordinate dispensability.

For columns, the fraction with a negative norm change ranges from 0.996 to 1.000 across all 15 conditions. The top 26/256 IDs account for 0.137–0.148 of summed absolute norm change. Thus the change is spread across many IDs rather than confined to a few zero or globally weak rows/columns. Specific extreme IDs and ranges are retained in column_concentration.csv; these are descriptive magnitude changes, not token usage or coordinate dispensability.

80% winners: 3 conditions, current-scale tau=.1 sparse−dense fraction differences -0.000183105 to +0.000808716; positive in 1/3. These are post hoc subsets of the same five seeds; pooling both sparsities would not add independent replication.

80% nonwinners: 2 conditions, current-scale tau=.1 sparse−dense fraction differences +0.000305176 to +0.000885010; positive in 2/2. These are post hoc subsets of the same five seeds; pooling both sparsities would not add independent replication.

85% winners: 1 condition, current-scale tau=.1 sparse−dense fraction differences -0.000534058 to -0.000534058; positive in 0/1. These are post hoc subsets of the same five seeds; pooling both sparsities would not add independent replication.

85% nonwinners: 4 conditions, current-scale tau=.1 sparse−dense fraction differences +0.000259399 to +0.001037598; positive in 4/4. These are post hoc subsets of the same five seeds; pooling both sparsities would not add independent replication.

row_statistics.csv and column_statistics.csv also include each statistic’s change from the corresponding condition initialization at every available step. endpoint_seed_summary.csv reports equal-seed means, seed ranges and descriptive seed standard deviations. Both historical and rerun ticket annotations are retained in sparse_dense_paired.csv. Independent numerical checks pass in independent_validation.json.

## Threshold-specific endpoint interpretation

At tau=.1, the equal-seed mean increase from initialization is 0.176086 / 0.182190 percentage points (80% / 85%) with the fixed initialization scale, versus 0.029602 / 0.032959 percentage points after current-RMS normalization. Most of that mean increase disappears after removing overall scale shrinkage; residual shape changes vary across seeds. Dense changes are +0.143738 percentage points with the initialization scale and −0.003967 with current RMS. These are descriptive changes of the measured distributions.

The sparse–dense near-zero differences depend on threshold and sparsity; there is no uniform increase at every threshold. The predefined current-scale endpoints are:

| Expert sparsity | tau | Mean sparse−dense fraction | Seed range | Positive / 5 |
|---|---:|---:|---|---:|
| 80% | 0.001 | -0.000119019 | -0.000244141 to +0.000045776 | 1/5 |
| 80% | 0.01 | -0.000405884 | -0.000823975 to -0.000091553 | 0/5 |
| 80% | 0.1 | +0.000335693 | -0.000183105 to +0.000885010 | 3/5 |
| 85% | 0.001 | -0.000021362 | -0.000274658 to +0.000213623 | 2/5 |
| 85% | 0.01 | +0.000003052 | -0.000183105 to +0.000106812 | 3/5 |
| 85% | 0.1 | +0.000369263 | -0.000534058 to +0.001037598 | 4/5 |

In particular, the 80% condition has a lower tau=.01 current-scale near-zero fraction in all five seeds (mean −0.000405884), whereas the 85% condition has almost no mean difference there (+0.000003052) and mixed signs. The fixed-initialization scale also shows a negative tau=.01 contrast for 80% in all five seeds (mean −0.000363159). Full-grid effects are in paired_threshold_curves.csv; these examples use the three thresholds specified in the request.

The winner pattern is not consistent across winning seeds: at 80%, winning seed 61 has a positive current-scale tau=.1 contrast (+0.000808716), like nonwinning seeds 17 and 29, while winning seeds 7 and 43 have negative contrasts. At 85%, the only winning seed is 7 (negative contrast); one winning seed cannot establish replication across winning seeds. RMS is lower for every sparse endpoint, including all winners and nonwinners. These annotations do not establish a mechanism or classifier.

No row or coordinate column is wholly near-zero by a global RMS <=.1×initial-RMS criterion. Positive per-entry near-zero fractions therefore do not indicate whole near-zero rows or columns. Signed histogram counts are exported in signed_histogram_counts.csv, and CDF plots include the smallest observed positive magnitude. Figures C/D/F were checked and redrawn with separate figure and panel titles.
