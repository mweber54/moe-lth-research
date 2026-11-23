# Gate-amplitude pattern decomposition

All seven specified conditions completed on the 85% sparsity grid. Eighteen canonical age/seed dense amplitude sources were reconstructed solely to generate artifacts and validated against Level 2. Each transformed amplitude schedule was replayed identically into its matched sparse and dense recipients.

The primary readout is the same original R100 assignments, acceptances, capacity priorities and gate amplitudes for every condition and both reference levels. Condition-matched and native readouts are secondary and are never mixed into primary contrasts or retention.

## Source identity and actual shuffle perturbation

| experts | seed | age | recurring_edge_fraction | token_conditioned_definition |
| --- | --- | --- | --- | --- |
| 8 | 7 | 0 | 0.904114 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 7 | 20 | 0.904114 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 7 | 100 | 0.904114 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 17 | 0 | 0.906772 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 17 | 20 | 0.906772 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 17 | 100 | 0.906772 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 29 | 0 | 0.913349 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 29 | 20 | 0.913349 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 8 | 29 | 100 | 0.913349 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 7 | 20 | 0.891428 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 7 | 60 | 0.891428 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 7 | 100 | 0.891428 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 17 | 20 | 0.898702 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 17 | 60 | 0.898702 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 17 | 100 | 0.898702 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 29 | 20 | 0.879441 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 29 | 60 | 0.879441 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |
| 12 | 29 | 100 | 0.879441 | exact repeated batch hash + token position + layer + donor-assigned expert; singleton edges retained unchanged |

| experts | phase | temporal_moved_edge_min | temporal_moved_edge_max | temporal_changed_value_min | temporal_changed_value_max | token_changed_value_min | token_changed_value_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | train | 0.889338 | 0.897039 | 0.889317 | 0.897036 | 0.999427 | 0.999458 |
| 8 | validation | 0.405407 | 0.484904 | 0.405405 | 0.484904 | 0.999381 | 0.999427 |
| 12 | train | 0.931521 | 0.939253 | 0.931268 | 0.93925 | 0.998725 | 0.999187 |
| 12 | validation | 0.408297 | 0.479225 | 0.408235 | 0.47917 | 0.998607 | 0.999131 |

## Final contrast hierarchy

| experts | target_age | control_age | level1_native_amplitude | global_mean | layer_mean | expert_layer_mean | token_conditioned | temporal | temporal_shuffle | token_shuffle | level2_r100_amplitude |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | -0.0654941 | -0.290065 | 0.380315 | 0.374996 | -5.54438 | 2.72292 | 1.08914 | 0.358503 | -0.00386188 |
| 8 | 20 | 100 | -0.318137 | 2.57997 | 0.550531 | -8.32255 | -3.50917 | -0.749412 | -1.46701 | -2.92834 | -0.0010511 |
| 12 | 60 | 20 | 0.930014 | -0.123947 | -0.545117 | 0.291677 | 2.87443 | -0.453466 | 0.651096 | 0.565427 | -0.000280834 |
| 12 | 60 | 100 | -0.421066 | 1.11486 | 0.781416 | 1.9114 | -0.853188 | -4.75764 | -0.164765 | -0.303916 | 0.00141751 |

## Paired contrasts

| experts | seed | target_age | control_age | amplitude_condition | target_gap | control_gap | contrast | favors_target | contrast_change_vs_level1 | contrast_change_vs_level2 | retention_score | near_zero_denominator | sign_flip_vs_level1 | level1_contrast | level2_contrast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 7 | 20 | 0 | level1_native_amplitude | 0.138233 | 0.0793076 | 0.0589256 | False | 0 | 0.0659399 | 1 | False | False | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | global_mean | 4.76854 | 5.34025 | -0.57171 | True | -0.630636 | -0.564696 | -8.5638 | False | True | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | layer_mean | 4.50117 | 5.248 | -0.746826 | True | -0.805751 | -0.739812 | -11.2195 | False | True | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | expert_layer_mean | 5.47623 | 4.80106 | 0.675166 | False | 0.616241 | 0.682181 | 10.3455 | False | False | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | token_conditioned | -6.55856 | -5.90584 | -0.652718 | True | -0.711643 | -0.645703 | -9.79231 | False | True | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | temporal | -39.0662 | -37.216 | -1.85019 | True | -1.90912 | -1.84318 | -27.9524 | False | True | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | temporal_shuffle | -4.77751 | -7.38334 | 2.60583 | False | 2.5469 | 2.61284 | 39.6246 | False | False | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | token_shuffle | -16.5313 | -8.61996 | -7.91138 | True | -7.9703 | -7.90436 | -119.872 | False | True | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 0 | level2_r100_amplitude | 0.257032 | 0.264046 | -0.00701421 | True | -0.0659399 | 0 | 0 | False | True | 0.0589256 | -0.00701421 |
| 8 | 7 | 20 | 100 | level1_native_amplitude | 0.138233 | 0.352033 | -0.2138 | True | 0 | -0.207575 | 1 | False | False | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | global_mean | 4.76854 | 2.83286 | 1.93568 | False | 2.14948 | 1.9419 | -9.35518 | False | True | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | layer_mean | 4.50117 | 4.50041 | 0.000766253 | False | 0.214566 | 0.00699069 | -0.0336779 | False | True | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | expert_layer_mean | 5.47623 | 12.3844 | -6.90819 | True | -6.69439 | -6.90196 | 33.2505 | False | False | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | token_conditioned | -6.55856 | -7.59171 | 1.03315 | False | 1.24695 | 1.03938 | -5.00723 | False | True | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | temporal | -39.0662 | -41.5652 | 2.49894 | False | 2.71274 | 2.50516 | -12.0687 | False | True | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | temporal_shuffle | -4.77751 | -4.61358 | -0.163939 | True | 0.0498601 | -0.157715 | 0.759797 | False | False | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | token_shuffle | -16.5313 | -12.8584 | -3.67293 | True | -3.45913 | -3.66671 | 17.6645 | False | False | -0.2138 | -0.00622444 |
| 8 | 7 | 20 | 100 | level2_r100_amplitude | 0.257032 | 0.263256 | -0.00622444 | True | 0.207575 | 0 | -0 | False | False | -0.2138 | -0.00622444 |
| 8 | 17 | 20 | 0 | level1_native_amplitude | 0.0672897 | 0.150209 | -0.0829194 | True | 0 | -0.0836922 | 1 | False | False | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | global_mean | 7.00447 | 7.04003 | -0.0355615 | True | 0.0473579 | -0.0363343 | 0.434142 | False | False | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | layer_mean | 11.1676 | 9.45187 | 1.71574 | False | 1.79866 | 1.71497 | -20.4914 | False | True | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | expert_layer_mean | -5.06461 | -8.29415 | 3.22954 | False | 3.31246 | 3.22876 | -38.579 | False | True | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | token_conditioned | -10.6556 | -1.47875 | -9.17681 | True | -9.09389 | -9.17758 | 109.659 | False | False | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | temporal | -39.2179 | -45.0592 | 5.84132 | False | 5.92424 | 5.84055 | -69.786 | False | True | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | temporal_shuffle | -6.30074 | -5.87836 | -0.422381 | True | -0.339461 | -0.423153 | 5.05607 | False | False | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | token_shuffle | -12.6346 | -21.1971 | 8.56246 | False | 8.64538 | 8.56169 | -102.3 | False | True | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 0 | level2_r100_amplitude | 0.20148 | 0.200707 | 0.000772865 | False | 0.0836922 | 0 | -0 | False | True | -0.0829194 | 0.000772865 |
| 8 | 17 | 20 | 100 | level1_native_amplitude | 0.0672897 | 0.263571 | -0.196281 | True | 0 | -0.197861 | 1 | False | False | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | global_mean | 7.00447 | 5.73326 | 1.27121 | False | 1.46749 | 1.26963 | -6.41677 | False | True | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | layer_mean | 11.1676 | 8.94954 | 2.21807 | False | 2.41435 | 2.21649 | -11.2022 | False | True | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | expert_layer_mean | -5.06461 | 3.25128 | -8.31589 | True | -8.11961 | -8.31747 | 42.0369 | False | False | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | token_conditioned | -10.6556 | -7.01541 | -3.64014 | True | -3.44386 | -3.64172 | 18.4054 | False | False | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | temporal | -39.2179 | -31.5799 | -7.63802 | True | -7.44174 | -7.6396 | 38.6108 | False | False | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | temporal_shuffle | -6.30074 | -3.98095 | -2.31979 | True | -2.12351 | -2.32137 | 11.7323 | False | False | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | token_shuffle | -12.6346 | -8.97692 | -3.65772 | True | -3.46144 | -3.6593 | 18.4943 | False | False | -0.196281 | 0.00158009 |
| 8 | 17 | 20 | 100 | level2_r100_amplitude | 0.20148 | 0.1999 | 0.00158009 | False | 0.197861 | 0 | -0 | False | True | -0.196281 | 0.00158009 |
| 8 | 29 | 20 | 0 | level1_native_amplitude | -0.172134 | 0.00035401 | -0.172488 | True | 0 | -0.167144 | 1 | False | False | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | global_mean | 5.72186 | 5.98478 | -0.262923 | True | -0.0904348 | -0.257579 | 1.54106 | False | False | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | layer_mean | 8.90751 | 8.73548 | 0.172027 | False | 0.344515 | 0.177371 | -1.06119 | False | True | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | expert_layer_mean | -2.63999 | 0.139723 | -2.77971 | True | -2.60723 | -2.77437 | 16.5987 | False | False | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | token_conditioned | -12.7853 | -5.98164 | -6.80363 | True | -6.63114 | -6.79828 | 40.6732 | False | False | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | temporal | -17.0205 | -21.1981 | 4.17763 | False | 4.35012 | 4.18297 | -25.0261 | False | True | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | temporal_shuffle | -4.41994 | -5.50391 | 1.08397 | False | 1.25646 | 1.08931 | -6.51722 | False | True | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | token_shuffle | -10.0443 | -10.4688 | 0.424424 | False | 0.596912 | 0.429768 | -2.57124 | False | True | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 0 | level2_r100_amplitude | 0.226776 | 0.23212 | -0.0053443 | True | 0.167144 | 0 | -0 | False | False | -0.172488 | -0.0053443 |
| 8 | 29 | 20 | 100 | level1_native_amplitude | -0.172134 | 0.372197 | -0.544331 | True | 0 | -0.545822 | 1 | False | False | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | global_mean | 5.72186 | 1.18884 | 4.53302 | False | 5.07735 | 4.53153 | -8.30221 | False | True | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | layer_mean | 8.90751 | 9.47475 | -0.567244 | True | -0.0229125 | -0.568735 | 1.04198 | False | False | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | expert_layer_mean | -2.63999 | 7.10358 | -9.74357 | True | -9.19923 | -9.74506 | 17.8539 | False | False | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | token_conditioned | -12.7853 | -4.86476 | -7.92051 | True | -7.37618 | -7.922 | 14.5139 | False | False | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | temporal | -17.0205 | -19.9113 | 2.89084 | False | 3.43517 | 2.88935 | -5.29357 | False | True | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | temporal_shuffle | -4.41994 | -2.50264 | -1.91731 | True | -1.37298 | -1.9188 | 3.51543 | False | False | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | token_shuffle | -10.0443 | -8.58998 | -1.45436 | True | -0.910027 | -1.45585 | 2.66726 | False | False | -0.544331 | 0.00149106 |
| 8 | 29 | 20 | 100 | level2_r100_amplitude | 0.226776 | 0.225285 | 0.00149106 | False | 0.545822 | 0 | -0 | False | True | -0.544331 | 0.00149106 |
| 12 | 7 | 60 | 20 | level1_native_amplitude | 0.390023 | -0.0157088 | 0.405732 | False | 0 | 0.407428 | 1 | False | False | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | global_mean | 8.54154 | 9.63559 | -1.09406 | True | -1.49979 | -1.09236 | -2.68111 | False | True | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | layer_mean | 16.9739 | 18.724 | -1.75007 | True | -2.15581 | -1.74838 | -4.29125 | False | True | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | expert_layer_mean | 11.341 | 13.9096 | -2.5686 | True | -2.97433 | -2.5669 | -6.30025 | False | True | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | token_conditioned | -10.8083 | -11.1953 | 0.386951 | False | -0.0187807 | 0.388648 | 0.953904 | False | False | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | temporal | -34.9904 | -34.654 | -0.336371 | True | -0.742103 | -0.334675 | -0.821433 | False | True | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | temporal_shuffle | -4.00997 | -4.85822 | 0.848254 | False | 0.442522 | 0.84995 | 2.08613 | False | False | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | token_shuffle | -15.3197 | -17.768 | 2.44829 | False | 2.04256 | 2.44998 | 6.01329 | False | False | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 20 | level2_r100_amplitude | 0.220295 | 0.221991 | -0.0016963 | True | -0.407428 | 0 | 0 | False | True | 0.405732 | -0.0016963 |
| 12 | 7 | 60 | 100 | level1_native_amplitude | 0.390023 | 0.510996 | -0.120973 | True | 0 | -0.12032 | 1 | False | False | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | global_mean | 8.54154 | 9.54069 | -0.999151 | True | -0.878178 | -0.998499 | 8.29867 | False | False | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | layer_mean | 16.9739 | 15.8104 | 1.16355 | False | 1.28452 | 1.1642 | -9.67583 | False | True | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | expert_layer_mean | 11.341 | 11.4293 | -0.0882495 | True | 0.0327236 | -0.0875968 | 0.72803 | False | False | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | token_conditioned | -10.8083 | -7.18983 | -3.61851 | True | -3.49754 | -3.61786 | 30.0685 | False | False | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | temporal | -34.9904 | -34.813 | -0.177328 | True | -0.0563552 | -0.176676 | 1.46838 | False | False | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | temporal_shuffle | -4.00997 | -4.22799 | 0.21802 | False | 0.338994 | 0.218673 | -1.81742 | False | True | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | token_shuffle | -15.3197 | -12.0459 | -3.27382 | True | -3.15284 | -3.27316 | 27.2037 | False | False | -0.120973 | -0.000652749 |
| 12 | 7 | 60 | 100 | level2_r100_amplitude | 0.220295 | 0.220947 | -0.000652749 | True | 0.12032 | 0 | -0 | False | False | -0.120973 | -0.000652749 |
| 12 | 17 | 60 | 20 | level1_native_amplitude | 0.174979 | -0.787432 | 0.962411 | False | 0 | 0.961202 | 1 | False | False | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | global_mean | 10.8536 | 10.9352 | -0.081606 | True | -1.04402 | -0.0828153 | -0.086158 | False | True | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | layer_mean | 12.3293 | 13.2526 | -0.923208 | True | -1.88562 | -0.924417 | -0.96173 | False | True | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | expert_layer_mean | 10.1608 | 10.516 | -0.355186 | True | -1.3176 | -0.356395 | -0.370781 | False | True | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | token_conditioned | -4.55266 | -8.13552 | 3.58285 | False | 2.62044 | 3.58165 | 3.72622 | False | False | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | temporal | -31.1125 | -41.2306 | 10.1181 | False | 9.15569 | 10.1169 | 10.5253 | False | False | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | temporal_shuffle | -2.72807 | -3.07605 | 0.347976 | False | -0.614435 | 0.346767 | 0.360764 | False | False | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | token_shuffle | -11.5124 | -10.7658 | -0.746655 | True | -1.70907 | -0.747864 | -0.778051 | False | True | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 20 | level2_r100_amplitude | 0.213278 | 0.212069 | 0.00120922 | False | -0.961202 | 0 | 0 | False | False | 0.962411 | 0.00120922 |
| 12 | 17 | 60 | 100 | level1_native_amplitude | 0.174979 | 0.30704 | -0.132061 | True | 0 | -0.132606 | 1 | False | False | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | global_mean | 10.8536 | 8.53737 | 2.31627 | False | 2.44833 | 2.31573 | -17.4633 | False | True | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | layer_mean | 12.3293 | 12.543 | -0.213663 | True | -0.081602 | -0.214208 | 1.61537 | False | False | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | expert_layer_mean | 10.1608 | 6.85636 | 3.30442 | False | 3.43649 | 3.30388 | -24.9151 | False | True | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | token_conditioned | -4.55266 | -4.67561 | 0.122944 | False | 0.255005 | 0.1224 | -0.923036 | False | True | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | temporal | -31.1125 | -20.5596 | -10.5529 | True | -10.4208 | -10.5534 | 79.585 | False | False | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | temporal_shuffle | -2.72807 | -1.8324 | -0.895667 | True | -0.763606 | -0.896212 | 6.75847 | False | False | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | token_shuffle | -11.5124 | -9.99982 | -1.51261 | True | -1.38055 | -1.51315 | 11.4109 | False | False | -0.132061 | 0.000544643 |
| 12 | 17 | 60 | 100 | level2_r100_amplitude | 0.213278 | 0.212733 | 0.000544643 | False | 0.132606 | 0 | -0 | False | True | -0.132061 | 0.000544643 |
| 12 | 29 | 60 | 20 | level1_native_amplitude | -0.692023 | -2.11392 | 1.4219 | False | 0 | 1.42225 | 1 | False | False | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | global_mean | 8.00115 | 7.19733 | 0.80382 | False | -0.618078 | 0.804175 | 0.565424 | False | False | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | layer_mean | 7.91326 | 6.87533 | 1.03793 | False | -0.383967 | 1.03829 | 0.730029 | False | False | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | expert_layer_mean | 8.07959 | 4.28077 | 3.79881 | False | 2.37692 | 3.79917 | 2.67123 | False | False | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | token_conditioned | -8.46041 | -13.1139 | 4.65347 | False | 3.23158 | 4.65383 | 3.27215 | False | False | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | temporal | -45.8939 | -34.7517 | -11.1421 | True | -12.564 | -11.1418 | -7.83389 | False | True | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | temporal_shuffle | -3.00997 | -3.76703 | 0.757057 | False | -0.664841 | 0.757412 | 0.532544 | False | False | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | token_shuffle | -13.5143 | -13.5089 | -0.00535147 | True | -1.42725 | -0.00499605 | -0.00351277 | False | True | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 20 | level2_r100_amplitude | 0.188714 | 0.189069 | -0.000355419 | True | -1.42225 | 0 | 0 | False | True | 1.4219 | -0.000355419 |
| 12 | 29 | 60 | 100 | level1_native_amplitude | -0.692023 | 0.318141 | -1.01016 | True | 0 | -1.01452 | 1 | False | False | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | global_mean | 8.00115 | 5.97368 | 2.02747 | False | 3.03764 | 2.02311 | -1.99415 | False | True | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | layer_mean | 7.91326 | 6.51889 | 1.39436 | False | 2.40453 | 1.39 | -1.3701 | False | True | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | expert_layer_mean | 8.07959 | 5.56156 | 2.51803 | False | 3.5282 | 2.51367 | -2.47768 | False | True | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | token_conditioned | -8.46041 | -9.39641 | 0.936001 | False | 1.94616 | 0.93164 | -0.918302 | False | True | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | temporal | -45.8939 | -42.3511 | -3.54273 | True | -2.53257 | -3.54709 | 3.49631 | False | False | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | temporal_shuffle | -3.00997 | -3.19333 | 0.183351 | False | 1.19352 | 0.178991 | -0.176428 | False | True | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | token_shuffle | -13.5143 | -17.389 | 3.87468 | False | 4.88484 | 3.87032 | -3.81491 | False | True | -1.01016 | 0.00436062 |
| 12 | 29 | 60 | 100 | level2_r100_amplitude | 0.188714 | 0.184353 | 0.00436062 | False | 1.01452 | 0 | -0 | False | True | -1.01016 | 0.00436062 |

## Scientific questions

1. Global scale: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | global_mean | -0.290065 | 0.269103 | 3 | 0 | -2.1962 | 5.54221 | 3 | 1 |
| 8 | 20 | 100 | global_mean | 2.57997 | 1.72371 | 0 | 3 | -8.02472 | 1.48873 | 3 | 3 |
| 12 | 60 | 20 | global_mean | -0.123947 | 0.949646 | 2 | 1 | -0.733948 | 1.71747 | 3 | 2 |
| 12 | 60 | 100 | global_mean | 1.11486 | 1.83648 | 1 | 2 | -3.71958 | 12.9674 | 3 | 2 |

2. Added layer structure: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | layer_mean | 0.380315 | 1.24443 | 1 | 2 | -10.924 | 9.71847 | 3 | 3 |
| 8 | 20 | 100 | layer_mean | 0.550531 | 1.47179 | 1 | 2 | -3.39798 | 6.78005 | 3 | 2 |
| 12 | 60 | 20 | layer_mean | -0.545117 | 1.43194 | 2 | 1 | -1.50765 | 2.55477 | 3 | 2 |
| 12 | 60 | 100 | layer_mean | 0.781416 | 0.869457 | 1 | 2 | -3.14352 | 5.85077 | 3 | 2 |

3. Added expert/layer structure: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | expert_layer_mean | 0.374996 | 3.01585 | 1 | 2 | -3.87828 | 30.2139 | 3 | 1 |
| 8 | 20 | 100 | expert_layer_mean | -8.32255 | 1.4177 | 3 | 0 | 31.0471 | 12.2411 | 3 | 0 |
| 12 | 60 | 20 | expert_layer_mean | 0.291677 | 3.23262 | 2 | 1 | -1.33327 | 4.56253 | 3 | 2 |
| 12 | 60 | 100 | expert_layer_mean | 1.9114 | 1.77583 | 1 | 2 | -8.88824 | 13.9719 | 3 | 2 |

4. Token-conditioned information: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | token_conditioned | -5.54438 | 4.39935 | 3 | 0 | 46.8465 | 59.9643 | 3 | 1 |
| 8 | 20 | 100 | token_conditioned | -3.50917 | 4.47827 | 2 | 1 | 9.30402 | 12.5457 | 3 | 1 |
| 12 | 60 | 20 | token_conditioned | 2.87443 | 2.21973 | 0 | 3 | 2.65076 | 1.48695 | 3 | 0 |
| 12 | 60 | 100 | token_conditioned | -0.853188 | 2.4291 | 1 | 2 | 9.40906 | 17.8916 | 3 | 2 |

5. Temporal information: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | temporal | 2.72292 | 4.04685 | 1 | 2 | -40.9215 | 25.0402 | 3 | 3 |
| 8 | 20 | 100 | temporal | -0.749412 | 5.96892 | 1 | 2 | 7.08285 | 27.5134 | 3 | 2 |
| 12 | 60 | 20 | temporal | -0.453466 | 10.6306 | 2 | 1 | 0.62331 | 9.26445 | 3 | 2 |
| 12 | 60 | 100 | temporal | -4.75764 | 5.29339 | 3 | 0 | 28.1832 | 44.5268 | 3 | 0 |

6. Temporal shuffling: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | temporal_shuffle | 1.08914 | 1.51411 | 1 | 2 | 12.7211 | 24.0069 | 3 | 1 |
| 8 | 20 | 100 | temporal_shuffle | -1.46701 | 1.1463 | 3 | 0 | 5.33584 | 5.70828 | 3 | 0 |
| 12 | 60 | 20 | temporal_shuffle | 0.651096 | 0.26644 | 0 | 3 | 0.993148 | 0.950444 | 3 | 0 |
| 12 | 60 | 100 | temporal_shuffle | -0.164765 | 0.633217 | 1 | 2 | 1.58821 | 4.55214 | 3 | 2 |

7. Token shuffling: common-readout results and three-seed consistency are below. A retention near one reproduces the reference contrast; near zero resembles Level 2.

| experts | target_age | control_age | amplitude_condition | mean_contrast | seed_sd | seeds_favoring_target | seeds_favoring_control | mean_retention_score | retention_seed_sd | valid_retention_seeds | ordering_flips_vs_level1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 20 | 0 | token_shuffle | 0.358503 | 8.23712 | 1 | 2 | -74.9144 | 63.2641 | 3 | 3 |
| 8 | 20 | 100 | token_shuffle | -2.92834 | 1.27653 | 3 | 0 | 12.942 | 8.90786 | 3 | 0 |
| 12 | 60 | 20 | token_shuffle | 0.565427 | 1.6722 | 2 | 1 | 1.74391 | 3.71762 | 3 | 2 |
| 12 | 60 | 100 | token_shuffle | -0.303916 | 3.72437 | 2 | 1 | 11.5999 | 15.5102 | 3 | 1 |

8. Closest reduced representation to full reference retention, by architecture/control:

| experts | control_age | amplitude_condition | recovering_seeds | valid_seeds | distance_from_full |
| --- | --- | --- | --- | --- | --- |
| 8 | 0 | global_mean | 0 | 3 | 3.55691 |
| 8 | 100 | layer_mean | 1 | 3 | 4.42596 |
| 12 | 20 | temporal_shuffle | 0 | 3 | 0.730942 |
| 12 | 100 | temporal_shuffle | 0 | 3 | 3.25078 |

9–10. Conditions satisfying 0.8–1.2 retention with preserved signs in all three seeds for all four contrasts: none. Per-contrast seed consistency is in retention_consistency.csv. This criterion is descriptive, not statistical significance.

11. First observed nontrivial contrast, divergence from Level 2, and half retention are listed in effect_onset.csv. The Level-1 common readout exists only at nine retained snapshots, so half-retention timing is limited to those observations. No interpolation is treated as evidence.

| experts | seed | control_age | amplitude_condition | first_nontrivial_step | first_divergence_from_level2_step | first_observed_half_retention_step |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 7 | 0 | global_mean | 50 | 50 | 1000 |
| 8 | 7 | 0 | layer_mean | 50 | 50 | 500 |
| 8 | 7 | 0 | expert_layer_mean | 50 | 50 | 100 |
| 8 | 7 | 0 | token_conditioned | 50 | 50 | 250 |
| 8 | 7 | 0 | temporal | 50 | 50 | 250 |
| 8 | 7 | 0 | temporal_shuffle | 50 | 50 | 100 |
| 8 | 7 | 0 | token_shuffle | 50 | 50 | 100 |
| 8 | 7 | 100 | global_mean | 50 | 50 | 100 |
| 8 | 7 | 100 | layer_mean | 50 | 50 | 100 |
| 8 | 7 | 100 | expert_layer_mean | 50 | 50 | 100 |
| 8 | 7 | 100 | token_conditioned | 50 | 50 | 100 |
| 8 | 7 | 100 | temporal | 50 | 50 | 100 |
| 8 | 7 | 100 | temporal_shuffle | 50 | 50 | 100 |
| 8 | 7 | 100 | token_shuffle | 50 | 50 | 100 |
| 8 | 17 | 0 | global_mean | 50 | 50 | 750 |
| 8 | 17 | 0 | layer_mean | 50 | 50 | 100 |
| 8 | 17 | 0 | expert_layer_mean | 50 | 50 | 100 |
| 8 | 17 | 0 | token_conditioned | 50 | 50 | 100 |
| 8 | 17 | 0 | temporal | 50 | 50 | 100 |
| 8 | 17 | 0 | temporal_shuffle | 50 | 50 | 100 |
| 8 | 17 | 0 | token_shuffle | 50 | 50 | 100 |
| 8 | 17 | 100 | global_mean | 50 | 50 | 100 |
| 8 | 17 | 100 | layer_mean | 50 | 50 | 100 |
| 8 | 17 | 100 | expert_layer_mean | 50 | 50 | 100 |
| 8 | 17 | 100 | token_conditioned | 50 | 50 | 250 |
| 8 | 17 | 100 | temporal | 50 | 50 | 100 |
| 8 | 17 | 100 | temporal_shuffle | 50 | 50 | 100 |
| 8 | 17 | 100 | token_shuffle | 50 | 50 | 100 |
| 8 | 29 | 0 | global_mean | 50 | 50 | 100 |
| 8 | 29 | 0 | layer_mean | 50 | 50 | 100 |
| 8 | 29 | 0 | expert_layer_mean | 50 | 50 | 100 |
| 8 | 29 | 0 | token_conditioned | 50 | 50 | 100 |
| 8 | 29 | 0 | temporal | 50 | 50 | 100 |
| 8 | 29 | 0 | temporal_shuffle | 50 | 50 | 100 |
| 8 | 29 | 0 | token_shuffle | 50 | 50 | 100 |
| 8 | 29 | 100 | global_mean | 50 | 50 | 100 |
| 8 | 29 | 100 | layer_mean | 50 | 50 | 100 |
| 8 | 29 | 100 | expert_layer_mean | 50 | 50 | 100 |
| 8 | 29 | 100 | token_conditioned | 50 | 50 | 250 |
| 8 | 29 | 100 | temporal | 50 | 50 | 1500 |
| 8 | 29 | 100 | temporal_shuffle | 50 | 50 | 100 |
| 8 | 29 | 100 | token_shuffle | 50 | 50 | 100 |
| 12 | 7 | 20 | global_mean | 50 | 50 | 100 |
| 12 | 7 | 20 | layer_mean | 50 | 50 | 100 |
| 12 | 7 | 20 | expert_layer_mean | 50 | 50 | 100 |
| 12 | 7 | 20 | token_conditioned | 50 | 50 | 250 |
| 12 | 7 | 20 | temporal | 50 | 50 | 1000 |
| 12 | 7 | 20 | temporal_shuffle | 50 | 50 | 100 |
| 12 | 7 | 20 | token_shuffle | 50 | 50 | 100 |
| 12 | 7 | 100 | global_mean | 50 | 50 | 100 |
| 12 | 7 | 100 | layer_mean | 50 | 50 | 100 |
| 12 | 7 | 100 | expert_layer_mean | 50 | 50 | 100 |
| 12 | 7 | 100 | token_conditioned | 50 | 50 | 100 |
| 12 | 7 | 100 | temporal | 50 | 50 | 500 |
| 12 | 7 | 100 | temporal_shuffle | 50 | 50 | 100 |
| 12 | 7 | 100 | token_shuffle | 50 | 50 | 100 |
| 12 | 17 | 20 | global_mean | 50 | 50 | 500 |
| 12 | 17 | 20 | layer_mean | 50 | 50 | 100 |
| 12 | 17 | 20 | expert_layer_mean | 50 | 50 | 100 |
| 12 | 17 | 20 | token_conditioned | 50 | 50 | 100 |
| 12 | 17 | 20 | temporal | 50 | 50 | 100 |
| 12 | 17 | 20 | temporal_shuffle | 50 | 50 | 100 |
| 12 | 17 | 20 | token_shuffle | 50 | 50 | 100 |
| 12 | 17 | 100 | global_mean | 50 | 50 | 100 |
| 12 | 17 | 100 | layer_mean | 50 | 50 | 100 |
| 12 | 17 | 100 | expert_layer_mean | 50 | 50 | 100 |
| 12 | 17 | 100 | token_conditioned | 50 | 50 | 100 |
| 12 | 17 | 100 | temporal | 50 | 50 | 750 |
| 12 | 17 | 100 | temporal_shuffle | 50 | 50 | 100 |
| 12 | 17 | 100 | token_shuffle | 50 | 50 | 100 |
| 12 | 29 | 20 | global_mean | 50 | 50 | 100 |
| 12 | 29 | 20 | layer_mean | 50 | 50 | 100 |
| 12 | 29 | 20 | expert_layer_mean | 50 | 50 | 100 |
| 12 | 29 | 20 | token_conditioned | 50 | 50 | 250 |
| 12 | 29 | 20 | temporal | 50 | 50 | nan |
| 12 | 29 | 20 | temporal_shuffle | 50 | 50 | 100 |
| 12 | 29 | 20 | token_shuffle | 50 | 50 | 100 |
| 12 | 29 | 100 | global_mean | 50 | 50 | 250 |
| 12 | 29 | 100 | layer_mean | 50 | 50 | 250 |
| 12 | 29 | 100 | expert_layer_mean | 50 | 50 | 250 |
| 12 | 29 | 100 | token_conditioned | 50 | 50 | 100 |
| 12 | 29 | 100 | temporal | 50 | 50 | 100 |
| 12 | 29 | 100 | temporal_shuffle | 50 | 50 | 250 |
| 12 | 29 | 100 | token_shuffle | 50 | 50 | 250 |

12. Expert-gradient ratios to Level 1 range from 0.1740 to 0.5635 across matched runs. Clipping, scale and nonfinite-step fractions are in gradient_scale_summary.csv. Router task gradients are absent; recorded router gradients are from the preserved weighted auxiliary loss. Gradient scaling alone does not establish a mechanism.

13–14. Canonical dense-source layer means range from 0.03074 to 0.13086; expert/layer means range from 0.00039 to 0.98953. Heatmaps and variance tables show where amplitude variation is concentrated descriptively. They do not identify independent layer/expert causal effects.

15–16. The contrast hierarchy and retention consistency identify which tested coarse or fine representations reproduce the age-conditioned effect. The verdict applies only to the canonical dense-source transformations tested here. Failure of reduced conditions is not proof that a full high-order canonical pattern is necessary: no full detached canonical-source recovery control was added.

17. Level 1 used online native amplitudes with task gradients; the canonical sources here are native proposals along dense Level-2 trajectories. Detachment and source-trajectory differences, preserved auxiliary gradients, and endogenous hidden/expert dynamics remain unresolved pathways. Token and temporal shuffles cannot by themselves establish necessity against that unmatched full-reference training procedure.

## Protocol and interpretation

All coefficients are detached; historical optimizer, clipping, GradScaler, masks, data order and auxiliary losses are preserved. Sources are canonical dense amplitude trajectories associated with router age, not sparse models own native proposals. Temporal shuffling is restricted to equal-length expert/layer blocks and leaves singleton strata fixed; it can also change token mapping across batches. Token means use exact recurring identities where at least 80% of accepted-edge observations recur; any tokenizer-ID substitution is explicitly recorded in each transform manifest.

Retention uses a preregistered denominator tolerance of 1e-5. Near-zero cases are flagged. Means and sample SD use three independent seeds; tokens, layers, experts and steps are mechanism observations. Some reference target/control contrasts already favor the control; preserving such a contrast is preservation of an age effect, not recovery of a favorable target advantage.

Raw secondary outcomes and readout_comparison.csv expose evaluation-policy differences. Descriptive nested variance components follow layer, expert within layer, time within expert/layer, then token residual; they are not independent causal variance contributions.

All new artifacts are in stable storage at /users/kent/student/mweber54/moe-lth-research/moe-lth-research/results/gate_amplitude_pattern_decomposition. No additional experiment was launched.

GATE-PATTERN RESULT SEED-CONDITIONED

**Most important implication for the paper:** The measured retention shows which information in the canonical age-conditioned dense amplitude trajectory is sufficient under detached replay and a common readout. It does not establish exclusive forward-value mediation in ordinary training or necessity of an untested full detached source pattern.

**Most important next experiment:** A full, untransformed detached replay of the same canonical dense amplitude sources would provide the missing matched reference for interpreting token/time/high-order necessity. This is proposed only and has not been launched.
