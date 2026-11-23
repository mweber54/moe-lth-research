# Post hoc evaluation amendment

This amendment preserves the registered native endpoint, original reports, and pilot feasibility failure. It is not a preregistered replacement primary result.

All results use existing seed-17 final step-5000 models, FP32, TF32 disabled, eval and inference mode. The same 31 complete validation batches contain 492,416 target tokens, including the original partial batch. Ten fixed validation permutations measure evaluation variability, not training replication. There is only one shuffle_101-trained recipient per variant.

## e32_seed17_dense

| Recipient | Evaluation I | Evaluation S mean (SD; range) |
|---|---:|---:|
| identity | 1.33868900 | 7.173069802017252 (0.007760641265864111; 7.161150254179417–7.186439628938219) |
| shuffle_101 | 1.71589751 | 1.7212376003888852 (0.0002849541616839769; 1.720932420892222–1.7216928060066554) |

Contextual CE:
- donor, native: 1.32989249
- identity, native: 4.00922772
- identity, donor_router_weights: 2.04987919
- shuffle_101, native: 1.94213159
- shuffle_101, donor_router_weights: 1.82020267

Donor S is an immediate evaluation perturbation, not recovery after shuffled training.
Donor S mean: 7.67090926.

Contrasts (mean; sample SD; range):
- A: 5.83438080312185; 0.007760641265864111; 5.822461255284016 to 5.8477506300428175
- B: -0.00534009025225175; 0.0002849541616839769; -0.0057952958700220325 to -0.005034910755588706
- C: 0.3825486014934835; 0.0002849541616839769; 0.3822434219968205 to 0.3830038071112538
- D: 5.829040712869599; 0.007870521424649616; 5.817298830438659 to 5.842624989397532
## e32_seed17_imp80

| Recipient | Evaluation I | Evaluation S mean (SD; range) |
|---|---:|---:|
| identity | 1.35384955 | 5.5137874495434644 (0.0043215148714338646; 5.505343157066938–5.52123813192845) |
| shuffle_101 | 1.76841649 | 1.7724531090597317 (0.00018164869987244114; 1.7721282556083626–1.772836446359246) |

Contextual CE:
- donor, native: 1.35100942
- identity, native: 6.14754215
- identity, donor_router_weights: 1.79455908
- shuffle_101, native: 1.80547763
- shuffle_101, donor_router_weights: 1.78735941

Donor S is an immediate evaluation perturbation, not recovery after shuffled training.
Donor S mean: 5.47109412.

Contrasts (mean; sample SD; range):
- A: 4.15993789871668; 0.0043215148714338646; 4.1514936062401535 to 4.167388581101665
- B: -0.004036616919596581; 0.00018164869987244114; -0.004419954219111011 to -0.003711763468227547
- C: 0.41860355823294715; 0.00018164869987244114; 0.41827870478157814 to 0.4189868955324616
- D: 4.1559012817970835; 0.004364046563974919; 4.147449390839253 to 4.163206841036471

## Interpretation

1. The identity native-endpoint anomaly is explained by policy compatibility: identity recipients recover near donor CE under donor slots, while native frozen routing performs poorly. Router weights alone only partially repair the incompatibility.
2. The matched-policy shuffle penalty (L_SS − L_II) is 0.38254860 for dense and 0.41860356 for IMP80. Positive values indicate worse corresponding-policy recovery in this seed.
3. Sparse minus dense differential penalty: 0.03605496 (SD 0.00034666; range 0.03549492557481981 to 0.036407847605279375). A positive value is consistent with greater sparse sensitivity to the combined training/evaluation protocol in this seed.
4. The crossed results warrant further investigation of this replay protocol, with any expansion decided after reviewing the interaction and compute cost. No additional recovery is authorized or launched here.

These comparisons condition on each variant’s own complete donor; donor policies differ across sparsities. The complete-slot intervention jointly changes expert identity, token-specific gates, and accepted/dropped tokens. It does not isolate expert selection or the training-time assignment effect. Replay training removes the native task-gradient path through gate coefficients.

Donor-generated routing requires an external complete donor. Recipient CE uses recipient embeddings, attention, normalization, experts, residuals and output head. These results establish neither standalone deployment performance nor a winning ticket. Better shuffled native endpoints do not show that random routing improves ordinary MoE recovery. They do not identify IMP specialization or a mechanism of sparse outperformance.

No source-seed confidence intervals or significance claims are justified at n=1. Evaluation permutations are paired across recipients and variants. Contrasts involving shuffled policies are computed per full-validation replicate before summarizing.

A = L_IS − L_II; B = L_SI − L_SS; C = L_SS − L_II; D = (L_IS − L_II) − (L_SS − L_SI). Per-batch losses and actual token counts are retained. Native router entropy, if exposed internally by replay, is not an imposed-policy statistic and is not used here.

Identity contextual evaluations and L_II are reused from the completed posthoc_validation diagnostic after verifying checkpoint SHA256, validation batch hashes and target counts. Donor native and identity replay are rechecked exactly on all batches. See numerical_checks.json and checkpoint_and_batch_provenance.json for checks and budget accounting.

The crossed interaction is pronounced: identity-trained recipients deteriorate substantially under shuffled evaluation, while shuffle-trained recipients have very similar CE under I and S. Thus the matched-policy difference must be read together with the full crossed table.

Compute review: main evaluation took 231.6 seconds on the H100 NVL (additional audit/report time excluded). Historical donor plus identity plus shuffle_101 recoveries took 5.75 dense and 7.07 IMP80 host-hours in total. These are observed historical wall times, not a guaranteed expansion estimate. Review the modified gradient path, combined slot intervention, and n=1 limitation before deciding whether more training is worth that cost. No expansion is launched.
