# Fixed-budget expert deletion reliability

**Conclusion: mixed results requiring a narrower claim.**

This is seed 17, the same nonmatching 32E IMP80 engineering case, with its matched unpruned recovery. No training, optimizer update, backward pass, recovery, mask search, permanent pruning, new source seed, predictor or follow-up launch occurred. Historical reports and checkpoints remain unchanged.

## Fixed design and probe provenance

Exactly 32 additional complete 128×128 training batches were frozen before new deletion measurements. The user authorized re-batching intact canonical 128-token sequences; historical shuffled minibatch membership is not claimed. The fixed availability-only sampler uses seed 171032 and excludes previous A/B windows, overlapping input-plus-target source spans and duplicate full-window content. All 4,096 new windows are source-disjoint, including shifted targets. The same batches are used for all eight trajectory/checkpoint combinations. Blocks 1–4 are discovery; 5–8 confirmation. Both are training-distribution contexts, possibly seen during training, not independent generalization or source-seed samples. See probe_manifest.csv and probe_overlap_audit.json.

The original layer, experts, coordinate groups, orderings and 0/1/2/4/8-group unions were retained: 52 unique weight unions plus four separate output-zero controls. Deletions are coordinate-matched across models, not density-matched. Step 2500 is verified pre-reset. Historical/rerun endpoint aliases count once. Source code, coordinates, numerical definitions and metadata were reused; prior A/B losses were not copied onto these new contexts.

## A–E: per-condition decision

Among 78 operationally covered whole-probe conditions with at least three distinct same-dose unions, 26 discovery-selected contrasts have a confirmation paired-batch 95% interval entirely above zero, and 1 entirely below zero. These are descriptive sampled-context intervals in one source seed, with multiple dependent panels and no multiplicity-adjusted or seed-level success claim. The full per-expert table below and condition_decisions.csv are primary; there is no global PASS.

A: discovery/confirmation reproducibility is judged within the same trajectory, checkpoint, expert and dose. B: a contrast interval crossing zero is unresolved at this budget, not evidence of harmlessness. C: functional-distortion/CE associations and cross-phase functional rankings are recorded in discovery_confirmation_rankings.csv for that same stratum; a functional output change need not order task damage. D: all eight blocks need at least 32 accepted positions for conditional-ranking eligibility; low-coverage whole-probe measurements remain visible. E: any reproduced contrast supports only a later explicitly fixed measurement confirmation in that named condition, not future persistence or permanent-pruning safety.

| Trajectory | Step | Expert | Groups | Accepted minimum/block | Discovery–confirmation rho | Selected confirmation contrast [95% interval] | A/B/E interpretation |
|---|---:|---:|---:|---:|---:|---|---|
| sparse | 25 | 0 | 1 | 26 | 1 | 1.9606e-06 [1.5013e-06, 2.424e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 25 | 0 | 2 | 26 | 1 | 3.865e-06 [2.4199e-06, 5.254e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 25 | 0 | 4 | 26 | 1 | 3.7771e-06 [2.3419e-06, 5.1589e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 25 | 0 | 8 | 26 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 25 | 8 | 1 | 0 | undefined | 1.8956e-08 [0, 5.6869e-08] | unidentified_discovery_tie |
| sparse | 25 | 8 | 2 | 0 | undefined | -1.8432e-08 [-5.5295e-08, 0] | unidentified_discovery_tie |
| sparse | 25 | 8 | 4 | 0 | undefined | -2.1368e-09 [-6.4104e-09, 0] | unidentified_discovery_tie |
| sparse | 25 | 8 | 8 | 0 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 25 | 16 | 1 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| sparse | 25 | 16 | 2 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| sparse | 25 | 16 | 4 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| sparse | 25 | 16 | 8 | 0 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 25 | 24 | 1 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| sparse | 25 | 24 | 2 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| sparse | 25 | 24 | 4 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| sparse | 25 | 24 | 8 | 0 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 250 | 0 | 1 | 877 | 0.5 | 3.8924e-07 [-4.5366e-07, 1.3951e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 0 | 2 | 877 | -0.4 | -4.1002e-07 [-1.7936e-06, 5.1991e-07] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 0 | 4 | 877 | 0.4 | 4.2229e-07 [-1.2849e-08, 1.1e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 0 | 8 | 877 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 250 | 8 | 1 | 2073 | 0.5 | 1.8859e-07 [-6.2191e-08, 4.106e-07] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 8 | 2 | 2073 | 0.8 | 1.6971e-06 [1.2617e-07, 3.3757e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 250 | 8 | 4 | 2073 | 0.8 | 1.9175e-06 [4.6632e-07, 3.5846e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 250 | 8 | 8 | 2073 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 250 | 16 | 1 | 1602 | 0.5 | 9.3237e-07 [3.7015e-07, 1.6899e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 250 | 16 | 2 | 1602 | 0.4 | 3.4581e-08 [-1.3845e-06, 1.676e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 16 | 4 | 1602 | -0.2 | -5.9105e-08 [-1.3916e-06, 7.4595e-07] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 16 | 8 | 1602 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 250 | 24 | 1 | 2560 | -0.5 | -1.5046e-07 [-9.9163e-07, 7.8529e-07] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 24 | 2 | 2560 | 0.2 | 2.3926e-07 [-1.0873e-06, 1.7218e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 24 | 4 | 2560 | 0 | -1.7312e-07 [-2.5435e-06, 2.4144e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 250 | 24 | 8 | 2560 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 1000 | 0 | 1 | 1840 | -0.5 | -8.429e-07 [-3.5167e-06, 1.4542e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 0 | 2 | 1840 | 0.6 | 1.7637e-06 [-1.7201e-07, 4.1652e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 0 | 4 | 1840 | 0.8 | 1.7346e-06 [3.662e-07, 3.1873e-06] | directional_contrast_reproduced_on_confirmation |
| sparse | 1000 | 0 | 8 | 1840 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 1000 | 8 | 1 | 1998 | 1 | 1.0059e-06 [-1.1607e-06, 3.3068e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 8 | 2 | 1998 | 0 | 4.3949e-07 [-3.4925e-06, 3.977e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 8 | 4 | 1998 | -0.8 | -7.9099e-07 [-5.4626e-06, 3.4857e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 8 | 8 | 1998 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 1000 | 16 | 1 | 2560 | 0.5 | 8.5871e-07 [-2.4555e-06, 3.6825e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 16 | 2 | 2560 | 0.8 | 4.0605e-06 [-1.3508e-06, 1.0334e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 16 | 4 | 2560 | 0.8 | 1.8831e-06 [-4.1842e-06, 9.5351e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 16 | 8 | 2560 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 1000 | 24 | 1 | 2280 | 0.5 | 1.0775e-06 [-1.1954e-06, 3.6019e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 24 | 2 | 2280 | 0.8 | 2.6487e-06 [-5.241e-07, 6.0953e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 24 | 4 | 2280 | 0.4 | 2.7794e-06 [-1.3097e-06, 7.2152e-06] | unresolved_at_fixed_sampling_budget |
| sparse | 1000 | 24 | 8 | 2280 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 2500 | 0 | 1 | 2415 | 0.5 | 7.4221e-06 [-3.5358e-06, 1.7829e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 0 | 2 | 2415 | 0.4 | 8.853e-06 [1.787e-06, 1.7648e-05] | directional_contrast_reproduced_on_confirmation |
| sparse | 2500 | 0 | 4 | 2415 | 1 | 1.7236e-05 [-4.1504e-07, 3.5459e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 0 | 8 | 2415 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 2500 | 8 | 1 | 2417 | 1 | 3.1677e-05 [1.3071e-05, 5.2461e-05] | directional_contrast_reproduced_on_confirmation |
| sparse | 2500 | 8 | 2 | 2417 | 0.8 | 5.1505e-05 [2.5553e-05, 8.2117e-05] | directional_contrast_reproduced_on_confirmation |
| sparse | 2500 | 8 | 4 | 2417 | 0.8 | 4.3739e-05 [2.1497e-05, 7.1012e-05] | directional_contrast_reproduced_on_confirmation |
| sparse | 2500 | 8 | 8 | 2417 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 2500 | 16 | 1 | 2449 | 1 | 1.5916e-05 [1.5231e-06, 2.8889e-05] | directional_contrast_reproduced_on_confirmation |
| sparse | 2500 | 16 | 2 | 2449 | 1 | 3.6084e-06 [-1.4368e-05, 2.0093e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 16 | 4 | 2449 | 0.8 | 4.196e-06 [-5.0784e-06, 1.2734e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 16 | 8 | 2449 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| sparse | 2500 | 24 | 1 | 2179 | 0.5 | 2.6978e-06 [-8.7273e-06, 1.4866e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 24 | 2 | 2179 | 0.8 | 1.4401e-05 [-7.8064e-06, 3.7858e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 24 | 4 | 2179 | 0.8 | 1.7386e-05 [-6.4666e-06, 4.1441e-05] | unresolved_at_fixed_sampling_budget |
| sparse | 2500 | 24 | 8 | 2179 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 25 | 0 | 1 | 321 | 1 | 4.197e-05 [3.5616e-05, 4.8881e-05] | directional_contrast_reproduced_on_confirmation |
| unpruned | 25 | 0 | 2 | 321 | 1 | 0.00019167 [7.3652e-05, 0.00035459] | directional_contrast_reproduced_on_confirmation |
| unpruned | 25 | 0 | 4 | 321 | 0.4 | 0.00018666 [9.8487e-05, 0.00033034] | directional_contrast_reproduced_on_confirmation |
| unpruned | 25 | 0 | 8 | 321 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 25 | 8 | 1 | 137 | 1 | 0.00012274 [0.00011486, 0.00013085] | directional_contrast_reproduced_on_confirmation |
| unpruned | 25 | 8 | 2 | 137 | 1 | 0.00013385 [0.00012562, 0.00014222] | directional_contrast_reproduced_on_confirmation |
| unpruned | 25 | 8 | 4 | 137 | 1 | 0.00013436 [0.00012624, 0.00014266] | directional_contrast_reproduced_on_confirmation |
| unpruned | 25 | 8 | 8 | 137 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 25 | 16 | 1 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| unpruned | 25 | 16 | 2 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| unpruned | 25 | 16 | 4 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| unpruned | 25 | 16 | 8 | 0 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 25 | 24 | 1 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| unpruned | 25 | 24 | 2 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| unpruned | 25 | 24 | 4 | 0 | undefined | 0 [0, 0] | unidentified_discovery_tie |
| unpruned | 25 | 24 | 8 | 0 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 250 | 0 | 1 | 2106 | 0.5 | 3.82e-07 [-3.3964e-06, 5.2173e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 0 | 2 | 2106 | 0 | 2.1604e-07 [-3.5815e-06, 4.1618e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 0 | 4 | 2106 | 0.4 | 5.2718e-06 [1.3057e-06, 8.9842e-06] | directional_contrast_reproduced_on_confirmation |
| unpruned | 250 | 0 | 8 | 2106 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 250 | 8 | 1 | 2036 | 0.5 | 2.453e-07 [-3.1835e-06, 3.802e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 8 | 2 | 2036 | 0.6 | 3.4429e-07 [-3.793e-06, 4.0493e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 8 | 4 | 2036 | 0.8 | 3.6553e-06 [-1.7359e-07, 8.1498e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 8 | 8 | 2036 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 250 | 16 | 1 | 2560 | -0.5 | -3.361e-07 [-4.5614e-06, 3.6191e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 16 | 2 | 2560 | 0.8 | 3.405e-06 [-4.2843e-06, 1.1321e-05] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 16 | 4 | 2560 | -0.4 | -2.6007e-06 [-7.9913e-06, 2.2727e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 16 | 8 | 2560 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 250 | 24 | 1 | 2560 | 1 | 7.8788e-06 [-8.2931e-08, 1.7961e-05] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 24 | 2 | 2560 | -0.4 | -6.1769e-06 [-1.4027e-05, 4.6136e-07] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 24 | 4 | 2560 | -0.2 | -8.0027e-07 [-6.7948e-06, 4.5255e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 250 | 24 | 8 | 2560 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 1000 | 0 | 1 | 1888 | 0.5 | 6.1553e-07 [-9.741e-07, 2.4013e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 0 | 2 | 1888 | 0 | 9.0422e-07 [-2.8799e-06, 3.7384e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 0 | 4 | 1888 | 0.8 | 2.9313e-06 [-1.9432e-06, 8.3081e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 0 | 8 | 1888 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 1000 | 8 | 1 | 2193 | 0.5 | 1.2486e-06 [7.5136e-07, 1.7141e-06] | directional_contrast_reproduced_on_confirmation |
| unpruned | 1000 | 8 | 2 | 2193 | 0.4 | 1.5494e-06 [7.1136e-07, 2.5605e-06] | directional_contrast_reproduced_on_confirmation |
| unpruned | 1000 | 8 | 4 | 2193 | -0.4 | -7.2839e-07 [-2.0389e-06, 6.4793e-07] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 8 | 8 | 2193 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 1000 | 16 | 1 | 314 | 0.5 | 1.9766e-07 [9.0784e-08, 3.0001e-07] | directional_contrast_reproduced_on_confirmation |
| unpruned | 1000 | 16 | 2 | 314 | 0.4 | 7.3565e-08 [-4.7152e-07, 4.2086e-07] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 16 | 4 | 314 | 0.4 | 9.7319e-08 [-4.6538e-07, 4.9955e-07] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 16 | 8 | 314 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 1000 | 24 | 1 | 2560 | 0.5 | 9.425e-08 [-9.0461e-07, 9.6514e-07] | unresolved_at_fixed_sampling_budget |
| unpruned | 1000 | 24 | 2 | 2560 | 1 | 3.7715e-06 [1.7431e-06, 5.8488e-06] | directional_contrast_reproduced_on_confirmation |
| unpruned | 1000 | 24 | 4 | 2560 | 0.6 | 1.7346e-06 [6.9521e-07, 3.0809e-06] | directional_contrast_reproduced_on_confirmation |
| unpruned | 1000 | 24 | 8 | 2560 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 2500 | 0 | 1 | 1798 | 0.5 | 1.4494e-06 [-2.3233e-06, 4.7819e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 2500 | 0 | 2 | 1798 | 0.8 | 2.948e-06 [-1.9415e-06, 7.7551e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 2500 | 0 | 4 | 1798 | 0.8 | 2.6058e-06 [-3.2139e-06, 7.4617e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 2500 | 0 | 8 | 1798 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 2500 | 8 | 1 | 1612 | 1 | 8.9417e-06 [5.6995e-06, 1.2383e-05] | directional_contrast_reproduced_on_confirmation |
| unpruned | 2500 | 8 | 2 | 1612 | 0.8 | 6.081e-06 [1.8186e-07, 1.1716e-05] | directional_contrast_reproduced_on_confirmation |
| unpruned | 2500 | 8 | 4 | 1612 | 0.8 | 8.7375e-06 [-8.5325e-08, 1.7692e-05] | unresolved_at_fixed_sampling_budget |
| unpruned | 2500 | 8 | 8 | 1612 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 2500 | 16 | 1 | 2443 | 0.5 | 4.3588e-06 [-4.4795e-07, 9.291e-06] | unresolved_at_fixed_sampling_budget |
| unpruned | 2500 | 16 | 2 | 2443 | 0.8 | 9.2172e-06 [2.2966e-06, 1.6543e-05] | directional_contrast_reproduced_on_confirmation |
| unpruned | 2500 | 16 | 4 | 2443 | 1 | 1.1194e-05 [5.2319e-06, 1.7342e-05] | directional_contrast_reproduced_on_confirmation |
| unpruned | 2500 | 16 | 8 | 2443 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |
| unpruned | 2500 | 24 | 1 | 2203 | -1 | -5.9301e-06 [-1.1115e-05, -2.2175e-07] | directional_contrast_reversed_candidate_context_dependence |
| unpruned | 2500 | 24 | 2 | 2203 | 1 | 7.5019e-06 [2.4819e-06, 1.3312e-05] | directional_contrast_reproduced_on_confirmation |
| unpruned | 2500 | 24 | 4 | 2203 | 0.8 | 5.3772e-06 [-1.9853e-06, 1.2989e-05] | unresolved_at_fixed_sampling_budget |
| unpruned | 2500 | 24 | 8 | 2203 | undefined | undefined [undefined, undefined] | undefined_singleton_no_equal_dose_contrast |

Each row retains expert identity. The eight-group dose contains one unique union per expert, so it cannot identify an equal-dose ranking or high-minus-low comparison. The 1-group panel has three unique unions; 2/4-group panels have four. Small panels and overlapping unions require caution even when Spearman is ±1. Zero dose and output-zero controls are not mixed into dose rankings.

## Sampling precision, near ties and context variation

All fixed pairwise contrasts—not only the discovery-selected extremes—are in paired_intervention_contrasts.csv. It contains discovery, confirmation and each of the eight blocks, with contrast sizes, paired complete-batch bootstrap intervals, target counts and zero-denominator frequencies. Selection used discovery only, with frozen ID tie-breaking; confirmation never reselected either extreme. discovery_confirmation_rankings.csv retains underlying intervention damages, numerical ties, bootstrap rank intervals/undefined frequencies, confirmation block signs/intervals and phase-specific leave-one-block-out sensitivities.

Every bootstrap uses 2,000 fixed-seed resamples of complete batches, paired across all interventions. Conditional ratios resample loss sums and accepted counts, then divide; conditional means with different denominators are not averaged. Undefined replicates remain undefined. These intervals measure variation over sampled training contexts in one seed, not model/seed generalization. Negative rank correlations alone do not establish context-dependent importance: compare the paired contrasts, their uncertainty, coverage and leave-one-block-out estimates before treating context dependence as a candidate explanation. No semantic or specialization labels were fitted.

uncertainty_and_tolerance_summary.csv reports continuous effects and intervals first, followed by interpretations relative to 0.0001 and 0.001 nats/token. An interval below a convention is compatible with below-convention damage on this probe only. Crossing is unresolved; exceeding is evidence relative to that convention. None defines harmlessness, permanent dispensability or pruning safety.

## Coverage and functional distortion

46 of 256 expert/trajectory/checkpoint/block cells fall below the operational minimum of 32 accepted positions. This threshold is a coverage flag, not proof of precision. Contributing complete-batch counts, maximum batch fractions, accepted-position HHI and effective batch counts are stored for all cells. No expert, checkpoint or batch was replaced after seeing coverage.

Whole-model CE changes can propagate beyond baseline-accepted positions through attention and later layers; accepted-position CE is not an isolated expert loss. Empty conditional populations are undefined, not zero. Negative damage is retained.

functional_distortion_summary.csv stores the squared output-difference numerator, baseline-output energy denominator, element counts and gate-weighted energies separately. Pooled RMS ratios form sqrt(sum numerator / sum denominator), with the original near-zero baseline rule, and are explicitly distinguished from the average of defined batch ratios. Distortion-versus-task-damage correlations stay within expert/checkpoint/dose and are not manufactured by pooling doses.

## Numerical checks and execution

128 representative repeats had maximum token-loss disagreement 0. Zero-dose equivalence, unmodified native baseline equivalence, exact intervention restoration, full model hashes, original IMP masks, RNG, module modes, no gradients and native targeted-layer routing/gates were checked. Deterministic repeats are not evidence of agreement across contexts.

A second exact-compatible optimization reused only the unmodified complete-batch prefix strictly upstream of layer1. Native layer1 and every downstream layer were still evaluated for each deletion, with complete-batch routing/capacity. All original profile effects, sequence statistics and functional energies matched bitwise. Prefix tensors were ephemeral GPU working memory, never saved as hidden-data caches. New cache identities also bind prefix_engine_sha256.

An engineering acceleration replaced dynamic integer assignment-index synchronization with stable integer grouping; native router probabilities, top-k capacity selection, gates, expert order, expert computation and index_add remained unchanged. All 56 registered profile interventions, token-loss baselines, paired sequence sums, CE statistics and output energies matched bitwise against the original implementation. Original frozen sources and statistical definitions were preserved. New caches additionally bind routing_engine_sha256; earlier exact-compatible caches retain their original implementation identities. No new probe contexts were introduced.

The sole worker moved from contended Viper to the established Adder H100 host after an exact all-intervention check on registered batch0, then returned when the original GPU became nearly idle. Baseline routing/gates, all CE statistics and paired sequence sums matched bitwise; output energies also matched exactly. The original Viper profile is retained as a contention-sensitive estimate. Combined main-worker elapsed time is 1776.1s, plus the original profile and host compatibility check. The transfer changed no contexts, groups, numerical protocol or statistical definitions.

One-batch profile estimated 19289.3s for measurement and 163.1 MB of compact caches. Actual main pass including table serialization: 416.1s on NVIDIA H100 NVL; analysis 34.8s; report/figures before sealing 18.9s. Cache storage 162.3 MB; all outputs before seal 220.5 MB. Profiling used one of the registered 32 batches and that cache was reused. No full hidden tensors or logits were saved.

Final category: **mixed results requiring a narrower claim**. No cross-seed expansion or later experiment follows automatically. This assay provides no specialization, lottery-ticket-mechanism, permanent-dispensability or pruning-algorithm claim.
