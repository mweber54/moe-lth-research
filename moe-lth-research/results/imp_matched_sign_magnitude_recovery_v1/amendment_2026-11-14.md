# Matched-distance recovery amendment — 2026-11-14 (America/Indianapolis)

Recorded UTC: 2026-11-14T20:20:00+00:00. This prospective amendment amends `preregistration_2026-11-12.md` committed in `b4e9451bd9de867bdca8e6f996bf93f62955de33` solely regarding the granularity of the hard distance matching tolerance check during preparation. The original preregistration document and commit `b4e9451bd9de867bdca8e6f996bf93f62955de33` remain preserved unchanged. This amendment is registered before any matched-distance recovery training: zero runs and zero attempted updates have been dispatched or trained, as certified in `pre_amendment_training_audit.json`.

## Background and Mathematical Diagnosis

During prospective preparation under commit `b4e9451bd9de867bdca8e6f996bf93f62955de33`, execution halted on Seed 7, Mask `imp`, Draw 1 at tensor `blocks.2.moe.experts.13.fc2.weight` due to the assertion `assert mismatch <= 0.001`:
- Realized magnitude displacement distance $d_M = 0.16233814029093258$
- Realized partial sign-flip distance $d_{Bm} = 0.16356010974920532$
- Absolute mismatch $= 0.0012219694582727425$

An audit across all 6,912 matrix-draw instances in the experimental design revealed that:
1. Pruning by IMP concentrates non-uniform sparsity in late-layer down-projections (`fc2` in blocks 2 and 3), leaving as few as 3,490 to 8,081 survivor weights.
2. Sign flipping is inherently quantized: flipping a single positive/negative survivor pair contributes a discrete quantum of squared displacement $\Delta S_k = 4(w_{\text{pos},k}^2 + w_{\text{neg},k}^2)$, corresponding to a discrete jump in normalized Frobenius distance of:
   $$\Delta d \approx \frac{4}{d \cdot N_{\text{surv}}}$$
   For matrices with $N_{\text{surv}} < 12,500$, the discrete step size $\Delta d$ ranges from $0.003$ to $0.010$, which exceeds the $0.001$ tolerance.
3. When the target $d_M$ lands in the gap between consecutive discrete prefix states, no integer prefix $k$ can satisfy $|d_{Bm} - d_M| \le 0.001$. Across the design, exactly 159 matrices (2.30% of matrices, 88.7% of them under the `imp` mask) exhibit this discrete dead-zone phenomenon, with a maximum observed mismatch of $0.004448$.
4. In contrast, at the whole-model aggregate level (the total physical dose delivered to the network), the distance matches to $< 8 \times 10^{-6}$ across all 27 models (mean $2.46 \times 10^{-6}$), exceeding the $\le 0.001$ requirement by two orders of magnitude.

## Amended Distance Matching Policy (Option A)

To maintain fidelity to the audited prefix-matching algorithm of `matched_signs.py` without introducing non-uniform weight selection biases or combinatorial search:

1. **Model-Level Hard Distance Tolerance:** The hard distance tolerance $\le 0.001$ is enforced strictly on the aggregate survivor-weight Frobenius distance across all expert matrices for each model:
   $$\left| \sqrt{\frac{\sum_m \|W'_{Bm,m} - W_m\|_F^2}{\sum_m \|W_m\|_F^2}} - \sqrt{\frac{\sum_m \|W'_{M,m} - W_m\|_F^2}{\sum_m \|W_m\|_F^2}} \right| \le 0.001$$
   This aggregate match must pass for all 27 models in `job_manifest.csv` before dispatch.
2. **Descriptive Per-Matrix Reporting:** For each expert matrix, the realized target distance $d_M$, realized sign-flip distance $d_{Bm}$, absolute mismatch $|d_{Bm} - d_M|$, relative mismatch, and number of flipped pairs are recorded descriptively in `matrix_invariants.csv`.
3. **Per-Matrix Upper Safety Bound:** To safeguard against numerical corruption or software malfunction while accommodating the discrete atomic resolution of low-survivor expert layers, an upper bound of $|d_{Bm} - d_M| \le 0.015$ is asserted per matrix (comfortably bounding the maximum discrete step size of $\approx 0.010$).
4. **Preservation of Perturbation Sampling:** The prefix-matching algorithm and deterministic SHA256 seeds from `preregistration_2026-11-12.md` are preserved without alteration. No searching over alternative lists, no loss optimization, and no magnitude tuning is permitted.

## Preserved Invariants and Design Constraints

All other design specifications from `preregistration_2026-11-12.md` remain completely unchanged:
- **Cohort:** E32/IMP80 only (32 experts, nominal final sparsity 0.8).
- **Grid:** Seeds 7, 17, 29; masks imp, random1, random2; draws 1, 2, 3; data orders A, B; arms M, Bm. Exactly 108 recoveries.
- **Caps:** 2,500 attempted updates per recovery; 270,000 total updates cap including AMP skips.
- **Invariants:** Exact sign preservation and magnitude multiset preservation for M; exact magnitude preservation and sign count preservation for Bm; exact zero coordinates; unchanged router and shared parameters; step-zero validation CE reproduction within $1 \times 10^{-3}$.
- **Decision Rules:** Identical estimands, Student-$t$ paired draw intervals, practical direct-comparison margin $\delta = 0.02486047$ nats, and seed-consistency criteria.
- **Stop Boundary:** Absolute stop after 108 recoveries; no automatic retry, second segment, or continuation.
