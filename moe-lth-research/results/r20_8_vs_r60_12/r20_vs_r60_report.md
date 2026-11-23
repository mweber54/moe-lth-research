# R20(8) vs R60(12) Developmental-State Comparison

This is a probe-only analysis; no sparse recovery was rerun. The direct 8E anchor is R20 from the same reference family as the cited e8 75/85% ticket results.

## Measured Comparison

The equal-weight range-normalized distance from 8E R20 is lower for 12E R20 than for 12E R60 in every seed: seed 7, 0.1040 versus 0.3110; seed 17, 0.1241 versus 0.2615; seed 29, 0.1592 versus 0.2835. Therefore 12E R60 is not the closest measured analogue of 8E R20. It is also farther in chronological cumulative exposure: 8E R20 has 43,008 mean logged assignments per expert, while 12E R20 has 28,672 and 12E R60 has 83,285.

The measured vector includes normalized routing entropy, margin, load CV, normalized token-ID/expert MI, assignment agreement to R100, functional JS to R100, and cumulative assignment exposure. R60 is later/more mature than the 8E R20 anchor on entropy, margin, specialization, stability-to-R100, functional distance-to-R100, and exposure; it is not a multimetric developmental match.

## Limits

Complete per-expert token-frequency diversity is unavailable because historical context logs are samples, not full routing histories. Only R0/R20/R100 are retained in the direct e8 ticket-result reference family, so the direct R10/R40 neighborhood around the e8 anchor is unavailable. These limitations restrict refinement, but do not reverse the measured all-seed ordering that 12E R20 is closer than 12E R60.

## Verdict

NO COMMON DEVELOPMENTAL STATE FOUND

**Most important implication for the paper:** The e8 R20 and e12 R60 sparse-recovery optima should be described as architecture-specific optimum ages, not evidence of a shared router/expert developmental clock. The later 12-expert optimum does not correspond to the same measured certainty, load, specialization, stabilization, functional-distance, or exposure state as e8 R20.

**Most important next experiment:** Rerun only the three e8 dense reference trajectories with the identical 2500-step protocol and checkpoints at R0/R10/R20/R40/R60/R80/R100, logging complete per-expert token identity/frequency histories and the same fixed-probe routing outputs; do not rerun sparse recovery.

NO COMMON DEVELOPMENTAL STATE FOUND
