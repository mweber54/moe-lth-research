# Mechanism Candidate Ranking

All rankings are descriptive. None establishes causation because token identities and intermediate router states were not saved.

1. **Coupled router adaptation (interaction)** - FOR: R20 is the only initialization with a consistently lower trainable final gap (0.1691 mean; seeds 0.1178, 0.2431, 0.1464) and it loses this advantage when frozen (0.2474 mean). Its trainable early router-gradient AUC is 185.1, above R0 (170.5) and R100 (175.5). AGAINST: this does not reveal the causal intermediate variable. Confidence: moderate as a descriptive interaction, low as mechanism.
2. **Layer-specific assignment/utilization adaptation** - FOR: aggregate assignment churn, utilization, and per-layer drop fractions are available at every 50 steps. AGAINST: aggregate counts cannot determine token-level reassignment or agreement with R0/R20/R100; frozen routing also changes as non-router weights recover. Confidence: low.
3. **Confidence/entropy/margin adaptation** - FOR: trainable R20 entropy decreases 0.5285 and margin rises 0.2176, versus frozen R20 changes of 0.1879 and 0.0642. AGAINST: trainable R0 changes more (-0.5786 entropy, +0.2576 margin) yet performs worse. Confidence: low.
4. **Router parameter drift** - FOR: trainable R20 has normalized final drift 0.9906; frozen is exactly zero. AGAINST: R0 drifts more (1.0525) and R100 less (0.6718), so magnitude alone does not track the effect. No intermediate parameter checkpoints support distance-to-reference analysis. Confidence: low.
5. **Assignment reorganization/stabilization** - FOR: trainable R20 final aggregate churn is 0.7767. AGAINST: R0 churn is higher (0.8454), R100 lower (0.6924), and frozen R20 is also high (0.7949); aggregate churn does not explain the ordering. Confidence: low.
6. **Expert-gradient dynamics** - FOR: R20 early expert-gradient AUC is 78.8 trainable versus 59.1 frozen. AGAINST: R100 is higher (112.0) without R20's performance; two expert-gradient records are invalid, and frozen lacks validity flags. Confidence: low.
7. **Capacity/token-drop reduction** - FOR: trainable R20 token drop falls by 0.0696. AGAINST: frozen aggregate capacity fields are absent, and R100 has a comparable drop decrease (0.0537). Confidence: low.

The R20 advantage appears after early transient recovery: mean trainable gaps are R0/R20/R100 = 0.0512/0.0721/0.1154 at step 500, then 0.1999/0.1764/0.2339 at step 1000 and 0.2857/0.1691/0.1891 at step 2500. Thus candidates must precede or coincide with the 500-1000 step separation and be seed-consistent.
