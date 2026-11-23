# Winning Ticket Rankings by Cohort

This document ranks candidate tickets by their absolute sparse-minus-dense validation cross-entropy gap within valid, distinct comparison cohorts. Never pool or compare across distinct recovery budgets, optimizer reset protocols, or discovery stages.

> [!NOTE]

> **Classification criteria:**

> - **Category A (Strict LTH-verified tickets):** 0 models established in this repository (all evaluations use validation loss, many involve optimizer reset, and sample sizes are small).

> - **Category B (Candidate tickets):** Sparse validation CE $\le$ Dense validation CE (gap $\le 0.000000$).

> - **Category C (Near-dense recoveries):** Sparse within 5% of dense CE ($0 < \text{relative gap} \le +5.0\%$).

> - **Weaker recoveries:** Relative gap $> +5.0\%$.


## Cohort 1: Primary 2,500-step IMP Final Recoveries (E8, E16, E32; 80% & 85% Sparsity)

| Rank | Architecture | Seed | Sparsity | Order | Dense CE | Sparse CE | Gap (Sparse − Dense) | Relative Gap | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | E16 | 17 | 80.0% | original | 1.479363 | 1.458223 | **-0.021140** | -1.43% | Category B |
| 2 | E32 | 17 | 80.0% | original | 1.496764 | 1.488434 | **-0.008330** | -0.56% | Category B |
| 3 | E32 | 61 | 80.0% | original | 1.524581 | 1.518804 | **-0.005777** | -0.38% | Category B |
| 4 | E32 | 7 | 80.0% | original | 1.483661 | 1.490267 | **+0.006606** | +0.45% | Category C |
| 5 | E32 | 29 | 80.0% | original | 1.489596 | 1.496653 | **+0.007057** | +0.47% | Category C |
| 6 | E32 | 43 | 80.0% | original | 1.481041 | 1.492978 | **+0.011937** | +0.81% | Category C |
| 7 | E32 | 7 | 85.0% | original | 1.483661 | 1.497846 | **+0.014185** | +0.96% | Category C |
| 8 | E32 | 17 | 85.0% | original | 1.496764 | 1.528476 | **+0.031713** | +2.12% | Category C |
| 9 | E16 | 43 | 80.0% | original | 1.452162 | 1.484748 | **+0.032586** | +2.24% | Category C |
| 10 | E8 | 29 | 80.0% | original | 1.449880 | 1.485237 | **+0.035358** | +2.44% | Category C |
| 11 | E16 | 17 | 85.0% | original | 1.479363 | 1.516907 | **+0.037544** | +2.54% | Category C |
| 12 | E8 | 43 | 80.0% | original | 1.486438 | 1.530619 | **+0.044181** | +2.97% | Category C |
| 13 | E16 | 29 | 80.0% | original | 1.428627 | 1.488630 | **+0.060003** | +4.20% | Category C |
| 14 | E32 | 43 | 85.0% | original | 1.481041 | 1.541849 | **+0.060808** | +4.11% | Category C |
| 15 | E16 | 43 | 85.0% | original | 1.452162 | 1.513762 | **+0.061600** | +4.24% | Category C |
| 16 | E32 | 29 | 85.0% | original | 1.489596 | 1.556545 | **+0.066949** | +4.49% | Category C |
| 17 | E8 | 17 | 85.0% | original | 1.525026 | 1.604084 | **+0.079058** | +5.18% | Weaker / Unsuccessful recovery |
| 18 | E8 | 61 | 80.0% | original | 1.473055 | 1.557290 | **+0.084235** | +5.72% | Weaker / Unsuccessful recovery |
| 19 | E32 | 61 | 85.0% | original | 1.524581 | 1.609297 | **+0.084716** | +5.56% | Weaker / Unsuccessful recovery |
| 20 | E8 | 17 | 80.0% | original | 1.525026 | 1.611851 | **+0.086824** | +5.69% | Weaker / Unsuccessful recovery |
| 21 | E8 | 61 | 85.0% | original | 1.473055 | 1.566409 | **+0.093354** | +6.34% | Weaker / Unsuccessful recovery |
| 22 | E16 | 7 | 80.0% | original | 1.479220 | 1.591078 | **+0.111858** | +7.56% | Weaker / Unsuccessful recovery |
| 23 | E16 | 29 | 85.0% | original | 1.428627 | 1.549095 | **+0.120467** | +8.43% | Weaker / Unsuccessful recovery |
| 24 | E16 | 7 | 85.0% | original | 1.479220 | 1.612969 | **+0.133749** | +9.04% | Weaker / Unsuccessful recovery |
| 25 | E16 | 61 | 80.0% | original | 1.458569 | 1.608024 | **+0.149455** | +10.25% | Weaker / Unsuccessful recovery |
| 26 | E16 | 61 | 85.0% | original | 1.458569 | 1.608058 | **+0.149489** | +10.25% | Weaker / Unsuccessful recovery |
| 27 | E8 | 7 | 80.0% | original | 1.488117 | 1.640241 | **+0.152125** | +10.22% | Weaker / Unsuccessful recovery |
| 28 | E8 | 43 | 85.0% | original | 1.486438 | 1.646748 | **+0.160310** | +10.78% | Weaker / Unsuccessful recovery |
| 29 | E8 | 7 | 85.0% | original | 1.488117 | 1.653694 | **+0.165577** | +11.13% | Weaker / Unsuccessful recovery |
| 30 | E8 | 29 | 85.0% | original | 1.449880 | 1.616458 | **+0.166579** | +11.49% | Weaker / Unsuccessful recovery |


## Cohort 2: Extended 5,000-step IMP Recoveries (E8, E16, E32; 80% & 85% Sparsity; Reset at 2500)

| Rank | Architecture | Seed | Sparsity | Order | Dense CE | Sparse CE | Gap (Sparse − Dense) | Relative Gap | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | E32 | 7 | 80.0% | original | 1.365141 | 1.329541 | **-0.035600** | -2.61% | Category B |
| 2 | E32 | 61 | 80.0% | original | 1.366185 | 1.339272 | **-0.026913** | -1.97% | Category B |
| 3 | E32 | 7 | 85.0% | original | 1.365141 | 1.343936 | **-0.021205** | -1.55% | Category B |
| 4 | E32 | 43 | 80.0% | original | 1.339491 | 1.337727 | **-0.001764** | -0.13% | Category B |
| 5 | E16 | 17 | 80.0% | original | 1.331802 | 1.332964 | **+0.001161** | +0.09% | Category C |
| 6 | E32 | 29 | 80.0% | original | 1.341397 | 1.345185 | **+0.003789** | +0.28% | Category C |
| 7 | E32 | 43 | 85.0% | original | 1.339491 | 1.352689 | **+0.013198** | +0.99% | Category C |
| 8 | E16 | 29 | 80.0% | original | 1.321925 | 1.339021 | **+0.017097** | +1.29% | Category C |
| 9 | E32 | 17 | 80.0% | original | 1.329892 | 1.351009 | **+0.021117** | +1.59% | Category C |
| 10 | E32 | 29 | 85.0% | original | 1.341397 | 1.366194 | **+0.024797** | +1.85% | Category C |
| 11 | E16 | 17 | 85.0% | original | 1.331802 | 1.359199 | **+0.027396** | +2.06% | Category C |
| 12 | E32 | 61 | 85.0% | original | 1.366185 | 1.395752 | **+0.029567** | +2.16% | Category C |
| 13 | E32 | 17 | 85.0% | original | 1.329892 | 1.366163 | **+0.036271** | +2.73% | Category C |
| 14 | E16 | 43 | 80.0% | original | 1.307075 | 1.347895 | **+0.040820** | +3.12% | Category C |
| 15 | E16 | 29 | 85.0% | original | 1.321925 | 1.368432 | **+0.046508** | +3.52% | Category C |
| 16 | E8 | 61 | 80.0% | original | 1.312370 | 1.359518 | **+0.047148** | +3.59% | Category C |
| 17 | E8 | 17 | 80.0% | original | 1.342707 | 1.393060 | **+0.050353** | +3.75% | Category C |
| 18 | E8 | 17 | 85.0% | original | 1.342707 | 1.395561 | **+0.052854** | +3.94% | Category C |
| 19 | E16 | 43 | 85.0% | original | 1.307075 | 1.363433 | **+0.056358** | +4.31% | Category C |
| 20 | E8 | 29 | 80.0% | original | 1.292806 | 1.352959 | **+0.060152** | +4.65% | Category C |
| 21 | E8 | 61 | 85.0% | original | 1.312370 | 1.373400 | **+0.061030** | +4.65% | Category C |
| 22 | E8 | 43 | 80.0% | original | 1.296372 | 1.364653 | **+0.068282** | +5.27% | Weaker / Unsuccessful recovery |
| 23 | E16 | 61 | 85.0% | original | 1.321252 | 1.391043 | **+0.069791** | +5.28% | Weaker / Unsuccessful recovery |
| 24 | E16 | 7 | 80.0% | original | 1.325997 | 1.399186 | **+0.073189** | +5.52% | Weaker / Unsuccessful recovery |
| 25 | E16 | 61 | 80.0% | original | 1.321252 | 1.400198 | **+0.078946** | +5.98% | Weaker / Unsuccessful recovery |
| 26 | E16 | 7 | 85.0% | original | 1.325997 | 1.417819 | **+0.091822** | +6.92% | Weaker / Unsuccessful recovery |
| 27 | E8 | 7 | 80.0% | original | 1.310935 | 1.417124 | **+0.106190** | +8.10% | Weaker / Unsuccessful recovery |
| 28 | E8 | 29 | 85.0% | original | 1.292806 | 1.401026 | **+0.108219** | +8.37% | Weaker / Unsuccessful recovery |
| 29 | E8 | 7 | 85.0% | original | 1.310935 | 1.424287 | **+0.113352** | +8.65% | Weaker / Unsuccessful recovery |
| 30 | E8 | 43 | 85.0% | original | 1.296372 | 1.418864 | **+0.122493** | +9.45% | Weaker / Unsuccessful recovery |


## Cohort 3: Matched Recovery-Order Permutation Stability (E32, 80% Sparsity, 5000 Steps; Orders A & B)

| Rank | Architecture | Seed | Sparsity | Order | Dense CE | Sparse CE | Gap (Sparse − Dense) | Relative Gap | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | E32 | 7 | 80.0% | A | 1.326870 | 1.322395 | **-0.004475** | -0.34% | Category B |
| 2 | E32 | 7 | 80.0% | B | 1.329698 | 1.344199 | **+0.014501** | +1.09% | Category C |
| 3 | E32 | 29 | 80.0% | A | 1.311695 | 1.342609 | **+0.030914** | +2.36% | Category C |
| 4 | E32 | 17 | 80.0% | B | 1.314434 | 1.347391 | **+0.032958** | +2.51% | Category C |
| 5 | E32 | 17 | 80.0% | A | 1.311574 | 1.348883 | **+0.037309** | +2.84% | Category C |


## Cohort 4: Lower-Sparsity Discovery Stages (Rounds 1–7; 20% to 67.2% Sparsity)

| Rank | Architecture | Seed | Sparsity | Order | Dense CE | Sparse CE | Gap (Sparse − Dense) | Relative Gap | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | E32 | 61 | 59.0% | original | 1.524581 | 1.422411 | **-0.102170** | -6.70% | Category B |
| 2 | E32 | 61 | 48.8% | original | 1.524581 | 1.424036 | **-0.100545** | -6.59% | Category B |
| 3 | E32 | 61 | 36.0% | original | 1.524581 | 1.440277 | **-0.084304** | -5.53% | Category B |
| 4 | E32 | 61 | 67.2% | original | 1.524581 | 1.442486 | **-0.082095** | -5.38% | Category B |
| 5 | E32 | 61 | 20.0% | original | 1.524581 | 1.467000 | **-0.057582** | -3.78% | Category B |
| 6 | E32 | 43 | 59.0% | original | 1.481041 | 1.424429 | **-0.056612** | -3.82% | Category B |
| 7 | E32 | 43 | 67.2% | original | 1.481041 | 1.425821 | **-0.055221** | -3.73% | Category B |
| 8 | E32 | 43 | 48.8% | original | 1.481041 | 1.428530 | **-0.052511** | -3.55% | Category B |
| 9 | E32 | 7 | 59.0% | original | 1.483661 | 1.432610 | **-0.051051** | -3.44% | Category B |
| 10 | E32 | 17 | 67.2% | original | 1.496764 | 1.450670 | **-0.046094** | -3.08% | Category B |
| 11 | E8 | 61 | 20.0% | original | 1.473055 | 1.428424 | **-0.044631** | -3.03% | Category B |
| 12 | E32 | 17 | 36.0% | original | 1.496764 | 1.452150 | **-0.044614** | -2.98% | Category B |
| 13 | E8 | 17 | 59.0% | original | 1.525026 | 1.481780 | **-0.043247** | -2.84% | Category B |
| 14 | E32 | 7 | 48.8% | original | 1.483661 | 1.440984 | **-0.042677** | -2.88% | Category B |
| 15 | E8 | 17 | 48.8% | original | 1.525026 | 1.482906 | **-0.042121** | -2.76% | Category B |
| 16 | E8 | 17 | 36.0% | original | 1.525026 | 1.483038 | **-0.041988** | -2.75% | Category B |
| 17 | E32 | 43 | 36.0% | original | 1.481041 | 1.440392 | **-0.040650** | -2.74% | Category B |
| 18 | E32 | 17 | 59.0% | original | 1.496764 | 1.458977 | **-0.037787** | -2.52% | Category B |
| 19 | E16 | 17 | 48.8% | original | 1.479363 | 1.441632 | **-0.037731** | -2.55% | Category B |
| 20 | E32 | 17 | 48.8% | original | 1.496764 | 1.459086 | **-0.037677** | -2.52% | Category B |
| 21 | E16 | 7 | 59.0% | original | 1.479220 | 1.444326 | **-0.034894** | -2.36% | Category B |
| 22 | E32 | 43 | 20.0% | original | 1.481041 | 1.449131 | **-0.031911** | -2.15% | Category B |
| 23 | E32 | 29 | 48.8% | original | 1.489596 | 1.460268 | **-0.029328** | -1.97% | Category B |
| 24 | E8 | 17 | 20.0% | original | 1.525026 | 1.496062 | **-0.028964** | -1.90% | Category B |
| 25 | E16 | 7 | 48.8% | original | 1.479220 | 1.451137 | **-0.028083** | -1.90% | Category B |
| 26 | E16 | 17 | 67.2% | original | 1.479363 | 1.452088 | **-0.027274** | -1.84% | Category B |
| 27 | E8 | 7 | 59.0% | original | 1.488117 | 1.460985 | **-0.027132** | -1.82% | Category B |
| 28 | E32 | 17 | 20.0% | original | 1.496764 | 1.472501 | **-0.024263** | -1.62% | Category B |
| 29 | E16 | 43 | 67.2% | original | 1.452162 | 1.427909 | **-0.024253** | -1.67% | Category B |
| 30 | E32 | 7 | 67.2% | original | 1.483661 | 1.460475 | **-0.023186** | -1.56% | Category B |
| 31 | E16 | 17 | 36.0% | original | 1.479363 | 1.456466 | **-0.022897** | -1.55% | Category B |
| 32 | E32 | 29 | 20.0% | original | 1.489596 | 1.468378 | **-0.021218** | -1.42% | Category B |
| 33 | E32 | 29 | 36.0% | original | 1.489596 | 1.468583 | **-0.021013** | -1.41% | Category B |
| 34 | E8 | 7 | 67.2% | original | 1.488117 | 1.467146 | **-0.020971** | -1.41% | Category B |
| 35 | E32 | 7 | 36.0% | original | 1.483661 | 1.465496 | **-0.018165** | -1.22% | Category B |
| 36 | E16 | 7 | 36.0% | original | 1.479220 | 1.461911 | **-0.017309** | -1.17% | Category B |
| 37 | E8 | 61 | 36.0% | original | 1.473055 | 1.456113 | **-0.016941** | -1.15% | Category B |
| 38 | E32 | 7 | 20.0% | original | 1.483661 | 1.467272 | **-0.016389** | -1.10% | Category B |
| 39 | E16 | 43 | 59.0% | original | 1.452162 | 1.438680 | **-0.013482** | -0.93% | Category B |
| 40 | E16 | 17 | 59.0% | original | 1.479363 | 1.466083 | **-0.013279** | -0.90% | Category B |
| 41 | E8 | 61 | 67.2% | original | 1.473055 | 1.461435 | **-0.011620** | -0.79% | Category B |
| 42 | E16 | 43 | 36.0% | original | 1.452162 | 1.440720 | **-0.011442** | -0.79% | Category B |
| 43 | E16 | 7 | 20.0% | original | 1.479220 | 1.469360 | **-0.009860** | -0.67% | Category B |
| 44 | E8 | 43 | 59.0% | original | 1.486438 | 1.476759 | **-0.009679** | -0.65% | Category B |
| 45 | E16 | 61 | 67.2% | original | 1.458569 | 1.450054 | **-0.008516** | -0.58% | Category B |
| 46 | E8 | 43 | 67.2% | original | 1.486438 | 1.479524 | **-0.006914** | -0.47% | Category B |
| 47 | E8 | 43 | 48.8% | original | 1.486438 | 1.479525 | **-0.006914** | -0.47% | Category B |
| 48 | E8 | 17 | 67.2% | original | 1.525026 | 1.519233 | **-0.005793** | -0.38% | Category B |
| 49 | E16 | 43 | 48.8% | original | 1.452162 | 1.447171 | **-0.004991** | -0.34% | Category B |
| 50 | E8 | 61 | 48.8% | original | 1.473055 | 1.469922 | **-0.003133** | -0.21% | Category B |
| 51 | E16 | 29 | 36.0% | original | 1.428627 | 1.426180 | **-0.002447** | -0.17% | Category B |
| 52 | E16 | 43 | 20.0% | original | 1.452162 | 1.450459 | **-0.001703** | -0.12% | Category B |
| 53 | E16 | 17 | 0.0% | original | 1.479363 | 1.479363 | **+0.000000** | +0.00% | Category B |
| 54 | E16 | 29 | 0.0% | original | 1.428627 | 1.428627 | **+0.000000** | +0.00% | Category B |
| 55 | E16 | 43 | 0.0% | original | 1.452162 | 1.452162 | **+0.000000** | +0.00% | Category B |
| 56 | E16 | 61 | 0.0% | original | 1.458569 | 1.458569 | **+0.000000** | +0.00% | Category B |
| 57 | E16 | 7 | 0.0% | original | 1.479220 | 1.479220 | **+0.000000** | +0.00% | Category B |
| 58 | E32 | 17 | 0.0% | original | 1.496764 | 1.496764 | **+0.000000** | +0.00% | Category B |
| 59 | E32 | 29 | 0.0% | original | 1.489596 | 1.489596 | **+0.000000** | +0.00% | Category B |
| 60 | E32 | 43 | 0.0% | original | 1.481041 | 1.481041 | **+0.000000** | +0.00% | Category B |
| 61 | E32 | 61 | 0.0% | original | 1.524581 | 1.524581 | **+0.000000** | +0.00% | Category B |
| 62 | E32 | 7 | 0.0% | original | 1.483661 | 1.483661 | **+0.000000** | +0.00% | Category B |
| 63 | E8 | 17 | 0.0% | original | 1.525026 | 1.525026 | **+0.000000** | +0.00% | Category B |
| 64 | E8 | 29 | 0.0% | original | 1.449880 | 1.449880 | **+0.000000** | +0.00% | Category B |
| 65 | E8 | 43 | 0.0% | original | 1.486438 | 1.486438 | **+0.000000** | +0.00% | Category B |
| 66 | E8 | 61 | 0.0% | original | 1.473055 | 1.473055 | **+0.000000** | +0.00% | Category B |
| 67 | E8 | 7 | 0.0% | original | 1.488117 | 1.488117 | **+0.000000** | +0.00% | Category B |
| 68 | E16 | 17 | 20.0% | original | 1.479363 | 1.480343 | **+0.000981** | +0.07% | Category C |
| 69 | E8 | 29 | 48.8% | original | 1.449880 | 1.453307 | **+0.003428** | +0.24% | Category C |
| 70 | E32 | 29 | 67.2% | original | 1.489596 | 1.495777 | **+0.006181** | +0.41% | Category C |
| 71 | E16 | 29 | 48.8% | original | 1.428627 | 1.435446 | **+0.006818** | +0.48% | Category C |
| 72 | E8 | 7 | 36.0% | original | 1.488117 | 1.495136 | **+0.007019** | +0.47% | Category C |
| 73 | E8 | 61 | 59.0% | original | 1.473055 | 1.480250 | **+0.007196** | +0.49% | Category C |
| 74 | E8 | 43 | 36.0% | original | 1.486438 | 1.494296 | **+0.007858** | +0.53% | Category C |
| 75 | E32 | 29 | 59.0% | original | 1.489596 | 1.498018 | **+0.008422** | +0.57% | Category C |
| 76 | E8 | 7 | 48.8% | original | 1.488117 | 1.496662 | **+0.008546** | +0.57% | Category C |
| 77 | E8 | 7 | 20.0% | original | 1.488117 | 1.496717 | **+0.008600** | +0.58% | Category C |
| 78 | E16 | 29 | 20.0% | original | 1.428627 | 1.440407 | **+0.011779** | +0.82% | Category C |
| 79 | E16 | 29 | 59.0% | original | 1.428627 | 1.441461 | **+0.012834** | +0.90% | Category C |
| 80 | E16 | 61 | 59.0% | original | 1.458569 | 1.473650 | **+0.015080** | +1.03% | Category C |
| 81 | E8 | 43 | 20.0% | original | 1.486438 | 1.504868 | **+0.018430** | +1.24% | Category C |
| 82 | E8 | 29 | 67.2% | original | 1.449880 | 1.483898 | **+0.034019** | +2.35% | Category C |
| 83 | E8 | 29 | 59.0% | original | 1.449880 | 1.485157 | **+0.035277** | +2.43% | Category C |
| 84 | E8 | 29 | 36.0% | original | 1.449880 | 1.488990 | **+0.039111** | +2.70% | Category C |
| 85 | E16 | 61 | 48.8% | original | 1.458569 | 1.499382 | **+0.040812** | +2.80% | Category C |
| 86 | E16 | 29 | 67.2% | original | 1.428627 | 1.472121 | **+0.043494** | +3.04% | Category C |
| 87 | E8 | 29 | 20.0% | original | 1.449880 | 1.498762 | **+0.048883** | +3.37% | Category C |
| 88 | E16 | 7 | 67.2% | original | 1.479220 | 1.555385 | **+0.076165** | +5.15% | Weaker / Unsuccessful recovery |
| 89 | E16 | 61 | 20.0% | original | 1.458569 | 1.535588 | **+0.077019** | +5.28% | Weaker / Unsuccessful recovery |
| 90 | E16 | 61 | 36.0% | original | 1.458569 | 1.545718 | **+0.087148** | +5.97% | Weaker / Unsuccessful recovery |

