# 12-Expert Router Development Preflight

Architecture: 12 experts, top-k=1, four MoE layers, d_model=256, expert hidden size=1024, capacity factor 1.25, learned linear router, auxiliary loss 0.1. Only router age and expert count study scope differ from the established protocol; all ages are fractions of each 2500-step dense reference. Sparse tickets use masks discovered from E_T and surviving values rewound from E_0. Existing compatible R0/R20/R100 e12 cells are reused; missing ages are new runs.
