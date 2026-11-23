# Known-mask transitions — COMPLETE

20/20 complete, 0 active, 0 pending, 0 failed.

Prospective controlled experiment motivated by previous exploratory findings; not
project-level preregistration. The existing IMP mask is privileged information.
No outcome-driven changes or additional conditions are authorized.

The conditions share initialization, attempted-token budget and final sparse mask,
but have unequal cumulative active-parameter budgets. Abrupt keeps dense capacity
longer than gradual; both exceed fixed early capacity. Zeros in dense tensors imply
no FLOP or wall-clock savings. IMP discovery cost remains an excluded oracle cost.

Source seed is the replication unit(n=5). Individual seed effects precede descriptive
means, medians, ranges and SD. Missing states/readouts remain missing. Historical
TF32 CE values are never substituted for the common FP32 TF32-off readout.

## Each seed

|Seed|Abrupt−fixed5000|Gradual−fixed5000|Gradual−abrupt5000|
|---|---:|---:|---:|
|7|0.0315539|0.0469191|0.0153652|
|17|-0.0216685|-0.00756331|0.0141052|
|29|0.0207117|0.0151378|-0.00557389|
|43|0.0109433|-0.00131076|-0.012254|
|61|0.0400994|0.021384|-0.0187154|

## Descriptive source-seed summaries

|Contrast|Observed seeds|Mean|Median|Range|SD|
|---|---:|---:|---:|---|---:|
|abrupt_minus_fixed_5000|5|0.0163279|0.0207117|-0.0216685 to 0.0400994|0.023921|
|gradual_minus_fixed_5000|5|0.0149134|0.0151378|-0.00756331 to 0.0469191|0.0214167|
|gradual_minus_abrupt_5000|5|-0.00141459|-0.00557389|-0.0187154 to 0.0153652|0.015464|

## Required conclusions

1. All three sparse paths reached the identical registered final IMP80 mask within each seed.

2–3. abrupt:5000 transition−fixed mean=0.0163279, median=0.0207117, range[-0.0216685,0.0400994], SD=0.023921; 4/5 positive deficits and 1/5 negative differences. Pre-reset2500 mean difference=0.0353454. These continuous seed effects describe persistence and sign consistency; no equivalence margin was registered.

4. abrupt observed post1000 transition−fixed gaps by seed=[0.4724641320483407, 0.48768719437151997, 0.4873125749314027, 0.5145659042946602, 0.528168287009239]; pre-reset2500=[0.0702650330054877, -0.0021798871516844898, 0.054740835400852283, 0.01581451919763177, 0.03808641276246627];2525=[0.07131669860868439, -0.010796501298380168, 0.05812380542124296, 0.01734183217821106, 0.03738674285985377];5000=[0.031553878368479005, -0.021668505814505412, 0.020711712416195516, 0.010943275713307665, 0.04009935665203512]. Compare these individual trajectories with immediate training-probe event damage; they measure different populations. Changes across2500 combine optimizer reset and further training and do not isolate either.

2–3. gradual:5000 transition−fixed mean=0.0149134, median=0.0151378, range[-0.00756331,0.0469191], SD=0.0214167; 3/5 positive deficits and 2/5 negative differences. Pre-reset2500 mean difference=0.0252355. These continuous seed effects describe persistence and sign consistency; no equivalence margin was registered.

4. gradual observed post1000 transition−fixed gaps by seed=[0.10637022987428191, 0.12295679481055721, 0.12277785443557665, 0.09567137765213785, 0.11019151841233477]; pre-reset2500=[0.06407608525715736, -0.005787488267864793, 0.04647098527477156, -0.005284658233798423, 0.02670248771002104];2525=[0.0652480412255505, -0.011526307373283418, 0.047207664551767836, -0.007648615910515355, 0.0300417874725869];5000=[0.04691907479535096, -0.007563311984102095, 0.015137823011559748, -0.0013107626792940863, 0.021383961667154727]. Compare these individual trajectories with immediate training-probe event damage; they measure different populations. Changes across2500 combine optimizer reset and further training and do not isolate either.

5. Capacity, removal timing, dense-training history and retained Adam state remain combined. Routing changes are accompanying diagnostics, not a causal routing intervention. Better transition results may reflect useful dense history at a higher capacity cost. Known-mask reachability does not establish online discovery or training savings.

6. These mixed path effects do not yet isolate a mechanism or justify expanding mask selection. Neither a new mask-selection experiment nor an additional transition intervention is recommended from these results alone. No follow-up is launched.
