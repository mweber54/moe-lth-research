# Step-zero findings

All 15 unmodified baselines reproduce the original logged CE exactly. Step zero is true original random initialization; every router/shared tensor comes from that complete checkpoint. No training was run.

Full-dose mean CE changes across five draws are shown separately for every seed and mask realization:

|Seed|Mask|A: magnitude placement|B: sign placement|C: constant magnitude|D: full shuffle|
|---|---|---|---|---|---|
|7|imp|-0.582389|4.788810|-0.562223|4.794809|
|7|random1|0.000055|0.000205|0.001620|-0.000521|
|7|random2|-0.002689|-0.002778|-0.001293|-0.003014|
|17|imp|-0.524422|4.701809|-0.502143|4.702954|
|17|random1|-0.001753|-0.003977|-0.000539|-0.004013|
|17|random2|-0.003167|-0.000839|-0.002759|-0.000503|
|29|imp|-0.626563|4.866801|-0.604728|4.862159|
|29|random1|-0.003132|-0.000984|-0.003166|-0.002106|
|29|random2|0.001536|-0.000237|0.001022|0.000594|
|43|imp|-0.532510|5.307119|-0.511350|5.301520|
|43|random1|-0.000445|-0.001384|0.000327|-0.002829|
|43|random2|-0.001353|0.000443|-0.000400|0.000715|
|61|imp|-0.472565|4.556000|-0.455111|4.554161|
|61|random1|0.001890|-0.000817|0.001842|0.001311|
|61|random2|-0.001806|-0.001249|-0.001332|-0.001098|

Distance-matched comparisons use sign-count-preserving positive/negative pair swaps selected by weights alone. They are an additional conditional sign-permutation distribution, distinct from the uniform subset permutation B. This limits any claim of a universal sign effect. All realized distances and paired effects are in distance_matched_comparisons.csv. Maximum absolute aggregate distance mismatch: 3.2973185e-05.

Distance-matched effects at each seed and A dose:

|Seed|Mask|A dose|A change|Matched B change|B minus A|
|---|---|---|---|---|---|
|7|imp|0.1|-0.058542|0.065537|0.124079|
|7|imp|0.25|-0.143089|0.166233|0.309322|
|7|imp|0.5|-0.287318|0.325191|0.612509|
|7|imp|1|-0.582389|0.638014|1.220404|
|7|random1|0.1|0.000444|0.000113|-0.000331|
|7|random1|0.25|0.000739|-0.001115|-0.001855|
|7|random1|0.5|0.001271|0.001231|-0.000040|
|7|random1|1|0.000055|-0.000912|-0.000967|
|7|random2|0.1|-0.000392|0.000041|0.000433|
|7|random2|0.25|-0.000546|-0.001052|-0.000506|
|7|random2|0.5|-0.000750|-0.001472|-0.000723|
|7|random2|1|-0.002689|0.000178|0.002867|
|17|imp|0.1|-0.051735|0.069104|0.120839|
|17|imp|0.25|-0.127908|0.170708|0.298616|
|17|imp|0.5|-0.258262|0.334016|0.592278|
|17|imp|1|-0.524422|0.652956|1.177378|
|17|random1|0.1|0.000330|-0.000202|-0.000532|
|17|random1|0.25|-0.000406|-0.001535|-0.001129|
|17|random1|0.5|-0.000435|-0.000460|-0.000025|
|17|random1|1|-0.001753|-0.003585|-0.001832|
|17|random2|0.1|-0.000599|-0.000073|0.000526|
|17|random2|0.25|-0.000644|-0.001505|-0.000861|
|17|random2|0.5|-0.001112|-0.001939|-0.000828|
|17|random2|1|-0.003167|-0.001290|0.001877|
|29|imp|0.1|-0.061556|0.072749|0.134306|
|29|imp|0.25|-0.152204|0.174119|0.326324|
|29|imp|0.5|-0.308806|0.346470|0.655276|
|29|imp|1|-0.626563|0.671665|1.298228|
|29|random1|0.1|0.000065|0.000082|0.000017|
|29|random1|0.25|0.000099|-0.000959|-0.001058|
|29|random1|0.5|-0.001470|-0.001167|0.000303|
|29|random1|1|-0.003132|0.000513|0.003644|
|29|random2|0.1|0.000061|0.000057|-0.000004|
|29|random2|0.25|0.000118|0.000122|0.000004|
|29|random2|0.5|0.000278|0.000383|0.000105|
|29|random2|1|0.001536|0.000220|-0.001316|
|43|imp|0.1|-0.055496|0.082400|0.137896|
|43|imp|0.25|-0.131431|0.202821|0.334252|
|43|imp|0.5|-0.264211|0.398444|0.662655|
|43|imp|1|-0.532510|0.786831|1.319341|
|43|random1|0.1|-0.000145|0.000204|0.000348|
|43|random1|0.25|0.000346|0.000237|-0.000109|
|43|random1|0.5|-0.000602|0.000957|0.001559|
|43|random1|1|-0.000445|0.000059|0.000504|
|43|random2|0.1|0.000080|-0.000674|-0.000754|
|43|random2|0.25|0.000039|0.000743|0.000704|
|43|random2|0.5|-0.001273|-0.000790|0.000483|
|43|random2|1|-0.001353|-0.000794|0.000559|
|61|imp|0.1|-0.047640|0.067772|0.115411|
|61|imp|0.25|-0.116431|0.164793|0.281224|
|61|imp|0.5|-0.236162|0.326235|0.562397|
|61|imp|1|-0.472565|0.635216|1.107781|
|61|random1|0.1|0.000168|0.000045|-0.000123|
|61|random1|0.25|-0.000092|-0.000297|-0.000204|
|61|random1|0.5|-0.000140|0.000829|0.000969|
|61|random1|1|0.001890|0.000291|-0.001599|
|61|random2|0.1|-0.000482|-0.000547|-0.000065|
|61|random2|0.25|-0.000460|-0.000046|0.000414|
|61|random2|0.5|-0.001561|0.000461|0.002023|
|61|random2|1|-0.001806|-0.002447|-0.000642|

Uniform-chance CE is ln(256) = 5.545177. The observed random-mask baseline CE is much larger; it is not near chance. No chance-floor interpretation explains the flat shuffle response. The high CE is a reproduced feature of this initialization, not a corrected or retuned metric. In particular, a null random-mask damage effect is conditional on a model with extremely poor initial predictions.

All mask, degree, sign/magnitude multiset and A/B/D energy assertions passed for each emitted result. All identity draws reproduce baseline CE exactly with zero routing changes. Gaussian directions match A/B distance independently in every matrix. Noise can change signs and weight energy; it is a distance control, not a matched-distribution control.

These results apply only to step zero and the tested initializations and masks. They do not establish which individual connections are necessary, a topology-only mechanism, or any explanation of training behavior or the steps 1500–2500 gap.
