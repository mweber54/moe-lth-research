# Surviving initial-value placement findings

Status: COMPLETE. 70 of 70 registered endpoints available; missing and failed outcomes are listed in completion.json.

Five-source mean interaction: 0.05485539 CE; sample SD 0.02960742; range [0.02285032, 0.10241370]. Orders averaged equally within each source.

Initial mean penalties: IMP 4.84591133, random -0.00182740 CE. Final mean penalties: IMP 0.05222047, random -0.00263492 CE.

Mean IMP advantage over random: original 0.05864581, shuffled 0.00379042 CE. Positive values favor IMP.

The observed average interaction is positive: IMP has a larger shuffling penalty than sampled random masks in these tested conditions. This descriptive five-source result is conditional on the registered sources, masks and recovery protocol; it does not establish a specific mechanism.

An average IMP advantage remains after shuffling. Exact original placement is not necessary for this residual advantage under the tested conditions; arbitrary replacement values and topology-only explanations are untested.

|Source|Order|Initial IMP penalty|Initial random penalty|Final IMP penalty|Final random penalty|Interaction|Original advantage|Shuffled advantage|
|---|---|---|---|---|---|---|---|---|

|7|A|4.798553|-0.001350|0.033246|0.002064|0.031182|0.060088|0.028906|

|7|B|4.798553|-0.001350|0.013111|-0.001408|0.014519|0.027060|0.012542|

|17|A|4.703588|-0.002593|0.037865|-0.001463|0.039328|0.043033|0.003705|

|17|B|4.703588|-0.002593|0.051028|0.000178|0.050849|0.047575|-0.003275|

|29|A|4.869319|-0.002123|0.046909|0.005287|0.041623|0.042647|0.001024|

|29|B|4.869319|-0.002123|0.046881|-0.000713|0.047594|0.039823|-0.007771|

|43|A|5.301603|-0.003272|0.060300|-0.011649|0.071949|0.071167|-0.000782|

|43|B|5.301603|-0.003272|0.044700|-0.001982|0.046683|0.036857|-0.009826|

|61|A|4.556494|0.000202|0.085441|0.001357|0.084084|0.102977|0.018893|

|61|B|4.556494|0.000202|0.102723|-0.018020|0.120743|0.115232|-0.005511|

imp individual nested endpoint penalties range from -0.00105741 to 0.11715540 CE; every realization is retained in realization_level_effects.csv.

random individual nested endpoint penalties range from -0.03226589 to 0.01623850 CE; every realization is retained in realization_level_effects.csv.

Applied updates per endpoint range from 4997 to 4999 of 5000 attempts; no compensation steps are added.

Absolute sparse CE and sparse-minus-unpruned CE are in endpoint_losses.csv. Initial and final penalties are reported separately in realization_level_effects.csv; trajectories retain every realization. Positive interaction means IMP is harmed more by shuffling than sampled random controls. Shared IMP-original controls are used once within each source/order contrast. Random masks and shuffles are nested, not independent source replications.

Initial damage changes the model function; a final deficit alone does not establish slower optimization. No initial-CE adjustment is used. The treatment does not distinguish sign, magnitude, weighted paths or compatibility with biases/router/shared components. IMP and random retained-value distributions and energy are not matched. No novelty or topology-only mechanism is claimed. No equivalence threshold or follow-up grid is introduced.

Projected and measured production costs are separate in compute_accounting.json. Update/AMP/reset records and native gradient logs preserve numerical-accounting differences.
