import json, glob, datetime
from fge_common import *

rows = [
 ('activation_geometry', 'completed', '16 steps x 5 seeds x 2 sparsities; sparse-native geometry not evaluated (historical n=128 table reused, labeled)'),
 ('fixed_gate_distortion', 'completed', 'same-step-dense gates (fixed_source, dense_native) and source-final gates (fixed_source)'),
 ('selection_state_pruning (C->D)', 'completed', '10 masks; C = IMP state that selected the mask'),
 ('gradient_geometry', 'completed', 'checkpoints 0/2500/5000, 16 batches, task and aux separated, 10 random-mask draws'),
 ('mask_structure', 'completed', 'nesting is structural; cross-seed coordinate overlap is raw only'),
 ('conditional connectivity', 'unavailable', 'no distinct same-mask endpoints exist; interpolation not run (connectivity/eligibility_report.md)'),
 ('mask-conditioned sparse-native activation geometry', 'excluded', 'not required for the primary fixed-source comparison; n=128 historical sparse-native reused'),
 ('historical track curves', 'excluded', 'weights identical to rerun at 0/2500/3000/4000/5000, not independent'),
 ('8E/12E gradient cohorts', 'excluded', 'different cohorts; not joined'),
]
table(OUT / 'coverage_and_exclusions.csv', [dict(family=a, status=b, note=c) for a, b, c in rows])
n_act = len(glob.glob(str(OUT / 'activation_geometry/seed_*/step_*/completed.json')))
n_grad = len(glob.glob(str(OUT / 'gradient_geometry/seed_*/completed.json')))
(OUT / 'completion.json').write_text(json.dumps(dict(
    status='COMPLETE', finished_utc=datetime.datetime.utcnow().isoformat() + 'Z', activation_steps_completed=n_act, activation_steps_expected=80,
    gradient_seeds_completed=n_grad, gradient_seeds_expected=5, connectivity='unavailable', training_performed=False,
    existing_artifacts_modified=False, implementation_hash=implementation_hash()), indent=2))
print(n_act, n_grad)
