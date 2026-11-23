#!/usr/bin/env bash
set -euo pipefail
study=results/imp_sign_repair_mask_routing_v3
cd "$(dirname "$(realpath "$0")")/../.."
python=.venv-highres/bin/python
"$python" -B - "$study" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
e=json.loads((p/"repair_outcome.json").read_text())
decision=e["positive_seeds_step250"]==3
(p/"cross_mask_repair_gate.json").write_text(json.dumps(dict(
    run=decision,rule="Positive full-validation step-250 repair interaction in all three seed means.",evidence=e),indent=2))
PY
if "$python" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["run"] else 1)' "$study/cross_mask_repair_gate.json"; then
    ssh -i "$HOME/.ssh/dataset_b_cluster" -o BatchMode=yes -o ConnectTimeout=15 adder.cs.kent.edu         "cd '$PWD' || exit; nohup setsid bash -c '$python -Bu $study/later.py connectivity_repair 1; echo \$? > $study/connectivity_repair_adder.status' >> '$study/connectivity_repair_adder.log' 2>&1 < /dev/null &"
    "$python" -Bu "$study/later.py" connectivity_repair 0 >> "$study/connectivity_repair_viper.log" 2>&1
    while [[ ! -s "$study/connectivity_repair_adder.status" ]]; do sleep 30; done
    test "$(cat "$study/connectivity_repair_adder.status")" = 0
fi
