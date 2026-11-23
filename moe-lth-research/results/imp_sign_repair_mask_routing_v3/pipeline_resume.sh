#!/usr/bin/env bash
set -euo pipefail
study=results/imp_sign_repair_mask_routing_v3
cd "$(dirname "$(realpath "$0")")/../.."
python=.venv-highres/bin/python
remote=adder.cs.kent.edu
ssh_options=(-i "$HOME/.ssh/dataset_b_cluster" -o BatchMode=yes -o ConnectTimeout=15)
wait_for_audit() {
    local pid expected
    pid=$(cat "$study/audit.pid")
    expected="$python -Bu $study/audit.py"
    while [[ ! -s "$study/audit_manifest.json" ]]; do
        if [[ "$(ps -p "$pid" -o args= || true)" != "$expected" ]]; then
            test -s "$study/audit_manifest.json" && break
            echo "Audit exited before producing a verified manifest"
            return 1
        fi
        sleep 10
    done
}
run_pair() {
    local script=$1 mode=$2 local_status=0 remote_state
    local done_file="$study/${script}_${mode}_adder.status"
    rm -f "$done_file"
    ssh "${ssh_options[@]}" "$remote" "cd '$PWD' || exit;
        nohup setsid bash -c '$python -Bu $study/$script.py $mode 1; echo \$? > $done_file' >> '$study/${script}_${mode}_adder.log' 2>&1 < /dev/null &"
    "$python" -Bu "$study/$script.py" "$mode" 0 >> "$study/${script}_${mode}_viper.log" 2>&1 || local_status=$?
    if [[ "$local_status" != 0 ]]; then
        echo "Viper failed in $script $mode with status $local_status"
        return "$local_status"
    fi
    while [[ ! -s "$done_file" ]]; do sleep 30; done
    remote_state=$(cat "$done_file")
    if [[ "$remote_state" != 0 ]]; then
        echo "Adder failed in $script $mode with status $remote_state"
        return "$remote_state"
    fi
}
"$python" -Bu "$study/prediction.py" >> "$study/prediction.log" 2>&1
run_pair replay repair
"$python" -Bu "$study/summaries.py" repair
"$python" -Bu "$study/verdict.py"
"$python" -Bu "$study/masks.py"
"$python" -Bu "$study/feasibility.py" connectivity
run_pair later connectivity
"$python" -Bu "$study/summaries.py" connectivity
"$python" -Bu "$study/verdict.py"
"$python" -Bu - "$PWD/$study" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
import later_report
later_report.sign_agreement("connectivity")
PY
bash "$study/conditional_repair.sh"
run_pair probes scaled
"$python" -Bu "$study/feasibility.py" routing
run_pair later routing
"$python" -Bu "$study/summaries.py" routing
"$python" -Bu "$study/verdict.py"
