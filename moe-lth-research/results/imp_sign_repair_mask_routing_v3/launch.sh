#!/usr/bin/env bash
set -euo pipefail
study=results/imp_sign_repair_mask_routing_v3
cd "$(dirname "$(realpath "$0")")/../.."
test -s "$study/audit_manifest.json"
mode="$1"
ssh -i ~/.ssh/dataset_b_cluster -o BatchMode=yes -o ConnectTimeout=15 adder.cs.kent.edu  "cd '$PWD' || exit; nohup setsid .venv-highres/bin/python -Bu '$study/replay.py' '$mode' 1 >> '$study/${mode}_adder.log' 2>&1 < /dev/null &"
exec .venv-highres/bin/python -Bu "$study/replay.py" "$mode" 0
