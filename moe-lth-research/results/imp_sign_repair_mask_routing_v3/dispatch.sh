#!/usr/bin/env bash
set -euo pipefail
study=results/imp_sign_repair_mask_routing_v3
cd "$(dirname "$(realpath "$0")")/../.."
expected=".venv-highres/bin/python -Bu $study/audit.py"
pid=$(cat "$study/audit.pid")
while [[ ! -s "$study/audit_manifest.json" ]]; do
    if [[ "$(ps -p "$pid" -o args= || true)" != "$expected" ]]; then
        test -s "$study/audit_manifest.json" && break
        echo "Audit exited without a verified manifest"
        exit 1
    fi
    sleep 10
done
bash "$study/launch.sh" free
while true; do
    completed=$(find "$study/free" -mindepth 2 -maxdepth 2 -name result.json | wc -l)
    [[ "$completed" == 42 ]] && break
    sleep 30
done
bash "$study/launch.sh" repair
