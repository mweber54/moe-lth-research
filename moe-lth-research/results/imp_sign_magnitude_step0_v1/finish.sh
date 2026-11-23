#!/usr/bin/env bash
set -euo pipefail
cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research
for analysis_seed in 7 17 29 43 61; do
    ANALYSIS_SEED="$analysis_seed" .venv-highres/bin/python -u results/imp_sign_magnitude_step0_v1/analyze.py >> "results/imp_sign_magnitude_step0_v1/analysis_seed${analysis_seed}.log" 2>&1
    ANALYSIS_SEED="$analysis_seed" .venv-highres/bin/python -u results/imp_sign_magnitude_step0_v1/matched_signs.py >> "results/imp_sign_magnitude_step0_v1/matched_seed${analysis_seed}.log" 2>&1
done
.venv-highres/bin/python results/imp_sign_magnitude_step0_v1/report.py > results/imp_sign_magnitude_step0_v1/report.log 2>&1
