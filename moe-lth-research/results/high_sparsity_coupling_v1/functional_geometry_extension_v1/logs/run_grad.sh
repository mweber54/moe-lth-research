cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research/scripts/functional_geometry_extension
for s in 7 17 29 43 61; do ../../.venv-highres/bin/python gradient_worker.py --seed $s >> ../../results/high_sparsity_coupling_v1/functional_geometry_extension_v1/logs/gradient_seed$s.log 2>&1; done
