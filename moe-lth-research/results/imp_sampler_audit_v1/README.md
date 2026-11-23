# CPU-only sampler audit v1

This directory is a new, versioned audit workspace. Historical v2/v3 files are
read-only inputs. No training is launched by any script here.

Run `bash run_cpu_audit.sh` from this directory. It compiles two distinct local
libraries, runs deterministic defect tests, reconstructs the sealed v2 pairs,
and reruns v3 connectivity feasibility with the original and validated
samplers under the frozen grid and gates.
