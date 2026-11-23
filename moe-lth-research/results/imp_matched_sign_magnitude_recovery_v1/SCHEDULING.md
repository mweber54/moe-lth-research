# Matched study GPU scheduling

The coordinator is CPU-only. It probes Adder and Viper independently, selects
one admitted host, and starts preparation there. A preparation lock and a
durable `preparation_started.json` marker prohibit concurrent preparation or
automatic retries after preparation has begun. Failed or interrupted
preparation, a sealed registration, an ambiguous worker launch, and an
existing STOP require manual review; never remove a marker as a retry shortcut.
The host worker lock and atomic job claims remain in force. Admission is
rechecked before every claim. A worker on a busy GPU waits without claiming.
An unreachable host at the CPU verification stage is skipped; a failure after
the worker launch is attempted is ambiguous and stops the entire study without
retry. An existing STOP is not replaced.

The old `coordinator_status.json` can remain stale after a coordinator is
stopped. Check the actual process and lock ownership rather than treating that
file as proof of a running coordinator.

Every telemetry-based admission is recorded in `gpu_admission/` with the host,
UTC time, policy, free and total VRAM, GPU utilization, external compute PIDs,
and decision. A failed telemetry or SSH probe is recorded as rejected with
unavailable telemetry. CPU probes do not initialize CUDA. Telemetry can change
after a check and is not a GPU reservation; it cannot protect against another
user starting work afterwards.

## Authorization and configuration

Default policy authorizes **no hosts**, and does not permit sharing. An operator
must obtain and verify an exclusive GPU allocation under the cluster policy
before listing the host in `exclusive_authorized_hosts`. The configuration is
an operator attestation, not proof of an allocation. Even then the original
idle check (no external compute process, low memory use, utilization at most
10%) and the host's minimum free VRAM must both pass.

Set `MATCHED_GPU_POLICY` identically on the coordinator and every manual
process. The coordinator explicitly forwards it to remote hosts and workers.
It is sealed in the registration, along with the scheduling code, and cannot
be changed mid-study. `min_free_mib` maps host FQDNs to integers in MiB; choose
thresholds from measured resource requirements and allocation terms, not the
illustrative values below. A missing threshold defaults to zero for exclusive
use and 53,248 MiB for shared use; neither establishes sufficient capacity.
A missing authorized-host entry prohibits use.

Sharing is disabled unless `allow_shared` is explicitly true **and** a
committed authorization/amendment JSON is supplied. The JSON must apply to
this experiment and original preregistration, identify authorized hosts, cite
the cluster authorization, and contain the protocol amendment text. It must
assert both `cluster_gpu_sharing_authorized` and
`preregistration_amended_for_shared_gpu`. Obtain the actual cluster
authorization and any required scientific approval first; those assertions
are not themselves proof of either. The code checks the file against the
specified Git commit and seals its hash. Never create a false attestation or
enable sharing merely because telemetry looks favorable. The existing
preregistration and amendment are not changed by this scheduler.

For shared admission the provisional baseline from the earlier order-stability
study is at least 53,248 MiB free in each of five samples, two seconds apart,
with mean utilization at most 60% and no sample over 90%. Operators may set
stricter limits. This baseline is not a measured peak for the matched
preparation or recovery workload and does not establish authorization or
guarantee safe co-tenancy. The current scheduler uses NVIDIA's reported free
memory, which can be lower than the earlier worker's total-minus-used estimate.

## CPU-only validation (safe to run now)

From the project directory:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-highres/bin/python results/imp_matched_sign_magnitude_recovery_v1/test_scheduler.py
PYTHONDONTWRITEBYTECODE=1 python3 results/imp_matched_sign_magnitude_recovery_v1/gpu_policy.py dry-run
```

The first command exercises real scheduling/dispatch control flow with
temporary directories and mocked hosts/workers, including failed dispatch and
STOP. The second reports synthetic admission examples. Neither probes a real
GPU, starts a worker, claims a job, or consumes training attempts.

## Manual deployment, only after review and authorization

Do not run the following until the changes and allocation have been reviewed.
These example thresholds are **not** measured requirements. Edit them and the
authorized host list to match the actual allocation:

```sh
cd /users/kent/student/mweber54/moe-lth-research/moe-lth-research
export MATCHED_GPU_POLICY='{"exclusive_authorized_hosts":["adder.cs.kent.edu"],"min_free_mib":{"adder.cs.kent.edu":20000,"viper.cs.kent.edu":20000}}'
PYTHONDONTWRITEBYTECODE=1 .venv-highres/bin/python -u results/imp_matched_sign_magnitude_recovery_v1/coordinator.py
```

The final command can prepare and start the registered 108 recovery runs; it
is **not** a dry-run. Stop for manual review if any STOP, registration,
preparation-started marker, manifest, variants, jobs, or launch already
exists. Existing lock files alone are not evidence of an active process; do
not delete them.

For a separately authorized shared allocation, first commit the actual
authorization/amendment JSON, for example with keys:

```json
{
  "experiment": "imp_matched_sign_magnitude_recovery_v1",
  "preregistration_commit": "b4e9451bd9de867bdca8e6f996bf93f62955de33",
  "cluster_gpu_sharing_authorized": true,
  "preregistration_amended_for_shared_gpu": true,
  "cluster_authorization_reference": "actual cluster approval reference",
  "protocol_amendment_text": "actual approved amendment text",
  "hosts": ["adder.cs.kent.edu"]
}
```

Only after that approval and commit, set
`MATCHED_GPU_POLICY` to a JSON object containing `allow_shared: true`,
`authorization_path` (relative to the project directory),
`authorization_commit` (the full 40-character Git commit), host minimums,
and optionally `max_shared_utilization` (0–90; default 90) and
`max_shared_mean_utilization` (0–60; default 60). Run the same
coordinator command above. These controls do not replace the cluster's
allocation mechanism.
