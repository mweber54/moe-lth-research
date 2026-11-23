"""Publish final report and order means after all sealed results arrive."""
import csv
import fcntl
from pathlib import Path
import time

from common_v2 import OUT, load, pilot as p


def main():
    while not (OUT / "matching_seal.json").is_file():
        dispatch = (OUT / "dispatch.log").read_text()
        if "Matching exited without a seal" in dispatch:
            raise SystemExit(dispatch.strip())
        time.sleep(30)
    report = load("neuron_matched_final_report", OUT / "report.py")
    assert report.OUT == OUT
    manifest = p.read(OUT / "inputs.json")
    assert p.digest(report.__file__) == manifest["hashes"][report.__file__]
    workers = [OUT / f"progress_{host}.cs.kent.edu.json" for host in ("viper","adder")]
    paths = [OUT / "jobs" / p.job_id(job) / "result.json" for job in manifest["jobs"]]
    last_count = None
    while True:
        count = sum(path.is_file() for path in paths)
        if count != last_count:
            print(f"{count} of {len(paths)} saved recoveries",flush=True)
            last_count = count
        if count == len(paths) and all(path.is_file() and p.read(path)["status"] == "complete" for path in workers):
            break
        if (OUT / "stop.request").exists():
            return
        time.sleep(60)
    with (OUT / "report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        report.main()
        endpoints = OUT / "paired_endpoints.csv"
        with endpoints.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 18
        for row in rows:
            for metric in ("high_minus_random","high_minus_baseline","random_minus_baseline"):
                row[metric] = float(row[metric])
        report.table("order_means.csv",report.summarize(rows,("order",)))
        artifacts = [endpoints, *(OUT / filename for filename in
                    ("per_run.csv","paired_results.csv","seed_order_means.csv","seed_means.csv",
                     "order_means.csv","endpoint_contrasts.pdf","recovery_trajectories.pdf","findings.md"))]
        p.write(OUT / "completion.json", dict(recoveries=len(paths),pairs=len(rows),
                 reporter=str(Path(report.__file__).resolve()),output=str(OUT),
                 artifacts={path.name:p.digest(path) for path in artifacts}))


if __name__ == "__main__":
    main()
