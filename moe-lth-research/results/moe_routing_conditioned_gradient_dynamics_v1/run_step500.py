"""Replay the validated seed7_A qualification probes at step 500."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_SCRIPT = (
    ROOT / "results/moe_gradient_cancellation_qualification_v1/run_pilot.py"
)
CONFIG = json.loads((HERE / "configuration.json").read_text())
PILOT_CONFIG = json.loads(
    (ROOT / "results/moe_gradient_cancellation_qualification_v1/configuration.json").read_text()
)
if PILOT_CONFIG["cell"] != {"seed": 7, "order": "A", "step": 250}:
    raise RuntimeError("qualification pilot registration differs from expected source")

spec = importlib.util.spec_from_file_location("qualification_collector", SOURCE_SCRIPT)
collector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(collector)

# Reuse the sealed implementation exactly, changing only output and checkpoint identity.
collector.HERE = HERE
collector.CONFIG = {
    **PILOT_CONFIG,
    "experiment": CONFIG["experiment"],
    "cell": {"seed": 7, "order": "A", "step": 500},
}
collector.STEP = 500
collector.SEED = 7
collector.ORDER = "A"
collector.CELL = "seed7_A"

if __name__ == "__main__":
    collector.main()
