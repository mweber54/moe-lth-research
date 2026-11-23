"""Load the archived runner and verify study-local module identities."""
import importlib.util
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
PILOT = OUT.with_name("imp_early_sign_reinforcement_v1")


def load(name, path):
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    return module


pilot = load("neuron_matched_pilot_run", PILOT / "run.py")
assert pilot.OUT == PILOT
