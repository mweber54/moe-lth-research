# Controlled token–expert matching

Protocol, audit, validation, frozen registration and findings live in `results/token_expert_matching_v1`. Native sealed artifacts are read-only. `replay.py` implements complete top1 slots and native recipient auxiliary gradients. `validate.py` runs canonical GPU equivalence, tracing/RNG, backward, reset and interrupted replay checks. `run.py` refuses production until validation is marked ready, enforces fresh attempts and checks every trace checksum. Use canonical `.venv-highres/bin/python`; CPU unit tests run in `.venv`.

```
.venv/bin/python -m pytest -q tests/test_token_expert_matching.py
.venv-highres/bin/python scripts/token_expert_matching_v1/validate.py
.venv-highres/bin/python scripts/token_expert_matching_v1/run.py --variant e32_seed17_dense --kind donor
```

Do not manually bypass `validation.json`, silently resume an uncertain attempt, infer training routes from probes, or expand before the seed17 pilot feasibility decision. Native segment interruption checkpoints are preserved; the production wrapper intentionally rejects automatic attempt reuse.
