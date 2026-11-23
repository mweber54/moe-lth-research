# Exact local rectangle feasibility v1

CPU-only follow-up. GPU training is not invoked. The task replays the frozen
one-million-proposal sampler to identify uncovered degree-control matrices at
dose 0.075, then exhaustively searches ordered high-corner pairs under the
sampler's local first-rectangle rules.

The declared bound is 100 million ordered pairs per matrix. Exhaustion proves
`provably_none`; reaching the bound yields `unresolved`; any witness proves
local feasibility only.
