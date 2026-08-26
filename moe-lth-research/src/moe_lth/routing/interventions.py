from __future__ import annotations

import torch

from moe_lth.models.router import RouteOverride

from .route_history import RouteHistory


SUPPORTED_MODES = {
    "learned",
    "fixed_random",
    "random_every_step",
    "replay",
    "swapped",
    "shuffled_usage",
    "deconfounded_shuffle",
    "graded_corruption",
}


class RoutingController:
    """Controls routing interventions during training.

    Supported modes:
      - learned: normal learned routing (no override)
      - fixed_random: frozen random router weights
      - random_every_step: fully random balanced routing each step
      - replay: exact replay of archived routing history
      - swapped: replay with expert ID swaps/shifts
      - shuffled_usage: replay with per-step global permutation (preserves counts)
      - deconfounded_shuffle: replay with per-expert identity shuffle (preserves counts)
      - graded_corruption: replay with parameterized fraction of assignments corrupted
    """

    def __init__(
        self,
        mode: str,
        num_layers: int,
        num_experts: int,
        seed: int,
        history: RouteHistory | None = None,
        swap_pairs: list[list[int]] | None = None,
        layer_swap_pairs: dict[int, list[list[int]]] | None = None,
        cyclic_shift: int = 0,
        layer_cyclic_shifts: dict[int, int] | None = None,
        corruption_fraction: float = 0.0,
    ):
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported routing mode {mode!r}. Expected one of {sorted(SUPPORTED_MODES)}")
        self.mode = mode
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.seed = seed
        self.history = history
        self.swap_pairs = swap_pairs or []
        self.layer_swap_pairs = layer_swap_pairs or {}
        self.cyclic_shift = int(cyclic_shift)
        self.layer_cyclic_shifts = layer_cyclic_shifts or {}
        self.corruption_fraction = float(corruption_fraction)

    def overrides(self, token_ids: torch.Tensor, step: int) -> list[torch.Tensor | None]:
        if self.mode in {"learned", "fixed_random"}:
            return [None] * self.num_layers
        return [self._layer_override(token_ids, step, layer_id) for layer_id in range(self.num_layers)]

    def _layer_override(self, token_ids: torch.Tensor, step: int, layer_id: int) -> torch.Tensor:
        if self.mode == "random_every_step":
            generator = torch.Generator(device=token_ids.device)
            generator.manual_seed(self.seed + step * 1009 + layer_id * 9176)
            total = token_ids.numel()
            balanced = torch.arange(total, device=token_ids.device).remainder(self.num_experts)
            permutation = torch.randperm(total, generator=generator, device=token_ids.device)
            return balanced[permutation].reshape_as(token_ids)

        if self.history is None:
            raise ValueError(f"Routing mode {self.mode!r} requires a replay history.")

        if hasattr(self.history, "get_primary_ids"):
            routes = self.history.get_primary_ids(step, layer_id, token_ids.device)
        else:
            routes = self.history.get(step, layer_id, token_ids.device)
        if routes.ndim == 1 and routes.numel() == token_ids.numel():
            routes = routes.reshape(token_ids.shape)
        if routes.shape != token_ids.shape:
            raise ValueError(
                f"Replay shape {tuple(routes.shape)} does not match batch shape {tuple(token_ids.shape)}."
            )

        if self.mode == "deconfounded_shuffle" and hasattr(self.history, "traces"):
            trace = self.history.get(step, layer_id)
            if hasattr(trace, "selected_expert_ids") and hasattr(trace, "gate_values"):
                primary = torch.from_numpy(trace.selected_expert_ids[:, 0].astype(__import__('numpy').int64)).to(token_ids.device)
                gate = torch.from_numpy(trace.gate_values.astype(__import__('numpy').float32)).to(token_ids.device)
                accepted = torch.from_numpy(trace.accepted_mask.astype(bool)).to(token_ids.device)
                shuffled, preserved_gate, preserved_accept = self._deconfounded_trace_shuffle(
                    primary,
                    gate,
                    accepted,
                    step,
                    layer_id,
                )
                if shuffled.numel() == token_ids.numel():
                    return RouteOverride(
                        expert_ids=shuffled.reshape(token_ids.shape),
                        gate_values=preserved_gate.reshape(token_ids.shape[0], token_ids.shape[1], 1),
                    )
                return self.transform_replayed(routes, step, layer_id)

        return self.transform_replayed(routes, step, layer_id)

    def _deconfounded_trace_shuffle(
        self,
        primary_ids: torch.Tensor,
        gate_values: torch.Tensor,
        accepted_mask: torch.Tensor,
        step: int,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .deconfounded import deconfounded_identity_shuffle

        shuffled, preserved_gate, preserved_accept = deconfounded_identity_shuffle(
            selected_expert_ids=primary_ids.reshape(primary_ids.shape[0], 1),
            gate_values=gate_values,
            accepted_mask=accepted_mask,
            step=step,
            layer_id=layer_id,
            seed=self.seed,
        )
        return shuffled.reshape_as(primary_ids), preserved_gate, preserved_accept

    def transform_replayed(self, routes: torch.Tensor, step: int, layer_id: int) -> torch.Tensor:
        if self.mode == "swapped":
            routes = routes.clone()
            for first, second in self.swap_pairs + self.layer_swap_pairs.get(layer_id, []):
                first_mask = routes == first
                second_mask = routes == second
                routes[first_mask] = second
                routes[second_mask] = first
            shift = self.layer_cyclic_shifts.get(layer_id, self.cyclic_shift)
            if shift:
                routes = (routes + int(shift)) % self.num_experts
            return routes

        if self.mode == "shuffled_usage":
            generator = torch.Generator(device=routes.device)
            generator.manual_seed(self.seed + step * 1009 + layer_id * 9176)
            flat = routes.flatten()
            return flat[torch.randperm(flat.numel(), generator=generator, device=flat.device)].reshape_as(routes)

        if self.mode == "deconfounded_shuffle":
            from .deconfounded import deconfounded_identity_shuffle_flat
            return deconfounded_identity_shuffle_flat(
                primary_ids=routes,
                step=step,
                layer_id=layer_id,
                seed=self.seed,
                num_experts=self.num_experts,
            )

        if self.mode == "graded_corruption":
            from .deconfounded import graded_route_corruption
            return graded_route_corruption(
                primary_ids=routes,
                corruption_fraction=self.corruption_fraction,
                step=step,
                layer_id=layer_id,
                seed=self.seed,
                num_experts=self.num_experts,
                preserve_counts=True,
            )

        return routes
