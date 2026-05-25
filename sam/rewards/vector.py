"""Seven-dimensional reward vector with anti-hack guards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arcengine import FrameData, GameState

from sam.planner.state_graph import frame_hash

DIM_NAMES = (
    "curiosity",
    "efficiency",
    "intuition",
    "proximity",
    "savvy",
    "competence",
    "recovery",
)

EFFICIENCY_FLOOR = 0.3


@dataclass
class RewardVector:
    curiosity: float = 0.0
    efficiency: float = 0.0
    intuition: float = 0.0
    proximity: float = 0.0
    savvy: float = 0.0
    competence: float = 0.0
    recovery: float = 0.0
    aux: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        out = {k: getattr(self, k) for k in DIM_NAMES}
        for i, v in enumerate(self.aux):
            out[f"aux_{i}"] = v
        return out

    def as_array(self) -> np.ndarray:
        base = np.array([getattr(self, k) for k in DIM_NAMES], dtype=np.float32)
        if self.aux:
            base = np.concatenate([base, np.array(self.aux, dtype=np.float32)])
        return base

    def weighted_sum(self, weights: np.ndarray | None = None) -> float:
        w = weights if weights is not None else np.ones(len(DIM_NAMES), dtype=np.float32)
        w = w[: len(DIM_NAMES)]
        w[1] = max(w[1], EFFICIENCY_FLOOR)
        vals = np.array([getattr(self, k) for k in DIM_NAMES], dtype=np.float32)
        return float(np.dot(w, vals))


class RewardVectorEngine:
    def __init__(self) -> None:
        self.hash_visits: dict[str, int] = {}
        self._prev_levels = 0
        self._prev_value = 0.0
        self._last_actions: list[int] = []
        self._game_over_seen = False
        self.weights = np.array([0.5, 1.0, 0.4, 0.6, 0.3, 2.0, 0.2], dtype=np.float32)

    def _curiosity(self, h: str) -> float:
        count = self.hash_visits.get(h, 0)
        self.hash_visits[h] = count + 1
        if count == 0:
            return 1.0
        return 0.0

    def step_env(
        self,
        before: FrameData,
        after: FrameData,
        action_id: int,
        wm_confidence_delta: float = 0.0,
        value_delta: float = 0.0,
    ) -> RewardVector:
        h = frame_hash(after)
        rv = RewardVector()
        rv.curiosity = self._curiosity(h)
        rv.efficiency = -1.0
        rv.intuition = float(np.clip(wm_confidence_delta, -1.0, 1.0))
        if after.levels_completed > before.levels_completed:
            rv.proximity = 1.0
            rv.competence = 10.0
        else:
            rv.proximity = float(np.clip(value_delta, -1.0, 1.0))
            if value_delta > 0 and h in self.hash_visits and self.hash_visits[h] > 2:
                rv.proximity = 0.0
        if len(self._last_actions) >= 3 and len(set(self._last_actions[-3:])) == 1:
            rv.savvy = -0.5
        else:
            rv.savvy = 0.2 if action_id in after.available_actions else -0.2
        if after.state == GameState.GAME_OVER:
            self._game_over_seen = True
            rv.competence = -2.0
        if self._game_over_seen and after.state == GameState.NOT_FINISHED:
            rv.recovery = 0.5
            self._game_over_seen = False
        self._prev_levels = after.levels_completed
        self._last_actions.append(action_id)
        self._last_actions = self._last_actions[-10:]
        return rv

    def step_internal(self, action_id: int, wm_confidence: float = 0.0) -> RewardVector:
        rv = RewardVector()
        rv.efficiency = -0.05
        rv.intuition = float(np.clip(wm_confidence - 0.5, -0.5, 0.5))
        rv.savvy = 0.1
        self._last_actions.append(action_id)
        return rv
