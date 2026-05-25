"""RewardVectorEngine: 7 fixed dimensions + scalar score."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REWARD_DIM_NAMES = (
    "curiosity",
    "efficiency",
    "intuition",
    "proximity",
    "savvy",
    "competence",
    "recovery",
)

DEFAULT_WEIGHTS = np.array([0.15, 0.35, 0.1, 0.15, 0.05, 0.15, 0.05], dtype=np.float32)
EFFICIENCY_FLOOR = 0.25


@dataclass
class RewardVector:
    values: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    aux: dict[str, float] = field(default_factory=dict)
    is_internal: bool = False

    def as_dict(self) -> dict[str, float]:
        out = {name: float(self.values[i]) for i, name in enumerate(REWARD_DIM_NAMES)}
        out.update(self.aux)
        return out


class RewardVectorEngine:
    """Hand-designed 7-dim reward with anti-hack guards."""

    def __init__(self) -> None:
        self.weights = DEFAULT_WEIGHTS.copy()
        self._visited_hashes: set[str] = set()
        self._prev_wm_conf: float = 0.5
        self._prev_value: float = 0.0
        self._action_outcomes: dict[tuple[str, int], set[str]] = {}
        self._last_hash: str | None = None
        self._noop_streak = 0

    def reset_level(self) -> None:
        self._prev_wm_conf = 0.5
        self._prev_value = 0.0
        self._last_hash = None
        self._noop_streak = 0

    def compute(
        self,
        *,
        state_hash: str,
        prev_hash: str | None,
        wm_confidence: float,
        value_delta: float,
        level_before: int,
        level_after: int,
        is_env_step: bool,
        is_internal: bool,
        game_over: bool,
        reset_after_over: bool,
        grid_changed: int,
        action_id: int | None = None,
    ) -> RewardVector:
        r = np.zeros(7, dtype=np.float32)

        # curiosity — first visit only
        if state_hash not in self._visited_hashes:
            r[0] = 1.0
            self._visited_hashes.add(state_hash)
        else:
            r[0] = 0.0

        # efficiency
        if is_env_step:
            r[1] = -1.0
        elif is_internal:
            r[1] = -0.05

        # intuition — WM confidence delta
        conf_delta = wm_confidence - self._prev_wm_conf
        r[2] = float(np.clip(conf_delta, -0.5, 0.5))
        self._prev_wm_conf = wm_confidence

        # proximity — value head delta + level progress
        prox = float(np.clip(value_delta, -1.0, 1.0))
        if level_after > level_before:
            prox += 2.0
        if prev_hash and state_hash == prev_hash and grid_changed == 0:
            prox = min(prox, 0.0)
        r[3] = prox
        self._prev_value += value_delta

        # savvy — action splits outcomes
        if action_id is not None and prev_hash is not None:
            key = (prev_hash, action_id)
            outcomes = self._action_outcomes.setdefault(key, set())
            outcomes.add(state_hash)
            if len(outcomes) > 1:
                r[4] = 0.3
            elif grid_changed > 0:
                r[4] = 0.1
            else:
                r[4] = -0.2
                self._noop_streak += 1
        else:
            r[4] = 0.0

        # competence — level win (env only)
        if is_env_step and level_after > level_before:
            r[5] = 5.0

        # recovery
        if game_over:
            r[5] -= 2.0
        if reset_after_over:
            r[6] = 0.5

        self._last_hash = state_hash
        vec = RewardVector(values=r, is_internal=is_internal)
        return vec

    def scalar(self, vec: RewardVector, aux: dict[str, float] | None = None) -> float:
        w = self.weights.copy()
        w[1] = max(w[1], EFFICIENCY_FLOOR)
        total = float(np.dot(w, vec.values))
        if aux:
            for v in aux.values():
                total += 0.05 * v
        return total

    def adapt_weights(self, meta_grad: np.ndarray, lr: float = 0.01) -> None:
        self.weights = self.weights + lr * meta_grad
        self.weights = np.clip(self.weights, 0.01, 1.0)
        self.weights /= self.weights.sum()
