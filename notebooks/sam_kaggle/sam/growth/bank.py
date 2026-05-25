"""Skill module bank with growth triggers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sam.config import MAX_SKILL_MODULES
from sam.growth.skill_module import SkillModule


@dataclass
class GrowthEvent:
    reason: str
    module_index: int


class SkillModuleBank:
    def __init__(self, latent_dim: int = 128, max_modules: int = MAX_SKILL_MODULES) -> None:
        self.latent_dim = latent_dim
        self.max_modules = max_modules
        self.modules: list[SkillModule] = []
        self.events: list[GrowthEvent] = []
        self._novel_hash_streak = 0
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> SkillModuleBank:
        self.device = device
        for mod in self.modules:
            mod.to(device)
        return self

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        for mod in self.modules:
            if self.device and next(mod.parameters()).device != out.device:
                mod.to(out.device)
            out = mod(out)
        return out

    def can_grow(self) -> bool:
        return len(self.modules) < self.max_modules

    def grow(self, reason: str) -> GrowthEvent | None:
        if not self.can_grow():
            return None
        mod = SkillModule(self.latent_dim)
        if self.device:
            mod.to(self.device)
        self.modules.append(mod)
        evt = GrowthEvent(reason=reason, module_index=len(self.modules) - 1)
        self.events.append(evt)
        return evt

    def on_novel_state(self, is_novel: bool) -> GrowthEvent | None:
        if is_novel:
            self._novel_hash_streak += 1
        else:
            self._novel_hash_streak = 0
        if self._novel_hash_streak >= 10 and self.can_grow():
            self._novel_hash_streak = 0
            return self.grow("novelty_rate")
        return None

    def on_wm_plateau(self, errors: list[float], threshold: float = 0.01) -> GrowthEvent | None:
        if len(errors) < 5:
            return None
        recent = errors[-5:]
        if max(recent) - min(recent) < threshold and self.can_grow():
            return self.grow("wm_plateau")
        return None

    def on_level_success(self) -> GrowthEvent | None:
        return self.grow("level_success")

    def on_game_win(self) -> GrowthEvent | None:
        return self.grow("game_win")

    def total_params(self) -> int:
        return sum(m.param_count() for m in self.modules)

    def modules_state_dict(self) -> list[dict]:
        return [m.state_dict() for m in self.modules]
