"""Bundled SAM core: encoder, world model, policy, value."""

from __future__ import annotations

import torch
import torch.nn as nn

from sam.config import MAX_TOTAL_PARAMS
from sam.core.encoder import GridEncoder
from sam.core.policy import ActionHead
from sam.core.value import ValueHead
from sam.core.world_model import WorldModel
from sam.growth.bank import SkillModuleBank
from sam.utils import count_params


class SamCore(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = GridEncoder(latent_dim=latent_dim)
        self.world_model = WorldModel(latent_dim=latent_dim)
        self.policy = ActionHead(latent_dim=latent_dim)
        self.value = ValueHead(latent_dim=latent_dim)
        self.skills = SkillModuleBank(latent_dim=latent_dim)

    def to(self, device: torch.device) -> SamCore:
        super().to(device)
        self.skills.to(device)
        return self

    def encode(self, frame: list[list[list[int]]]) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder(frame)
            return self.skills.apply(z)

    def total_params(self) -> int:
        base = (
            count_params(self.encoder)
            + count_params(self.world_model)
            + count_params(self.policy)
            + count_params(self.value)
        )
        return base + self.skills.total_params()

    def within_budget(self) -> bool:
        return self.total_params() <= MAX_TOTAL_PARAMS
