"""SkillModule: literal growth unit attached to encoder, WM, policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class SkillModule(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden: int = 32) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
            nn.Sigmoid(),
        )
        self.residual = nn.Linear(latent_dim, latent_dim, bias=False)
        nn.init.zeros_(self.residual.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        g = self.gate(z)
        return (z + g * self.residual(z)).squeeze(0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
