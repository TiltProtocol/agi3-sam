"""Value head for scalar and vector reward prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

REWARD_DIMS = 7


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.scalar = nn.Linear(latent_dim, 1)
        self.vector = nn.Linear(latent_dim, REWARD_DIMS)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.scalar(z).squeeze(-1), self.vector(z)
