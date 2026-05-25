"""Low-rank per-game adapter on latent vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class GameAdapter(nn.Module):
    """LoRA-style adapter: z' = z + scale * (B @ A @ z)."""

    def __init__(self, dim: int, rank: int = 16, scale: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.scale = scale
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.scale * self.up(self.down(z))
