"""World model p(z' | z, a)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sam.core.adapter import GameAdapter


class WorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        num_actions: int = 8,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_embed = nn.Embedding(num_actions, 32)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        self.adapter = GameAdapter(latent_dim)
        self.confidence_head = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor, action_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        a = torch.tensor([action_id], device=z.device, dtype=torch.long)
        ae = self.action_embed(a).expand(z.size(0), -1)
        z_next = self.net(torch.cat([z, ae], dim=-1))
        z_next = self.adapter(z_next)
        conf = torch.sigmoid(self.confidence_head(z_next))
        return z_next.squeeze(0), conf.squeeze()

    def predict_error(self, z: torch.Tensor, z_target: torch.Tensor, action_id: int) -> float:
        z_pred, _ = self.forward(z, action_id)
        if z_target.dim() == 1:
            z_target = z_target.unsqueeze(0)
        return float(torch.nn.functional.mse_loss(z_pred, z_target).item())
