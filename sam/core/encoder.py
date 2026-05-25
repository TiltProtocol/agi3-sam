"""GridEncoder: 64x64 grid → latent z."""

from __future__ import annotations

import torch
import torch.nn as nn

from sam.core.adapter import GameAdapter


class GridEncoder(nn.Module):
    """CNN encoder for ARC grids (colors 0-15 one-hot)."""

    def __init__(self, latent_dim: int = 128, num_colors: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.conv = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)
        self.adapter = GameAdapter(latent_dim)

    def grid_to_tensor(self, frame: list[list[list[int]]], device: torch.device | None = None) -> torch.Tensor:
        """Convert frame layers to one-hot tensor [B, C, H, W]."""
        if not frame:
            h, w, c = 64, 64, 1
            grid = torch.zeros(1, c, h, w)
            return grid.to(device) if device else grid
        layer = frame[0]
        h, w = len(layer), len(layer[0]) if layer else 0
        if h == 0 or w == 0:
            t = torch.zeros(1, self.num_colors, 64, 64)
            return t.to(device) if device else t
        tensor = torch.zeros(h, w, dtype=torch.long)
        for y in range(h):
            for x in range(w):
                tensor[y, x] = min(max(int(layer[y][x]), 0), self.num_colors - 1)
        one_hot = torch.nn.functional.one_hot(tensor, self.num_colors).float()
        one_hot = one_hot.permute(2, 0, 1).unsqueeze(0)
        if h != 64 or w != 64:
            one_hot = torch.nn.functional.interpolate(
                one_hot, size=(64, 64), mode="nearest"
            )
        return one_hot.to(device) if device else one_hot

    def forward(self, frame: list[list[list[int]]]) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self.grid_to_tensor(frame, device=device)
        h = self.conv(x)
        z = self.fc(h.flatten(1))
        return self.adapter(z)

    def encode_batch(self, frames: list[list[list[list[int]]]]) -> torch.Tensor:
        tensors = [self.grid_to_tensor(f) for f in frames]
        batch = torch.cat(tensors, dim=0)
        h = self.conv(batch)
        z = self.fc(h.flatten(1))
        return self.adapter(z)
