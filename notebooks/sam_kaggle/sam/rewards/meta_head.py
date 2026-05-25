"""MetaRewardHead: gated hybrid auxiliary reward dimensions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

COMMIT_CORR_THRESHOLD = 0.3
COMMIT_MIN_EPISODES = 20
MAX_AUX_DIMS = 2


@dataclass
class AuxDimension:
    name: str
    weight: float = 0.05


class MetaRewardHead(nn.Module):
    """Proposes up to 2 auxiliary scalars; commits only if correlated with competence."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.proposer = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, MAX_AUX_DIMS),
            nn.Tanh(),
        )
        self.committed: list[AuxDimension] = []
        self._pending_episodes: list[tuple[list[float], list[float]]] = []

    def propose(self, z: torch.Tensor) -> dict[str, float]:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        raw = self.proposer(z).squeeze(0).detach().flatten()
        out: dict[str, float] = {}
        for i in range(min(MAX_AUX_DIMS, raw.numel())):
            out[f"aux_{i}"] = float(raw[i].item())
        for dim in self.committed:
            out[dim.name] = dim.weight * float(raw[0].item()) if raw.numel() else 0.0
        return out

    def record_episode(self, aux_trace: list[float], competence_trace: list[float]) -> None:
        if len(aux_trace) < 5:
            return
        corr = np.corrcoef(aux_trace, competence_trace)[0, 1]
        if np.isnan(corr):
            return
        if abs(corr) >= COMMIT_CORR_THRESHOLD and len(self.committed) < MAX_AUX_DIMS:
            name = f"committed_aux_{len(self.committed)}"
            self.committed.append(AuxDimension(name=name, weight=0.05 * np.sign(corr)))

    def state_dict_committed(self) -> list[dict]:
        return [{"name": d.name, "weight": d.weight} for d in self.committed]
