"""Bundled SAM neural core with param counting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from sam.config import MAX_TOTAL_PARAMS
from sam.core.encoder import GridEncoder
from sam.core.policy import ActionHead
from sam.core.value import ValueHead
from sam.core.world_model import WorldModel
from sam.growth.bank import SkillModuleBank


@dataclass
class SAMCore:
    encoder: GridEncoder
    world_model: WorldModel
    policy: ActionHead
    value: ValueHead
    skill_bank: SkillModuleBank
    device: torch.device

    @classmethod
    def create(cls, latent_dim: int = 128, device: str | None = None) -> SAMCore:
        dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        encoder = GridEncoder(latent_dim=latent_dim).to(dev)
        wm = WorldModel(latent_dim=latent_dim).to(dev)
        policy = ActionHead(latent_dim=latent_dim).to(dev)
        value = ValueHead(latent_dim=latent_dim).to(dev)
        bank = SkillModuleBank(latent_dim=latent_dim)
        return cls(encoder, wm, policy, value, bank, dev)

    def param_count(self) -> int:
        core = sum(
            p.numel()
            for m in (self.encoder, self.world_model, self.policy, self.value)
            for p in m.parameters()
        )
        return core + self.skill_bank.total_params()

    def assert_param_budget(self) -> None:
        n = self.param_count()
        if n > MAX_TOTAL_PARAMS:
            raise ValueError(f"SAM param count {n} exceeds cap {MAX_TOTAL_PARAMS}")

    def micro_update(
        self,
        z: torch.Tensor,
        z_next: torch.Tensor,
        action_id: int,
        reward_vec: torch.Tensor,
        lr: float = 1e-4,
        modules_only: bool = False,
    ) -> float:
        self.encoder.train()
        self.world_model.train()
        self.value.train()
        z_pred, _ = self.world_model(z, action_id)
        loss = nn.functional.mse_loss(z_pred, z_next.detach())
        _, pred_r = self.value(z)
        if pred_r.shape != reward_vec.shape:
            reward_vec = reward_vec[: pred_r.shape[-1]]
        loss = loss + 0.1 * nn.functional.mse_loss(pred_r, reward_vec.detach())
        params = []
        if modules_only and self.skill_bank.modules:
            params = list(self.skill_bank.modules[-1].parameters())
        else:
            params = (
                list(self.world_model.parameters())
                + list(self.value.parameters())
            )
        if not params:
            return 0.0
        opt = torch.optim.Adam(params, lr=lr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "world_model": self.world_model.state_dict(),
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                "skill_modules": self.skill_bank.modules_state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            return
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(data["encoder"])
        self.world_model.load_state_dict(data["world_model"])
        self.policy.load_state_dict(data["policy"])
        self.value.load_state_dict(data["value"])
