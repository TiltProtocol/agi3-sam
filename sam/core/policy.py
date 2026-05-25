"""Action head with dynamic masking."""

from __future__ import annotations

import torch
import torch.nn as nn

from arcengine import GameAction


class ActionHead(nn.Module):
    def __init__(self, latent_dim: int = 128, num_actions: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.logits = nn.Linear(latent_dim, num_actions)
        self.coord_head = nn.Linear(latent_dim, 64 * 64)

    def masked_logits(
        self, z: torch.Tensor, available_actions: list[int]
    ) -> torch.Tensor:
        if z.dim() == 1:
            z = z.unsqueeze(0)
        raw = self.logits(z).squeeze(0)
        mask = torch.full((self.num_actions,), float("-inf"), device=raw.device)
        for aid in available_actions:
            if 0 <= aid < self.num_actions:
                mask[aid] = 0.0
        return raw + mask

    def select_action(
        self,
        z: torch.Tensor,
        available_actions: list[int],
        temperature: float = 1.0,
    ) -> int:
        logits = self.masked_logits(z, available_actions)
        if temperature <= 0:
            return int(logits.argmax().item())
        probs = torch.softmax(logits / temperature, dim=-1)
        idx = int(torch.multinomial(probs, 1).item())
        return idx

    def to_game_action(self, action_id: int, game_id: str) -> GameAction:
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.action_data.game_id = game_id
        elif action.is_complex():
            flat = self.coord_head
            action.set_data({"game_id": game_id, "x": 32, "y": 32})
        return action
