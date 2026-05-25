"""Checkpoint save/load for SAM core."""

from __future__ import annotations

from pathlib import Path

import torch

from sam.core.sam_core import SamCore


def save_sam_checkpoint(core: SamCore, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder": core.encoder.state_dict(),
        "world_model": core.world_model.state_dict(),
        "policy": core.policy.state_dict(),
        "value": core.value.state_dict(),
        "skills": core.skills.modules_state_dict(),
        "total_params": core.total_params(),
    }
    torch.save(payload, path)


def load_sam_checkpoint(
    core: SamCore,
    path: str | Path,
    *,
    device: torch.device | None = None,
) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    core.encoder.load_state_dict(payload["encoder"])
    core.world_model.load_state_dict(payload["world_model"])
    core.policy.load_state_dict(payload["policy"])
    core.value.load_state_dict(payload["value"])
    for i, state in enumerate(payload.get("skills", [])):
        while len(core.skills.modules) <= i:
            core.skills.grow("checkpoint_restore")
        core.skills.modules[i].load_state_dict(state)
